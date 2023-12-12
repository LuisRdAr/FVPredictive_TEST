import pandas as pd
import math
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm
import json
import sys
import os

if __name__ == "__main__":
    # Carga, desde la ruta de ejecución, de los parámetros para conexión a la base de datos  
    root_path = os.getcwd()
    params = None
    for filename in os.listdir(root_path):
        if "params.json" in filename:
            with open(os.path.join(root_path, filename)) as f:
                params = json.load(f)
    if params is None:
        print("No se ha encontrado el archivo de parámetros para la conexión a la base de datos")
        sys.exit()
    data_path = os.path.join(root_path, params["data_path"])
    schema_name = params["schema"]

    # Conexión con la base de datos
    try:
        password = params['password'].replace('@', '%40')
        engine = create_engine(f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}')
        conn = psycopg2.connect(
            database = params['dbname'],
            user = params['user'],
            password = params['password'],
            host = params['host'],  
            port = params['port'])
        cur = conn.cursor()
    except psycopg2.Error as e:
        print("Error al conectarse a PostgreSQL:", e)
        sys.exit()

    # Lectura del número de registros a procesar para proceder por trozos
    consulta_sql = f"SELECT COUNT(*) FROM {schema_name}.inversores_raw WHERE procesado = false;"
    count = pd.read_sql_query(consulta_sql, engine).values[0][0]
    if count == 0:
        print(f"No se han encontrado nuevos registros para procesar en {schema_name}.inversores_raw")
        sys.exit()

    chunk_size = 50000
    for i in tqdm(range(0, count, chunk_size), total = math.ceil(count/chunk_size)):
        # Lectura de los datos de meteo sin procesar limitados por el chunk_size
        consulta_sql = f"SELECT * FROM {schema_name}.inversores_raw WHERE procesado = false ORDER BY id LIMIT {chunk_size};"
        inv_df = pd.read_sql_query(consulta_sql, engine)
        if inv_df.empty:
            continue
            
        # Comprobación si el id del dispositivo está ya registrado y registro en caso de no ser así
        for id in inv_df["dispositivo_id"].unique():
            id_int = int(id)
            cur.execute(f"""SELECT * FROM {schema_name}.dispositivos WHERE dispositivo_id = %s;""", (id_int,))
            resultados = cur.fetchall()
            if not resultados:
                print("Generando entradas de nuevos dispositivos")
                dispositivo = inv_df[(inv_df["dispositivo_id"] == id_int)][["parque_id", 
                                                                            "dispositivo_id", 
                                                                            "nombre_dispositivo", 
                                                                            "ref", 
                                                                            "ubicacion", 
                                                                            "descripcion_dispositivo"]].drop_duplicates()
                cur.execute(f"""INSERT INTO {schema_name}.dispositivos
                            VALUES(%s, %s, %s, %s, %s, %s);""", 
                            tuple(attr for attr in dispositivo.values[0]))
            conn.commit()
        
        # Descarte de parámetros redundantes (relativos a la tabla parque o dispositivos)
        inv_df = inv_df.drop(columns = ["parque_id",
                            "descripcion_parque", 
                            "localizacion_parque",
                            "potencia_max", 
                            "num_paneles"])
        inv_df = inv_df.drop(columns = ["nombre_dispositivo", 
                            "ref", 
                            "descripcion_dispositivo", 
                            "ubicacion"])
        
        # Separacion del DataFrame inversores e inversores_detalle
        column_list = ["id"]+inv_df.columns[inv_df.columns.str.contains("amp_dc_in")].to_list()
        amp_df = inv_df[column_list].copy()
        inv_df = inv_df.drop(columns = inv_df.columns[inv_df.columns.str.contains("amp_dc_in")])

        amp_df = amp_df.melt(id_vars = ["id"])
        amp_df = amp_df.rename(columns={"variable":"entrada_id", "value": "amp_dc"})
        amp_df = amp_df.dropna()
        amp_df["entrada_id"] = amp_df["entrada_id"].apply(lambda x: x[9:])
        amp_df = amp_df.reset_index(drop = True)

        # Iteración sobre el dataframe resultante, volcado en las tablas inversores e 
        # inversores_detalle y actualización de la tabla raw
        for row in tqdm(inv_df.itertuples(), position = 1, leave = False, total = inv_df.shape[0]):
            try:
                cur.execute(f"""INSERT INTO {schema_name}.inversores(
                            id, dispositivo_id, datetime_utc, status, alarma, estado, 
                            potencia_act, potencia_rea, cos_phi, vol_12, vol_13, vol_23, amp_1, 
                            amp_2, amp_3, frec, lim_act, lim_rea, vol_dc_bus, aisl_dc, energia_dia)
                        VALUES 
                            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s);""",
                        (row[1], row[2], row[3], row[5], row[6], row[7], row[8], row[9], row[10], row[11],
                        row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22]))
                amp_id_df = amp_df[amp_df["id"] == row[1]]
                for row_amp in amp_id_df.itertuples():
                    cur.execute(f"""INSERT INTO {schema_name}.inversores_detalle(
                            id, entrada_id, amp_dc)
                        VALUES 
                            (%s, %s, %s);""",
                        (row_amp[1], row_amp[2], row_amp[3]))
                cur.execute(f"""UPDATE {schema_name}.inversores_raw
                            SET procesado = true,
                                datetime_procesado = NOW()
                            WHERE id = %s""",
                            (row[1],))
                conn.commit()
            except psycopg2.Error as e:
                print("Error:", e)
                print("\tID:", row[1])
                conn.rollback()

        del(inv_df, amp_df)
    cur.close()
    conn.close()
