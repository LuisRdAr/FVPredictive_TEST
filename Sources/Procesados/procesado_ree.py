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
    consulta_sql = f"SELECT COUNT(*) FROM {schema_name}.ree_raw WHERE procesado = false;"
    count = pd.read_sql_query(consulta_sql, engine).values[0][0]
    if count == 0:
        print(f"No se han encontrado nuevos registros para procesar en {schema_name}.contador_red_raw")
        sys.exit()
        
    chunk_size = 50000
    for i in tqdm(range(0, count, chunk_size), total = math.ceil(count/chunk_size)):
        # Lectura de los datos de meteo sin procesar limitados por el chunk_size
        consulta_sql = f"SELECT * FROM {schema_name}.ree_raw WHERE procesado = false ORDER BY id LIMIT {chunk_size};"
        ree_df = pd.read_sql_query(consulta_sql, engine)

        # Comprobación si el id del dispositivo está ya registrado y registro en caso de no ser así
        for id in ree_df["dispositivo_id"].unique():
            id_int = int(id)
            cur.execute(f"""SELECT * FROM {schema_name}.dispositivos WHERE dispositivo_id = %s;""", (id_int,))
            resultados = cur.fetchall()
            if not resultados:
                print("Generando entradas de nuevos dispositivos")
                dispositivo = ree_df[(ree_df["dispositivo_id"] == id_int)][["parque_id", 
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
        ree_df = ree_df.drop(columns = ["parque_id",
                            "descripcion_parque", 
                            "localizacion_parque",
                            "potencia_max", 
                            "num_paneles"])
        ree_df = ree_df.drop(columns = ["nombre_dispositivo", 
                            "ref", 
                            "descripcion_dispositivo", 
                            "ubicacion"])

        # Iteración sobre el dataframe resultante, volcado en la tabla ree y actualización de la tabla raw
        for row in tqdm(ree_df.itertuples(), position = 1, leave=False, total = ree_df.shape[0]):
            try:
                cur.execute(f"""INSERT INTO {schema_name}.ree(
                        id, dispositivo_id, datetime_utc, med_id, status, alarma, motivo,
                        consigna_pot_act_ree, consigna_pot_act_planta, consigna_fdp_ree, consigna_fdp_planta)
                    VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                    (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]))
                cur.execute(f"""UPDATE {schema_name}.ree_raw
                        SET procesado = true,
                            datetime_procesado = NOW()
                        WHERE id = %s""",
                        (row[1],))
                conn.commit()
            except psycopg2.Error as e:
                print("Error:", e)
                print("\tID:", row[1])
                conn.rollback()
        
        del(ree_df)
    cur.close()
    conn.close()
