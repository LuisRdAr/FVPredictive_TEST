import pandas as pd
import math
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine
from tqdm import tqdm
import json
import sys
import os

if __name__ == "__main__":
    # Carga, desde la ruta de ejecución, de los parámetros para conexión a la base de datos  
    ROOT_PATH = os.getcwd()
    params = None
    for filename in os.listdir(ROOT_PATH):
        if "params.json" in filename:
            with open(os.path.join(ROOT_PATH, filename)) as f:
                params = json.load(f)
    if params is None:
        print("No se ha encontrado el archivo de parámetros para la conexión a la base de datos")
        sys.exit()
    else:
        print(f"Parámetros de la planta {params['schema'].capitalize()} cargados correctamente")
    data_path = os.path.join(ROOT_PATH, params["data_path"])
    SCHEMA_NAME = params["schema"]

    # Conexión con la base de datos
    try:
        PASSWORD = params['password'].replace('@', '%40')
        ENGINE_STRING = f'postgresql://{params["user"]}:{PASSWORD}' +  \
            f'@{params["host"]}:{params["port"]}/{params["dbname"]}'
        engine = create_engine(ENGINE_STRING)
        conn = psycopg2.connect(
            database = params['dbname'],
            user = params['user'],
            password = params['password'],
            host = params['host'],  
            port = params['port'])
        cur = conn.cursor()
        print(f"Conexión a la base de datos {params['dbname']} (esquema {SCHEMA_NAME}) establecida")
    except Exception as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    # Lectura del número de registros a procesar para proceder por trozos
    consulta_sql = f"SELECT COUNT(*) FROM {SCHEMA_NAME}.inversores_raw WHERE procesado = false;"
    count = pd.read_sql_query(consulta_sql, engine).values[0][0]
    if count == 0:
        print(f"No se han encontrado nuevos registros para procesar en {SCHEMA_NAME}.inversores_raw")
        sys.exit()

    CHUNK_SIZE = 5000000
    for i in tqdm(range(0, count, CHUNK_SIZE), total = math.ceil(count/CHUNK_SIZE)):
        # Lectura de los datos de meteo sin procesar limitados por el chunk_size
        consulta_sql = f"""SELECT * 
                            FROM {SCHEMA_NAME}.inversores_raw 
                            WHERE procesado = false 
                            ORDER BY id LIMIT {CHUNK_SIZE};"""
        inv_df = pd.read_sql_query(consulta_sql, engine)
        inv_df = inv_df.drop(columns = ["procesado", "datetime_procesado"])
            
        # Comprobación si el id del dispositivo está ya registrado y registro en caso de no ser así
        for pair in inv_df[["parque_id", "dispositivo_id"]].drop_duplicates().values:
            id_par = int(pair[0])
            id_inv = int(pair[1])
            cur.execute(f"""SELECT * 
                            FROM {SCHEMA_NAME}.dispositivos 
                            WHERE parque_id = {id_par} 
                                AND dispositivo_id = {id_inv};""")
            resultados = cur.fetchall()
            if not resultados:
                print("Generando entradas de nuevos dispositivos")
                dispositivo = inv_df[(inv_df["parque_id"] == id_par) & 
                                    (inv_df["dispositivo_id"] == id_inv)][["parque_id", 
                                                                        "dispositivo_id", 
                                                                        "nombre_dispositivo", 
                                                                        "ref", 
                                                                        "ubicacion", 
                                                                        "descripcion_dispositivo"]]\
                                                                            .drop_duplicates()
                print(dispositivo)
                cur.execute(f"""INSERT INTO {SCHEMA_NAME}.dispositivos
                            VALUES(%s, %s, %s, %s, %s, %s);""", 
                            tuple(attr for attr in dispositivo.values[0]))
            conn.commit()
        
        # Descarte de parámetros redundantes (relativos a la tabla parque o dispositivos)
        inv_df.drop(columns = ["descripcion_parque", 
                            "localizacion_parque",
                            "potencia_max", 
                            "num_paneles"], inplace = True)
        inv_df.drop(columns = ["nombre_dispositivo", 
                            "ref", 
                            "descripcion_dispositivo", 
                            "ubicacion"], inplace = True)
        
        # Separacion del DataFrame inversores e inversores_detalle
        amp_columns = inv_df.filter(like = "amp_dc_in").columns
        column_list = ["id"] + amp_columns.tolist()
        amp_df = inv_df[column_list].copy()
        inv_df = inv_df.drop(columns = amp_columns)

        amp_df = amp_df.melt(id_vars = ["id"])
        amp_df = amp_df.rename(columns={"variable":"entrada_id", "value": "amp_dc"})
        amp_df = amp_df.dropna()
        amp_df["entrada_id"] = amp_df["entrada_id"].apply(lambda x: x[9:])
        amp_df = amp_df.reset_index(drop = True)

        # Escritura de los datos en la base de datos
        try:
            dtypes_inv = {
                'id': sqlalchemy.types.INTEGER(),
                'parque_id': sqlalchemy.types.SMALLINT(),
                'dispositivo_id': sqlalchemy.types.SMALLINT(),
                'datetime_utc': sqlalchemy.types.DateTime(timezone=True),
                'med_id': sqlalchemy.types.INTEGER(),
                'status': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'alarma': sqlalchemy.types.SMALLINT(),
                'estado': sqlalchemy.types.SMALLINT(),
                'potencia_act': sqlalchemy.types.SMALLINT(),
                'potencia_rea': sqlalchemy.types.SMALLINT(),
                'cos_phi': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'vol_12': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'vol_13': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'vol_23': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'amp_1': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'amp_2': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'amp_3': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'frec': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'lim_act': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'lim_rea': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'vol_dc_bus': sqlalchemy.types.SMALLINT(),
                'aisl_dc': sqlalchemy.types.INTEGER(),
                'energia_dia': sqlalchemy.types.INTEGER()}
            keys_inv = list(dtypes_inv.keys())
            inv_df = inv_df.drop(columns=[col for col in inv_df.columns if col not in keys_inv])
            inv_df.to_sql('inversores', 
                          engine, 
                          if_exists = 'append', 
                          index = False, 
                          schema = SCHEMA_NAME, 
                          dtype = dtypes_inv, 
                          chunksize = 100000)
        except Exception as error:
            print(f"Error en la escritura de datos en la tabla inversores: \n\t{error}")
            sys.exit()
        
        try:
            dtypes_amp = {
                'id': sqlalchemy.types.INTEGER(),
                'entrada_id': sqlalchemy.types.SMALLINT(),
                'amp_dc': sqlalchemy.types.Float(precision=3, asdecimal=True)}
            keys_amp = list(dtypes_amp.keys())
            amp_df = amp_df.drop(columns=[col for col in amp_df.columns if col not in keys_amp])
            amp_df.to_sql('inversores_detalle', 
                          engine, 
                          if_exists = 'append', 
                          index = False, 
                          schema = SCHEMA_NAME, 
                          dtype = dtypes_amp, chunksize = 100000)
        except Exception as error:
            print(f"Error en la escritura de datos en la tabla inversores_detalle: \n\t{error}")
            sys.exit()

        # Actualización de la tabla inversores_raw para indicar que los registros han sido 
            # procesados y la fecha de procesado
        # Se actualiza por trozos para evitar bloqueos de la tabla pasando un array de las ids a 
            # actualizar
        ids = inv_df["id"].tolist()
        CHUNK_SIZE_UPDATE = 100000
        id_chunks = [tuple(ids[i:i + CHUNK_SIZE]) for i in range(0, len(ids), CHUNK_SIZE_UPDATE)]
        for id_chunk in id_chunks:
            try:
                cur.execute(f"""UPDATE {SCHEMA_NAME}.inversores_raw
                            SET procesado = true,
                                datetime_procesado = NOW()
                            WHERE id IN %s""",
                            (id_chunk,))
                conn.commit()
            except psycopg2.Error as e:
                print("Error:", e)
                conn.rollback()
        del(inv_df, amp_df)
    cur.close()
    conn.close()
