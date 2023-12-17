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
    root_path = os.getcwd()
    params = None
    for filename in os.listdir(root_path):
        if "params.json" in filename:
            with open(os.path.join(root_path, filename)) as f:
                params = json.load(f)
    if params is None:
        print("No se ha encontrado el archivo de parámetros para la conexión a la base de datos")
        sys.exit()
    else:
        print(f"Parámetros de la planta {params['schema'].capitalize()} cargados correctamente")
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
        
    chunk_size = 2500000
    for i in tqdm(range(0, count, chunk_size), total = math.ceil(count/chunk_size)):
        # Lectura de los datos de ree sin procesar limitados por el chunk_size
        consulta_sql = f"SELECT * FROM {schema_name}.ree_raw WHERE procesado = false ORDER BY id LIMIT {chunk_size};"
        ree_df = pd.read_sql_query(consulta_sql, engine)
        ree_df = ree_df.drop(columns = ["procesado", "datetime_procesado"])

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
                print(dispositivo)
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
        
        # Volcado de datos en la tabla ree
        try:
            dtypes_ree = {
                'id': sqlalchemy.types.INTEGER(),
                'dispositivo_id': sqlalchemy.types.SMALLINT(),
                'datetime_utc': sqlalchemy.types.DateTime(timezone=True),
                'med_id': sqlalchemy.types.INTEGER(),
                'status': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'alarma': sqlalchemy.types.SMALLINT(),
                'motivo': sqlalchemy.types.SMALLINT(),
                'consigna_pot_act_ree': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'consigna_pot_act_planta': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'consigna_fdp_ree': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'consigna_fdp_planta': sqlalchemy.types.Float(precision=3, asdecimal=True)}
            keys_ree = list(dtypes_ree.keys())
            ree_df = ree_df.drop(columns=[col for col in ree_df.columns if col not in keys_ree])
            ree_df.to_sql('ree', engine, if_exists = 'append', index = False, schema = schema_name, dtype = dtypes_ree)

        except Exception as error:
            print("Error en la escritura de datos en la tabla ree: \n\t{}".format(error))
            sys.exit()

        # Actualización de la tabla ree_raw para indicar que los registros han sido procesados y la fecha de procesado
        # Se actualiza por trozos para evitar bloqueos de la tabla pasando un array de las ids a actualizar
        ids = ree_df["id"].tolist()
        chunk_size_update = 100000
        id_chunks = [tuple(ids[i:i + chunk_size]) for i in range(0, len(ids), chunk_size_update)]
        for id_chunk in id_chunks:
            try:
                cur.execute(f"""UPDATE {schema_name}.ree_raw
                            SET procesado = true,
                                datetime_procesado = NOW()
                            WHERE id IN %s""",
                            (id_chunk,))
                conn.commit()
            except psycopg2.Error as e:
                print("Error:", e)
                conn.rollback()
        del(ree_df)
    cur.close()
    conn.close()