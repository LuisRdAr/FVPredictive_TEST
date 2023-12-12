import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm
import json
import sys
import os
import requests


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

    # Apertura de conexión a la base de datos
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

    # Carga de fechas de los datos a procesar y coordenadas del parque
    consulta_sql = f"SELECT DISTINCT(datetime_utc::DATE) FROM {schema_name}.meteo_raw WHERE procesado = false ORDER BY 1;"
    fechas = pd.read_sql_query(consulta_sql, engine)
    consulta_sql = f"""SELECT latitud, longitud FROM {schema_name}.parques;"""
    gps = pd.read_sql_query(consulta_sql, engine).values
    lat = gps[0][0]
    lon = gps[0][1]

    # Cálculo de las horas de salida y puesta del sol para las coords dadas y volcado en la base de datos
    for fecha in tqdm(fechas.itertuples(), position = 1, leave=False, total = fechas.shape[0]):
        r = requests.get('https://api.sunrise-sunset.org/json', params={'lat': lat, 'lng': lon, 'date': fecha[1].strftime("%Y-%m-%d")}).json()['results']
        hora_salida_sol = pd.to_datetime(fecha[1].strftime("%Y-%m-%d") + " " + r['sunrise'], utc = True)
        hora_puesta_sol = pd.to_datetime(fecha[1].strftime("%Y-%m-%d") + " " + r['sunset'], utc = True)
        cur.execute(f"""INSERT INTO {schema_name}.daylight
                    (date, sunrise, sunset)
                    VALUES 
                        (%s, %s, %s);""",
                    (fecha[1], hora_salida_sol, hora_puesta_sol))
        conn.commit()
    cur.close()
    conn.close()

        
