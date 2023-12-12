import pandas as pd
import numpy as np
import math
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm
import json
import sys
import os
import pvlib
import pickle
import datetime as dt

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
        print(f"Conexión con la base de datos {params['dbname']} de PostgreSQL completada con éxito")
    except psycopg2.Error as e:
        print("Error al conectarse a PostgreSQL:", e)
        sys.exit()

    # Carga de las coordenadas gps del parque e instanciación del objeto pvlib.location para obtener la posición del Sol
    consulta_sql = f"""SELECT latitud, longitud FROM {schema_name}.parques;"""
    gps = pd.read_sql_query(consulta_sql, engine).values
    lat = gps[0][0]
    lon = gps[0][1]
    parque = pvlib.location.Location(latitude=lat, longitude=lon, tz="Europe/Madrid")

    # Carga de los modelos para estimación de las irradiancias esperadas
    irradiance_path = os.path.join(root_path,'Modelos/irradiancias_esperadas')
    with open(os.path.join(irradiance_path, 'POA/XGBRegressor-2023-10-26 09:28:39.773066/model.model'), 'rb') as f:
        model_poa = pickle.load(f)
    with open(os.path.join(irradiance_path, 'HOR/XGBRegressor-2023-10-26 09:31:55.932239/model.model'), 'rb') as f:
        model_hor = pickle.load(f)

    # Lectura del número de registros a procesar para proceder por trozos
    consulta_sql = f"SELECT COUNT(*) FROM {schema_name}.meteo_raw WHERE procesado = false;"
    count = pd.read_sql_query(consulta_sql, engine).values[0][0]
    if count == 0:
        print(f"No se han encontrado nuevos registros para procesar en {schema_name}.meteo_raw")
        sys.exit()

    chunk_size = 50000
    for i in tqdm(range(0, count, chunk_size), total = math.ceil(count/chunk_size)):
        # Lectura de los datos de meteo sin procesar limitados por el chunk_size
        consulta_sql = f"""WITH daylight AS(
                                SELECT date, sunrise, sunset
                                    FROM galisteo.daylight)
                            SELECT * 
                            FROM galisteo.meteo_raw AS met 
                            JOIN daylight AS d 
                                ON met.datetime_utc::DATE = d.date
                            WHERE procesado = false  
                            ORDER BY met.datetime_utc, met.dispositivo_id
                            LIMIT {chunk_size};"""
        meteo_df = pd.read_sql_query(consulta_sql, engine)

        # Comprobación si el id del dispositivo está ya registrado y registro en caso de no ser así
        for id in meteo_df["dispositivo_id"].unique():
            id_int = int(id)
            cur.execute(f"""SELECT * FROM {schema_name}.dispositivos WHERE dispositivo_id = %s;""", (id_int,))
            resultados = cur.fetchall()
            if not resultados:
                print("Generando entradas de nuevos dispositivos")
                dispositivo = meteo_df[(meteo_df["dispositivo_id"] == id_int)][["parque_id", 
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
        meteo_df = meteo_df.drop(columns = ["parque_id",
                            "descripcion_parque", 
                            "localizacion_parque",
                            "potencia_max", 
                            "num_paneles"])
        meteo_df = meteo_df.drop(columns = ["nombre_dispositivo", 
                            "ref", 
                            "descripcion_dispositivo", 
                            "ubicacion"])
        
        # Compleción de los registros relativos al viento y humedad (solo una estación registra los datos)
        meteo_df[["vel_viento", "dir_viento", "hum_rel"]] = meteo_df[["vel_viento", "dir_viento", "hum_rel"]].bfill(limit = 1).ffill(limit = 1)

        # Cálculo de la elevación y azimuth para los instantes dados
        solar_position = parque.get_solarposition(pd.date_range(start = meteo_df["datetime_utc"].min(), 
                                                    end = meteo_df["datetime_utc"].max(),
                                                    freq="30s", 
                                                    tz="utc"))[["elevation", "azimuth"]]
        meteo_df = pd.merge(meteo_df, solar_position, left_on = "datetime_utc", right_index = True)
        meteo_df["daylight"] = np.where((meteo_df["datetime_utc"] >= meteo_df["sunrise"]) & (meteo_df["datetime_utc"] < meteo_df["sunset"]),
                                        True,
                                        False)

        # Cálculo de las irradiancias esperadas (mediante modelo) y comparativa con valor real debido a nubosidad
        meteo_df["day_of_year"] = meteo_df["datetime_utc"].dt.dayofyear
        meteo_df["dia_año_sen"] = np.sin(meteo_df["day_of_year"] * (2*np.pi/365))
        meteo_df["dia_año_cos"] = np.cos(meteo_df["day_of_year"] * (2*np.pi/365))

        meteo_df["poa_pred"] = model_poa.predict(meteo_df[['temp_amb', 'vel_viento', 'dir_viento',
                                                        'hum_rel', 'elevation', 'azimuth', 
                                                        'dia_año_sen', 'dia_año_cos', "daylight"]])
        meteo_df["poa_pred"] = meteo_df["poa_pred"].rolling(window=30).mean().fillna(0)
        meteo_df["hor_pred"] = model_hor.predict(meteo_df[['temp_amb', 'vel_viento', 'dir_viento',
                                                        'hum_rel', 'elevation', 'azimuth',
                                                        'dia_año_sen', 'dia_año_cos', "daylight"]])
        meteo_df["hor_pred"] = meteo_df["hor_pred"].rolling(window=30).mean().fillna(0)
        meteo_df['mean_poa'] = meteo_df[['rad_poa']].rolling(window=30).mean().fillna(0)
        meteo_df['mean_hor'] = meteo_df[['rad_hor']].rolling(window=30).mean().fillna(0)
        meteo_df['ratio_poa'] = np.where((meteo_df["datetime_utc"] > (meteo_df["sunrise"] + dt.timedelta(minutes = 15))) & 
                                            (meteo_df["datetime_utc"] < (meteo_df["sunset"] - dt.timedelta(minutes = 15))) &
                                            (np.abs(meteo_df['mean_poa'] - meteo_df['poa_pred']) > 25), 
                                        meteo_df['mean_poa'] * 100 / meteo_df['poa_pred'], 
                                        100)
        meteo_df['ratio_hor'] = np.where((meteo_df["datetime_utc"] > (meteo_df["sunrise"] + dt.timedelta(minutes = 15))) & 
                                            (meteo_df["datetime_utc"] < (meteo_df["sunset"] - dt.timedelta(minutes = 15))) &
                                            (np.abs(meteo_df['mean_hor'] - meteo_df['hor_pred']) > 10), 
                                        meteo_df['mean_hor'] * 100 / meteo_df['hor_pred'], 
                                        100)
        ratio = []
        dict_meteo = {col: meteo_df.columns.to_list().index(col)+1 for col in meteo_df.columns}
        for row in meteo_df.itertuples():
            if ((row[dict_meteo["ratio_poa"]] < 95) & (row[dict_meteo["ratio_hor"]] < 95)) | \
                    ((row[dict_meteo["ratio_poa"]] > 100) & (row[dict_meteo["ratio_hor"]] > 100)):
                ratio.append(row[dict_meteo["ratio_poa"]])
            else:
                ratio.append(100)
        meteo_df["ratio_mean"] = ratio

        # Filtrado de outliers siguiendo el criterio de las desviaciones estandar. Agrupación 
        # de rangos normales por daylight y mes del año.
        outlier_entrada = []
        std_param_t = 3
        std_param_f = 6
        lista_attr = ["rad_poa", "rad_hor", "rad_celda1", "rad_celda2", 
                    "temp_amb", "temp_panel1", "temp_panel2", "vel_viento", "dir_viento", "hum_rel"]
        for column in lista_attr:
            consulta_sql = f"""SELECT EXTRACT(MONTH FROM datetime_utc AT TIME ZONE '+00:00') AS month, 
                                        AVG({column}) + {std_param_t} * STDDEV({column}) AS upper_lim,  
                                        AVG({column}) - {std_param_t} * STDDEV({column}) AS lower_lim
                                    FROM {schema_name}.meteo_raw AS m
                                    JOIN {schema_name}.daylight AS d
                                        ON m.datetime_utc::DATE = d.date
                                    WHERE m.datetime_utc BETWEEN d.sunrise AND d.sunset
                                    GROUP BY EXTRACT(MONTH FROM datetime_utc AT TIME ZONE '+00:00')
                                    ORDER BY 1;"""

            metrics = pd.read_sql_query(consulta_sql, engine)
            meteo_copy_df = meteo_df[meteo_df["daylight"] == True][["id", "datetime_utc", column]]
            meteo_copy_df = pd.merge(meteo_copy_df, metrics, left_on=meteo_copy_df["datetime_utc"].dt.month, right_on="month")
            outliers = meteo_copy_df[
                            (meteo_copy_df[column] > meteo_copy_df["upper_lim"]) | 
                            (meteo_copy_df[column] < meteo_copy_df["lower_lim"])]["id"].values
            outlier_entrada.extend(outliers)
            
            consulta_sql = f"""SELECT EXTRACT(MONTH FROM datetime_utc AT TIME ZONE '+00:00') AS month, 
                                        AVG({column}) + {std_param_f} * STDDEV({column}) AS upper_lim,  
                                        AVG({column}) - {std_param_f} * STDDEV({column}) AS lower_lim
                                    FROM {schema_name}.meteo_raw AS m
                                    JOIN {schema_name}.daylight AS d
                                        ON m.datetime_utc::DATE = d.date
                                    WHERE m.datetime_utc NOT BETWEEN d.sunrise AND d.sunset
                                    GROUP BY EXTRACT(MONTH FROM datetime_utc AT TIME ZONE '+00:00')
                                    ORDER BY 1;"""

            metrics = pd.read_sql_query(consulta_sql, engine)
            meteo_copy_df = meteo_df[meteo_df["daylight"] == False][["id", "datetime_utc", column]]
            meteo_copy_df = pd.merge(meteo_copy_df, metrics, left_on=meteo_copy_df["datetime_utc"].dt.month, right_on="month")
            outliers = meteo_copy_df[
                            (meteo_copy_df[column] > meteo_copy_df["upper_lim"]) | 
                            (meteo_copy_df[column] < meteo_copy_df["lower_lim"])]["id"].values
            outlier_entrada.extend(outliers)
        outlier_entrada = set(outlier_entrada)
        meteo_df = meteo_df[~(meteo_df["id"].isin(outlier_entrada))]

        # Iteración sobre el dataframe resultante, volcado en la tabla meteo y actualización de la tabla raw
        for row in tqdm(meteo_df.itertuples(), position = 1, leave=False, total = meteo_df.shape[0]):
            try:
                cur.execute(f"""INSERT INTO {schema_name}.meteo(
                        id, dispositivo_id, datetime_utc, med_id, status, alarma, rad_poa, rad_hor, rad_celda1, rad_celda2, 
                        temp_amb, temp_panel1, temp_panel2, vel_viento, dir_viento, hum_rel, elevation, azimuth, daylight, 
                        cloud_impact)
                            VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                    (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11],
                    row[12], row[13], row[14], row[15], row[16], row[22], row[23], row[24], row[-1]))
                cur.execute(f"""UPDATE {schema_name}.meteo_raw
                        SET procesado = true,
                            datetime_procesado = NOW()
                        WHERE id = %s""",
                        (row[1],))
                conn.commit()
            except psycopg2.Error as e:
                print("Error:", e)
                print("\tID:", row[1])
                conn.rollback()
        del(meteo_df)
    cur.close()
    conn.close()
