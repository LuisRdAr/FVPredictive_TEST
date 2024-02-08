import pandas as pd
import numpy as np
import math
import psycopg2
import sqlalchemy
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
        print(f"Conexión a la base de datos {params['dbname']} (esquema {schema_name}) establecida")
    except Exception as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    # Carga de las coordenadas gps del parque e instanciación del objeto pvlib.location para obtener la posición del Sol
    consulta_sql = f"""SELECT latitud, longitud FROM {schema_name}.parques;"""
    gps = pd.read_sql_query(consulta_sql, engine).values[0]
    lat = gps[0]
    lon = gps[1]
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

    chunk_size = 2500000
    outliers_iqr = 0
    outliers_off = 0
    for i in tqdm(range(0, count, chunk_size), total = math.ceil(count/chunk_size)):
        # Lectura de los datos de meteo sin procesar limitados por el chunk_size
        consulta_sql = f"""WITH daylight AS(
                                SELECT date, sunrise, sunset
                                    FROM {schema_name}.daylight)
                            SELECT id, parque_id, dispositivo_id, nombre_dispositivo, ref, ubicacion, 
                                    descripcion_dispositivo, datetime_utc AT TIME ZONE 'UTC+0' AS datetime_utc, 
                                    med_id, status, alarma, rad_poa, rad_hor, rad_celda1, rad_celda2,
                                    temp_amb, temp_panel1, temp_panel2, vel_viento, dir_viento,
                                    hum_rel, presion_atm, procesado, datetime_procesado, 
                                    date AT TIME ZONE 'UTC+0' AS date, sunrise,
                                    sunset 
                            FROM {schema_name}.meteo_raw AS met 
                            JOIN daylight AS d 
                                ON met.datetime_utc::DATE = d.date
                            WHERE procesado = false  
                            ORDER BY met.datetime_utc, met.dispositivo_id
                            LIMIT {chunk_size};"""
        meteo_df = pd.read_sql_query(consulta_sql, engine)
        ids = meteo_df["id"].tolist()
        meteo_df["datetime_utc"] = pd.to_datetime(meteo_df["datetime_utc"], utc = True)

        # Comprobación si el id del dispositivo está ya registrado y registro en caso de no ser así
        for pair in meteo_df[["parque_id", "dispositivo_id"]].drop_duplicates().values:
            id_par = int(pair[0])
            id_dis = int(pair[1])
            cur.execute(f"""SELECT * FROM {schema_name}.dispositivos WHERE parque_id = {id_par} AND dispositivo_id = {id_dis};""")
            resultados = cur.fetchall()
            if not resultados:
                print("Generando entradas de nuevos dispositivos")
                dispositivo = meteo_df[(meteo_df["parque_id"] == id_par) & 
                                    (meteo_df["dispositivo_id"] == id_dis)][["parque_id", 
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
        meteo_df.drop(columns = ["nombre_dispositivo", 
                                "ref", 
                                "descripcion_dispositivo", 
                                "ubicacion"], inplace = True)
        
        # Descarte de valores anómalos basándose en la regla de IQR con parámetro 1.5
        # Se calculan los percentiles 25 y 75 con SQL para cada parámetro y se descartan los registros
        # que se encuentren fuera del rango superior (criterio natural para límite inferior)
        lista_attr = ["rad_poa", "rad_hor", "rad_celda1", "rad_celda2", "temp_amb", "temp_panel1", 
                    "temp_panel2", "vel_viento", "dir_viento", "hum_rel", "presion_atm"]
        metrics = pd.DataFrame()
        for column in lista_attr:
            consulta_sql = f"""SELECT percentile_cont(0.25) WITHIN GROUP(ORDER BY {column}) AS percentile_25,
                                        percentile_cont(0.75) WITHIN GROUP(ORDER BY {column}) AS percentile_75
                                    FROM {schema_name}.meteo_raw AS m
                                    JOIN {schema_name}.daylight AS d
                                        ON m.datetime_utc::DATE = d.date
                                    WHERE m.datetime_utc BETWEEN d.sunrise AND d.sunset;"""

            metrics = pd.concat([metrics,pd.read_sql_query(consulta_sql, engine)])
        metrics["column"] = lista_attr
        metrics["upper_lim"] = metrics["percentile_75"] + 1.5 * (metrics["percentile_75"] - metrics["percentile_25"])
        metrics["lower_lim"] = metrics["percentile_25"] - 1.5 * (metrics["percentile_75"] - metrics["percentile_25"])

        clean_meteo_df = meteo_df[(meteo_df["rad_poa"] >= 0) & (meteo_df["rad_poa"] <= metrics[metrics["column"] == "rad_poa"]["upper_lim"].values[0]) &
                        (meteo_df["rad_hor"] >= 0) & (meteo_df["rad_hor"] <= metrics[metrics["column"] == "rad_hor"]["upper_lim"].values[0]) &
                        (meteo_df["rad_celda1"] >= 0) & (meteo_df["rad_celda1"] <= metrics[metrics["column"] == "rad_celda1"]["upper_lim"].values[0]) &
                        (meteo_df["rad_celda2"] >= 0) & (meteo_df["rad_celda2"] <= metrics[metrics["column"] == "rad_celda2"]["upper_lim"].values[0]) &
                        (meteo_df["temp_amb"] >= -273.15) & (meteo_df["temp_amb"] <= metrics[metrics["column"] == "temp_amb"]["upper_lim"].values[0]) &
                        (meteo_df["temp_panel1"] >= -273.15) & (meteo_df["temp_panel1"] <= metrics[metrics["column"] == "temp_panel1"]["upper_lim"].values[0]) &
                        (meteo_df["temp_panel2"] >= -273.15) & (meteo_df["temp_panel2"] <= metrics[metrics["column"] == "temp_panel2"]["upper_lim"].values[0]) &
                        (meteo_df["vel_viento"] >= 0) & (meteo_df["vel_viento"] <= 150) &
                        (meteo_df["dir_viento"] >= 0) & (meteo_df["dir_viento"] <= 360) &
                        (meteo_df["hum_rel"] >= 0) & (meteo_df["hum_rel"] <= 100)]
        
        # Obtención de los instantes en los que las medidas de presion_atm del dispositivo 41 se encuentren fuera del rango IQR
        outliers_presion_dt = clean_meteo_df[(clean_meteo_df["dispositivo_id"] == 41) & 
                            ((clean_meteo_df["presion_atm"] <= metrics[metrics["column"] == "presion_atm"]["lower_lim"].values[0]) |
                                (clean_meteo_df["presion_atm"] >= metrics[metrics["column"] == "presion_atm"]["upper_lim"].values[0]))]["datetime_utc"].values

        # Obtener el valor medio de presion_atm y hum_rel para los registros con dispositivo_id igual a 42 y 43 para instantes dados
        # en los que las medidas de presion_atm del dispositivo 41 se encuentren fuera del rango IQR
        df_42_43 = clean_meteo_df[(clean_meteo_df["dispositivo_id"] != 41) & (clean_meteo_df["datetime_utc"].isin(outliers_presion_dt))][["datetime_utc", "hum_rel", "presion_atm"]]
        df_42_43_grouped = df_42_43.groupby(["datetime_utc"]).mean().reset_index()

        # Asignar el valor medio a los registros medidos por los dispositivos 42 y 43 en los instantes donde el dispositivo 41 no funcione correctamente
        clean_meteo_df.loc[(clean_meteo_df["datetime_utc"].isin(outliers_presion_dt)) & (clean_meteo_df["dispositivo_id"] == 41), "presion_atm"] = df_42_43_grouped["presion_atm"]
        clean_meteo_df.loc[(clean_meteo_df["datetime_utc"].isin(outliers_presion_dt)) & (clean_meteo_df["dispositivo_id"] == 41), "hum_rel"] = df_42_43_grouped["hum_rel"]       
                
        # Proceso análogo para los registros con dispositivo_id igual a 42 y 43
        outliers_presion_dt = clean_meteo_df[(clean_meteo_df["dispositivo_id"] != 41) & 
                            ((clean_meteo_df["presion_atm"] <= metrics[metrics["column"] == "presion_atm"]["lower_lim"].values[0]) |
                                (clean_meteo_df["presion_atm"] >= metrics[metrics["column"] == "presion_atm"]["upper_lim"].values[0]))]["datetime_utc"].unique().tolist()
        df_41 = clean_meteo_df[(clean_meteo_df["dispositivo_id"] == 41) & (clean_meteo_df["datetime_utc"].isin(outliers_presion_dt))][["datetime_utc", "hum_rel", "presion_atm"]]
        clean_meteo_df.loc[(clean_meteo_df["datetime_utc"].isin(outliers_presion_dt)) & (clean_meteo_df["dispositivo_id"] == 42), "presion_atm"] = df_41["presion_atm"]
        clean_meteo_df.loc[(clean_meteo_df["datetime_utc"].isin(outliers_presion_dt)) & (clean_meteo_df["dispositivo_id"] == 42), "hum_rel"] = df_41["hum_rel"]       
        clean_meteo_df.loc[(clean_meteo_df["datetime_utc"].isin(outliers_presion_dt)) & (clean_meteo_df["dispositivo_id"] == 43), "presion_atm"] = df_41["presion_atm"]
        clean_meteo_df.loc[(clean_meteo_df["datetime_utc"].isin(outliers_presion_dt)) & (clean_meteo_df["dispositivo_id"] == 43), "hum_rel"] = df_41["hum_rel"]       
        
        ##########
        # INTENTAR GENERALIZAR ESTE PROCESO PARA VARIAS COLUMNAS. LAS TEMPERATURAS PRESENTAN UN COMPORTAMIENTO ANÁLOGO
        # (SE PUEDE COMPROBAR AGRUPANDO POR HORAS Y DISPOSITIVOS Y POSTERIORMENTE POR HORAS Y MIRAS LA STD)
        ##########

        outliers_iqr += meteo_df.shape[0] - clean_meteo_df.shape[0]

        # Cálculo de la elevación y azimuth para los instantes dados
        solar_position = parque.get_solarposition(pd.date_range(start = clean_meteo_df["datetime_utc"].min(), 
                                                    end = clean_meteo_df["datetime_utc"].max(),
                                                    freq="30s", 
                                                    tz="utc"))[["elevation", "azimuth"]]
        clean_meteo_df = pd.merge(clean_meteo_df, solar_position, left_on = "datetime_utc", right_index = True, how = "inner")
        clean_meteo_df["daylight"] = np.where((clean_meteo_df["datetime_utc"] >= clean_meteo_df["sunrise"]) & 
                                              (clean_meteo_df["datetime_utc"] < clean_meteo_df["sunset"]),
                                        True,
                                        False)

        # Cálculo de las irradiancias esperadas (mediante modelo) y comparativa con valor real debido a nubosidad
        clean_meteo_df["day_of_year"] = clean_meteo_df["datetime_utc"].dt.dayofyear
        clean_meteo_df["dia_año_sen"] = np.sin(clean_meteo_df["day_of_year"] * (2*np.pi/365))
        clean_meteo_df["dia_año_cos"] = np.cos(clean_meteo_df["day_of_year"] * (2*np.pi/365))

        clean_meteo_df["poa_pred"] = model_poa.predict(clean_meteo_df[['temp_amb', 'vel_viento', 'dir_viento',
                                                        'hum_rel', 'elevation', 'azimuth', 
                                                        'dia_año_sen', 'dia_año_cos', 'daylight']])
        clean_meteo_df["poa_pred"] = clean_meteo_df["poa_pred"].rolling(window=30).mean().fillna(0)
        clean_meteo_df["mean_poa"] = clean_meteo_df[["rad_poa"]].rolling(window=30).mean().fillna(0)
        clean_meteo_df["ratio_poa"] = np.where((clean_meteo_df["datetime_utc"] > (clean_meteo_df["sunrise"] + dt.timedelta(minutes = 15))) & 
                                            (clean_meteo_df["datetime_utc"] < (clean_meteo_df["sunset"] - dt.timedelta(minutes = 15))) &
                                            (np.abs(clean_meteo_df["mean_poa"] - clean_meteo_df["poa_pred"]) > 25), 
                                        clean_meteo_df["mean_poa"] * 100 / clean_meteo_df["poa_pred"], 
                                        100)
        
        clean_meteo_df["hor_pred"] = model_hor.predict(clean_meteo_df[['temp_amb', 'vel_viento', 'dir_viento',
                                                        'hum_rel', 'elevation', 'azimuth',
                                                        'dia_año_sen', 'dia_año_cos', "daylight"]])
        clean_meteo_df["hor_pred"] = clean_meteo_df["hor_pred"].rolling(window=30).mean().fillna(0)
        clean_meteo_df["mean_hor"] = clean_meteo_df[['rad_hor']].rolling(window=30).mean().fillna(0)
        clean_meteo_df["ratio_hor"] = np.where((clean_meteo_df["datetime_utc"] > (clean_meteo_df["sunrise"] + dt.timedelta(minutes = 15))) & 
                                            (clean_meteo_df["datetime_utc"] < (clean_meteo_df["sunset"] - dt.timedelta(minutes = 15))) &
                                            (np.abs(clean_meteo_df["mean_hor"] - clean_meteo_df["hor_pred"]) > 10), 
                                        clean_meteo_df["mean_hor"] * 100 / clean_meteo_df["hor_pred"], 
                                        100)
        
        ratio = []
        for row in clean_meteo_df.itertuples():
            if ((row.ratio_poa < 95) & (row.ratio_hor < 95)) | \
                    ((row.ratio_poa > 100) & (row.ratio_hor > 100)):
                ratio.append(row.ratio_poa)
            else:
                ratio.append(100)
        clean_meteo_df["cloud_impact"] = ratio

        # Volcado de datos en la tabla meteo
        try:
            dtypes_meteo = {
                'id': sqlalchemy.types.INTEGER(),
                'parque_id': sqlalchemy.types.SMALLINT(),
                'dispositivo_id': sqlalchemy.types.SMALLINT(),
                'datetime_utc': sqlalchemy.types.DateTime(timezone=True),
                'med_id': sqlalchemy.types.INTEGER(),
                'status': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'alarma': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'rad_poa': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'rad_hor': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'rad_celda1': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'rad_celda2': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'temp_amb': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'temp_panel1': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'temp_panel2': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'vel_viento': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'dir_viento': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'hum_rel': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'presion_atm': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'elevation': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'azimuth': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'daylight': sqlalchemy.types.Boolean(),
                'cloud_impact': sqlalchemy.types.Float(precision=3, asdecimal=True)}
            keys_meteo = list(dtypes_meteo.keys())
            clean_meteo_df = clean_meteo_df.drop(columns=[col for col in clean_meteo_df.columns if col not in keys_meteo])
            clean_meteo_df.to_sql('meteo', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes_meteo, chunksize = 100000)
        except Exception as error:
            print("Error en la escritura de datos en la tabla meteo: \n\t{}".format(error))
            sys.exit()

        # Actualización de la tabla meteo_raw para indicar que los registros han sido procesados y la fecha de procesado
        # Se actualiza por trozos para evitar bloqueos de la tabla pasando un array de las ids a actualizar
        ids = meteo_df["id"].tolist()
        chunk_size_update = 100000
        id_chunks = [tuple(ids[i:i + chunk_size]) for i in range(0, len(ids), chunk_size_update)]
        for id_chunk in id_chunks:
            try:
                cur.execute(f"""UPDATE {schema_name}.meteo_raw
                            SET procesado = true,
                                datetime_procesado = NOW()
                            WHERE id IN %s""",
                            (id_chunk,))
                conn.commit()
            except psycopg2.Error as e:
                print("Error:", e)
                conn.rollback()
        del(meteo_df)
    cur.close()
    conn.close()
    print("Número de registros descartados por IQR: ", outliers_iqr)
    print("Número de registros descartados por outliers de sensores: ", outliers_off)
    print("Número de registros cargados inicialmente: ", count)