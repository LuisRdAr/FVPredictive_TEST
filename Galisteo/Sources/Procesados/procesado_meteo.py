#####
# EN LUGAR DE DESCARTAR LOS REGISTROS CON VALORES ANÓMALOS, SE PUEDE PLANTEAR UNA CARGA ETIQUETADA 
# BOOLEANA QUE INDIQUE EL FALLO DE LOS SENSORES. -> HECHO PARCIALMENTE
# SI SE ESTUDIA EL NÚMERO DE MEDIDAS ERRONEAS EN UN REGISTRO, SI SOLO FALLA UNA SE PODRÍA APLICAR 
# ALGÚN TIPO DE MODELO DE IMPUTACIÓN PARA SOBREESCRIBIR EL VALOR ATIPICO. SI FALLAN MÁS DE UNA 
# MEDIDA, SE PODRÍA ETIQUETAR COMO REGISTRO NO FIABLE.
# MÁS TARDE SE PODRÍA UTILIZAR PARA ENTRENAR UN MODELO DE CLASIFICACIÓN QUE PERMITA IDENTIFICAR 
# LOS REGISTROS QUE NO SON FIABLES. ESTO PERMITIRÍA NO DESCARTAR LOS REGISTROS Y PODER UTILIZARLOS 
# PARA EL ANÁLISIS DE DATOS. 
# TAMBIÉN SE PODRÍA TRATAR DE INTERPOLAR PARA OBTENER LOS VALORES QUE DEBERÍAN DE DARSE EN LA 
# REALIDAD.
#####

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
from sklearn.cluster import KMeans
import datetime as dt


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
    SCHEMA_NAME = params["schema"]

    # Conexión con la base de datos
    try:
        PASSWORD = params['password'].replace('@', '%40')
        ENGINE_STRING = f'postgresql://{params["user"]}:{PASSWORD}' + \
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

    # Carga de las coordenadas gps del parque e instanciación del objeto pvlib.location para 
    # obtener la posición del Sol
    consulta_sql = f"""SELECT latitud, longitud FROM {SCHEMA_NAME}.parques;"""
    gps = pd.read_sql_query(consulta_sql, engine).values[0]
    lat, lon = gps
    parque = pvlib.location.Location(latitude=lat, longitude=lon, tz="Europe/Madrid")

    # Carga de los modelos para estimación de las irradiancias esperadas
    IRRADIANCE_PATH = os.path.join(ROOT_PATH,'Modelos/irradiancias_esperadas')
    POA_PATH = 'POA/XGBRegressor-2023-10-26 09:28:39.773066/model.model'
    HOR_PATH = 'HOR/XGBRegressor-2023-10-26 09:31:55.932239/model.model'
    with open(os.path.join(IRRADIANCE_PATH, POA_PATH), 'rb') as f:
        model_poa = pickle.load(f)
    with open(os.path.join(IRRADIANCE_PATH, HOR_PATH), 'rb') as f:
        model_hor = pickle.load(f)

    # Lectura del número de registros a procesar para proceder por trozos
    consulta_sql = f"SELECT COUNT(*) FROM {SCHEMA_NAME}.meteo_raw WHERE procesado = false;"
    count = pd.read_sql_query(consulta_sql, engine).values[0][0]
    if count == 0:
        print(f"No se han encontrado nuevos registros para procesar en {SCHEMA_NAME}.meteo_raw")
        sys.exit()

    CHUNK_SIZE = 7500000
    outliers_iqr = 0
    outliers_off = 0
    for i in tqdm(range(0, count, CHUNK_SIZE), total = math.ceil(count/CHUNK_SIZE)):
        # Lectura de los datos de meteo sin procesar limitados por el chunk_size
        consulta_sql = f"""WITH daylight AS(
                                SELECT date, sunrise, sunset
                                    FROM {SCHEMA_NAME}.daylight)
                            SELECT id, 
                                    parque_id, 
                                    dispositivo_id, 
                                    nombre_dispositivo, 
                                    ref, 
                                    ubicacion, 
                                    descripcion_dispositivo, 
                                    datetime_utc AT TIME ZONE 'UTC+0' AS datetime_utc, 
                                    med_id, 
                                    status, 
                                    alarma, 
                                    rad_poa, 
                                    rad_hor, 
                                    rad_celda1, 
                                    rad_celda2,
                                    temp_amb, 
                                    temp_panel1, 
                                    temp_panel2, 
                                    vel_viento, 
                                    dir_viento,
                                    hum_rel, 
                                    procesado, 
                                    datetime_procesado, 
                                    sunrise, 
                                    sunset 
                            FROM {SCHEMA_NAME}.meteo_raw AS met 
                            JOIN daylight AS d 
                                ON met.datetime_utc::DATE = d.date
                            WHERE procesado = false  
                            ORDER BY met.datetime_utc, met.dispositivo_id
                            LIMIT {CHUNK_SIZE};"""
        meteo_df = pd.read_sql_query(consulta_sql, engine)
        ids = meteo_df["id"].tolist()
        meteo_df["datetime_utc"] = pd.to_datetime(meteo_df["datetime_utc"], utc = True)
        meteo_df["date"] = meteo_df["datetime_utc"].dt.date

        # Comprobación si el id del dispositivo está ya registrado y registro en caso de no ser así
        for id in meteo_df["dispositivo_id"].unique():
            id_int = int(id)
            cur.execute(f"""SELECT * FROM {SCHEMA_NAME}.dispositivos WHERE dispositivo_id = %s;""",
                        (id_int,))
            resultados = cur.fetchall()
            if not resultados:
                print("Generando entradas de nuevos dispositivos")
                dispositivo = meteo_df[(meteo_df["dispositivo_id"] == id_int)][["parque_id", 
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
        meteo_df.drop(columns = ["nombre_dispositivo", 
                                 "ref", 
                                 "descripcion_dispositivo", 
                                 "ubicacion"], inplace = True)
        
        # Cálculo de la elevación y azimuth para los instantes dados
        solar_position=parque.get_solarposition(pd.date_range(start=meteo_df["datetime_utc"].min(), 
                                                    end=meteo_df["datetime_utc"].max(),
                                                    freq="30s", 
                                                    tz="utc"))[["elevation", "azimuth"]]
        meteo_df = pd.merge(meteo_df, solar_position, 
                            left_on = "datetime_utc", right_index = True, how = "inner")
        meteo_df["daylight"] = (meteo_df["datetime_utc"] >= meteo_df["sunrise"]) \
                                 & (meteo_df["datetime_utc"] < meteo_df["sunset"])

        # SE IMPLEMENTÓ UN CODIGO PARA BONETE AL SER FRECUENTES LOS FALLOS DE LOS SENSORES
        # QUEDA PENDIENTE DE DESARROLLAR UN CÓDIGO VÁLIDO GENERALIZABLE A OTROS PARQUES Y 
        # QUIZÁS CON RANGOS HORARIOS MÁS REDUCIDOS
        meteo_df["status_srl"] = 0

        lista_attr = ["rad_poa", "rad_hor", "rad_celda1", "rad_celda2", 
                    "temp_amb", "temp_panel1", "temp_panel2", "vel_viento", "dir_viento", "hum_rel"]
        metrics = pd.DataFrame()
        metrics["column"] = lista_attr
        percentile_25 = []
        percentile_75 = []
        consulta_sql = f"""SELECT COUNT(*) FROM {SCHEMA_NAME}.meteo"""
        count = pd.read_sql_query(consulta_sql, engine).values[0][0]
        if count == 0:
            for column in lista_attr:
                percentile_25.append(meteo_df[meteo_df["status_srl"] == 0][column].quantile(0.25))
                percentile_75.append(meteo_df[meteo_df["status_srl"] == 0][column].quantile(0.75))
        else:
            for column in lista_attr:
                consulta_sql = f"""
                    SELECT percentile_cont(0.25) WITHIN GROUP(ORDER BY {column}) AS percentile_25,
                        percentile_cont(0.75) WITHIN GROUP(ORDER BY {column}) AS percentile_75
                    FROM {SCHEMA_NAME}.meteo AS m
                    JOIN {SCHEMA_NAME}.daylight AS d
                        ON m.datetime_utc::DATE = d.date
                    WHERE m.datetime_utc BETWEEN d.sunrise AND d.sunset AND
                        (m.dispositivo_id = 41 OR m.dispositivo_id = 42 OR m.dispositivo_id = 43) 
                        AND (m.vel_viento != 'NaN' AND m.dir_viento != 'NaN' AND m.hum_rel != 'NaN')
                        AND status_srl = 0;"""
                values = pd.read_sql_query(consulta_sql, engine).values[0]
                percentile_25.append(values[0])
                percentile_75.append(values[1])
        metrics["percentile_25"] = percentile_25
        metrics["percentile_75"] = percentile_75   
        metrics["upper_lim"] = metrics["percentile_75"] \
                                + 3 * (metrics["percentile_75"] - metrics["percentile_25"])
        metrics["lower_lim"] = metrics["percentile_25"] \
                                - 1.5 * (metrics["percentile_75"] - metrics["percentile_25"])

        meteo_df["status_srl"] = \
            np.where((meteo_df["rad_poa"] < \
                            metrics[metrics["column"]=="rad_poa"]["lower_lim"].values[0]) | 
                        (meteo_df["rad_poa"] > \
                                metrics[metrics["column"]=="rad_poa"]["upper_lim"].values[0]), 
                    2, 
                    meteo_df["status_srl"])
        meteo_df["status_srl"] = \
            np.where((meteo_df["rad_hor"] < \
                            metrics[metrics["column"] == "rad_hor"]["lower_lim"].values[0]) | 
                        (meteo_df["rad_hor"] > \
                                metrics[metrics["column"] == "rad_hor"]["upper_lim"].values[0]),
                    2,
                    meteo_df["status_srl"])
        meteo_df["status_srl"] = \
            np.where((meteo_df["rad_celda1"] < \
                            metrics[metrics["column"] == "rad_celda1"]["lower_lim"].values[0]) | 
                        (meteo_df["rad_celda1"] > \
                                metrics[metrics["column"] == "rad_celda1"]["upper_lim"].values[0]),
                    2,
                    meteo_df["status_srl"])
        meteo_df["status_srl"] = \
            np.where((meteo_df["rad_celda2"] < \
                            metrics[metrics["column"] == "rad_celda2"]["lower_lim"].values[0]) | 
                        (meteo_df["rad_celda2"] > \
                                metrics[metrics["column"] == "rad_celda2"]["upper_lim"].values[0]),
                    2,
                    meteo_df["status_srl"])
        meteo_df["status_srl"] = \
            np.where((meteo_df["temp_amb"] < \
                            metrics[metrics["column"] == "temp_amb"]["lower_lim"].values[0]) | 
                        (meteo_df["temp_amb"] > \
                                metrics[metrics["column"] == "temp_amb"]["upper_lim"].values[0]),
                    2,
                    meteo_df["status_srl"])
        meteo_df["status_srl"] = \
            np.where((meteo_df["temp_panel1"] < \
                            metrics[metrics["column"] == "temp_panel1"]["lower_lim"].values[0]) | 
                        (meteo_df["temp_panel1"] > \
                                metrics[metrics["column"] == "temp_panel1"]["upper_lim"].values[0]),
                    2,
                    meteo_df["status_srl"])
        meteo_df["status_srl"] = \
            np.where((meteo_df["temp_panel2"] < \
                            metrics[metrics["column"] == "temp_panel2"]["lower_lim"].values[0]) | 
                        (meteo_df["temp_panel2"] > \
                                metrics[metrics["column"] == "temp_panel2"]["upper_lim"].values[0]),
                    2,
                    meteo_df["status_srl"])
        meteo_df["status_srl"] = \
            np.where((meteo_df["rad_poa"] == 0) & (meteo_df["rad_hor"] == 0) & 
                        (meteo_df["rad_celda1"] == 0) & (meteo_df["rad_celda2"] == 0) &
                        (meteo_df["temp_amb"] == 0) & 
                        (meteo_df["temp_panel1"] == 0) & (meteo_df["temp_panel2"] == 0),
                    2,
                    meteo_df["status_srl"])

        # Se separan los registros de la estación que mide la humedad y los registros de viento y se
        # etiquetan los registros fuera del rango natural
        hum_df = meteo_df[meteo_df["dispositivo_id"] == 42][["datetime_utc",
                                                             "hum_rel", 
                                                             "vel_viento", 
                                                             "dir_viento"]]
        hum_df = hum_df.dropna(subset=["hum_rel"])
        hum_df["status_srl"] = 0
        hum_df["status_srl"] = np.where((hum_df["hum_rel"] >= 0) & 
                                            (hum_df["hum_rel"] <= 100) &
                                            (hum_df["vel_viento"] >= 0) & 
                                            (hum_df["vel_viento"] <= 100) &
                                            (hum_df["dir_viento"] >= 0) & 
                                            (hum_df["dir_viento"] <= 360), 
                                        0, 
                                        3)

        # Se unen los dataframes de irradiancia y humedad y viento
        prep_meteo_df = pd.merge(meteo_df.drop(columns=["vel_viento", "dir_viento", "hum_rel"]), 
                                 hum_df, 
                                 on = ["datetime_utc"], 
                                 how = "left")
        prep_meteo_df["status_srl_y"] = prep_meteo_df["status_srl_y"].fillna(4).astype(int)
        prep_meteo_df["status_srl"] = np.where((prep_meteo_df["status_srl_x"] == 0) &
                                                    ((prep_meteo_df["status_srl_y"] == 3) | 
                                                        (prep_meteo_df["status_srl_y"] == 4)), 
                                                prep_meteo_df["status_srl_y"],
                                                np.where((prep_meteo_df["status_srl_x"] != 0) &
                                                    ((prep_meteo_df["status_srl_y"] == 3) | 
                                                        (prep_meteo_df["status_srl_y"] == 4)),
                                                    5,
                                                    prep_meteo_df["status_srl_x"]))
        prep_meteo_df.drop(columns = ["status_srl_x", "status_srl_y"], inplace = True)

        # Cálculo de las irradiancias esperadas (mediante modelo) y comparativa con valor real 
        # debido a nubosidad
        prep_meteo_df["day_of_year"] = prep_meteo_df["datetime_utc"].dt.dayofyear
        prep_meteo_df["dia_año_sen"] = np.sin(prep_meteo_df["day_of_year"] * (2*np.pi/365))
        prep_meteo_df["dia_año_cos"] = np.cos(prep_meteo_df["day_of_year"] * (2*np.pi/365))

        prep_meteo_df["poa_pred"] = model_poa.predict(prep_meteo_df[['temp_amb', 
                                                                     'vel_viento', 
                                                                     'dir_viento',
                                                                     'hum_rel', 
                                                                     'elevation', 
                                                                     'azimuth', 
                                                                     'dia_año_sen', 
                                                                     'dia_año_cos', 
                                                                     'daylight']])
        prep_meteo_df["poa_pred"] = prep_meteo_df["poa_pred"].rolling(window=30).mean().fillna(0)
        prep_meteo_df["mean_poa"] = prep_meteo_df[["rad_poa"]].rolling(window=30).mean().fillna(0)
        t_delta = dt.timedelta(minutes = 15)
        prep_meteo_df["ratio_poa"] = np.where((prep_meteo_df["datetime_utc"] > \
                                                    (prep_meteo_df["sunrise"] + t_delta)) & 
                                                (prep_meteo_df["datetime_utc"] < \
                                                    (prep_meteo_df["sunset"] - t_delta)) &
                                                (np.abs(prep_meteo_df["mean_poa"] \
                                                        - prep_meteo_df["poa_pred"]) > 25), 
                                        prep_meteo_df["mean_poa"] * 100 / prep_meteo_df["poa_pred"], 
                                        100)
        
        prep_meteo_df["hor_pred"] = model_hor.predict(prep_meteo_df[['temp_amb', 
                                                                     'vel_viento', 
                                                                     'dir_viento',
                                                                     'hum_rel', 
                                                                     'elevation', 
                                                                     'azimuth',
                                                                     'dia_año_sen', 
                                                                     'dia_año_cos',
                                                                     'daylight']])
        prep_meteo_df["hor_pred"] = prep_meteo_df["hor_pred"].rolling(window=30).mean().fillna(0)
        prep_meteo_df["mean_hor"] = prep_meteo_df[['rad_hor']].rolling(window=30).mean().fillna(0)
        prep_meteo_df["ratio_hor"] = np.where((prep_meteo_df["datetime_utc"] > \
                                                    (prep_meteo_df["sunrise"] + t_delta)) & 
                                                (prep_meteo_df["datetime_utc"] < \
                                                        (prep_meteo_df["sunset"] - t_delta)) &
                                                (np.abs(prep_meteo_df["mean_hor"] \
                                                        - prep_meteo_df["hor_pred"]) > 10), 
                                        prep_meteo_df["mean_hor"] * 100 / prep_meteo_df["hor_pred"], 
                                        100)
        
        ratio = []
        for row in prep_meteo_df.itertuples():
            if ((row.ratio_poa < 95) & (row.ratio_hor < 95)) | \
                    ((row.ratio_poa > 100) & (row.ratio_hor > 100)):
                ratio.append(row.ratio_poa)
            else:
                ratio.append(100)
        prep_meteo_df["cloud_impact"] = [min(r, 100) for r in ratio]

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
                'elevation': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'azimuth': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'daylight': sqlalchemy.types.Boolean(),
                'cloud_impact': sqlalchemy.types.Float(precision=3, asdecimal=True),
                'status_srl': sqlalchemy.types.SMALLINT()}
            keys_meteo = list(dtypes_meteo.keys())
            dropped_cols = [col for col in prep_meteo_df.columns if col not in keys_meteo]
            prep_meteo_df = prep_meteo_df.drop(columns = dropped_cols)
            prep_meteo_df.to_sql('meteo', 
                                 engine, 
                                 if_exists = 'append', 
                                 index = False, 
                                 schema = SCHEMA_NAME, 
                                 dtype = dtypes_meteo, 
                                 chunksize = 100000)
        except Exception as error:
            print("Error en la escritura de datos en la tabla meteo: \n\t{}".format(error))
            sys.exit()

        # Actualización de la tabla meteo_raw para indicar que los registros han sido procesados y 
            # la fecha de procesado.
        # Se actualiza por trozos para evitar bloqueos de la tabla pasando un array de las ids a 
            # actualizar.
        ids = meteo_df["id"].tolist()
        CHUNK_SIZE_UPDATE = 100000
        id_chunks = [tuple(ids[i:i + CHUNK_SIZE]) for i in range(0, len(ids), CHUNK_SIZE_UPDATE)]
        for id_chunk in id_chunks:
            try:
                cur.execute(f"""UPDATE {SCHEMA_NAME}.meteo_raw
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
    print("Número de registros con etiquetado por anomalía en el rango: ", 
          prep_meteo_df[(prep_meteo_df["status_srl"] != 0) & \
                        (prep_meteo_df["status_srl"] != 1)].shape[0])
    print("Número de registros con etiquetado por malfuncionamiento de piranómetro: ", 
          prep_meteo_df[(prep_meteo_df["status_srl"] == 1)].shape[0])
    print("Número de registros cargados inicialmente: ", count)
