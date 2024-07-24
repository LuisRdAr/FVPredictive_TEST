#####
# EN LUGAR DE DESCARTAR LOS REGISTROS CON VALORES ANÓMALOS, SE PUEDE PLANTEAR UNA CARGA ETIQUETADA BOOLEANA 
# QUE INDIQUE EL FALLO DE LOS SENSORES.
# SI SE ESTUDIA EL NÚMERO DE MEDIDAS ERRONEAS EN UN REGISTRO, SI SOLO FALLA UNA SE PODRÍA APLICAR ALGÚN TIPO DE MODELO DE IMPUTACIÓN
# PARA SOBREESCRIBIR EL VALOR ATIPICO. SI FALLAN MÁS DE UNA MEDIDA, SE PODRÍA ETIQUETAR COMO REGISTRO NO FIABLE.
# MÁS TARDE SE PODRÍA UTILIZAR PARA ENTRENAR UN MODELO DE CLASIFICACIÓN QUE PERMITA IDENTIFICAR LOS REGISTROS
# QUE NO SON FIABLES. ESTO PERMITIRÍA NO DESCARTAR LOS REGISTROS Y PODER UTILIZARLOS PARA EL ANÁLISIS DE DATOS.
# TAMBIÉN SE PODRÍA TRATAR DE INTERPOLAR PARA OBTENER LOS VALORES QUE DEBERÍAN DE DARSE EN LA REALIDAD.
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


def abnormal_daily_behaviour(df, attr):
    mean, max_val, count = attr
    
    day_df = df[df["daylight"]]
    night_df = df[~df["daylight"]]
    # Se crea una instancia de KMeans y ajusta el modelo sobre el conjunto de datos medios durante el día
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    day_df["comp_group"] = kmeans.fit_predict(day_df[[mean]])

    # Asociacion de los dispositivos a cada grupo y calculo de algunas métricas para cada grupo
    devices_group_0 = day_df[day_df["comp_group"] == 0]["dispositivo_id"].unique()
    devices_group_1 = day_df[day_df["comp_group"] == 1]["dispositivo_id"].unique()

    max_group_0 = df[df["dispositivo_id"].isin(devices_group_0)][max_val].max()
    max_group_1 = df[df["dispositivo_id"].isin(devices_group_1)][max_val].max()
    
    mean_group_0_day = day_df[day_df["dispositivo_id"].isin(devices_group_0)][mean].mean()
    mean_group_1_day = day_df[day_df["dispositivo_id"].isin(devices_group_1)][mean].mean()

    mean_group_0_night = night_df[night_df["dispositivo_id"].isin(devices_group_0)][mean].mean()
    mean_group_1_night = night_df[night_df["dispositivo_id"].isin(devices_group_1)][mean].mean()

    # Asignación de la etiqueta de fallo a los dispositivos en función del comportamiento de los grupos
    if max_group_0 > 1750:
        day_df["failure"] = day_df["comp_group"] == 0
    elif max_group_1 > 1750:
        day_df["failure"] = day_df["comp_group"] == 1
    elif mean_group_0_night > 1:
        day_df["failure"] = day_df["comp_group"] == 0
    elif mean_group_1_night > 1:
        day_df["failure"] = day_df["comp_group"] == 1
    elif mean_group_0_night > 1 and mean_group_1_night > 1:
        day_df["failure"] = mean_group_0_night > mean_group_1_night
    elif mean_group_0_day > mean_group_1_day:
        day_df["failure"] = day_df["comp_group"] == 1
    else:
        day_df["failure"] = day_df["comp_group"] == 0

    return day_df[["failure"]]


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
    lat, lon = gps
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

    chunk_size = 7500000
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
                                    hum_rel, procesado, datetime_procesado, sunrise, sunset 
                            FROM {schema_name}.meteo_raw AS met 
                            JOIN daylight AS d 
                                ON met.datetime_utc::DATE = d.date
                            WHERE procesado = false  
                            ORDER BY met.datetime_utc, met.dispositivo_id
                            LIMIT {chunk_size};"""
        meteo_df = pd.read_sql_query(consulta_sql, engine)
        ids = meteo_df["id"].tolist()
        meteo_df["datetime_utc"] = pd.to_datetime(meteo_df["datetime_utc"], utc = True)
        meteo_df["date"] = meteo_df["datetime_utc"].dt.date

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
        meteo_df.drop(columns = ["nombre_dispositivo", 
                                 "ref", 
                                 "descripcion_dispositivo", 
                                 "ubicacion"], inplace = True)
        
        # Cálculo de la elevación y azimuth para los instantes dados
        rad_df = meteo_df[(meteo_df["dispositivo_id"] >= 40) & (meteo_df["dispositivo_id"] < 50)].copy()
        solar_position = parque.get_solarposition(pd.date_range(start = rad_df["datetime_utc"].min(), 
                                                    end = rad_df["datetime_utc"].max(),
                                                    freq="30s", 
                                                    tz="utc"))[["elevation", "azimuth"]]
        rad_df = pd.merge(rad_df, solar_position, left_on = "datetime_utc", right_index = True, how = "inner")
        rad_df["daylight"] = (rad_df["datetime_utc"] >= rad_df["sunrise"]) & (rad_df["datetime_utc"] < rad_df["sunset"])

        # Clasificación de los días donde al menos uno de los piranómetros no funcione correctamente
        # Se calcula el coeficiente de variación de las irradiancias para cada instante y se agrupa para obtener el 
        # comportamiento diario de este. A partir de las medias y desviaciones se clusteriza en dos grupos.
        dt_agg_df = rad_df[rad_df["daylight"]].groupby([rad_df["datetime_utc"]])[["rad_poa", "rad_hor"]].agg({"mean", "std"})
        dt_agg_df.columns = ['_'.join(col).strip() for col in dt_agg_df.columns.values]
        dt_agg_df["rad_poa_coef_var"] = np.where(dt_agg_df["rad_poa_std"] != 0,dt_agg_df["rad_poa_std"]/dt_agg_df["rad_poa_mean"], 0)
        dt_agg_df["rad_hor_coef_var"] = np.where(dt_agg_df["rad_hor_std"] != 0,dt_agg_df["rad_hor_std"]/dt_agg_df["rad_hor_mean"], 0)
        dt_agg_df = dt_agg_df.reset_index()
        dt_agg_df["datetime_utc"] = pd.to_datetime(dt_agg_df["datetime_utc"])
        daily_agg_df = dt_agg_df.groupby([dt_agg_df["datetime_utc"].dt.date])[["rad_poa_coef_var", "rad_hor_coef_var"]].agg({"mean", "std"})
        daily_agg_df.columns = ['_'.join(col).strip() for col in daily_agg_df.columns.values]
        del dt_agg_df

        data = daily_agg_df[["rad_poa_coef_var_mean", "rad_hor_coef_var_mean", "rad_poa_coef_var_std", "rad_hor_coef_var_std"]].values
        kmeans = KMeans(n_clusters = 2, random_state = 0, n_init=10).fit(data)
        del data
        daily_agg_df["cluster"] = kmeans.labels_
        correct_cluster = daily_agg_df[daily_agg_df["rad_poa_coef_var_mean"] == daily_agg_df["rad_poa_coef_var_mean"].min()]["cluster"].values[0]
        incorrect_cluster = daily_agg_df[daily_agg_df["rad_poa_coef_var_mean"] == daily_agg_df["rad_poa_coef_var_mean"].max()]["cluster"].values[0]
    
        # Con los días clasificados se coge aquellos que muestran comportamiento anómalo y se calculan los valores medios
        # de irradiancia para cada dispositivo en esos días. Se calculan los outliers mediante un clustering 
        # y se etiquetan los registros de modo que el cluster con comportamiento más bajo sea el incorrecto.
        incorrect_behaviour = daily_agg_df[(daily_agg_df["cluster"] == incorrect_cluster)][["cluster"]].reset_index()
        correct_behaviour = daily_agg_df[(daily_agg_df["cluster"] == correct_cluster)][["cluster"]].reset_index()
        del daily_agg_df
        rad_incorrect_df = rad_df[rad_df["datetime_utc"].dt.date.isin(incorrect_behaviour["datetime_utc"])]
        daily_inv_agg_df = rad_incorrect_df.groupby(["dispositivo_id", rad_incorrect_df["date"], "daylight"])["rad_poa"].agg({"mean", "max", "count"}).reset_index()
        outlier = daily_inv_agg_df.groupby("date").apply(lambda x: abnormal_daily_behaviour(x, ["mean", "max", "count"])).reset_index(level = 0, drop = True)
        daily_failed_stations_df = pd.merge(daily_inv_agg_df[["date", "dispositivo_id"]], outlier, left_index = True, right_index = True).sort_values(by = ["date","dispositivo_id"])

        # Se genera un dataframe que contenga todos los dias y dispositivos, y si tienen comportamiento fallido o no
        correct_behaviour = correct_behaviour.merge(pd.DataFrame(daily_inv_agg_df["dispositivo_id"].unique(), columns = ["dispositivo_id"]), how = "cross")
        correct_behaviour["failure"] = False
        correct_behaviour.rename(columns = {"datetime_utc": "date"}, inplace = True)
        correct_behaviour.drop(columns = ["cluster"], inplace = True)
        daily_failed_stations_df = pd.concat([daily_failed_stations_df, correct_behaviour], axis = 0).sort_values(by = ["date", "dispositivo_id"]).reset_index(drop = True)

        # Se asigna un código de fallo en registros con valores anómalos basándose en la regla de IQR con parámetro 1.5
        # Se calculan los percentiles 25 y 75 con SQL para cada parámetro y se descartan los registros
        # que se encuentren fuera del rango superior (criterio natural para límite inferior)
        rad_df = rad_df.merge(daily_failed_stations_df, on = ["date", "dispositivo_id"], how = "inner")
        rad_df["status_srl"] = rad_df["failure"].astype(int)
        rad_df.drop(columns = ["failure"], inplace = True)

        non_failed_rad_df = rad_df[rad_df["status_srl"] == 0]

        lista_attr = ["rad_poa", "rad_hor", "rad_celda1", "rad_celda2", 
                    "temp_amb", "temp_panel1", "temp_panel2", "vel_viento", "dir_viento", "hum_rel"]
        metrics = pd.DataFrame()
        metrics["column"] = lista_attr
        percentile_25 = []
        percentile_75 = []
        consulta_sql = f"""SELECT COUNT(*) FROM {schema_name}.meteo"""
        count = pd.read_sql_query(consulta_sql, engine).values[0][0]
        if count == 0:
            for column in lista_attr:
                percentile_25.append(non_failed_rad_df[column].quantile(0.25))
                percentile_75.append(non_failed_rad_df[column].quantile(0.75))
        else:
            for column in lista_attr:
                consulta_sql = f"""SELECT percentile_cont(0.25) WITHIN GROUP(ORDER BY {column}) AS percentile_25,
                                        percentile_cont(0.75) WITHIN GROUP(ORDER BY {column}) AS percentile_75
                                    FROM {schema_name}.meteo AS m
                                    JOIN {schema_name}.daylight AS d
                                        ON m.datetime_utc::DATE = d.date
                                    WHERE m.datetime_utc BETWEEN d.sunrise AND d.sunset AND
                                        (m.dispositivo_id = 41 OR m.dispositivo_id = 42 OR m.dispositivo_id = 43) AND
                                        (m.vel_viento != 'NaN' AND m.dir_viento != 'NaN' AND m.hum_rel != 'NaN') AND
                                        status_srl = 0;"""
                values = pd.read_sql_query(consulta_sql, engine).values[0]
                percentile_25.append(values[0])
                percentile_75.append(values[1])
        metrics["percentile_25"] = percentile_25
        metrics["percentile_75"] = percentile_75   
        metrics["upper_lim"] = metrics["percentile_75"] + 1.5 * (metrics["percentile_75"] - metrics["percentile_25"])
        metrics["lower_lim"] = metrics["percentile_25"] - 1.5 * (metrics["percentile_75"] - metrics["percentile_25"])

        clean_rad_df = non_failed_rad_df[(non_failed_rad_df["rad_poa"] >= max([0, metrics[metrics["column"] == "rad_poa"]["lower_lim"].values[0]])) & 
                                    (non_failed_rad_df["rad_poa"] <= metrics[metrics["column"] == "rad_poa"]["upper_lim"].values[0]) &
                                (non_failed_rad_df["rad_hor"] >= max([0, metrics[metrics["column"] == "rad_hor"]["lower_lim"].values[0]])) & 
                                    (non_failed_rad_df["rad_hor"] <= metrics[metrics["column"] == "rad_hor"]["upper_lim"].values[0]) &
                                (non_failed_rad_df["rad_celda1"] >= max([0, metrics[metrics["column"] == "rad_celda1"]["lower_lim"].values[0]])) & 
                                    (non_failed_rad_df["rad_celda1"] <= metrics[metrics["column"] == "rad_celda1"]["upper_lim"].values[0]) &
                                (non_failed_rad_df["rad_celda2"] >= max([0, metrics[metrics["column"] == "rad_celda2"]["lower_lim"].values[0]])) & 
                                    (non_failed_rad_df["rad_celda2"] <= metrics[metrics["column"] == "rad_celda2"]["upper_lim"].values[0]) &
                                (non_failed_rad_df["temp_amb"] >= max([-273.15, metrics[metrics["column"] == "temp_amb"]["lower_lim"].values[0]])) & 
                                    (non_failed_rad_df["temp_amb"] <= metrics[metrics["column"] == "temp_amb"]["upper_lim"].values[0]) &
                                (non_failed_rad_df["temp_panel1"] >= max([-273.15, metrics[metrics["column"] == "temp_panel1"]["lower_lim"].values[0]])) & 
                                    (non_failed_rad_df["temp_panel1"] <= metrics[metrics["column"] == "temp_panel1"]["upper_lim"].values[0]) &
                                (non_failed_rad_df["temp_panel2"] >= max([-273.15, metrics[metrics["column"] == "temp_panel2"]["lower_lim"].values[0]])) & 
                                    (non_failed_rad_df["temp_panel2"] <= metrics[metrics["column"] == "temp_panel2"]["upper_lim"].values[0])]
        rad_df.loc[non_failed_rad_df.index.difference(clean_rad_df.index), "status_srl"] = 2
        outliers_iqr += non_failed_rad_df.shape[0] - clean_rad_df.shape[0]

        # Se separan los registros de la estación que mide la humedad y los registros de viento y se etiquetan los registros fuera del rango natural
        hum_df = meteo_df[meteo_df["dispositivo_id"] == 41][["datetime_utc","hum_rel"]]
        hum_df = hum_df.dropna(subset=["hum_rel"])
        hum_df["status_srl"] = 0
        hum_df["status_srl"] = np.where((hum_df["hum_rel"] >= 0) & (hum_df["hum_rel"] <= 100), 0, 3)
   
        # Se separan los registros de la estación que mide la velocidad y dirección del viento y se etiquetan los registros fuera del rango natural
        ane_df = meteo_df[((meteo_df["dispositivo_id"] >= 50) & (meteo_df["dispositivo_id"] < 60)) | 
                                (meteo_df["dispositivo_id"] == 41)].copy()
        ane_df = ane_df.dropna(subset=["vel_viento", "dir_viento"])
        ane_df["status_srl"] = 0
        ane_df["status_srl"] = np.where((ane_df["vel_viento"] >= 0) & (ane_df["vel_viento"] <= 100) &
                                (ane_df["dir_viento"] >= 0) & (ane_df["dir_viento"] <= 360), 0, 3)
        # Se calculan los valores medios de viento y dirección para cada instante considerando solo aquellos registros naturales
        # y se etiquetan los registros que no se encuentren en el rango natural
        ane_agg_df = ane_df[ane_df["status_srl"] == 0][["datetime_utc", "vel_viento", "dir_viento"]].groupby("datetime_utc").mean().reset_index()
        ane_agg_df["status_srl"] = 0
        ane_df = ane_agg_df.merge(ane_df[["datetime_utc"]].drop_duplicates(), on = "datetime_utc", how = "right")
        ane_df["status_srl"] = np.where(ane_df["status_srl"].isnull(), 3, ane_df["status_srl"]).astype(int)

        # Se unen los dataframes de humedad y viento y se etiquetan los registros que no se encuentren en el rango natural
        merged_df = ane_df.merge(hum_df, on = "datetime_utc", how = "outer")
        merged_df["status_srl_x"] = merged_df["status_srl_x"].astype(int)
        merged_df["status_srl_y"] = merged_df["status_srl_y"].fillna(3).astype(int)
        merged_df["status_srl"] = np.where((merged_df["status_srl_x"] == 0) & (merged_df["status_srl_y"] == 0) , 0, 3)
        merged_df.drop(columns = ["status_srl_x", "status_srl_y"], inplace = True)
        
        # Se unen los dataframes de irradiancia y humedad y viento
        prep_meteo_df = pd.merge(rad_df.drop(columns=["vel_viento", "dir_viento", "hum_rel"]), merged_df, on = ["datetime_utc"], how = "left")
        prep_meteo_df["status_srl_y"] = prep_meteo_df["status_srl_y"].fillna(3).astype(int)
        prep_meteo_df["status_srl"] = np.where((prep_meteo_df["status_srl_y"] == 3), 3, prep_meteo_df["status_srl_x"]) 
        prep_meteo_df.drop(columns = ["status_srl_x", "status_srl_y"], inplace = True)

        # Cálculo de las irradiancias esperadas (mediante modelo) y comparativa con valor real debido a nubosidad
        prep_meteo_df["day_of_year"] = prep_meteo_df["datetime_utc"].dt.dayofyear
        prep_meteo_df["dia_año_sen"] = np.sin(prep_meteo_df["day_of_year"] * (2*np.pi/365))
        prep_meteo_df["dia_año_cos"] = np.cos(prep_meteo_df["day_of_year"] * (2*np.pi/365))

        prep_meteo_df["poa_pred"] = model_poa.predict(prep_meteo_df[['temp_amb', 'vel_viento', 'dir_viento',
                                                        'hum_rel', 'elevation', 'azimuth', 
                                                        'dia_año_sen', 'dia_año_cos', 'daylight']])
        prep_meteo_df["poa_pred"] = prep_meteo_df["poa_pred"].rolling(window=30).mean().fillna(0)
        prep_meteo_df["mean_poa"] = prep_meteo_df[["rad_poa"]].rolling(window=30).mean().fillna(0)
        prep_meteo_df["ratio_poa"] = np.where((prep_meteo_df["datetime_utc"] > (prep_meteo_df["sunrise"] + dt.timedelta(minutes = 15))) & 
                                            (prep_meteo_df["datetime_utc"] < (prep_meteo_df["sunset"] - dt.timedelta(minutes = 15))) &
                                            (np.abs(prep_meteo_df["mean_poa"] - prep_meteo_df["poa_pred"]) > 25), 
                                        prep_meteo_df["mean_poa"] * 100 / prep_meteo_df["poa_pred"], 
                                        100)
        
        prep_meteo_df["hor_pred"] = model_hor.predict(prep_meteo_df[['temp_amb', 'vel_viento', 'dir_viento',
                                                        'hum_rel', 'elevation', 'azimuth',
                                                        'dia_año_sen', 'dia_año_cos', "daylight"]])
        prep_meteo_df["hor_pred"] = prep_meteo_df["hor_pred"].rolling(window=30).mean().fillna(0)
        prep_meteo_df["mean_hor"] = prep_meteo_df[['rad_hor']].rolling(window=30).mean().fillna(0)
        prep_meteo_df["ratio_hor"] = np.where((prep_meteo_df["datetime_utc"] > (prep_meteo_df["sunrise"] + dt.timedelta(minutes = 15))) & 
                                            (prep_meteo_df["datetime_utc"] < (prep_meteo_df["sunset"] - dt.timedelta(minutes = 15))) &
                                            (np.abs(prep_meteo_df["mean_hor"] - prep_meteo_df["hor_pred"]) > 10), 
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
            prep_meteo_df = prep_meteo_df.drop(columns=[col for col in prep_meteo_df.columns if col not in keys_meteo])
            prep_meteo_df.to_sql('meteo', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes_meteo, chunksize = 100000)
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
    print("Número de registros con etiquetado por anomalía en el rango: ", prep_meteo_df[(prep_meteo_df["status_srl"] == 2) | (prep_meteo_df["status_srl"] == 3)].shape[0])
    print("Número de registros con etiquetado por malfuncionamiento de piranómetro: ", prep_meteo_df[(prep_meteo_df["status_srl"] == 1)].shape[0])
    print("Número de registros cargados inicialmente: ", count)
