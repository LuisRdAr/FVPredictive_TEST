import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import json
import sys
import os
import shutil
import requests
from tqdm import tqdm
import gc

def procesar_directorio(initial_path, cadena):
    """
    Busca mediante recursividad en el directorio inicial todos los directorios hasta encontrar un directorio que contenga
    la cadena de texto "cadena". Una vez encontrado, busca en el directorio todos los ficheros .csv y los concatena en un
    único dataframe. Por último, mueve el directorio encontrado a la carpeta "Procesados" y devuelve el dataframe.
    
    Parameters:
        initial_path (str): Ruta del directorio inicial.
        cadena (str): Cadena de texto a buscar en los nombres de los directorios.
    
    Returns:
        pandas.DataFrame: Dataframe que contiene la concatenación de todos los ficheros .csv encontrados.
    """
    
    df = pd.DataFrame()
    for filename in os.listdir(initial_path):
        complete_path = os.path.join(initial_path, filename)
        if (os.path.isdir(complete_path)) and (cadena in filename):
            for internal_filename in os.listdir(complete_path):
                if internal_filename.endswith(".csv"):
                    csv_path = os.path.join(complete_path, internal_filename)
                    try:
                        temp_df = pd.read_csv(csv_path, delimiter = ";", low_memory = False)
                        if temp_df.shape[1] == 1:
                            temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory = False)
                    except pd.errors.ParserError:
                        temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory = False)
                    df = pd.concat([df, temp_df])
                    if not os.path.exists(complete_path.replace("Nuevos", "Procesados")):
                        os.makedirs(complete_path.replace("Nuevos", "Procesados"))
                    shutil.move(csv_path, csv_path.replace("Nuevos", "Procesados"))   
        elif (os.path.isfile(complete_path)) and (cadena in filename):
            csv_path = os.path.join(initial_path, filename)
            try:
                temp_df = pd.read_csv(csv_path, delimiter = ";", low_memory = False)
                if temp_df.shape[1] == 1:
                    temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory = False)
            except pd.errors.ParserError:
                temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory = False)
            df = pd.concat([df, temp_df])
            if not os.path.exists(initial_path.replace("Nuevos", "Procesados")):
                os.makedirs(initial_path.replace("Nuevos", "Procesados"))
            shutil.move(csv_path, csv_path.replace("Nuevos", "Procesados"))
        elif (os.path.isdir(complete_path)):
            partial_df = procesar_directorio(complete_path, cadena)
            if not partial_df.empty:
                df = pd.concat([df, partial_df])
        else:
            continue
    return df

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

    # Carga y comprobación de la existencia de nuevos ficheros para procesar
    df = procesar_directorio(data_path, 'CL_UNCL')
    if df.empty:
        print(f"No se han encontrado nuevos ficheros para procesar en {data_path}")
        sys.exit()
    print(f"Se han encontrado {df.shape[0]} nuevos registros para procesar")

    columnas = df.columns

    # Crear un diccionario para almacenar grupos de columnas y los nombres de las columnas para el dataframe reestructurado
    grupos_columnas = {}
    col_names = []

    # Iterar sobre las columnas y agruparlas por el patrón común
    for columna in columnas:
        if columna != "Datetime":
            if columna.split(" - ")[1] not in col_names:
                col_names.append(columna.split(" - ")[1])
            else:
                pass
        
            grupo = columna.split(" - ")[0]
            if grupo not in grupos_columnas:
                grupos_columnas[grupo] = [columna]
            else:
                grupos_columnas[grupo].append(columna)

    # Crear un dataframe para cada grupo de columnas y concatenarlos en un único dataframe
    meteo_aux_df = pd.DataFrame()
    for grupo, columnas_grupo in grupos_columnas.items():
        single_meteo_df = pd.concat([df["Datetime"], df[columnas_grupo].copy().rename(columns={col: col.split(" - ")[1] for col in columnas_grupo})], axis = 1)
        single_meteo_df["disp_name"] = grupo
        meteo_aux_df = pd.concat([meteo_aux_df, single_meteo_df], axis=0)

    # Parseo de fechas y ordenación del dataframe
    meteo_aux_df["Datetime"] = pd.to_datetime(meteo_aux_df["Datetime"], utc = True)
    meteo_aux_df = meteo_aux_df.dropna(subset = ['Temperature Uncleaned Cell(ºC)', 'Temperature Cleaned Cell(ºC)']).sort_values(by = ["Datetime", "disp_name"]).reset_index(drop = True)
    del single_meteo_df
    del df
    gc.collect()

    # Carga y comprobación de la existencia de nuevos ficheros para procesar
    df = procesar_directorio(data_path, 'MET')
    if df.empty:
        print(f"No se han encontrado nuevos ficheros para procesar en {data_path}")
        sys.exit()
    print(f"Se han encontrado {df.shape[0]} nuevos registros para procesar")

    columnas = df.columns

    # Crear un diccionario para almacenar grupos de columnas y los nombres de las columnas para el dataframe reestructurado
    grupos_columnas = {}
    col_names = []

    # Iterar sobre las columnas y agruparlas por el patrón común
    for columna in columnas:
        if columna != "Datetime":
            if columna.split(" - ")[1] not in col_names:
                col_names.append(columna.split(" - ")[1])
            else:
                pass
        
            grupo = columna.split(" - ")[0]
            if grupo not in grupos_columnas:
                grupos_columnas[grupo] = [columna]
            else:
                grupos_columnas[grupo].append(columna)

    # Crear un dataframe para cada grupo de columnas y concatenarlos en un único dataframe
    meteo_df = pd.DataFrame()
    for grupo, columnas_grupo in grupos_columnas.items():
        single_meteo_df = pd.concat([df["Datetime"], df[columnas_grupo].copy().rename(columns={col: col.split(" - ")[1] for col in columnas_grupo})], axis = 1)
        single_meteo_df["disp_name"] = grupo
        meteo_df = pd.concat([meteo_df, single_meteo_df], axis=0)

    # Parseo de fechas y ordenación del dataframe
    meteo_df["Datetime"] = pd.to_datetime(meteo_df["Datetime"], utc = True)
    meteo_df = meteo_df.dropna(subset = ["Radiation Pyranometer 2"]).sort_values(by = ["Datetime", "disp_name"]).reset_index(drop = True)
    num_duplicates = meteo_df[meteo_df.duplicated(subset = ["Datetime", "disp_name"])].shape[0]
    meteo_df = meteo_df.drop_duplicates(subset = ["Datetime", "disp_name"], keep = "last").reset_index(drop = True)
    meteo_df = pd.merge(meteo_df, meteo_aux_df, how = "inner", on = ["Datetime", "disp_name"])
    del meteo_aux_df
    del single_meteo_df
    del df
    gc.collect()

    # Borrado de duplicados
    num_duplicates = meteo_df[meteo_df.duplicated(subset = ["Datetime", "disp_name"])].shape[0]
    meteo_df = meteo_df.drop_duplicates(subset = ["Datetime", "disp_name"], keep = "last").reset_index(drop = True)

    # Asignación de identificadores a los dispositivos y parque
    meteo_df["parque_id"] = meteo_df["disp_name"].str[3]
    meteo_df["dispositivo_id"] = meteo_df["dispositivo_id"] = meteo_df.groupby("parque_id")["disp_name"].transform(lambda x: x.astype("category").cat.codes + 41)
    meteo_df.drop("disp_name", axis=1, inplace=True)

    # Procesado de columnas: conversión de tipos
    columns = []
    for column in meteo_df.columns:
        try:
            meteo_df[column] = meteo_df[column].str.replace(",", ".").astype(float)
            if meteo_df[column].iloc[0].is_integer():
                meteo_df[column] = meteo_df[column].astype(int)
        except Exception as e:
            pass
        columns.append(column.strip().replace(" ", "_").split("(")[0].lower())
    meteo_df.columns = columns

    meteo_df = meteo_df.rename(columns={'datetime': "datetime_utc", 
                                'wind_speed_1': "vel_viento", 
                                'wind_direction_1': "dir_viento",
                                'average_external_ambient_temperature': "temp_amb", 
                                'average_humidity': "hum_rel",
                                'average_atmospheric_pressure': "presion_atm", 
                                'radiation_pyranometer_1': "rad_hor",
                                'radiation_pyranometer_2': "rad_poa",
                                'average atmospheric pressure': "presion_atm",
                                'temperature_uncleaned_cell': "temp_panel1",
                                'temperature_cleaned_cell': "temp_panel2",
                                'radiation_uncleaned_cell': "rad_celda1",
                                'radiation_cleaned_cell': "rad_celda2"})

    # Conexión a la base de datos y carga de datos
    try:
        password = params['password'].replace('@', '%40')
        engine = create_engine(f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}')
        print(f"Conexión a la base de datos {params['dbname']} (esquema {schema_name}) establecida")
    except Exception as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    # Comprobación de la existencia de registros duplicados en la base de datos
    check_query = f"SELECT datetime_utc, parque_id, dispositivo_id FROM {schema_name}.meteo_raw"
    check_df = pd.read_sql_query(check_query, engine)
    check_df["datetime_utc"] = pd.to_datetime(check_df["datetime_utc"], utc = True)
    merged_df = meteo_df.merge(check_df, how='left', indicator=True, left_on = ["datetime_utc", "parque_id","dispositivo_id"], right_on = ["datetime_utc", "parque_id", "dispositivo_id"])
    meteo_df = merged_df[merged_df["_merge"] == "left_only"]
    meteo_df = meteo_df.drop(columns = "_merge")

    print(f"Se han encontrado {num_duplicates} registros duplicados")
    print(f"Se han encontrado {merged_df.shape[0] - meteo_df.shape[0]} registros ya existentes en la base de datos")

    del merged_df
    fechas_viejas = pd.to_datetime(check_df["datetime_utc"], utc = True).dt.date.unique()
    del check_df
    gc.collect()

    # Ordenación del dataframe por fecha y dispositivo
    meteo_df.sort_values(by = ["datetime_utc", "dispositivo_id"], inplace = True)
    meteo_df.reset_index(drop = True, inplace = True)
    
    # Volcado del dataframe en la base de datos
    try:
        dtypes = {
            'parque_id': sqlalchemy.types.SMALLINT(),
            'descripcion_parque': sqlalchemy.types.VARCHAR(length=25),
            'localizacion_parque': sqlalchemy.types.VARCHAR(length=25),
            'potencia_max': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'num_paneles': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'dispositivo_id': sqlalchemy.types.SMALLINT(),
            'nombre_dispositivo': sqlalchemy.types.VARCHAR(length=25),
            'ref': sqlalchemy.types.VARCHAR(length=25),
            'ubicacion': sqlalchemy.types.VARCHAR(length=25),
            'descripcion_dispositivo': sqlalchemy.types.VARCHAR(length=25),
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
            'presion_atm': sqlalchemy.types.Float(precision=3, asdecimal=True),}
        keys = list(dtypes.keys())
        meteo_df = meteo_df.drop(columns=[col for col in meteo_df.columns if col not in keys])
        meteo_df.to_sql('meteo_raw', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes)
    except Exception as error:
        print(f"Error en volcado del dataframe: \n\t{error}")
        sys.exit()

    # Calculo de horas de salida y puesta del sol para los nuevos días cargados y volcado en la base de datos
    # Carga de fechas de los datos a procesar y coordenadas del parque
    # Se descartan las fechas que ya existen en la base de datos
    fechas_nuevas = meteo_df["datetime_utc"].dt.date.unique()
    fechas = pd.DataFrame(fechas_nuevas, columns = ["date"]).merge(pd.DataFrame(fechas_viejas, columns = ["date"]), how = "left", indicator = True)
    fechas = fechas[fechas["_merge"] == "left_only"].drop(columns = "_merge")
    consulta_sql = f"""SELECT latitud, longitud FROM {schema_name}.parques;"""
    gps = pd.read_sql_query(consulta_sql, engine).values
    lat = gps[0][0]
    lon = gps[0][1]

    # Cálculo de las horas de salida y puesta del sol para las coords dadas usando la API sunrise_sunset y volcado en la base de datos
    hora_salida_sol = []
    hora_puesta_sol = []
    print("Calculando horas de salida y puesta del sol")
    for fecha in tqdm(fechas.itertuples(), total = fechas.shape[0]):
        consulta_sql = f"""SELECT * FROM {schema_name}.daylight WHERE date = '{fecha[1].strftime("%Y-%m-%d")}';"""
        check_date = pd.read_sql_query(consulta_sql, engine)
        if not check_date.empty:
            continue
        r = requests.get('https://api.sunrise-sunset.org/json', params={'lat': lat, 'lng': lon, 'date': fecha[1].strftime("%Y-%m-%d")}).json()['results']
        hora_salida_sol.append(pd.to_datetime(fecha[1].strftime("%Y-%m-%d") + " " + r['sunrise'], utc = True))
        hora_puesta_sol.append(pd.to_datetime(fecha[1].strftime("%Y-%m-%d") + " " + r['sunset'], utc = True))
    daylight = pd.DataFrame({"date": fechas["date"], "sunrise": hora_salida_sol, "sunset": hora_puesta_sol})
    daylight.to_sql('daylight', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes)