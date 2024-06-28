import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import json
import sys
import os
import shutil
import requests
from tqdm import tqdm


def procesar_directorio(initial_path, cadena):
    """
    Busca mediante recursividad en el directorio inicial todos los directorios hasta encontrar un directorio que contenga
    la cadena de texto "cadena". Una vez encontrado, busca en el directorio todos los ficheros .csv y los concatena en un
    único dataframe. Por último, mueve el directorio encontrado a la carpeta "Procesados" y devuelve el dataframe.
    """
    df = pd.DataFrame()
    for filename in os.listdir(initial_path):
        complete_path = os.path.join(initial_path, filename)
        if (os.path.isdir(complete_path)) and (cadena in filename):
            for internal_filename in os.listdir(complete_path):
                if internal_filename.endswith(".csv"):
                    csv_path = os.path.join(complete_path, internal_filename)
                    try:
                        temp_df = pd.read_csv(csv_path, delimiter = ";")
                        if temp_df.shape[1] == 1:
                            temp_df = pd.read_csv(csv_path, delimiter = ",")
                    except pd.errors.ParserError:
                        temp_df = pd.read_csv(csv_path, delimiter = ",")
                    df = pd.concat([df, temp_df])
                    if not os.path.exists(complete_path.replace("Nuevos", "Procesados")):
                        os.makedirs(complete_path.replace("Nuevos", "Procesados"))
                    shutil.move(csv_path, csv_path.replace("Nuevos", "Procesados"))   
        elif (os.path.isfile(complete_path)) and (cadena in filename):
            csv_path = os.path.join(initial_path, filename)
            try:
                temp_df = pd.read_csv(csv_path, delimiter = ";")
                if temp_df.shape[1] == 1:
                    temp_df = pd.read_csv(csv_path, delimiter = ",")
            except pd.errors.ParserError:
                temp_df = pd.read_csv(csv_path, delimiter = ",")
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
    df = procesar_directorio(data_path, 'meteo')
    if df.empty:
        print(f"No se han encontrado nuevos ficheros para procesar en {data_path}")
        sys.exit()
    print(f"Se han encontrado {df.shape[0]} nuevos registros para procesar")

    # Conversión de tipos y normalizacion de nombres de columnas
    columns = []
    for column in df.columns:
        try:
            df[column] = df[column].str.replace(",", ".").astype(float)
            if df[column].iloc[0].is_integer():
                df[column] = df[column].astype(int)
        except Exception as e:
            pass
        columns.append(column.strip().replace(" ", "_").lower())
    df.columns = columns

    # Parseo de fechas y ordenación del dataframe
    for column in df.columns:
        if ('date' in column.lower()):
            df[column] = pd.to_datetime(df[column], utc=True)
            df = df.rename(columns = {column: "datetime_utc", "device_id": "dispositivo_id"})

    # Descarte de registros anteriores a 2022
    df = df[df["datetime_utc"].dt.year >= 2022]

    # Conexión a la base de datos y carga de datos
    try:
        password = params['password'].replace('@', '%40')
        engine = create_engine(f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}')
        print(f"Conexión a la base de datos {params['dbname']} (esquema {schema_name}) establecida")
    except Exception as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    # Eliminación de registros duplicados
    num_duplicates = df[df.duplicated(subset = ["datetime_utc", "dispositivo_id"])].shape[0]
    df = df.drop_duplicates(subset = ["datetime_utc", "dispositivo_id"], keep = "last").reset_index(drop = True)

    # Comprobación de la existencia de registros ya presentes en la base de datos
    check_query = f"SELECT datetime_utc, dispositivo_id FROM {schema_name}.meteo_raw"
    check_df = pd.read_sql_query(check_query, engine)
    check_df["datetime_utc"] = pd.to_datetime(check_df["datetime_utc"], utc=True)
    merged_df = df.merge(check_df, how='left', indicator=True, left_on = ["datetime_utc", "dispositivo_id"], right_on = ["datetime_utc", "dispositivo_id"])
    df = merged_df[merged_df["_merge"] == "left_only"]
    df = df.drop(columns = "_merge")

    print(f"Se han encontrado {num_duplicates} registros duplicados")
    print(f"Se han encontrado {merged_df.shape[0] - df.shape[0]} registros ya existentes en la base de datos")

    # Ordenación del dataframe por fecha y dispositivo
    df = df.sort_values(by = ["datetime_utc", "dispositivo_id"])\
            .reset_index(drop = True)
    
    # Asignación de id de parque
    df["parque_id"] = 1

    # Volcado del dataframe en la base de datos
    try:
        df = df.rename(columns={"rad_cel1": "rad_celda1", 
                            "rad_cel2": "rad_celda2"})
        dtypes = {
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
            'hum_rel': sqlalchemy.types.Float(precision=3, asdecimal=True)}
        keys = list(dtypes.keys())
        df = df.drop(columns=[col for col in df.columns if col not in keys])
        df.to_sql('meteo_raw', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes)
    except Exception as error:
        print(f"Error en volcado del dataframe: \n\t{error}")
        sys.exit()

    # Calculo de horas de salida y puesta del sol para los nuevos días cargados y volcado en la base de datos
    # Carga de fechas de los datos a procesar y coordenadas del parque
    # Se descartan las fechas que ya existen en la base de datos
    fechas_nuevas = df["datetime_utc"].dt.date.unique()
    fechas_viejas = pd.to_datetime(check_df["datetime_utc"], utc = True).dt.date.unique()
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
        r = requests.get('https://api.sunrise-sunset.org/json', params={'lat': lat, 'lng': lon, 'date': fecha[1].strftime("%Y-%m-%d")}).json()['results']
        hora_salida_sol.append(pd.to_datetime(fecha[1].strftime("%Y-%m-%d") + " " + r['sunrise'], utc = True))
        hora_puesta_sol.append(pd.to_datetime(fecha[1].strftime("%Y-%m-%d") + " " + r['sunset'], utc = True))
    daylight = pd.DataFrame({"date": fechas["date"], "sunrise": hora_salida_sol, "sunset": hora_puesta_sol})
    daylight.to_sql('daylight', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes)