import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import json
import sys
import os
import shutil


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
    df = procesar_directorio(data_path, 'inv')
    if df.empty:
        print(f"No se han encontrado nuevos ficheros para procesar en {data_path}")
        sys.exit()
    print(f"Se han encontrado {df.shape[0]} nuevos registros para procesar")

    # Procesado de columnas: conversión de tipos
    columns = []
    for column in df.columns:
        try:
            df[column] = df[column].str.replace(",", ".").astype(float)
            if df[column].iloc[0].is_integer():
                df[column] = df[column].astype(int)
        except Exception as e:
            pass
        if "(i.)" in column:
            columns.append(column[:-4].strip().replace(" ", "_").lower())
        else:
            columns.append(column.strip().replace(" ", "_").lower())
    df.columns = columns

    # Parseo de fechas y ordenación del dataframe
    for column in df.columns:
        if ('date' in column.lower()):
            df[column] = pd.to_datetime(df[column], utc=True)
            df = df.rename(columns = {column: "datetime_utc", "device_id": "dispositivo_id"})

    num_na_pot_act = df[df["pot_act"].isna()].shape[0]
    df = df.dropna(subset = ["pot_act"])
    num_duplicates = df[df.duplicated(subset = ["datetime_utc", "dispositivo_id"])].shape[0]
    df = df.drop_duplicates(subset = ["datetime_utc", "dispositivo_id"], keep = "last").reset_index(drop = True)

    # Conexión a la base de datos y carga de datos
    try:
        password = params['password'].replace('@', '%40')
        engine = create_engine(f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}')
        print(f"Conexión a la base de datos {params['dbname']} (esquema {schema_name}) establecida")
    except Exception as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    # Comprobación de la existencia de registros duplicados en la base de datos
    check_query = f"SELECT datetime_utc, dispositivo_id FROM {schema_name}.inversores_raw"
    check_df = pd.read_sql_query(check_query, engine)
    check_df["datetime_utc"] = pd.to_datetime(check_df["datetime_utc"], utc=True)
    merged_df = df.merge(check_df, how='left', indicator=True, left_on = ["datetime_utc", "dispositivo_id"], right_on = ["datetime_utc", "dispositivo_id"])
    df = merged_df[merged_df["_merge"] == "left_only"]
    df = df.drop(columns = "_merge")
    print(f"Se han encontrado {num_na_pot_act} registros con potencia activa nula")
    print(f"Se han encontrado {num_duplicates} registros duplicados")
    print(f"Se han encontrado {merged_df.shape[0] - df.shape[0]} registros ya existentes en la base de datos")

    # Ordenación del dataframe por fecha y dispositivo
    df = df.sort_values(by = ["datetime_utc", "dispositivo_id"])\
            .reset_index(drop = True)
    
    if "parque_id" not in df.columns:
        df["parque_id"] = 1

    # Volcado del dataframe a la base de datos
    try:
        df = df.rename(columns={"descripcion": "descripcion_parque", 
                            "localizacion": "localizacion_parque", 
                            "nombre": "nombre_dispositivo", 
                            "descripcion.1": "descripcion_dispositivo", 
                            "pot_act": "potencia_act", 
                            "pot_reac": "potencia_rea", 
                            "volt_12": "vol_12",
                            "volt_13": "vol_13",
                            "volt_23": "vol_23",
                            "lim_react": "lim_rea", 
                            "volt_dc_bus": "vol_dc_bus"})
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
            'amp_dc_in1': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in2': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in3': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in4': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in5': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in6': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in7': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in8': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in9': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in10': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in11': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in12': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in13': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in14': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in15': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in16': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in17': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in18': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in19': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in20': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in21': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in22': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in23': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'amp_dc_in24': sqlalchemy.types.Float(precision=3, asdecimal=True),
            'aisl_dc': sqlalchemy.types.INTEGER(),
            'energia_dia': sqlalchemy.types.INTEGER()}
        keys = list(dtypes.keys())
        df = df.drop(columns=[col for col in df.columns if col not in keys])
        df.to_sql('inversores_raw', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes, chunksize = 100000)

    except Exception as error:
        print(f"Error en volcado del dataframe: \n\t{error}")
        sys.exit()
