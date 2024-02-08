import pandas as pd
import numpy as np
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
                        temp_df["parque_id"] = filename.split("_")[0]
                    except pd.errors.ParserError:
                        temp_df = pd.read_csv(csv_path, delimiter = ",")
                        temp_df["parque_id"] = filename.split("_")[0]
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
                temp_df["parque_id"] = filename.split("_")[0]
            except pd.errors.ParserError:
                temp_df = pd.read_csv(csv_path, delimiter = ",")
                temp_df["parque_id"] = filename.split("_")[0]
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
    root_path = "/home/upo/Desktop/Test_FVPredictive/FVPredictive_TEST/Alcazar"
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
    df = procesar_directorio(data_path, 'REG')
    if df.empty:
        print(f"No se han encontrado nuevos ficheros para procesar en {data_path}")
        sys.exit()
    print(f"Se han encontrado {df.shape[0]} nuevos registros para procesar")

    # Procesado de columnas
    df.rename(columns = {"REE - Active Power(MW)": "consigna_pot_act_planta",
                         "REE - Reason()": "motivo",
                         "Datetime": "datetime_utc"}, inplace = True)
    df["consigna_pot_act_planta"] = df["consigna_pot_act_planta"].str.replace(",", ".").astype(float)
    df.loc[df["consigna_pot_act_planta"] > 1000, "consigna_pot_act_planta"] /= 1000
    df.loc[(df["consigna_pot_act_planta"].isna()) & 
           (df["motivo"] == "NO LIMITACIÓN"), "consigna_pot_act_planta"] = df["consigna_pot_act_planta"].max()
    df["motivo"] = np.where((df["motivo"] == "NO LIMITACIÓN"), "NO LIMITACIÓN", 
                            np.where((df["motivo"] == "CONGESTIÓN EN LA EVACUACIÓN DE GENERACIÓN"), "CONGESTIÓN EN LA EVACUACIÓN DE GENERACIÓN", "OTRAS"))
    df["motivo"] = df["motivo"].astype("category").cat.codes
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["dispositivo_id"] = 76
    df["nombre_dispositivo"] = "Despacho"
    df["parque_id"] = df["parque_id"].str[-1].astype(int)

    num_duplicates = df[df.duplicated(subset = ["parque_id", "datetime_utc", "dispositivo_id"])].shape[0]
    df = df.drop_duplicates(subset = ["parque_id", "datetime_utc", "dispositivo_id"], keep = "last").reset_index(drop = True)
    num_na = df["consigna_pot_act_planta"].isna().sum()
    df.dropna(subset = ["consigna_pot_act_planta"], inplace = True)

    # Conexión a la base de datos y carga de datos
    try:
        password = params['password'].replace('@', '%40')
        engine = create_engine(f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}')
        print(f"Conexión a la base de datos {params['dbname']} (esquema {schema_name}) establecida")
    except Exception as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    # Comprobación de la existencia de registros duplicados en la base de datos
    check_query = f"SELECT datetime_utc, dispositivo_id FROM {schema_name}.ree_raw"
    check_df = pd.read_sql_query(check_query, engine)
    check_df["datetime_utc"] = pd.to_datetime(check_df["datetime_utc"], utc=True)
    merged_df = df.merge(check_df, how='left', indicator=True, left_on = ["datetime_utc", "dispositivo_id"], right_on = ["datetime_utc", "dispositivo_id"])
    df = merged_df[merged_df["_merge"] == "left_only"]
    df = df.drop(columns = "_merge")

    print(f"Se han encontrado {num_duplicates} registros duplicados")
    print(f"Se han encontrado {num_na} registros con valores nulos en la consigna de potencia activa")
    print(f"Se han encontrado {merged_df.shape[0] - df.shape[0]} registros ya existentes en la base de datos")

    # Ordenación del dataframe por fecha y dispositivo
    df = df.sort_values(by = ["datetime_utc", "parque_id", "dispositivo_id"])\
            .reset_index(drop = True)

    # Volcado del dataframe en la base de datos
    try:
        dtypes = {
            'parque_id': sqlalchemy.types.SMALLINT(),
            'dispositivo_id': sqlalchemy.types.SMALLINT(),
            'datetime_utc': sqlalchemy.types.DateTime(timezone=True),
            'motivo': sqlalchemy.types.SMALLINT(),
            'consigna_pot_act_planta': sqlalchemy.types.Float(precision=3, asdecimal=True)}
        keys = list(dtypes.keys())
        df = df.drop(columns=[col for col in df.columns if col not in keys])
        df.to_sql('ree_raw', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes)

    except Exception as error:
        print(f"Error en volcado del dataframe: \n\t{error}")
        sys.exit()