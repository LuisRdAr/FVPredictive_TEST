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
                        temp_df = pd.read_csv(csv_path, delimiter = ";", low_memory=False)
                        if temp_df.shape[1] == 1:
                            temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory=False)
                    except pd.errors.ParserError:
                        temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory=False)
                    df = pd.concat([df, temp_df])
                    if not os.path.exists(complete_path.replace("Nuevos", "Procesados")):
                        os.makedirs(complete_path.replace("Nuevos", "Procesados"))
                    shutil.move(csv_path, csv_path.replace("Nuevos", "Procesados"))   
        elif (os.path.isfile(complete_path)) and (cadena in filename):
            csv_path = os.path.join(initial_path, filename)
            try:
                temp_df = pd.read_csv(csv_path, delimiter = ";", low_memory=False)
                if temp_df.shape[1] == 1:
                    temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory=False)
            except pd.errors.ParserError:
                temp_df = pd.read_csv(csv_path, delimiter = ",", low_memory=False)
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
    df = procesar_directorio(data_path, 'IAZ')
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
    inv_df = pd.DataFrame()
    for grupo, columnas_grupo in grupos_columnas.items():
        single_inv_df = pd.concat([df["Datetime"], df[columnas_grupo].copy().rename(columns={col: col.split(" - ")[1] for col in columnas_grupo})], axis = 1)
        single_inv_df["disp_name"] = grupo
        inv_df = pd.concat([inv_df, single_inv_df], axis=0)

    # Asignación de identificadores a los dispositivos y parque
    inv_df["parque_id"] = inv_df["disp_name"].str[3]
    inv_df["dispositivo_id"] = inv_df.groupby("parque_id")["disp_name"].transform(lambda x: x.astype("category").cat.codes + 41)
    inv_df.drop("disp_name", axis=1, inplace=True)
    inv_df["parque_id"] = inv_df["parque_id"].astype('uint8')
    inv_df["dispositivo_id"] = inv_df["dispositivo_id"].astype('uint8')

    # Procesado de columnas: conversión de tipos
    columns = []
    for column in inv_df.columns:
        try:
            inv_df[column] = inv_df[column].str.replace(",", ".").astype(float)
            if inv_df[column].iloc[0].is_integer():
                inv_df[column] = inv_df[column].astype(int)
        except Exception as e:
            pass
        columns.append(column.strip().replace(" ", "_").split("(")[0].lower())
    inv_df.columns = columns

    inv_df.rename(columns={'datetime': 'datetime_utc', 
                                'dc_voltage': 'vol_dc_bus', 
                                'voltage_phase_ab': 'vol_12',
                                'voltage_phase_bc': 'vol_23',
                                'voltage_phase_ca': 'vol_31',
                                'current_phase_a': 'amp_1',
                                'current_phase_b': 'amp_2',
                                'current_phase_c': 'amp_3',
                                'active_power': 'potencia_act',
                                'reactive_power': 'potencia_rea'}, inplace = True)
    inv_df.rename(columns={column: f"amp_dc_in{str(int(column.split('_')[-1]))}" for column in inv_df.columns if "current_dc_chanel" in column}, inplace=True)

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
    inv_df = inv_df.drop(columns=[col for col in inv_df.columns if col not in keys])

    # Parseo de fechas y eliminación de registros duplicados
    inv_df["datetime_utc"] = pd.to_datetime(inv_df["datetime_utc"], utc = True)
    num_na_pot_act = inv_df[inv_df["potencia_act"].isna()].shape[0]
    inv_df = inv_df.dropna(subset = ["potencia_act"])
    num_duplicates = inv_df[inv_df.duplicated(subset = ["datetime_utc", "dispositivo_id"])].shape[0]
    inv_df = inv_df.drop_duplicates(subset = ["datetime_utc", "dispositivo_id"], keep = "last").reset_index(drop = True)

    # Conexión a la base de datos y carga de datos
    try:
        password = params['password'].replace('@', '%40')
        engine = create_engine(f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}')
        print(f"Conexión a la base de datos {params['dbname']} (esquema {schema_name}) establecida")
    except Exception as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    # Comprobación de la existencia de registros duplicados en la base de datos
    check_query = f"SELECT datetime_utc, parque_id, dispositivo_id FROM {schema_name}.inversores_raw"
    check_df = pd.read_sql_query(check_query, engine)
    check_df["datetime_utc"] = pd.to_datetime(check_df["datetime_utc"], utc=True)
    merged_df = inv_df.merge(check_df, how='left', indicator=True, left_on = ["datetime_utc", "parque_id", "dispositivo_id"], right_on = ["datetime_utc", "parque_id", "dispositivo_id"])
    inv_df = merged_df[merged_df["_merge"] == "left_only"]
    inv_df = inv_df.drop(columns = "_merge")

    print(f"Se han encontrado {num_duplicates} registros duplicados")
    print(f"Se han encontrado {merged_df.shape[0] - inv_df.shape[0]} registros ya existentes en la base de datos")

    # Ordenación del dataframe por fecha y dispositivo
    inv_df.sort_values(by = ["datetime_utc", "dispositivo_id"], inplace = True)
    inv_df.reset_index(drop = True, inplace = True)
    
    # Volcado del dataframe en la base de datos
    try:
        inv_df.to_sql('inversores_raw', engine, if_exists='append', index = False, schema = schema_name, dtype=dtypes)
    except Exception as error:
        print(f"Error en volcado del dataframe: \n\t{error}")
        sys.exit()
