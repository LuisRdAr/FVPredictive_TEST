import pandas as pd
import psycopg2
import json
import sys
import os
import shutil
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
                    temp_df = pd.read_csv(csv_path, delimiter = ";")
                    df = pd.concat([df, temp_df])
            if not os.path.exists(complete_path.replace("Nuevos", "Procesados")):
                os.makedirs(complete_path.replace("Nuevos", "Procesados"))
            shutil.move(complete_path, complete_path.replace("Nuevos", "Procesados"))   
        elif (os.path.isfile(complete_path)) and (cadena in filename):
            csv_path = os.path.join(initial_path, filename)
            temp_df = pd.read_csv(csv_path, delimiter = ";")
            if temp_df.shape[1] == 1:
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
    data_path = os.path.join(root_path, params["data_path"])
    schema_name = params["schema"]

    # Carga y comprobación de la existencia de nuevos ficheros para procesar
    df = procesar_directorio(data_path, 'inv')
    if df.empty:
        print(f"No se han encontrado nuevos ficheros para procesar en {data_path}")
        sys.exit()

    # Parseo de fechas y ordenación del dataframe
    for column in df.columns:
        if ('date' in column.lower()):
            df[column] = pd.to_datetime(df[column], utc=True)
            df = df.rename(columns = {column: "datetime", "device_id": "dispositivo_id"})
    df = df.sort_values(by = ["datetime", "dispositivo_id"])\
            .reset_index(drop = True)

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

    df = df.dropna(subset = ["pot_act"])

    # Conexión a la base de datos y carga de datos
    try:
        conn = psycopg2.connect(
            database = params['dbname'],
            user = params['user'],
            password = params['password'],
            host = params['host'],  
            port = params['port'])
        cur = conn.cursor()
        print("Conexión exitosa a PostgreSQL")
    except (Exception, psycopg2.Error) as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
        sys.exit()

    for row in tqdm(df.itertuples(), total = df.shape[0]):
        try:
            cur.execute(f"""INSERT INTO {schema_name}.inversores_raw(
                    parque_id, descripcion_parque, localizacion_parque, potencia_max, 
                    num_paneles, dispositivo_id, nombre_dispositivo, ref, ubicacion,
                    descripcion_dispositivo, datetime_utc, med_id, status, alarma, estado, 
                    potencia_act, potencia_rea, cos_phi, vol_12, vol_13, vol_23, amp_1, 
                    amp_2, amp_3, frec, lim_act, lim_rea, vol_dc_bus, amp_dc_in1, amp_dc_in2,
                    amp_dc_in3, amp_dc_in4, amp_dc_in5, amp_dc_in6, amp_dc_in7, amp_dc_in8, 
                    amp_dc_in9, amp_dc_in10, amp_dc_in11, amp_dc_in12, amp_dc_in13, amp_dc_in14,
                    amp_dc_in15, amp_dc_in16, amp_dc_in17, amp_dc_in18, amp_dc_in19, amp_dc_in20, 
                    amp_dc_in21, amp_dc_in22, amp_dc_in23, amp_dc_in24, aisl_dc, energia_dia)
                VALUES 
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                (row.parque_id, row.descripcion_parque, row.localizacion_parque, row.potencia_max, 
                row.num_paneles, row.dispositivo_id, row.nombre_dispositivo, row.ref, row.ubicacion,
                row.descripcion_dispositivo, row.datetime, row.med_id, row.status, row.alarma, row.estado,
                row.pot_act, row.pot_reac, row.cos_phi, row.volt_12, row.volt_13, row.volt_23, row.amp_1,
                row.amp_2, row.amp_3, row.frec, row.lim_act, row.lim_react, row.volt_dc_bus, row.amp_dc_in1, 
                row.amp_dc_in2, row.amp_dc_in3, row.amp_dc_in4, row.amp_dc_in5, row.amp_dc_in6, row.amp_dc_in7,
                row.amp_dc_in8, row.amp_dc_in9, row.amp_dc_in10, row.amp_dc_in11, row.amp_dc_in12, row.amp_dc_in13,
                row.amp_dc_in14, row.amp_dc_in15, row.amp_dc_in16, row.amp_dc_in17, row.amp_dc_in18, row.amp_dc_in19,
                row.amp_dc_in20, row.amp_dc_in21, row.amp_dc_in22, row.amp_dc_in23, row.amp_dc_in24, row.aisl_dc,
                row.energia_dia))
            conn.commit()
        except AttributeError:
            # En caso de que alguna de las columnas no exista porque el source no las incluya, se inserta el registro sin ella
            cur.execute(f"""INSERT INTO {schema_name}.inversores_raw(
                    dispositivo_id, datetime_utc, med_id, status, alarma, estado, 
                    potencia_act, potencia_rea, cos_phi, vol_12, vol_13, vol_23, amp_1, 
                    amp_2, amp_3, frec, lim_act, lim_rea, vol_dc_bus, amp_dc_in1, amp_dc_in2,
                    amp_dc_in3, amp_dc_in4, amp_dc_in5, amp_dc_in6, amp_dc_in7, amp_dc_in8, 
                    amp_dc_in9, amp_dc_in10, amp_dc_in11, amp_dc_in12, amp_dc_in13, amp_dc_in14,
                    amp_dc_in15, amp_dc_in16, amp_dc_in17, amp_dc_in18, amp_dc_in19, amp_dc_in20, 
                    amp_dc_in21, amp_dc_in22, amp_dc_in23, amp_dc_in24, aisl_dc, energia_dia)
                VALUES 
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                (row.dispositivo_id, row.datetime, row.med_id, row.status, row.alarma, row.estado,
                row.pot_act, row.pot_reac, row.cos_phi, row.volt_12, row.volt_13, row.volt_23, row.amp_1,
                row.amp_2, row.amp_3, row.frec, row.lim_act, row.lim_react, row.volt_dc_bus, row.amp_dc_in1, 
                row.amp_dc_in2, row.amp_dc_in3, row.amp_dc_in4, row.amp_dc_in5, row.amp_dc_in6, row.amp_dc_in7,
                row.amp_dc_in8, row.amp_dc_in9, row.amp_dc_in10, row.amp_dc_in11, row.amp_dc_in12, row.amp_dc_in13,
                row.amp_dc_in14, row.amp_dc_in15, row.amp_dc_in16, row.amp_dc_in17, row.amp_dc_in18, row.amp_dc_in19,
                row.amp_dc_in20, row.amp_dc_in21, row.amp_dc_in22, row.amp_dc_in23, row.amp_dc_in24, row.aisl_dc,
                row.energia_dia))
            conn.commit()
        except Exception as error:
            print("Error: ", error)
            print(row)
            conn.rollback()
    cur.close()
    conn.close()
    print("Conexión a PostgreSQL cerrada")
    
