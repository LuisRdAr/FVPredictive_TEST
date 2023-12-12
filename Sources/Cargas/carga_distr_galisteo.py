import pandas as pd
import psycopg2
import json
import sys
import os


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

    # Inicialización del dataframe y lectura del csv
    df = pd.DataFrame()
    df = pd.read_csv(data_path, sep = ";")
    df["parque_id"] = 1
    df = df[["parque_id"] + df.columns.to_list()[:-1]]

    # Renombrado de columnas
    columns = []
    for column in df.columns:
        if "nº" in column:
            columns.append(column.replace("nº ", "num_").strip().lower())
        else:
            columns.append(column.strip().lower().replace(" ", "_"))
    df.columns = columns

    # Procesado y formateo de columnas
    df["potencia"] = df["potencia"].str.replace(",", "").astype("float")
    df["dispositivo_id"] = df["cbs"].str.split("-").str[1].astype(int) + 20
    df["entrada_id"] = df["cbs"].str.split("-").str[2].astype(int)
    df = df.drop(columns = ["cbs"])

    # Apertura de conexión a la base de datos y carga de datos
    try:
        conn = psycopg2.connect(
            database = params['dbname'],
            user = params['user'],
            password = params['password'],
            host = params['host'],  
            port = params['port'])
        cur = conn.cursor()
        print("Conexión exitosa a PostgreSQL")

        for row in df.itertuples():
            try:
                cur.execute(f"""INSERT INTO {schema_name}.distrib_inversores(
                        parque_id, subcampo, dispositivo_id, entrada_id, tipo, num_strings,
                        num_modulos, potencia_mod, potencia)
                    VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                    (row.parque_id, row.subcampo, row.dispositivo_id, row.entrada_id, row.tipo, row.num_strings,
                     row.num_modulos, row.potencia_mod, row.potencia))
                conn.commit()
            except Exception as error:
                print("Error: ", error)
                conn.rollback()
        cur.close()
        conn.close()
        print("Conexión a PostgreSQL cerrada")
    except (Exception, psycopg2.Error) as error:
        print("Error en la apertura de base de datos: \n\t{}".format(error))
