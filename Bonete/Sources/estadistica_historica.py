#####
# INFERIR GRUPOS QUE NO TENGAN DATOS PARA ESTABLECER UNA ESTADÍSTICA DE REFERENCIA.
# EN BONETE NO HAY DATOS EN OCTUBRE, POR EJEMPLO, PERO SE PUEDEN INFERIR LOS GRUPOS
# A PARTIR DE LOS DATOS DE OTROS MESES.
#####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import sqlalchemy
import psycopg2
import json
import os
import sys
import gc


def generate_rad_bins(df, columns, lower_lim, upper_lim, lower_threshold, upper_threshold, initial_width):
    bins = []
    bin_count = ((df["rad_poa"] >= lower_lim) & (df["rad_poa"] < upper_lim)).sum()
    width = upper_lim - lower_lim
    if bin_count > lower_threshold and bin_count < upper_threshold:
        bins.append(upper_lim)
    elif bin_count <= lower_threshold:
        pass
    elif width >= initial_width/2:
        mid = (lower_lim + upper_lim)/2
        bins.extend(generate_rad_bins(df, columns, lower_lim, mid, lower_threshold, upper_threshold, initial_width))
        bins.extend(generate_rad_bins(df, columns, mid, upper_lim, lower_threshold, upper_threshold, initial_width))
    else:
        bins.append(upper_lim)

    bins = [int(i) if i%1 == 0 else i for i in bins]
    return bins


# Carga de los parametros de conexión a la base de datos
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

months = np.arange(1, 13)
for month in months:
    print(f"Mes {month}")
    # Consulta principal: se devuelven los amperajes de corriente continua asociados a cada entrada de cada inversor y la potencia
    # activa junto con la radiación, la nubosidad (limitada a 100% dado que la versión actual sobrepasa este valor y está pendiente 
    # de corrección) y la consigna de potencia actual de la planta (normalizada entre 0 y 1) en cada instante de tiempo.
    # Se cargan los datos en chunks para evitar problemas de memoria.
    main_query = f"""
        WITH inv AS (
            SELECT 
                datetime_utc,
                dispositivo_id,
                inv.id, 
                amp_dc,
                det.entrada_id,
                potencia_act
            FROM {schema_name}.inversores AS inv
            JOIN {schema_name}.inversores_detalle AS det
                ON inv.id = det.id
            WHERE (alarma = 0)
                AND (estado = 6)
                AND EXTRACT(MONTH FROM datetime_utc) = {month}
        ),
        met AS (
            SELECT 
                datetime_utc,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rad_poa) AS rad_poa,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LEAST(cloud_impact, 100)/100) AS cloud_impact,
                BOOL_OR(daylight) AS daylight
            FROM {schema_name}.meteo
            WHERE daylight = true
            GROUP BY datetime_utc
        )
        SELECT 
            inv.dispositivo_id,
            inv.entrada_id,
            inv.datetime_utc,   
            num_strings,
            amp_dc,
            potencia_act,
            rad_poa,
            cloud_impact,
            consigna_pot_act_planta/MAX(consigna_pot_act_planta) OVER (PARTITION BY inv.dispositivo_id) AS consigna_pot_act_planta
        FROM inv
        JOIN {schema_name}.distrib_inversores dist
            ON dist.dispositivo_id = inv.dispositivo_id
                AND dist.entrada_id = inv.entrada_id
        JOIN {schema_name}.dispositivos AS disp
            ON disp.dispositivo_id = inv.dispositivo_id
        JOIN met
            ON met.datetime_utc = inv.datetime_utc
        JOIN {schema_name}.ree AS ree
            ON ree.datetime_utc = inv.datetime_utc
        ORDER BY 4, 2, 3;"""
    chunksize = 500000
    chunks = []
    for chunk in pd.read_sql(main_query, engine, chunksize=chunksize):
        chunks.append(chunk)
    df = pd.concat(chunks)
    del chunk, chunks
    gc.collect()

    if df.empty:
        print(f"No hay datos para el mes {month}")
        continue
    else:
        # Se normalizan las intensidades de corriente entrantes en función del número de strings asociados a cada una de estas.
        # Se desnormaliza la intensidad de corriente en función de la consigna para modelar el comportamiento en caso de no existir 
        # consigna (valor igual a 1) y se categoriza en función de si la consigna es del 0% (0), 100% (1) o valores intermedios (2).
        # Se convierte la columna datetime_utc a tipo datetime.
        df["amp_dc"] = df["amp_dc"]/df["num_strings"]
        df["consigna_pot_act_planta"] = df["consigna_pot_act_planta"].fillna(1)  # Se rellenan los valores nulos con 1 al ser el valor más habitual 
                                                                            # (se podría hacer una imputación más sofisticada si la planta lo requiriese)
        df["amp_dc_reesc"] = np.where(df["consigna_pot_act_planta"] != 0, df["amp_dc"]/df["consigna_pot_act_planta"], df["amp_dc"])
        df["potencia_act_reesc"] = np.where(df["consigna_pot_act_planta"] != 0, df["potencia_act"]/df["consigna_pot_act_planta"], df["potencia_act"])
        df["consigna_cat"] = np.where(df["consigna_pot_act_planta"] == 0, 0, np.where(df["consigna_pot_act_planta"] == 1, 1, 2))
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc = True)

        # Se generan automaticamente los contenedores para la irradiancia inclinada en función de los valores estadísticos de la
        # distribución. Dado que los valores de irradiancia en la meseta son los más frecuentes se generan más contenedores en
        # esta este rango de valores
        initial_width = 50
        upper_threshold = df.shape[0] / ((df["rad_poa"].max()) / initial_width)
        lower_threshold = df.shape[0] / ((df["rad_poa"].max()) / (initial_width/4))
        upper_lim = np.floor(df["rad_poa"].max() / initial_width) * initial_width

        bins = [-np.Inf]
        for i in range(0, int(upper_lim), initial_width):
            bins.extend(generate_rad_bins(df, ["rad_poa"], i, i + initial_width, lower_threshold, upper_threshold, initial_width))
        bins[-1] = np.Inf

        labels = [str(bins[i]) + "-" + str(bins[i+1]) for i in range(len(bins)-1)]
        labels[0] = str(0) + "-" + str(bins[1])
        labels[-1] = str(bins[-2]) + "+"
        df["rad_bins"] = pd.cut(df["rad_poa"], 
            bins = bins,
            labels = labels)
        print(f"\tNúmero de contenedores de irradiancia generados: {len(labels)}")

        # Se generan los contenedores para la nubosidad
        df["cloud_impact_bins"] = pd.cut(df["cloud_impact"],
                                        bins = [0, 0.2, 0.40, 0.60, 0.80, 1],
                                        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
        

        # Se genera el dataframe con los datos estadísticos de las intensidades de corriente para cada grupo (cuantiles, count, IQR y lims)
        # Se añaden dichas estadísticas al dataframe original para filtrar aquellos registros atípicos siguiendo un criterio basado
            # en excluir aquellos valores fuera del rango definido por un radio igual al IQR de la distribución centrado en la mediana.
        group_stats = df.groupby(["rad_bins", "cloud_impact_bins", "consigna_cat", "dispositivo_id"], observed=True)["amp_dc_reesc"] \
                            .quantile(q = [0.25, 0.5, 0.75]) \
                                .unstack().reset_index() 
        group_stats["count"] = df.groupby(["rad_bins", "cloud_impact_bins", "consigna_cat", "dispositivo_id"], observed=True)["amp_dc_reesc"].count().values
        group_stats["IQR"] = group_stats[0.75] - group_stats[0.25]
        group_stats["upper_limit"] = group_stats[0.5] + group_stats["IQR"]
        group_stats["lower_limit"] = group_stats[0.5] - group_stats["IQR"]

        df_stats = df.merge(group_stats, on = ["rad_bins", "cloud_impact_bins", "consigna_cat", "dispositivo_id"], how = "left")

        df_stats["outlier"] = np.where((df_stats["amp_dc_reesc"] > df_stats["upper_limit"]) | 
                                            (df_stats["amp_dc_reesc"] < df_stats["lower_limit"]), 
                                    True, 
                                    False)
        filtered_df = df_stats[df_stats["outlier"] == False]

        # Se generan las estadísticas para cada grupo habiendo descartado aquellos valores atípicos
        amp_stats = filtered_df.groupby(["rad_bins", 
                                "cloud_impact_bins", 
                                "consigna_cat", 
                                "dispositivo_id"
                                ], observed=True)["amp_dc_reesc"].agg(["mean", "median", "std", "count"]).reset_index()
        amp_stats = amp_stats.rename(columns = {"cloud_impact_bins": "cloud_bins",
                                        "mean": "media", 
                                        "median": "mediana", 
                                        "std": "std", 
                                        "count": "count"})
        del group_stats, df_stats, filtered_df
        gc.collect()
        
        # Se repite el proceso para la potencia activa pero sin agrupar por dispositivo
        group_stats = df.groupby(["rad_bins", "cloud_impact_bins", "consigna_cat", "dispositivo_id"], observed=True)["potencia_act_reesc"] \
                            .quantile(q = [0.25, 0.5, 0.75]) \
                                .unstack().reset_index() 
        group_stats["count"] = df.groupby(["rad_bins", "cloud_impact_bins", "consigna_cat", "dispositivo_id"], observed=True)["potencia_act_reesc"].count().values
        group_stats["IQR"] = group_stats[0.75] - group_stats[0.25]
        group_stats["upper_limit"] = group_stats[0.5] + group_stats["IQR"]
        group_stats["lower_limit"] = group_stats[0.5] - group_stats["IQR"]
        print(f"\tNúmero de grupos generados: {group_stats.shape[0]}")

        df_stats = df.merge(group_stats, on = ["rad_bins", "cloud_impact_bins", "consigna_cat", "dispositivo_id"], how = "left")
        df_stats["outlier"] = np.where((df_stats["potencia_act_reesc"] > df_stats["upper_limit"]) | 
                                            (df_stats["potencia_act_reesc"] < df_stats["lower_limit"]), 
                                    True, 
                                    False)

        filtered_df = df_stats[df_stats["outlier"] == False]

        pot_stats = filtered_df.groupby(["rad_bins", 
                                "cloud_impact_bins", 
                                "consigna_cat",
                                "dispositivo_id"
                                ], observed=True)["potencia_act_reesc"].agg(["mean", "median", "std", "count"]).reset_index()
        pot_stats = pot_stats.rename(columns = {"cloud_impact_bins": "cloud_bins",
                                        "mean": "media", 
                                        "median": "mediana", 
                                        "std": "std", 
                                        "count": "count"})
        del group_stats, df_stats, filtered_df
        gc.collect()
        
        # Union de las estadísticas de amperaje y potencia por grupos
        stats = pd.merge(amp_stats, pot_stats, on = ["rad_bins", "cloud_bins", "consigna_cat", "dispositivo_id"], how = "inner", suffixes=("_amp", "_pot"))
        stats["mes"] = month
        del amp_stats, pot_stats
        gc.collect()
        print("\tEstadísticas generadas")
        # Se escriben las estadísticas en la tabla de estadística histórica
        try:
            dtypes = {
                "rad_bins": sqlalchemy.types.VARCHAR(length=10),
                "cloud_bins": sqlalchemy.types.VARCHAR(length=10),
                "consigna_cat": sqlalchemy.types.VARCHAR(length=1),
                "dispositivo_id":sqlalchemy.types.SMALLINT(),
                "mes": sqlalchemy.types.SMALLINT(),
                "media_amp": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "mediana_amp": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "std_amp": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "media_pot": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "mediana_pot": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "std_pot": sqlalchemy.types.Float(precision=3, asdecimal=True)
            }
            stats = stats.drop(columns = stats.columns.difference(list(dtypes.keys())))
            stats.to_sql("historico_stats", engine, schema = schema_name, if_exists = "append", index = False, dtype = dtypes)
            del stats
            gc.collect()
            print(f"\tEstadísticas del mes {month} escritas en la tabla de estadística histórica")
        except Exception as error:
                print("Error en la escritura de datos en la tabla de estadística histórica: \n\t{}".format(error))
                sys.exit()

        
    

    
