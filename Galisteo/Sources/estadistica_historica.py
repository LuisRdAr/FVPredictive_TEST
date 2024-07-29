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


def generate_rad_bins(df, columns, lower_lim, upper_lim, 
                      lower_threshold, upper_threshold, initial_width):
    bins = []
    bin_count = ((df["rad_poa"] >= lower_lim) & (df["rad_poa"] < upper_lim)).sum()
    width = upper_lim - lower_lim
    if bin_count > lower_threshold and bin_count < upper_threshold:
        bins.append(upper_lim)
    elif bin_count <= lower_threshold:
        pass
    elif width >= initial_width/2:
        mid = (lower_lim + upper_lim)/2
        bins.extend(generate_rad_bins(df, columns, lower_lim, mid, 
                                      lower_threshold, upper_threshold, initial_width))
        bins.extend(generate_rad_bins(df, columns, mid, upper_lim, 
                                      lower_threshold, upper_threshold, initial_width))
    else:
        bins.append(upper_lim)

    bins = [int(i) if i%1 == 0 else i for i in bins]
    return bins


# Carga de los parametros de conexión a la base de datos
ROOT_PATH = os.getcwd()
params = None
for filename in os.listdir(ROOT_PATH):
    if "params.json" in filename:
        with open(os.path.join(ROOT_PATH, filename)) as f:
            params = json.load(f)
if params is None:
    print("No se ha encontrado el archivo de parámetros para la conexión a la base de datos")
    sys.exit()
SCHEMA_NAME = params["schema"]

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

months = np.arange(1, 13)
for month in months:
    print(f"Mes {month}")
    # Consulta principal: se devuelven los amperajes de corriente continua asociados a cada entrada 
        # de cada inversor y la potencia activa junto con la radiación, la nubosidad (limitada a 
        # 100% dado que la versión actual sobrepasa este valor y está pendiente  de corrección) y la 
        # consigna de potencia actual de la planta (normalizada entre 0 y 1) en cada instante de 
        # tiempo.
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
            FROM {SCHEMA_NAME}.inversores AS inv
            JOIN {SCHEMA_NAME}.inversores_detalle AS det
                ON inv.id = det.id
            WHERE (alarma = 0)
                AND (estado = 6)
                AND EXTRACT(MONTH FROM datetime_utc) = {month}
        ),
        met AS (
            SELECT 
                datetime_utc,
                dispositivo_id,
                rad_poa,
                cloud_impact/100 AS cloud_impact
            FROM {SCHEMA_NAME}.meteo
            WHERE daylight = true
                AND status_srl = 0
                AND EXTRACT(MONTH FROM datetime_utc) = {month}
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
            consigna_pot_act_planta/MAX(consigna_pot_act_planta) 
                OVER (PARTITION BY inv.dispositivo_id) AS consigna_pot_act_planta
        FROM inv
        JOIN {SCHEMA_NAME}.distrib_inversores dist
            ON dist.dispositivo_id = inv.dispositivo_id
                AND dist.entrada_id = inv.entrada_id
        JOIN {SCHEMA_NAME}.dispositivos AS disp
            ON disp.dispositivo_id = inv.dispositivo_id
        JOIN met
            ON met.datetime_utc = inv.datetime_utc
                AND met.dispositivo_id = disp.meteo_cercana_id
        JOIN {SCHEMA_NAME}.ree AS ree
            ON ree.datetime_utc = inv.datetime_utc
        ORDER BY 3, 1, 2;"""
    chunksize = 500000
    chunks = []
    for chunk in pd.read_sql(main_query, engine, chunksize=chunksize):
        chunks.append(chunk)
    df = pd.concat(chunks)
    del chunk, chunks
    gc.collect()

    if df.empty:
        print(f"No hay datos para el mes {month}")
    else:
        # Se normalizan las intensidades de corriente entrantes en función del número de strings 
            # asociados a cada una de estas.
        # Se desnormaliza la intensidad de corriente en función de la consigna para modelar el 
            # comportamiento en caso de no existir  consigna (valor igual a 1) y se categoriza en 
            # función de si la consigna es del 0% (0), 100% (1) o valores intermedios (2).
        # Se convierte la columna datetime_utc a tipo datetime.
        df["amp_dc"] = df["amp_dc"]/df["num_strings"]
        df["consigna_pot_act_planta"] = df["consigna_pot_act_planta"].fillna(1)  
            # Se rellenan los valores nulos con 1 al ser el valor más habitual (se podría hacer una 
            # imputación más sofisticada si la planta lo requiriese)
        df["amp_dc_reesc"] = np.where(df["consigna_pot_act_planta"] != 0, 
                                      df["amp_dc"]/df["consigna_pot_act_planta"], 
                                      df["amp_dc"])
        df["potencia_act_reesc"] = np.where(df["consigna_pot_act_planta"] != 0, 
                                            df["potencia_act"]/df["consigna_pot_act_planta"], 
                                            df["potencia_act"])
        df["consigna_cat"] = np.where(df["consigna_pot_act_planta"] == 0, 
                                      0, 
                                      np.where(df["consigna_pot_act_planta"] == 1, 
                                               1, 
                                               2))
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc = True)

        # Se generan automaticamente los contenedores para la irradiancia inclinada en función de 
        # los valores estadísticos de la distribución. Dado que los valores de irradiancia en la 
        # meseta son los más frecuentes se generan más contenedores en este rango de valores.
        initial_width = 50
        upper_threshold = df.shape[0] / ((df["rad_poa"].max()) / initial_width)
        lower_threshold = df.shape[0] / ((df["rad_poa"].max()) / (initial_width/4))
        upper_lim = np.floor(df["rad_poa"].max() / initial_width) * initial_width

        bins = [-np.Inf]
        for i in range(0, int(upper_lim), initial_width):
            bins.extend(generate_rad_bins(df, ["rad_poa"], i, i + initial_width, 
                                          lower_threshold, upper_threshold, initial_width))
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
        
        # Se genera el dataframe con los datos estadísticos de las intensidades de corriente para 
            # cada grupo (cuantiles, count, IQR y lims)
        # Se añaden dichas estadísticas al dataframe original para filtrar aquellos registros 
            # atípicos siguiendo un criterio basado en excluir aquellos valores fuera del rango 
            # definido por un radio igual al IQR de la distribución centrado en la mediana.
        # Crear un DataFrame vacío para almacenar las estadísticas
        index = pd.MultiIndex(levels=[[], [], [], []], 
                              codes=[[], [], [], []], 
                              names=["rad_bins", "cloud_impact_bins", 
                                     "consigna_cat", "dispositivo_id"])
        group_stats_df = pd.DataFrame(index=index, columns=["count", "lower_limit", "upper_limit"])

        group_stats = df.groupby(["rad_bins", 
                                  "cloud_impact_bins", 
                                  "consigna_cat", 
                                  "dispositivo_id"], observed=True)["amp_dc_reesc"]

        for group, data in group_stats:
            data = data[data >= 0]
            if data.empty:
                continue
            bootstrap_lower_limits = []
            bootstrap_upper_limits = []
            
            for _ in range(1000):
                sample = data.sample(frac=1, replace=True)
                lower_percentile = np.percentile(sample, 25)
                upper_percentile = np.percentile(sample, 75)
                iqr = upper_percentile - lower_percentile
                bootstrap_lower_limits.append(lower_percentile - 1.5*iqr)
                bootstrap_upper_limits.append(upper_percentile + 1.5*iqr)
            
            count = data.shape[0]
            lower_limit = max(np.mean(bootstrap_lower_limits), 0)
            upper_limit = np.mean(bootstrap_upper_limits)

            group_stats_df.loc[group] = [count, lower_limit, upper_limit]

        group_stats_df = group_stats_df.rename(columns={"lower_limit": "lower_amp", 
                                                        "upper_limit": "upper_amp"})  
        group_stats_df = group_stats_df.reset_index()    
        
        # Se repite el proceso para la potencia activa
        group_stats = df.groupby(["rad_bins", 
                                  "cloud_impact_bins", 
                                  "consigna_cat", 
                                  "dispositivo_id"], observed=True)["potencia_act_reesc"]

        for group, data in group_stats:
            data = data[data >= 0]
            if data.empty:
                continue
            bootstrap_lower_limits = []
            bootstrap_upper_limits = []
            
            for _ in range(1000):
                sample = data.sample(frac=1, replace=True)
                lower_percentile = np.percentile(sample, 25)
                upper_percentile = np.percentile(sample, 75)
                iqr = upper_percentile - lower_percentile
                bootstrap_lower_limits.append(lower_percentile - 1.5*iqr)
                bootstrap_upper_limits.append(upper_percentile + 1.5*iqr)
            
            lower_limit = max(np.mean(bootstrap_lower_limits), 0)
            upper_limit = np.mean(bootstrap_upper_limits)
            
            group_stats_df.loc[
                (group_stats_df['rad_bins'] == group[0]) &
                (group_stats_df['cloud_impact_bins'] == group[1]) &
                (group_stats_df['consigna_cat'] == group[2]) &
                (group_stats_df['dispositivo_id'] == group[3]),
                ["lower_pot", "upper_pot"]] = [lower_limit, upper_limit]

        group_stats_df = group_stats_df.reset_index()
        group_stats_df = group_stats_df.rename(columns={"cloud_impact_bins": "cloud_bins"})
            
        del group_stats
        gc.collect()

        group_stats_df["mes"] = month

        group_stats_df = group_stats_df.dropna(subset=["lower_amp", "upper_amp", 
                                                       "lower_pot", "upper_pot"])
        
        print("\tEstadísticas generadas")
        # Se escriben las estadísticas en la tabla de estadística histórica
        try:
            dtypes = {
                "rad_bins": sqlalchemy.types.VARCHAR(length=10),
                "cloud_bins": sqlalchemy.types.VARCHAR(length=10),
                "consigna_cat": sqlalchemy.types.VARCHAR(length=1),
                "dispositivo_id":sqlalchemy.types.SMALLINT(),
                "mes": sqlalchemy.types.SMALLINT(),
                "lower_amp": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "upper_amp": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "lower_pot": sqlalchemy.types.Float(precision=3, asdecimal=True),
                "upper_pot": sqlalchemy.types.Float(precision=3, asdecimal=True)
            }
            group_stats_df.drop(columns = group_stats_df.columns.difference(list(dtypes.keys())),
                                inplace = True)
            group_stats_df.to_sql("historico_stats", 
                                  engine, 
                                  schema = SCHEMA_NAME, 
                                  if_exists = "append", 
                                  index = False, 
                                  dtype = dtypes)
            del group_stats_df
            gc.collect()
            print(f"\tEstadísticas del mes {month} escritas en la tabla de estadística histórica")
        except Exception as error:
            print(f"Error en la escritura de datos en la tabla de estadística histórica: \n\t{error}")
            sys.exit()