import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
import json
import pickle
import os
import sys
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from functools import partial
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, \
                            r2_score as r2
import time
import gc



def rad_cut(df, bins):
    """
    Funcion para categorizar la irradiancia en intervalos definidos por mes y cargados desde la 
    base de datos.
    La funcion recibe el dataframe agrupado por mes y los limites de los intervalos de irradiancia,
    genera unas labels para cada intervalo yañade una columna al dataframe con la categorizacion 
    de la irradiancia.
    """
    bins = np.sort(bins)
    labels = [str(int(bins[i])) + "-" + str(int(bins[i+1])) if 
                    np.isfinite(bins[i]) and np.isfinite(bins[i+1]) else 
                        str(bins[i]) + "-" + str(bins[i+1]) for i in range(len(bins)-1)]
    labels[-1] = str(bins[-2]) + "+"
    labels[0] = labels[0].replace("-inf", "0")
    df["rad_bins"] = pd.cut(df["rad_poa"], bins=bins, labels=labels)
    return df

def discriminador(row):
    """
    Funcion para discriminar los outliers en las columnas de amperajes asociadas a un registro 
    instantaneo.
    La funcion recibe una fila del dataframe y devuelve la misma fila con los outliers cambiados 
    por NaN y una columna adicional con la cantidad de outliers encontrados.
    """
    median = np.median(row)
    iqr = np.percentile(row, 75) - np.percentile(row, 25)
    coef_iqr = 3
    outlier = np.where((row < median - coef_iqr*iqr) | (row > median + coef_iqr*iqr), 1, 0)
    filtered_amps = np.where(outlier == 0, row, np.nan)
    row.loc[row.index] = filtered_amps
    row["n_outliers"] = outlier.sum()

    return row

def objective(space, train_set, cv_folds):
    """
    Funcion objetivo para la optimizacion de hiperparametros de un modelo XGBoost.
    La funcion recibe un conjunto de hiperparametros, un conjunto de entrenamiento, 
    el numero de folds y devuelve el error cuadratico medio de la validacion cruzada
    junto con un status y el numero de iteraciones optimas para dicha combinacion.
    """
    n_estimators = int(space.pop('n_estimators'))
    params = {'objective': 'reg:squarederror',
            'device': DEVICE,
            'tree_method': "hist",
            'random_state': 42,
            'n_jobs': -1}
    params = {**params, **space}
    if 'max_depth' in params:
        params['max_depth'] = int(params['max_depth'])
    if 'min_child_weight' in params:
        params['min_child_weight'] = max(1, int(params['min_child_weight']))

    cv_results = xgb.cv(params, 
                        train_set, 
                        nfold = cv_folds, 
                        num_boost_round = n_estimators, 
                        early_stopping_rounds = 100, 
                        metrics = 'rmse', 
                        as_pandas = True)
    return {'loss': cv_results['test-rmse-mean'].min(), 
            'status': STATUS_OK, 
            'n_rounds': cv_results.shape[0]}

def optimize_hyperparameters(space, 
                             dtrain,
                             STALL_LIMIT = 5, 
                             MAX_EVALS_PER_RUN = 250, 
                             gamma_algo = 1, 
                             cv_folds = 5,
                             verbose = False):
    """
    Funcion para la optimizacion de hiperparametros de un modelo XGBoost.
    La funcion recibe un espacio de busqueda, un conjunto de entrenamiento, el numero de folds,
    el limite de iteraciones sin mejora, el numero maximo de evaluaciones por iteracion, el factor
    de reduccion de gamma, el dispositivo de entrenamiento y si se desea mostrar informacion por
    pantalla.
    """
    best_loss = np.inf
    trials = Trials()
    stall_counter = 0
    num_evals = 0
    upper_limit = (MAX_EVALS_PER_RUN * STALL_LIMIT) * 5

    while stall_counter < STALL_LIMIT and num_evals < upper_limit:
        best = fmin(fn = lambda space: objective(space, 
                                                 train_set = dtrain, 
                                                 cv_folds = cv_folds), 
                    space = space, 
                    algo = partial(tpe.suggest, gamma=gamma_algo),
                    max_evals = num_evals + MAX_EVALS_PER_RUN, 
                    trials = trials,
                    verbose = verbose)
        best_params = space_eval(space, best)
        new_loss = trials.best_trial['result']['loss']
        if new_loss < best_loss:
            best_loss = new_loss
            stall_counter = 0
        else:
            stall_counter += 1
        num_evals += MAX_EVALS_PER_RUN
        gamma_algo = max(0.25, gamma_algo - (0.5/(2*STALL_LIMIT)))

    best_params["n_estimators"] = trials.best_trial['result']['n_rounds']
    print(f"\tNumero de evaluaciones realizadas: {num_evals}")
    print(f"\tBest params: {best_params}")
    print(f"\tBest trial: {trials.best_trial}")
    print(f"\tBest loss: {best_loss}", end="\n\n")

    return best_params



print("Configuracion para el entrenamiento de modelos de estimacion de intensidad de corriente " + \
      "a la entrada de inversores en plantas fotovoltaicas:", end="\n\n")
MODEL_NAME = "XGBRegressor"

valid_responses = ["y", "n"]
repeat = True
while repeat:
    OPTIMIZATION = ""
    while OPTIMIZATION not in valid_responses:
        OPTIMIZATION = input("¿Desea optimizar el modelo? (Y/N): ").lower()
    OPTIMIZATION = OPTIMIZATION == "y"

    valid_devices = ["cpu", "cuda"]
    DEVICE = ""
    if OPTIMIZATION:
        # Comprobacion de la disponibilidad de GPU para el entrenamiento
        if torch.cuda.is_available():
            while DEVICE not in valid_devices:
                DEVICE_BOOL = input("¿Desea utilizar GPU para el entrenamiento? (Y/N): ").lower()
                if DEVICE_BOOL == "y":
                    DEVICE = "cuda"
                else:
                    DEVICE = "cpu"
        STALL_LIMIT = 0
        while STALL_LIMIT < 1 or STALL_LIMIT > 11:
            STALL_LIMIT = int(input("Introduzca el limite de bloques de iteraciones sin mejora (2-10): "))
        MAX_EVALS_PER_RUN = 0
        while MAX_EVALS_PER_RUN < 10 or MAX_EVALS_PER_RUN > 251:
            MAX_EVALS_PER_RUN = int(input("Introduzca el numero de evaluaciones por bloque (10-250): "))
        CV_FOLDS = 0
        while CV_FOLDS < 2 or CV_FOLDS > 11:
            CV_FOLDS = int(input("Introduzca el numero de folds para la validacion cruzada (2-10): "))
    else:
        DEVICE = 'cpu'

    print("\nLas opciones seleccionadas son:")
    print(f"\tModelo: {MODEL_NAME}")
    print(f"\tOptimizacion: {OPTIMIZATION}")
    if OPTIMIZATION:
        print(f"\t\tLimite de bloques de iteraciones sin mejora: {STALL_LIMIT}")
        print(f"\t\tNumero de evaluaciones por bloque: {MAX_EVALS_PER_RUN}")
        print(f"\t\tNumero de folds para la validacion cruzada: {CV_FOLDS}")
    print(f"\tEntrenamiento: {DEVICE}", end="\n\n")

    correct = ""
    while correct not in valid_responses:
        correct = input("¿Son correctas las opciones seleccionadas? (Y/N): ").lower()
    if correct == "y":
        repeat = False
    else:
        continue    
print("Configuracion establecida", end="\n\n")

# Carga del fichero de parametros para conexion con la base de datos
ROOT_PATH = os.getcwd()
ROOT_PATH = "/home/upo/Desktop/Test_FVPredictive/FVPredictive_TEST/Bonete"
params = None
for filename in os.listdir(ROOT_PATH):
    if "params.json" in filename:
        with open(os.path.join(ROOT_PATH, filename)) as f:
            params = json.load(f)
if params is None:
    print("No se ha encontrado el archivo de parametros para la conexion a la base de datos")
    sys.exit()
DATA_PATH = os.path.join(ROOT_PATH, params["data_path"])
SCHEMA_NAME = params["schema"]

password = params['password'].replace('@', '%40')
engine = create_engine(
    f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}'
    )
print(f"Conexion a la base de datos {params['dbname']} (esquema {SCHEMA_NAME}) establecida")


INTERVALO_MIN = 15      # Intervalo de promediado en minutos
# Carga de los datos de entrenamiento
main_query = f"""
    WITH inv AS (
        SELECT 
            date_trunc('hour', datetime_utc) + 
                INTERVAL '{INTERVALO_MIN} min' * floor(date_part('minute', datetime_utc) / 
                {INTERVALO_MIN}) as datetime_utc_rounded,
            dispositivo_id,
            MIN(inv.id) AS id, 
            AVG(potencia_act) as potencia_act, 
            AVG(amp_dc) as amp_dc,
            det.entrada_id
        FROM {SCHEMA_NAME}.inversores AS inv
        JOIN {SCHEMA_NAME}.inversores_detalle AS det
            ON inv.id = det.id
        WHERE (alarma = 0)
            AND (estado = 6)
            AND EXTRACT(YEAR FROM datetime_utc) != 2024
        GROUP BY datetime_utc_rounded, dispositivo_id, det.entrada_id 
    ),
    met AS (
        SELECT 
            date_trunc('hour', datetime_utc) + 
                INTERVAL '{INTERVALO_MIN} min' * floor(date_part('minute', datetime_utc) / 
                {INTERVALO_MIN}) as datetime_utc_rounded, 
            dispositivo_id,
            AVG(rad_poa) AS rad_poa,
            AVG((rad_celda1 - rad_celda2))/NULLIF(AVG(rad_celda1), 0) AS rad_soiling, --porcentual
            AVG(temp_amb) AS temp_amb, 
            AVG((temp_panel1 + temp_panel2)/2)  AS temp_panel,
            AVG((LEAST(cloud_impact, 100)/100)) AS cloud_impact
        FROM {SCHEMA_NAME}.meteo
            WHERE daylight = true
                AND status_srl = 0
        GROUP BY dispositivo_id, datetime_utc_rounded
    ),
    met_gr AS (
        SELECT
            datetime_utc_rounded,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rad_poa) AS rad_poa,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rad_soiling) AS rad_soiling,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY temp_amb) AS temp_amb,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY temp_panel) AS temp_panel,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cloud_impact) AS cloud_impact
        FROM met
        GROUP BY datetime_utc_rounded
    ),
    ree AS (
        SELECT
            date_trunc('hour', datetime_utc) +
                INTERVAL '{INTERVALO_MIN} min' * floor(date_part('minute', datetime_utc) / 
                {INTERVALO_MIN}) as datetime_utc_rounded,
            AVG(consigna_pot_act_planta) as consigna_pot_act_planta
        FROM {SCHEMA_NAME}.ree
        GROUP BY datetime_utc_rounded
    )
    SELECT 
        inv.id,
        inv.dispositivo_id,
        inv.entrada_id,
        inv.datetime_utc_rounded as datetime_utc,  
        rad_poa,
        rad_soiling,
        temp_amb, 
        temp_panel,
        cloud_impact,
        consigna_pot_act_planta/MAX(consigna_pot_act_planta) 
            OVER (PARTITION BY inv.dispositivo_id) AS consigna_pot_act_planta,
        potencia_act,
        amp_dc,  
        num_strings
    FROM inv
    JOIN {SCHEMA_NAME}.distrib_inversores dist
        ON dist.dispositivo_id = inv.dispositivo_id
            AND dist.entrada_id = inv.entrada_id
    JOIN met_gr
        ON met_gr.datetime_utc_rounded = inv.datetime_utc_rounded
    JOIN ree
        ON ree.datetime_utc_rounded = inv.datetime_utc_rounded
    ORDER BY 4, 2, 3;"""

CHUNKSIZE = 100000
chunks = pd.read_sql_query(main_query, engine, chunksize = CHUNKSIZE)
main_df = pd.DataFrame()
for chunk in chunks:
    main_df = pd.concat([main_df, chunk], ignore_index = True)
del chunks, chunk
gc.collect()

# Normalizacion de la entrada de corriente continua, formateo de fechas y escalado de potencia (kW -> W)
print(f"Carga inicial de {main_df.shape[0]} registros")
print("Entrada de corriente continua normalizada segun el numero de strings")
main_df["amp_dc"] = main_df["amp_dc"]/main_df["num_strings"]
main_df["datetime_utc"] = pd.to_datetime(main_df["datetime_utc"], utc = True)
main_df["mes"] = main_df["datetime_utc"].dt.month
main_df["potencia_act"] = main_df["potencia_act"] * 1000

# Inicio del descarte de potencias y amperajes atipicos basado en la estadistica historica. 
# Se cargan las medias y desviaciones estandar de potencia y amperaje para cada dispositivo,
# mes, rango de irradiancia, rango de nubosidad y categoria de consigna
print("Descarte de potencias y amperajes atipicos segun los registros historicos")
stats_query = f"""
    SELECT
        rad_bins,
        cloud_bins,
        consigna_cat,
        dispositivo_id,
        mes,
        media_amp,
        std_amp,
        media_pot,
        std_pot
    FROM {SCHEMA_NAME}.historico_stats;"""
stats = pd.read_sql(stats_query, engine)
stats["mes"] = stats["mes"].astype(int)
stats["consigna_cat"] = stats["consigna_cat"].astype(int)
stats["dispositivo_id"] = stats["dispositivo_id"].astype(int)
stats["media_pot"] = stats["media_pot"] * 1000
stats["std_pot"] = stats["std_pot"] * 1000

# Se carga en un diccionario una lista con las labels de los contenedores de irradiancia 
# para cada mes. Con ellas se generan los contenedores de irradiancia en si para cada mes.
labels_month = stats[["mes", "rad_bins"]].drop_duplicates() \
                                            .groupby("mes") \
                                                .apply(lambda x: x["rad_bins"].unique())
labels_month = {k:v.tolist() for k,v in zip(labels_month.index, labels_month.values.tolist())}
bins_month = {}
for k,v in labels_month.items():
    bins = [-np.inf]
    for cut in v:
        try:
            bins.append(float(cut.split("-")[1]))
        except:
            bins.append(np.inf)
    bins = np.sort(bins)
    bins_month[k] = bins

# Se clasifica la irradiancia de cada registro en funcion de los contendor de irradiancia 
# que le corresponda segun su mes.
grouped_df = main_df[["rad_poa", "mes"]].groupby(["mes"])
for month, group in grouped_df:
    month = month[0]
    main_df.loc[group.index, "rad_bins"] = pd.cut(group["rad_poa"], 
                                                  bins=bins_month[month], 
                                                  labels=labels_month[month])
    main_df["rad_bins"] = main_df["rad_bins"].astype('object')

main_df["rad_bins"] = main_df["rad_bins"].astype('category')

# Se contenerizan la nubosidad y consigna y fusionan el dataframe principal con las estadisticas 
# historicas por dichos contenedores.
main_df["cloud_bins"] = pd.cut(main_df["cloud_impact"], 
                               bins=[0, 0.2, 0.4, 0.6, 0.8, 1], 
                               labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
                               ).astype("category")
main_df["consigna_cat"] = np.where(main_df["consigna_pot_act_planta"] == 0, 
                                   0, 
                                   np.where(main_df["consigna_pot_act_planta"] == 1, 
                                            1, 
                                            2)).astype("int8")

df_with_stats = main_df.merge(stats, 
                                on=["rad_bins", 
                                         "cloud_bins", 
                                         "consigna_cat", 
                                         "dispositivo_id", 
                                         "mes"], 
                                how="left") 
del stats

# Se descartan los registros cuyas potencias y amperajes esten fuera del rango de la media 
# mas/menos 3 desviaciones estandar.
df_with_stats["stats_pot_outlier"] = np.where(df_with_stats["consigna_cat"] == 0,
                                        np.where(np.abs(df_with_stats["potencia_act"] 
                                                    - df_with_stats["media_pot"]) 
                                                    > 3*df_with_stats["std_pot"], 
                                                "1", 
                                                "0"),
                                        np.where(np.abs(df_with_stats["potencia_act"] 
                                                    - (df_with_stats["media_pot"]
                                                        /df_with_stats["consigna_pot_act_planta"])) 
                                                    > 3*df_with_stats["std_pot"], 
                                                "1", 
                                                "0"))
n_outliers_pot = df_with_stats["stats_pot_outlier"].astype(int).sum()
print(f"\tNumero de outliers en potencia: {n_outliers_pot}")
df_with_stats = df_with_stats[df_with_stats["stats_pot_outlier"] == "0"]

df_with_stats["stats_amp_outlier"] = np.where(df_with_stats["consigna_cat"] == 0,
                                        np.where(np.abs(df_with_stats["amp_dc"] 
                                                    - df_with_stats["media_amp"])
                                                    > 3*df_with_stats["std_amp"], 
                                                "1", 
                                                "0"),
                                        np.where(np.abs(df_with_stats["amp_dc"] 
                                                    - (df_with_stats["media_amp"]
                                                       /df_with_stats["consigna_pot_act_planta"])) 
                                                    > 3*df_with_stats["std_amp"], 
                                                "1", 
                                                "0"))
n_outliers_amp = df_with_stats["stats_amp_outlier"].astype(int).sum()
print(f"\tNumero de outliers en amperaje: {n_outliers_amp}")
df_with_stats = df_with_stats[df_with_stats["stats_amp_outlier"] == "0"]

# Pivotado de las entradas de corriente continua. El dataframe cargado tiene un registro por cada
# entrada de corriente, por lo que se pivotara para tener un registro instantaneo por dispositivo
amp_df = df_with_stats.pivot(index=["dispositivo_id", "datetime_utc"], 
                             columns='entrada_id', 
                             values='amp_dc')
amp_df.columns = ["amp_dc_" + str(col) for col in amp_df.columns]
amp_columns = amp_df.columns
print(f"Numero de registros del dataframe tras pivotar: {amp_df.shape[0]}")

del df_with_stats
gc.collect()

# Tras pivotar descarte de registros donde alguna entrada fuese atipica (y por tanto ahora nula)
amp_df["n_outliers_amp"] = amp_df.isnull().sum(axis=1)
n_outliers_amp = amp_df[amp_df["n_outliers_amp"] != 0].shape[0]
print(f"\tNumero de registros con amperajes atipicos segun el historico: {n_outliers_amp}")
amp_df = amp_df[amp_df["n_outliers_amp"] <= 2]
amp_df = amp_df.apply(lambda row: row.fillna(row[amp_columns].mean()), axis=1)
n_outliers_amp = amp_df[amp_df['n_outliers_amp'] > 0].shape[0]
print(f"\t\tDe los cuales {n_outliers_amp} son corregidos con la media de las entradas restantes")
amp_df = amp_df.drop(columns=["n_outliers_amp"])

# Analisis comparativo de los amperajes de las entradas de corriente continua
amp_df = amp_df[amp_columns].apply(discriminador, axis=1)
n_outliers_amp = amp_df[amp_df["n_outliers"] != 0].shape[0]
print(f"\tNumero de registros con amperajes atipicos segun el analisis comparativo: {n_outliers_amp}")
amp_df = amp_df[amp_df["n_outliers"] <= 2]
amp_df = amp_df.apply(lambda row: row.fillna(row[amp_columns].mean()), axis=1)
n_outliers_amp = amp_df[amp_df['n_outliers'] > 0].shape[0]
print(f"\t\tDe los cuales {n_outliers_amp} son corregidos con la media de las entradas restantes")
amp_df = amp_df.drop(columns=["n_outliers"])

# Descarte de registros con corriente muy negativa
q1 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.25)
q3 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
n_corriente_neg = amp_df[amp_df < lower_bound].dropna(how='all').shape[0]
print(f"\tNumero de registros con corriente negativa: {n_corriente_neg}")
amp_df = amp_df[amp_df >= lower_bound].dropna(how='any')
amp_df[(amp_df >= lower_bound) & (amp_df <= 0)] = 0
main_df = main_df.drop(columns=["entrada_id", "amp_dc"]) \
                    .drop_duplicates(subset=["id", "datetime_utc"]) \
                        .set_index(["dispositivo_id", "datetime_utc"])
main_df = main_df.merge(amp_df, left_index=True, right_index=True, how="inner")

# Descarte de registros con potencia activa negativa
q1 = main_df[main_df['potencia_act'] < 0]['potencia_act'].quantile(0.25)
q3 = main_df[main_df['potencia_act'] < 0]['potencia_act'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
n_potencia_neg = main_df[main_df["potencia_act"] < lower_bound].dropna(how='all').shape[0]
print(f"\tRegistros descartados por potencia activa negativa: {n_potencia_neg}")
main_df = main_df[main_df["potencia_act"] >= lower_bound]
main_df.loc[main_df["potencia_act"] < 0, "potencia_act"] = 0

# Descarte de registros con diferencias de radiacion anomalas
IQR_soiling = main_df["rad_soiling"].quantile(0.75) - main_df["rad_soiling"].quantile(0.25)
lower_bound_soiling = main_df["rad_soiling"].quantile(0.25) - 3 * IQR_soiling
upper_bound_soiling = main_df["rad_soiling"].quantile(0.75) + 3 * IQR_soiling
outliers_rad_diff = main_df[(main_df["rad_soiling"] <= lower_bound_soiling) 
                            | (main_df["rad_soiling"] >= upper_bound_soiling)].shape[0]
main_df = main_df[(main_df["rad_soiling"] > lower_bound_soiling) 
                  & (main_df["rad_soiling"] < upper_bound_soiling)]
print(f"\tRegistros descartados por diferencias de radiacion anomalas: {outliers_rad_diff}")

# Asignacion de variables temporales
main_df["dia_año"] = main_df.index.get_level_values("datetime_utc").dayofyear
main_df["dia_año_sen"] = np.sin(main_df["dia_año"] * (2*np.pi/365))
main_df["hora_seg"] = main_df.index.get_level_values("datetime_utc").hour * 3600 + \
                        main_df.index.get_level_values("datetime_utc").minute * 60 + \
                        main_df.index.get_level_values("datetime_utc").second
main_df["hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))

# Manipulacion de coeficiente de nubosidad y de sobrecalentamiento
main_df = main_df[main_df["cloud_impact"]>=0]
main_df["coef_pot"] = (main_df["temp_panel"] - 25) * (-0.0045)
main_df["coef_isc"] = (main_df["temp_panel"] - 25) * (0.0005)

print("\nInicio del entrenamiento de los modelos con salida única por dispositivo", end="\n\n")

for inv_id in np.sort(main_df.index.get_level_values("dispositivo_id").unique()):
    # Construccion del dataframe de entrenamiento y validacion por dispositivo.
    # Se descartan atributos no necesarios para el entrenamiento, se pivotan las entradas de
    # corriente continua para tener un registro por dispositivo y entrada y se separan los datos 
    # en conjuntos de entrenamiento y validacion.
    print(f"Dispositivo {inv_id}")
    disp_df = main_df[main_df.index.get_level_values("dispositivo_id") == inv_id].copy()
    disp_df = disp_df.dropna()
    disp_df = disp_df.drop(columns = ["id",
                                    "potencia_act",
                                    "num_strings",
                                    "dia_año",
                                    "hora_seg",
                                    "mes",
                                    "rad_bins",
                                    "cloud_bins",
                                    "consigna_cat",
                                    ])
    disp_df = disp_df.melt(value_vars = amp_columns, 
                           var_name = "entrada_id", 
                           value_name = "amp_dc", 
                           ignore_index = False) \
                                .merge(disp_df.drop(columns=amp_columns), 
                                       left_index=True, 
                                       right_index=True)
    disp_df["entrada_id"] = disp_df["entrada_id"].str.replace("amp_dc_", "").astype(int)
    disp_df = disp_df.set_index("entrada_id", append=True)

    train_df, validation_df = train_test_split(
                                disp_df, 
                                test_size = 0.2,
                                stratify = disp_df.index.get_level_values("datetime_utc").hour, 
                                random_state = 42)
    print(f"\tRegistros de entrenamiento: {train_df.shape[0]}")
    print(f"\tRegistros de validacion: {validation_df.shape[0]}")

    # Separacion de features y target
    y = train_df[["amp_dc"]].copy()
    y_val = validation_df[["amp_dc"]].copy()
    X = train_df.drop(columns = "amp_dc")
    X_val = validation_df.drop(columns = "amp_dc")

    # Estandarizacion/normalizacion de variables numericas y codificacion de variables categoricas
    perc_attr = ['cloud_impact', 
                 'consigna_pot_act_planta']
    std_attr = []
    norm_attr = ['rad_poa', 
                 'rad_soiling', 
                 'temp_amb', 
                 'temp_panel', 
                 'coef_pot',
                 'coef_isc',] 
    cat_attr = []

    transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])
    transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 
    transformador_numerico_norm = Pipeline([('minmax_scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                   ('norm', transformador_numerico_norm, norm_attr),
                                                ('std', transformador_numerico_std, std_attr)],
                                remainder='passthrough')
    
    # Preparacion de los datos para el entrenamiento
    X_prep = preprocessor.fit_transform(X)
    dtrain = xgb.DMatrix(X_prep, label=y)

    if OPTIMIZATION:
        # En caso de optimizacion de hiperparametros se realiza una busqueda de los mismos
        # mediante el metodo de optimizacion de hiperparametros de Hyperopt. Se realiza una
        # busqueda de 250 iteraciones por cada conjunto de hiperparametros, con un limite de 5
        # iteraciones sin mejora por defecto. Se realiza una validacion cruzada de 5 folds por 
        # defecto y se muestra informacion por pantalla.
        # Con los hiperparametros optimos se entrena el modelo y se guarda en un pipeline junto
        # con el preprocesador.
        start_time = time.time()
        print("\tOptimizacion de los hiperparametros")
        space = {'n_estimators': hp.quniform('n_estimators', 100, 3000, 5),
                'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
                'max_depth': hp.quniform("max_depth", 3, 30, 1),
                'gamma': hp.uniform ('gamma', 0, 25),
                'min_child_weight' : hp.quniform('min_child_weight', 1, 30, 1),
                'subsample' : hp.uniform('subsample', 0.4, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 50),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 50)}
        best_params = optimize_hyperparameters(space, 
                                               dtrain, 
                                               STALL_LIMIT = STALL_LIMIT, 
                                               MAX_EVALS_PER_RUN = MAX_EVALS_PER_RUN, 
                                               cv_folds = CV_FOLDS, 
                                               verbose = True)
        end_time = time.time()
        exec_time = end_time - start_time
        exec_hours = int(exec_time // 3600)
        exec_min = int((exec_time % 3600) // 60)
        exec_sec = int(exec_time % 60)
        print(f"Tiempo de ejecucion: {exec_hours} horas, {exec_min} minutos y {exec_sec} segundos")

        final_params = {"device": DEVICE,
                    "objective": 'reg:squarederror', 
                    "tree_method": "hist",
                    "random_state": 42,
                    "n_jobs": -1}
        final_params = {**final_params, **best_params}
        train_params = final_params.copy()
        final_params.pop('n_estimators')
        final_params['max_depth'] = int(final_params['max_depth'])
        final_params['min_child_weight'] = max(1, int(final_params['min_child_weight']))
        model = xgb.train(final_params, dtrain, num_boost_round = int(best_params['n_estimators']))
    else:
        # En caso de no optimizacion de hiperparametros se entrena el modelo con los hiperparametros
        # por defecto y se guarda en un pipeline junto con el preprocesador.
        train_params = {'device': DEVICE,
                'objective': 'reg:squarederror',
                'tree_method': "hist",
                'random_state': 42,
                'n_jobs': -1}
        model = xgb.train(train_params, dtrain) 

    pipeline_model = Pipeline([('preprocessor', preprocessor),
                        ('regressor', model)])
    
    # Se cargan el numero de strings asociado a cada entrada de corriente continua para la 
    # desnormalizacion. Se realiza la prediccion de la intensidad de corriente a la entrada
    # de los inversores, se combina con el valor real, se desnormaliza y se calcula la diferencia.
    consulta_sql = f"""SELECT entrada_id, 
                                num_strings
                            FROM {SCHEMA_NAME}.distrib_inversores
                            WHERE dispositivo_id = {inv_id};"""
    num_strings = pd.read_sql_query(consulta_sql, engine)

    X_val_prep = xgb.DMatrix(pipeline_model.named_steps['preprocessor'].transform(X_val))
    prediction_df = y_val.copy()
    prediction_df["y_pred"] = pipeline_model.named_steps['regressor'].predict(X_val_prep)
    prediction_df = prediction_df.reset_index().merge(num_strings, on="entrada_id")
    prediction_df["amp_dc"] = prediction_df["amp_dc"] * prediction_df["num_strings"]
    prediction_df["y_pred"] = prediction_df["y_pred"] * prediction_df["num_strings"]
    prediction_df["y_diff"] = prediction_df["amp_dc"] - prediction_df["y_pred"]

    # Calculo de metricas de evaluacion
    rmse_score = round(mse(prediction_df["amp_dc"], prediction_df["y_pred"], squared=False), 3)
    mae_score = round(mae(prediction_df["amp_dc"], prediction_df["y_pred"]), 3)
    r2_score = round(r2(prediction_df["amp_dc"], prediction_df["y_pred"]), 3)
    metricas = {"RMSE": rmse_score, "MAE": mae_score, "R2": r2_score}
    print(f"\tMetricas de validacion: {metricas}", end="\n\n")

    # Guardado del modelo y de las metricas
    algoritmo = pipeline_model.named_steps["regressor"].__class__.__name__
    columnas = [col_name.split("__")[1] for col_name in preprocessor.get_feature_names_out()]
    importance_scores = model.get_score(importance_type='gain')
    total_gain = np.array([v for k,v in importance_scores.items()]).sum()
    importancia = {k: v/total_gain for k, v in importance_scores.items()}
    path = os.path.join(ROOT_PATH, 
                        f"Modelos/amperaje_promedio/Inversor_{inv_id - 20}",
                        f"Nuevos/{algoritmo}-{pd.Timestamp.now()}/")
    os.makedirs(path)
    with open(path+'model.model', "wb") as archivo_salida:
        pickle.dump(pipeline_model, archivo_salida)

    # Generacion de graficos: comparativa de valores reales y predichos, histograma de diferencias
    # y matriz de correlacion
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(prediction_df["amp_dc"], prediction_df["y_pred"], marker = ".")
    plt.plot([min(prediction_df["amp_dc"]), max(prediction_df["amp_dc"])], 
             [min(prediction_df["amp_dc"]), max(prediction_df["amp_dc"])], 
             color='black', 
             linestyle='-', 
             linewidth=1)
    ax.text(0.95, 
            0.05, 
            f"RMSE: {rmse_score}\nMAE: {mae_score}\nR2: {r2_score}", 
            transform = plt.gca().transAxes, 
            fontsize = 10,
            verticalalignment = 'bottom', 
            horizontalalignment = 'right', 
            bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("Comparacion de valores reales y predichos")
    plt.savefig(path + "scatter_validacion.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.histplot(prediction_df["y_diff"], kde=True, stat='percent')
    ax.axvline(x = 0, color = 'black', linestyle = '--', linewidth = 0.35, label = 'x=0')
    ax.text(0.95, 
            0.95, 
            f"Media: {round(prediction_df['y_diff'].mean(), 2)}\nDesviacion estandar: {round(prediction_df['y_diff'].std(), 2)}", 
            transform = ax.transAxes, 
            fontsize = 10, 
            verticalalignment = 'top', 
            horizontalalignment = 'right', 
            bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.title('Histograma de las diferencias entre valores reales y predichos')
    plt.xlabel('Diferencia')
    plt.ylabel('Porcentaje')
    plt.savefig(path + "histograma_diferencias.png")

    plt.figure(figsize=(12, 8))
    sns.heatmap(train_df[perc_attr + std_attr + norm_attr + y.columns.tolist()].corr(), 
                annot = True, 
                fmt = ".2f", 
                cmap = "coolwarm")
    plt.xticks(rotation = 45)
    plt.title("Matriz de correlacion")
    plt.savefig(path + "matriz_correlacion.png")

    # Distribucion de los registros por hora del dia
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(disp_df.index.get_level_values("datetime_utc").hour, 
                 bins = disp_df.index.get_level_values("datetime_utc").hour.nunique(), 
                 kde = False, 
                 stat = 'percent')
    plt.title("Distribucion de registros por hora del dia")
    plt.xlabel("Hora")
    plt.xticks(disp_df.index.get_level_values("datetime_utc").hour.unique())
    plt.ylabel("Porcentaje")
    plt.savefig(path + "registros_por_hora.png")

    # Generacion de graficos: comparativa de RMSE y RMSE relativo por hora
    rmse_list = []
    rrmse_list = []
    hora_list = []
    for group in prediction_df.reset_index().groupby(["dispositivo_id", 
                                                      prediction_df.reset_index()["datetime_utc"].dt.hour]):
        hora_list.append(group[0][1])
        rmse_score = round(mse(group[1]["amp_dc"], 
                               group[1]["y_pred"], 
                               squared=False), 
                            2)
        rmse_list.append(rmse_score)
        mean_amp_dc = group[1]["amp_dc"].mean()
        if mean_amp_dc == 0:
            rrmse_score = 0
        else:
            rrmse_score = round((mse(group[1]["amp_dc"], 
                                     group[1]["y_pred"], 
                                     squared=False)*100/mean_amp_dc), 
                                2)
        rrmse_list.append(rrmse_score)
        
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Hora')
    ax1.set_ylabel('RMSE', color=color)
    ax1.plot(hora_list, rmse_list, color=color, linewidth=1)
    sns.scatterplot(x=hora_list, y=rmse_list, color=color, ax=ax1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('RMSE Relativo', color=color)
    if max(rrmse_list) - min(rrmse_list) > 0.25:
        import matplotlib.ticker as ticker
        ax2.set_yscale('log')
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
    ax2.plot(hora_list, rrmse_list, color=color, linewidth=1)
    sns.scatterplot(x=hora_list, y = rrmse_list, color=color, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Comparativa de RMSE y RMSE Relativo por hora para el inversor {inv_id - 20}')
    plt.xticks(hora_list)
    ax1.grid(True, which='minor', color='gray', linewidth=0.5)
    ax2.grid(True, which='minor', color='gray', linewidth=0.5)
    plt.savefig(path + "rmse_por_hora.png")

    # Generacion de graficos: comparativa de RMSE y RMSE relativo por entrada
    rmse_list = []
    rrmse_list = []
    entrada_list = []
    for group in prediction_df.reset_index().groupby(["dispositivo_id", "entrada_id"]):
        entrada_list.append(group[0][1])
        rmse_score = round(mse(group[1]["amp_dc"], 
                               group[1]["y_pred"], 
                               squared=False), 
                            2)
        rmse_list.append(rmse_score)
        mean_amp_dc = group[1]["amp_dc"].mean()
        if mean_amp_dc == 0:
            rrmse_score = 0
        else:
            rrmse_score = round((mse(group[1]["amp_dc"], 
                                     group[1]["y_pred"], 
                                     squared=False)*100/mean_amp_dc), 
                                2)
        rrmse_list.append(rrmse_score)
        
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Entrada')
    ax1.set_ylabel('RMSE', color=color)
    ax1.plot(entrada_list, rmse_list, color=color, linewidth=1)
    sns.scatterplot(x=entrada_list, y=rmse_list, color=color, ax=ax1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('RMSE Relativo', color=color)
    ax2.plot(entrada_list, rrmse_list, color=color, linewidth=1)
    sns.scatterplot(x=entrada_list, y=rrmse_list, color=color, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'RMSE por entrada para el inversor {inv_id - 20}')
    plt.xticks(entrada_list)
    plt.tight_layout()
    ax1.grid(True, which='major', color='gray', linewidth=0.5)
    ax2.grid(True, which='minor', color='gray', linewidth=0.5)
    plt.savefig(path + "rmse_entrada.png")

    target_unnormalized = y.reset_index().merge(num_strings, on="entrada_id")
    target_unnormalized["amp_dc"] = target_unnormalized["amp_dc"] * target_unnormalized["num_strings"]

    with open(path+'informe_modelo.json', 'w') as archivo_json:
        informe = {"optimizacion": OPTIMIZATION,
                "intervalo_min": INTERVALO_MIN,
                "metricas": metricas,
                "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia.values())}.items(), 
                                                    key=lambda item: item[1], 
                                                    reverse=True)),
                "hiperparametros": {k:v for k,v in train_params.items() if v != None},
                "training_input_description": train_df[perc_attr + std_attr].describe() \
                                                    .loc[["mean", "std", "min", "max"]] \
                                                        .to_dict(),
                "training_target_description": target_unnormalized["amp_dc"].describe().to_dict(),
                }
        json.dump(informe, archivo_json)
    plt.close('all')
