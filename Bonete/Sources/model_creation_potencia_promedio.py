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
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import time
import gc


def rad_cut(df, bins):
    """
    Funcion para categorizar la irradiancia en intervalos definidos por mes y cargados desde
    la base de datos.
    La función recibe el dataframe agrupado por mes y los limites de los intervalos de irradiancia,
    genera unas labels 
    """
    bins = np.sort(bins)
    labels = [str(int(bins[i])) + "-" + str(int(bins[i+1])) if np.isfinite(bins[i]) and np.isfinite(bins[i+1]) else str(bins[i]) + "-" + str(bins[i+1]) for i in range(len(bins)-1)]
    labels[-1] = str(bins[-2]) + "+"
    labels[0] = labels[0].replace("-inf", "0")
    df["rad_bins"] = pd.cut(df["rad_poa"], bins=bins, labels=labels)
    return df

def discriminador(row):
    median = np.median(row)
    IQR = np.percentile(row, 75) - np.percentile(row, 25)
    coef_iqr = 3
    outlier = np.where((row < median - coef_iqr*IQR) | (row > median + coef_iqr*IQR), 1, 0)
    filtered_amps = np.where(outlier == 0, row, np.nan)
    row.loc[row.index] = filtered_amps
    row["n_outliers"] = outlier.sum()

    return row

def objective(space, train_set, cv_folds):
    n_estimators = int(space.pop('n_estimators'))
    params = {'device': device,
            'objective': 'reg:squarederror',
            'tree_method': "hist",
            'random_state': 42,
            'n_jobs': -1}
    params = {**params, **space}
    if 'max_depth' in params:
        params['max_depth'] = int(params['max_depth'])
    if 'min_child_weight' in params:
        params['min_child_weight'] = max(1, int(params['min_child_weight']))

    cv_results = xgb.cv(params, train_set, nfold = cv_folds, num_boost_round = n_estimators, early_stopping_rounds = 100, metrics = 'rmse', as_pandas = True)
    return {'loss': cv_results['test-rmse-mean'].min(), 'status': STATUS_OK, 'n_rounds': cv_results.shape[0]}

def optimize_hyperparameters(space, dtrain, STALL_LIMIT = 5, MAX_EVALS_PER_RUN = 250, gamma_algo = 0.75, cv_folds = 5, verbose = False):
    best_loss = np.inf
    trials = Trials()
    stall_counter = 0
    num_evals = 0
    upper_limit = (MAX_EVALS_PER_RUN * (STALL_LIMIT - 1)) * 10

    while stall_counter < STALL_LIMIT and num_evals < upper_limit:
        best = fmin(fn=lambda space: objective(space, train_set=dtrain, cv_folds=cv_folds), 
                    space=space, 
                    algo=partial(tpe.suggest, gamma=gamma_algo),
                    max_evals=num_evals + MAX_EVALS_PER_RUN, 
                    trials=trials,
                    verbose=verbose)
        best_params = space_eval(space, best)
        new_loss = trials.best_trial['result']['loss']
        if new_loss < best_loss:
            best_loss = new_loss
            stall_counter = 0
        else:
            stall_counter += 1
        num_evals += MAX_EVALS_PER_RUN
        gamma_algo = max(0.25, gamma_algo - 0.05)

    best_params["n_estimators"] = trials.best_trial['result']['n_rounds']
    print(f"\tNúmero de evaluaciones realizadas: {num_evals}")
    print(f"\tBest params: {best_params}")
    print(f"\tBest trial: {trials.best_trial}")
    print(f"\tBest loss: {best_loss}", end="\n\n")

    return best_params



print("Configuración para el entrenamiento de modelos de estimación de potencia activa de salida de inversores en plantas fotovoltaicas:", end="\n\n")
model_name = "XGBRegressor"

valid_responses = ["y", "n"]
optimizacion = ""
while optimizacion not in valid_responses:
    optimizacion = input("¿Desea optimizar el modelo? (Y/N): ").lower()
optimizacion = optimizacion == "y"

valid_devices = ["cpu", "cuda"]
device = ""
if optimizacion:
    # Comprobación de la disponibilidad de GPU para el entrenamiento
    if torch.cuda.is_available():
        while device not in valid_devices:
            device_bool = input("¿Desea utilizar GPU para el entrenamiento? (Y/N): ").lower()
            if device_bool == "y":
                device = "cuda"
            else:
                device = "cpu"
else:
    device = 'cpu'

print(f"\nLas opciones seleccionadas son: \nModelo: {model_name} \nOptimización: {optimizacion} \nEntrenamiento: {device}", end="\n\n")

# Carga del fichero de parámetros para conexión con la base de datos
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
print(f"Conexión a la base de datos {params['dbname']} (esquema {schema_name}) establecida")

INTERVALO_MIN = 15      # Intervalo de promediado en minutos
NUM_MOD_STRING = 30     # Estimación del número de módulos en cada string
SUP_MOD = 2             # Superficie de cada módulo en m^2
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
        FROM {schema_name}.inversores AS inv
        JOIN {schema_name}.inversores_detalle AS det
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
            AVG((rad_celda1 - rad_celda2))/NULLIF(AVG(rad_celda1), 0) AS rad_soiling,   --valor porcentual
            AVG(temp_amb) AS temp_amb, 
            AVG((temp_panel1 + temp_panel2)/2)  AS temp_panel,
            AVG((LEAST(cloud_impact, 100)/100)) AS cloud_impact
        FROM {schema_name}.meteo
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
        FROM {schema_name}.ree
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
        consigna_pot_act_planta/MAX(consigna_pot_act_planta) OVER (PARTITION BY inv.dispositivo_id) AS consigna_pot_act_planta,
        potencia_act,
        amp_dc,  
        num_strings
    FROM inv
    JOIN {schema_name}.distrib_inversores dist
        ON dist.dispositivo_id = inv.dispositivo_id
            AND dist.entrada_id = inv.entrada_id
    JOIN met_gr
        ON met_gr.datetime_utc_rounded = inv.datetime_utc_rounded
    JOIN ree
        ON ree.datetime_utc_rounded = inv.datetime_utc_rounded
    ORDER BY 4, 2, 3;"""

chunksize = 100000
chunks = pd.read_sql_query(main_query, engine, chunksize=chunksize)
main_df = pd.DataFrame()
for chunk in chunks:
    main_df = pd.concat([main_df, chunk], ignore_index = True)
del chunks, chunk
gc.collect()

# Normalización de la entrada de corriente continua, formateo de fechas y escalado de potencia
print(f"Carga inicial de {main_df.shape[0]} registros")
print("Entrada de corriente continua normalizada según el número de strings")
main_df["amp_dc"] = main_df["amp_dc"]/main_df["num_strings"]
main_df["datetime_utc"] = pd.to_datetime(main_df["datetime_utc"], utc = True)
main_df["mes"] = main_df["datetime_utc"].dt.month
main_df["potencia_act"] = main_df["potencia_act"] * 1000

# # Pivotado de las entradas de corriente continua. El dataframe cargado tiene un registro por cada
# # entrada de corriente, por lo que se pivotará para tener un registro instántaneo por dispositivo
# amp_df = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
# amp_df.columns = ["amp_dc_" + str(col) for col in amp_df.columns]
# print(f"Número de registros del dataframe tras pivotar: {amp_df.shape[0]}")

# main_df = main_df.drop(columns=["amp_dc", "entrada_id", "num_strings"]).drop_duplicates()
# main_df = main_df.set_index(["dispositivo_id", "datetime_utc"])
# gc.collect()

# Descarte de potencias y amperajes atípicos basado en la estadística histórica. Se cargan las medias y
# desviaciones estándar de potencia y amperaje para cada dispositivo, mes, rango de irradiancia, 
# rango de nubosidad y categoría de consigna
print("Descarte de potencias y amperajes atípicos según los registros históricos")
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
    FROM {schema_name}.historico_stats;"""
stats = pd.read_sql(stats_query, engine)
stats["mes"] = stats["mes"].astype(int)
stats["consigna_cat"] = stats["consigna_cat"].astype(int)
stats["dispositivo_id"] = stats["dispositivo_id"].astype(int)
stats["media_pot"] = stats["media_pot"] * 1000
stats["std_pot"] = stats["std_pot"] * 1000

# Se carga en un diccionario las labels de los contenedores de irradiancia para cada mes
# Con ellos se generan los contenedores de irradiancia en sí para cada mes
labels_month = stats[["mes", "rad_bins"]].drop_duplicates().groupby("mes").apply(lambda x: x["rad_bins"].unique())
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

# Se clasifica la irradiancia de cada registro en función de los contendor de irradiancia correspondiente al mes
grouped_df = main_df[["rad_poa", "mes"]].groupby(["mes"])
for month, group in grouped_df:
    month = month[0]
    main_df.loc[group.index, "rad_bins"] = pd.cut(group["rad_poa"], bins=bins_month[month], labels=labels_month[month])
    main_df["rad_bins"] = main_df["rad_bins"].astype('object')

main_df["rad_bins"] = main_df["rad_bins"].astype('category')

# Se contenerizan la nubosidad y consigna y fusionan el dataframe principal con las estadísticas historicas por dichos contenedores
main_df["cloud_bins"] = pd.cut(main_df["cloud_impact"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]).astype("category")
main_df["consigna_cat"] = np.where(main_df["consigna_pot_act_planta"] == 0, 0, np.where(main_df["consigna_pot_act_planta"] == 1, 1, 2)).astype("int8")

df_with_stats = main_df.merge(stats, on=["rad_bins", "cloud_bins", "consigna_cat", "dispositivo_id", "mes"], how="left") 
del stats

# Se descartan los registros cuyas potencias y amperajes estén fuera del rango de la media más/menos 3 desviaciones estándar
df_with_stats["stats_pot_outlier"] = np.where(df_with_stats["consigna_cat"] == 0,
                                            np.where(np.abs(df_with_stats["potencia_act"] - df_with_stats["media_pot"]) > 3*df_with_stats["std_pot"], 
                                                     "1", 
                                                     "0"),
                                            np.where(np.abs(df_with_stats["potencia_act"] - (df_with_stats["media_pot"]/df_with_stats["consigna_pot_act_planta"])) > 3*df_with_stats["std_pot"], 
                                                     "1", 
                                                     "0"))
n_outliers_pot = df_with_stats["stats_pot_outlier"].astype(int).sum()
print(f"Número de outliers en potencia: {n_outliers_pot}")
df_with_stats = df_with_stats[df_with_stats["stats_pot_outlier"] == "0"]

df_with_stats["stats_amp_outlier"] = np.where(df_with_stats["consigna_cat"] == 0,
                                            np.where(np.abs(df_with_stats["amp_dc"] - df_with_stats["media_amp"]) > 3*df_with_stats["std_amp"], 
                                                     "1", 
                                                     "0"),
                                            np.where(np.abs(df_with_stats["amp_dc"] - (df_with_stats["media_amp"]/df_with_stats["consigna_pot_act_planta"])) > 3*df_with_stats["std_amp"], 
                                                     "1", 
                                                     "0"))
n_outliers_amp = df_with_stats["stats_amp_outlier"].astype(int).sum()
print(f"Número de outliers en amperaje: {n_outliers_amp}")
df_with_stats = df_with_stats[df_with_stats["stats_amp_outlier"] == "0"]

# Pivotado de las entradas de corriente continua. El dataframe cargado tiene un registro por cada
# entrada de corriente, por lo que se pivotará para tener un registro instántaneo por dispositivo
amp_df = df_with_stats.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
amp_df.columns = ["amp_dc_" + str(col) for col in amp_df.columns]
amp_columns = amp_df.columns
print(f"Número de registros del dataframe tras pivotar: {amp_df.shape[0]}")

del df_with_stats
gc.collect()

# Tras pivotar descarte de registros donde alguna entrada fuese atípica (y por tanto ahora nula)
amp_df["n_outliers_amp"] = amp_df.isnull().sum(axis=1)
n_outliers_amp = amp_df[amp_df["n_outliers_amp"] != 0].shape[0]
print(f"Número de registros con amperajes atípicos según el histórico: {n_outliers_amp}")
amp_df = amp_df[amp_df["n_outliers_amp"] <= 2]
amp_df = amp_df.apply(lambda row: row.fillna(row[amp_columns].median()), axis=1)
print(f"\tDe los cuales {amp_df[amp_df['n_outliers_amp'] >0].shape[0]} son corregidos con la mediana de las entradas restantes")
amp_df = amp_df.drop(columns=["n_outliers_amp"])

# Análisis comparativo de los amperajes de las entradas de corriente continua
amp_df = amp_df[amp_columns].apply(discriminador, axis=1)
n_outliers_amp = amp_df[amp_df["n_outliers"] != 0].shape[0]
print(f"Número de registros con amperajes atípicos según el análisis comparativo: {n_outliers_amp}")
amp_df = amp_df[amp_df["n_outliers"] <= 2]
amp_df = amp_df.apply(lambda row: row.fillna(row[amp_columns].median()), axis=1)
print(f"\tDe los cuales {amp_df[amp_df['n_outliers'] >0].shape[0]} son corregidos con la mediana de las entradas restantes")
amp_df = amp_df.drop(columns=["n_outliers"])

# Descarte de registros con corriente muy negativa
q1 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.25)
q3 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
n_corriente_neg = amp_df[amp_df < lower_bound].dropna(how='all').shape[0]
print(f"Número de registros con corriente negativa: {n_corriente_neg}")
amp_df = amp_df[amp_df >= lower_bound].dropna(how='any')
amp_df[(amp_df >= lower_bound) & (amp_df <= 0)] = 0
main_df = main_df.drop(columns=["entrada_id", "amp_dc"]).drop_duplicates(subset=["id", "datetime_utc"]).set_index(["dispositivo_id", "datetime_utc"])
main_df = main_df.merge(amp_df, left_index=True, right_index=True, how="inner")

# Descarte de registros con potencia activa negativa
q1 = main_df[main_df['potencia_act'] < 0]['potencia_act'].quantile(0.25)
q3 = main_df[main_df['potencia_act'] < 0]['potencia_act'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
n_potencia_neg = main_df[main_df["potencia_act"] < lower_bound].dropna(how='all').shape[0]
print(f"Registros descartados por potencia activa negativa: {n_potencia_neg}")
main_df = main_df[main_df["potencia_act"] >= lower_bound]
main_df.loc[main_df["potencia_act"] < 0, "potencia_act"] = 0

# Descarte de registros con diferencias de radiación anómalas
IQR_soiling = main_df["rad_soiling"].quantile(0.75) - main_df["rad_soiling"].quantile(0.25)
lower_bound_soiling = main_df["rad_soiling"].quantile(0.25) - 3*IQR_soiling
upper_bound_soiling = main_df["rad_soiling"].quantile(0.75) + 3*IQR_soiling
outliers_rad_diff = main_df[(main_df["rad_soiling"] <= lower_bound_soiling) | (main_df["rad_soiling"] >= upper_bound_soiling)].shape[0]
main_df = main_df[(main_df["rad_soiling"] > lower_bound_soiling) & (main_df["rad_soiling"] < upper_bound_soiling)]
print(f"Registros descartados por diferencias de radiación anómalas: {outliers_rad_diff}")

# Asignación de variables temporales
main_df["dia_año"] = main_df.index.get_level_values("datetime_utc").dayofyear
main_df["dia_año_sen"] = np.sin(main_df["dia_año"] * (2*np.pi/365))
main_df["hora_seg"] = main_df.index.get_level_values("datetime_utc").hour * 3600 + \
                        main_df.index.get_level_values("datetime_utc").minute * 60 + \
                        main_df.index.get_level_values("datetime_utc").second
main_df["hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))

# Manipulación de coeficiente de nubosidad y de sobrecalentamiento
main_df = main_df[main_df["cloud_impact"]>=0]
main_df["coef_overheat"] = ((main_df["temp_panel"] - 25) * (-0.004)).clip(lower=0)

for inv_id in np.sort(main_df.index.get_level_values("dispositivo_id").unique()):
    # Separación de los datos de entrenamiento y validación
    print()
    print(f"Dispositivo {inv_id}")
    disp_df = main_df[main_df.index.get_level_values("dispositivo_id") == inv_id].copy()
    disp_df = disp_df.dropna()

    train_df, validation_df = train_test_split(disp_df, 
                                               test_size = 0.2,
                                               stratify = disp_df.index.get_level_values("datetime_utc").hour, 
                                               random_state = 42)
    
    print(f"\tRegistros de entrenamiento: {train_df.shape[0]}")
    print(f"\tRegistros de validación: {validation_df.shape[0]}")

    # Separación de input y target
    y = train_df[["potencia_act"]].copy()
    y_val = validation_df[["potencia_act"]].copy()
    amp_columns = train_df.filter(like="amp_dc").columns.tolist()
    X = train_df.drop(columns = amp_columns+["id",
                                            "num_strings",
                                            "mes",
                                            "rad_bins",
                                            "cloud_bins",
                                            "consigna_cat",
                                            "dia_año",
                                            "hora_seg",
                                            "potencia_act"
                                            ])
    X_val = validation_df.drop(columns = amp_columns+["id",
                                                    "num_strings",
                                                    "mes",
                                                    "rad_bins",
                                                    "cloud_bins",
                                                    "consigna_cat",
                                                    "dia_año",
                                                    "hora_seg",
                                                    "potencia_act"
                                                    ])

    # Estandarización/normalización de variables numéricas y codificación de variables categóricas
    perc_attr = ['cloud_impact', 'consigna_pot_act_planta']
    std_attr = []
    norm_attr = ['rad_poa', 'temp_amb', 'rad_soiling', 'temp_panel', 'coef_overheat']
    cat_attr = []

    transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))]) # Introducir manualmente catergorías?
    transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 
    transformador_numerico_norm = Pipeline([('minmax_scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                   ('norm', transformador_numerico_norm, norm_attr),
                                                ('std', transformador_numerico_std, std_attr)],
                                remainder='passthrough')
    X_prep = preprocessor.fit_transform(X)
    dtrain = xgb.DMatrix(X_prep, label=y)

    if optimizacion:
        start_time = time.time()
        print("\tOptimización de los hiperparámetros")
        space = {'n_estimators': hp.quniform('n_estimators', 100, 3000, 5),
                'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
                'max_depth': hp.quniform("max_depth", 3, 30, 1),
                'gamma': hp.uniform ('gamma', 0, 25),
                'min_child_weight' : hp.quniform('min_child_weight', 1, 30, 1),
                'subsample' : hp.uniform('subsample', 0.4, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 50),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 50)}
        best_params = optimize_hyperparameters(space, dtrain, STALL_LIMIT = 10, MAX_EVALS_PER_RUN = 250, cv_folds = 10, verbose = True)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_hours = int(execution_time // 3600)
        execution_minutes = int((execution_time % 3600) // 60)
        execution_seconds = int(execution_time % 60)
        print(f"Tiempo de ejecución: {execution_hours} horas, {execution_minutes} minutos y {execution_seconds} segundos")

        final_params = {"device": device,
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
        train_params = {'device': device,
                'objective': 'reg:squarederror',
                'tree_method': "hist",
                'random_state': 42,
                'n_jobs': -1}
        model = xgb.train(train_params, dtrain) 

    pipeline_model = Pipeline([('preprocessor', preprocessor),
                        ('regressor', model)])

    prediction_df = y_val.copy()
    dval = xgb.DMatrix(pipeline_model.named_steps['preprocessor'].transform(X_val))
    prediction_df["y_pred"] = pipeline_model.named_steps['regressor'].predict(dval)
    prediction_df["y_diff"] = prediction_df["potencia_act"] - prediction_df["y_pred"]

    rmse_score = round(mse(prediction_df["potencia_act"], prediction_df["y_pred"], squared = False),3)
    mae_score = round(mae(prediction_df["potencia_act"], prediction_df["y_pred"]),3)
    r2_score = round(r2(prediction_df["potencia_act"], prediction_df["y_pred"]),3)
    metricas = {"RMSE": rmse_score, "MAE": mae_score, "R2": r2_score}
    print(f"RMSE: {rmse_score}", 
            f"MAE: {mae_score}",
            f"R2: {r2_score}",
            sep = "\n")
    
    # Guardado del modelo y de las métricas
    algoritmo = pipeline_model.named_steps["regressor"].__class__.__name__
    columnas = [col_name.split("__")[1] for col_name in preprocessor.get_feature_names_out()]
    importance_scores = model.get_score(importance_type='gain')
    total_gain = np.array([v for k,v in importance_scores.items()]).sum()
    importancia = {k: v/total_gain for k, v in importance_scores.items()}
    path = os.path.join(root_path, f"Modelos/potencia_inversor_promedio/Inversor_{inv_id - 20}/Nuevos/{algoritmo}-{pd.Timestamp.now()}/")
    os.makedirs(path)
    with open(path+'model.model', "wb") as archivo_salida:
        pickle.dump(pipeline_model, archivo_salida)

    # Cálculo de RMSE por rangos de irradiancia
    rad_bins = [0, 100, 250, 500, np.inf]
    rad_labels = ["0-100", "100-250", "250-500", "500+"]
    prediction_df["rad_bin"] = pd.cut(validation_df["rad_poa"], bins = rad_bins, labels = rad_labels)
    rmse_rad = {}
    for group in prediction_df.groupby("rad_bin"):
        rmse_rad[group[0]] = round(mse(group[1]["potencia_act"], group[1]["y_pred"], squared=False), 2)
    
    # Generación de gráficos: comparativa de valores reales y predichos, histograma de diferencias y matriz de correlación
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()
    plt.scatter(prediction_df["potencia_act"], prediction_df["y_pred"], marker = ".")
    plt.plot([min(prediction_df["potencia_act"]), max(prediction_df["potencia_act"])], [min(prediction_df["potencia_act"]), max(prediction_df["potencia_act"])], color='black', linestyle='-', linewidth=1)
    ax.text(0.95, 0.05, f"RMSE: {rmse_score}\nMAE: {mae_score}\nR2: {r2_score}", transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("Comparación de valores reales y predichos")
    plt.savefig(path + "scatter_validacion.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()
    ax = sns.histplot(prediction_df["potencia_act"] - prediction_df["y_pred"], kde=True, stat='percent')
    ax.axvline(x=0, color='black', linestyle='--', linewidth = 0.35, label='x=0')
    prediction_df["y_diff"] = prediction_df["potencia_act"] - prediction_df["y_pred"]
    ax.text(0.95, 0.95, f"Media: {round(prediction_df['y_diff'].mean(), 2)}\nDesviación estándar: {round(prediction_df['y_diff'].std(), 2)}", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Histograma de las diferencias entre valores reales y predichos')
    plt.xlabel('Diferencia')
    plt.ylabel('Porcentaje')
    plt.savefig(path + "histogram_validacion.png")

    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    sns.heatmap(train_df[perc_attr + std_attr + norm_attr + y.columns.tolist()].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Matriz de correlación")
    plt.savefig(path + "correlacion.png")

    # Distribución de los registros por hora del día
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()
    sns.histplot(disp_df.index.get_level_values("datetime_utc").hour, bins=disp_df.index.get_level_values("datetime_utc").hour.nunique(), kde=False, stat='percent')
    plt.title("Distribución de registros por hora del día")
    plt.xlabel("Hora")
    plt.ylabel("Porcentaje")
    plt.savefig(path + "distribucion_hora.png")

     # Generación de gráficos: comparativa de RMSE y RMSE relativo por hora
    rmse_list = []
    rrmse_list = []
    hora_list = []
    for group in prediction_df.reset_index().groupby(["dispositivo_id", prediction_df.reset_index()["datetime_utc"].dt.hour]):
        hora_list.append(group[0][1])
        rmse_score = round(mse(group[1]["potencia_act"], group[1]["y_pred"], squared=False), 2)
        rmse_list.append(rmse_score)
        mean_potencia_act = group[1]['potencia_act'].mean()
        if mean_potencia_act == 0:
            rrmse_score = 0
        else:
            rrmse_score = round((mse(group[1]["potencia_act"], group[1]["y_pred"], squared=False)*100/mean_potencia_act), 2)
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
    plt.tight_layout()
    ax1.grid(True, which='minor', color='gray', linewidth=0.5)
    ax2.grid(True, which='minor', color='gray', linewidth=0.5)
    plt.savefig(path + "rmse_hora.png")
    plt.close("all")

    with open(path+'informe_modelo.json', 'w') as archivo_json:
        informe = {"optimizacion": optimizacion,
                    "intervalo_min": INTERVALO_MIN,
                    "metricas": metricas,
                    "metricas_rad": rmse_rad,
                    "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia.values())}.items(), key=lambda item: item[1], reverse=True)),
                    "hiperparametros": {k:v for k,v in train_params.items() if v != None},
                    "training_input_description": train_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                    "training_target_description": train_df["potencia_act"].describe().to_dict(),
                    }
        json.dump(informe, archivo_json)