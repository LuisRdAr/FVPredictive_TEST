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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from functools import partial
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, \
                            r2_score as r2
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import time

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
    return {'loss': cv_results['test-rmse-mean'].min(), 'status': STATUS_OK}

def discriminador(x, norm):

    if norm:
        amp_dc_norm = x["amp_dc"]
    else:
        amp_dc_norm = x["amp_dc"] / x["num_strings"]

    amp_dc_std = 3 * amp_dc_norm.std()
    amp_dc_mean = amp_dc_norm.mean()

    outlier = (amp_dc_norm > amp_dc_mean + amp_dc_std) | (amp_dc_norm < amp_dc_mean - amp_dc_std)

    results = x[["id", "datetime_utc", "dispositivo_id", "entrada_id"]]
    results["outlier"] = outlier
    results = results.reset_index(drop=True)
    return results



cuda_flag = torch.cuda.is_available()
print("Configuración para el entrenamiento de modelos de predicción de potencia activa de salida en inversores fotovoltaicos:", end="\n\n")
model_name = "XGBRegressor"

valid_responses = ["y", "n"]
normalizacion = True
optimizacion = False

# Comprobación de la disponibilidad de GPU para el entrenamiento
valid_devices = ["cpu", "cuda"]
device = ""
if cuda_flag and model_name == "XGBRegressor":
    while device not in valid_devices:
        device_bool = input("¿Desea utilizar GPU para el entrenamiento? (Y/N): ").lower()
        if device_bool == "y":
            device = "cuda"
        else:
            device = "cpu"
else:
    device = 'cpu'

print(f"\nLas opciones seleccionadas son: \nModelo: {model_name} \nNormalización: {normalizacion} \nOptimización: {optimizacion} \nEntrenamiento: {device}", end="\n\n")

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

seeds = np.random.randint(0, 10000, size=1000)
num_mod_string = 30
sup_mod = 2
# Carga de los datos de entrenamiento
main_query = f"""
           WITH f AS (
            SELECT *
                FROM {schema_name}.inversores
                WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 15 = 0)
                    AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                    AND (alarma = 0)
                    AND (estado = 6)
                ORDER BY datetime_utc)
        SELECT f.id, f.dispositivo_id, det.entrada_id, f.datetime_utc, potencia_act, num_strings, 
                rad_poa, rad_hor, rad_celda1, rad_celda2, temp_amb, temp_panel1, temp_panel2, cloud_impact, 
                motivo, consigna_pot_act_planta, amp_dc
            FROM f
            JOIN {schema_name}.inversores_detalle AS det
                ON f.id = det.id
            JOIN {schema_name}.distrib_inversores dist
                ON  dist.dispositivo_id = f.dispositivo_id
                    AND dist.entrada_id = det.entrada_id
            JOIN {schema_name}.dispositivos AS disp
                ON disp.dispositivo_id = f.dispositivo_id
            JOIN {schema_name}.meteo AS met
                ON met.dispositivo_id = disp.meteo_cercana_id
                    AND met.datetime_utc = f.datetime_utc
            JOIN {schema_name}.ree AS ree
                ON ree.datetime_utc = f.datetime_utc
            WHERE daylight = true
            ORDER BY 4, 2, 3;"""

chunksize = 100000
chunks = pd.read_sql_query(main_query, engine, chunksize=chunksize)
main_df = pd.DataFrame()
for chunk in chunks:
    main_df = pd.concat([main_df, chunk], ignore_index = True)

# Normalización de la entrada de corriente continua, formateo de fechas y escalado de potencia
print(f"\tCarga inicial de {main_df.shape[0]} registros")
if normalizacion:
    print("\tEntrada de corriente continua normalizada según el número de strings")
    main_df["amp_dc"] = main_df["amp_dc"]/main_df["num_strings"]
else:
    print("\tEntrada de corriente continua sin normalizar")
main_df["datetime_utc"] = pd.to_datetime(main_df["datetime_utc"], utc = True)
main_df["potencia_act"] = main_df["potencia_act"] * 1000

# Búsqueda de outliers en la corriente continua basándose en la desviación dentro del comportamiento del inversor en un instante dado
outliers = main_df.groupby(['datetime_utc', 'dispositivo_id']).apply(lambda x: discriminador(x, normalizacion)).reset_index(drop=True)
print(f"\tRegistros descartados por outlier de corriente: {main_df[(main_df['id'].isin(outliers[outliers['outlier'] == True]['id'].unique()))].shape[0]}")
main_df = main_df[~(main_df["id"].isin(outliers[outliers["outlier"] == True]["id"].unique()))]

# Agrupación de las entradas de corriente continua por inversor y por hora en una tabla dinámica
pivot_table = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc').sum(axis = 1)
main_df = pd.merge(main_df, pivot_table.reset_index(), on=["dispositivo_id", "datetime_utc"]).drop(columns=["entrada_id", "num_strings", "amp_dc"]).drop_duplicates().rename(columns={0: "amp_dc"})

# Descarte de datos con potencias negativas o muy grandes
print(f"\tRegistros descartados por potencia negativa: {main_df[(main_df['potencia_act'] < 0)].shape[0]}")
main_df = main_df[(main_df['potencia_act'] > 0)]
potencia_act_mean = main_df["potencia_act"].mean() 
potencia_act_std = main_df["potencia_act"].std()
print(f"\tRegistros descartados por corrientes altas: {main_df[(main_df['potencia_act'] > potencia_act_mean + 3 * potencia_act_std)].shape[0]}")
main_df = main_df[(main_df['potencia_act'] < potencia_act_mean + 3 * potencia_act_std)]

# Descarte de datos con corrientes negativa o muy grandes
print(f"\tRegistros descartados por corriente negativa: {main_df[(main_df['amp_dc'] < 0)].shape[0]}")
main_df = main_df[(main_df['amp_dc'] > 0)]
amp_dc_mean = main_df["amp_dc"].mean() 
amp_dc_std = main_df["amp_dc"].std()
print(f"\tRegistros descartados por corrientes altas: {main_df[(main_df['amp_dc'] > amp_dc_mean + 3 * amp_dc_std)].shape[0]}")
main_df = main_df[(main_df['amp_dc'] < amp_dc_mean + 3 * amp_dc_std)]

# Búsqueda de outliers basándose en la potencia activa y la potencia solar
num_strings_inv = f"""SELECT dispositivo_id, SUM(num_strings) as num_strings 
                        FROM {schema_name}.distrib_inversores
                        GROUP BY dispositivo_id;"""
num_strings_inv = pd.read_sql_query(num_strings_inv, engine).sort_values(by="dispositivo_id")
potencia_df = pd.merge(main_df[["dispositivo_id", "datetime_utc", "potencia_act", "rad_poa"]], num_strings_inv, on="dispositivo_id")
potencia_df["potencia_solar"] = potencia_df["rad_poa"] * potencia_df["num_strings"] * num_mod_string * sup_mod
potencia_df["outlier_1"] = np.where(potencia_df["potencia_act"] > 0.20 * potencia_df["potencia_solar"], True, False)
main_df = main_df.merge(potencia_df[["dispositivo_id", "datetime_utc", "outlier_1"]], on = ["dispositivo_id", "datetime_utc"])
print(f"\tRegistros descartados por outlier de potencia: {main_df[main_df['outlier_1'] == True].shape[0]}")
main_df = main_df[main_df["outlier_1"] == False].drop(columns = ["outlier_1"])

# Manipulación de variables meteorológicas
main_df["rad_diff"] = (main_df["rad_celda1"] - main_df["rad_celda2"])
main_df["temp_panel"] = (main_df["temp_panel1"] + main_df["temp_panel2"]) / 2
main_df = main_df.drop(columns = ["rad_celda1", "rad_celda2", "temp_panel1", "temp_panel2"])

# Manipulación de variables de consigna
main_df["motivo"] = main_df["motivo"].apply(lambda x: 0 if x == 0 else (2 if x == 7 else 1))
main_query = f"""
    SELECT MAX(consigna_pot_act_planta)
        FROM {schema_name}.ree AS ree;"""
max_pot_act = pd.read_sql_query(main_query, engine).values[0][0]
main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_planta"] / max_pot_act   # Valor normalizado

# Asignación de variables temporales
main_df["dia_año"] = main_df["datetime_utc"].dt.dayofyear
main_df["dia_año_sen"] = np.sin(main_df["dia_año"] * (2*np.pi/365))
main_df["hora_seg"] = main_df["datetime_utc"].dt.hour * 3600 + \
                        main_df["datetime_utc"].dt.minute * 60 + \
                        main_df["datetime_utc"].dt.second
main_df["hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))

# Escalado de variables porcentuales
main_df[['cloud_impact']] = main_df[['cloud_impact']].apply(lambda x: x/100)

rmse_seeds = []
mae_seeds = []
for seed in tqdm(seeds, total = len(seeds)):
    rmse_inv = []
    mae_inv = []
    for inv_id in np.sort(main_df["dispositivo_id"].unique()):
        # Separación de los datos de entrenamiento y validación
        disp_df = main_df[main_df["dispositivo_id"] == inv_id].copy()
        disp_df.dropna()

        train_df, validation_df = train_test_split(disp_df, test_size = 0.2, random_state = seed)

        # Descarte de variables que no se usarán en el entrenamiento y separación de input y target
        processed_df = train_df.drop(columns = ["id",
                                        "dia_año",
                                        "hora_seg",
                                        "amp_dc"
                                        ]).set_index(["datetime_utc",
                                                        "dispositivo_id"])
        processed_val_df = validation_df.drop(columns = ["id",
                                        "dia_año",
                                        "hora_seg",
                                        "amp_dc"
                                        ]).set_index(["datetime_utc",
                                                        "dispositivo_id"])
        y = processed_df[["potencia_act"]]
        y_val = processed_val_df[["potencia_act"]]
        X = processed_df.drop(columns = ["potencia_act"])
        X_val = processed_val_df.drop(columns = ["potencia_act"])

        # Estandarización/normalización de variables numéricas y codificación de variables categóricas
        perc_attr = ['cloud_impact', 'consigna_pot_act_planta']
        std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'rad_diff', 'temp_panel']
        cat_attr = ['motivo']

        transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))]) # Introducir manualmente catergorías?
        transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 

        preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                    ('std', transformador_numerico_std, std_attr)],
                                    remainder='passthrough')
        X_prep = preprocessor.fit_transform(X)
        dtrain = xgb.DMatrix(X_prep, label=y)
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
        
        rmse_inv.append(rmse_score)
        mae_inv.append(mae_score)
    rmse_seeds.append(np.mean(rmse_inv))
    mae_seeds.append(np.mean(mae_inv))
print()
print(f"RMSE: {np.mean(rmse_seeds)}")
print(f"MAE: {np.mean(mae_seeds)}", end="\n\n")
