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

def discriminador(x, norm):

    if norm:
        amp_dc_norm = x["amp_dc"]
    else:
        amp_dc_norm = x["amp_dc"] / x["num_strings"]

    amp_dc_std = 3 * amp_dc_norm.std()
    amp_dc_mean = amp_dc_norm.mean()

    outlier = (amp_dc_norm > amp_dc_mean + amp_dc_std) | (amp_dc_norm < amp_dc_mean - amp_dc_std)

    results = x[["id", "datetime_utc", "dispositivo_id", "entrada_id"]]
    results["outlier_2"] = outlier
    results = results.reset_index(drop=True)
    return results


cuda_flag = torch.cuda.is_available()
print("Configuración para el entrenamiento de modelos de predicción de entrada de corriente continua en inversores fotovoltaicos:", end="\n\n")
# valid_models = {"1": "XGBRegressor", "2": "RandomForestRegressor"}
# model_name = ""
# while model_name not in valid_models.keys():
#     model_name = input("Ingrese el valor numérico para el tipo de modelo que desea utilizar (XGBRegressor[1], RandomForestRegressor[2]): ")
# model_name = valid_models[model_name]
model_name = "XGBRegressor"

valid_responses = ["y", "n"]
normalizacion = ""
while normalizacion not in valid_responses:
    normalizacion = input("¿Desea normalizar el target? (Y/N): ").lower()
normalizacion = normalizacion == "y"

optimizacion = ""
while optimizacion not in valid_responses:
    optimizacion = input("¿Desea optimizar el modelo? (Y/N): ").lower()
optimizacion = optimizacion == "y"

use_cv = True
# if optimizacion:
#     while use_cv not in valid_responses:
#         use_cv = input("\t¿Desea utilizar validación cruzada? (Y/N): ").lower()
#     use_cv = use_cv == "y"

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

num_mod_string = 30
sup_mod = 2
ids_parques = [1,2,3]
for id_parque in ids_parques:
    print(f"Entrenamiento para el parque fotovoltaico {id_parque}")
    # Carga de los datos de entrenamiento
    main_query = f"""
                WITH inv AS (
                    SELECT 
                        date_trunc('hour', datetime_utc) + 
                        INTERVAL '15 min' * floor(date_part('minute', datetime_utc) /15) as datetime_utc_rounded, 
                        parque_id, 
                        dispositivo_id,
                        MIN(inv.id) AS id, 
                        AVG(potencia_act) as potencia_act, 
                        AVG(amp_dc) as amp_dc,
                        det.entrada_id
                    FROM alcazar.inversores AS inv
                    JOIN alcazar.inversores_detalle AS det
                        ON inv.id = det.id
                    WHERE parque_id = 1
                    GROUP BY parque_id, datetime_utc_rounded, dispositivo_id, det.entrada_id 
                ),
                met AS (
                    SELECT 
                        parque_id,
                        date_trunc('hour', datetime_utc) + 
                        INTERVAL '15 min' * floor(date_part('minute', datetime_utc) / 15) as datetime_utc_rounded, 
                        AVG(rad_poa) AS rad_poa, 
                        AVG(rad_hor) AS rad_hor, 
                        AVG(rad_celda1) AS rad_celda1,
                        AVG(rad_celda2) AS rad_celda2, 
                        AVG(temp_amb) AS temp_amb, 
                        AVG(temp_panel1) AS temp_panel1,
                        AVG(temp_panel2) AS temp_panel2, 
                        AVG(cloud_impact) AS cloud_impact, 
                        AVG(presion_atm) AS presion_atm,
                        BOOL_OR(daylight) AS daylight
                    FROM alcazar.meteo
                    WHERE parque_id = 1
                        AND daylight = true
                    GROUP BY parque_id, datetime_utc_rounded
                )
                SELECT 
                    inv.id,
                    inv.parque_id,
                    inv.dispositivo_id,
                    inv.entrada_id,
                    inv.datetime_utc_rounded as datetime_utc, 
                    potencia_act,  
                    num_strings, 
                    rad_poa,
                    rad_hor, 
                    rad_celda1, 
                    rad_celda2, 
                    temp_amb, 
                    temp_panel1, 
                    temp_panel2, 
                    presion_atm,
                    cloud_impact,
                    motivo,
                    consigna_pot_act_planta,
                    amp_dc
                FROM inv
                JOIN alcazar.distrib_inversores dist
                    ON dist.parque_id = inv.parque_id
                        AND dist.dispositivo_id = inv.dispositivo_id
                        AND dist.entrada_id = inv.entrada_id
                JOIN met
                    ON met.parque_id = inv.parque_id
                        AND met.datetime_utc_rounded = inv.datetime_utc_rounded
                JOIN alcazar.ree AS ree
                    ON ree.datetime_utc = inv.datetime_utc_rounded
                        AND ree.parque_id = inv.parque_id
                ORDER BY 5, 2, 3, 4;"""
    
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

    # Manipulación de variables meteorológicas
    main_df["rad_diff"] = (main_df["rad_celda1"] - main_df["rad_celda2"])
    main_df["temp_panel"] = (main_df["temp_panel1"] + main_df["temp_panel2"]) / 2
    main_df = main_df.drop(columns = ["rad_celda1", "rad_celda2", "temp_panel1", "temp_panel2"])

    # Búsqueda de outliers basándose en la potencia activa y la potencia solar
    potencia_df = main_df.groupby(["dispositivo_id", "datetime_utc"]).agg({"num_strings": "sum",
                                                                "rad_poa": "mean",
                                                                "potencia_act": "mean"
                                                                }).reset_index()
    potencia_df["potencia_solar"] = potencia_df["rad_poa"] * potencia_df["num_strings"] * num_mod_string * sup_mod 
    potencia_df["outlier_1"] = np.where(potencia_df["potencia_act"] > 0.20 * potencia_df["potencia_solar"], True, False)
    main_df = main_df.merge(potencia_df[["dispositivo_id", "datetime_utc", "outlier_1"]], on = ["dispositivo_id", "datetime_utc"])
    print(f"\tRegistros descartados por outlier de potencia: {main_df[main_df['outlier_1'] == True].shape[0]}")
    main_df = main_df[main_df["outlier_1"] == False]

    # Descarte de datos con corrientes negativa o muy grandes
    print(f"\tRegistros descartados por corriente negativa: {main_df[(main_df['id'].isin(main_df[(main_df['amp_dc'] < -0.1)]['id'].unique()))].shape[0]}")
    if normalizacion:
        main_df = main_df[~(main_df["id"].isin(main_df[(main_df["amp_dc"] < -0.1)]["id"].unique()))]
    else:
        main_df = main_df[~(main_df["id"].isin(main_df[(main_df["amp_dc"] < -1)]["id"].unique()))]
    amp_dc_mean = main_df["amp_dc"].mean() 
    amp_dc_std = main_df["amp_dc"].std()
    print(f"\tRegistros descartados por corrientes altas: {main_df[(main_df['id'].isin(main_df[(main_df['amp_dc'] > amp_dc_mean + 3 * amp_dc_std)]['id'].unique()))].shape[0]}")
    main_df = main_df[~(main_df["id"].isin(main_df[(main_df["amp_dc"] > amp_dc_mean + 3 * amp_dc_std)]["id"].unique()))]

    # Manipulación de variables de consigna
    main_query = f"""
        SELECT MAX(consigna_pot_act_planta)
            FROM {schema_name}.ree AS ree;"""
    max_pot_act = pd.read_sql_query(main_query, engine).values[0][0]
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_planta"] / max_pot_act 

    # Asignación de variables temporales
    main_df["dia_año"] = main_df["datetime_utc"].dt.dayofyear
    main_df["dia_año_sen"] = np.sin(main_df["dia_año"] * (2*np.pi/365))
    main_df["hora_seg"] = main_df["datetime_utc"].dt.hour * 3600 + \
                            main_df["datetime_utc"].dt.minute * 60 + \
                            main_df["datetime_utc"].dt.second
    main_df["hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))

    # Búsqueda de outliers en la corriente continua basándose en la desviación dentro del comportamiento del inversor en un instante dado
    outliers = main_df.groupby(['datetime_utc', 'dispositivo_id']).apply(lambda x: discriminador(x, normalizacion)).reset_index(drop=True)
    print(f"\tRegistros descartados por outlier de corriente: {main_df[(main_df['id'].isin(outliers[outliers['outlier_2'] == True]['id'].unique()))].shape[0]}")
    main_df = main_df[~(main_df["id"].isin(outliers[outliers["outlier_2"] == True]["id"].unique()))]

    # Escalado de variables porcentuales
    main_df[['cloud_impact', 'consigna_pot_act_planta']] = main_df[['cloud_impact', 'consigna_pot_act_planta']].apply(lambda x: x/100)

    # Agrupación de las entradas de corriente continua por inversor y por hora en una tabla dinámica
    pivot_table = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc').sum(axis = 1)
    complete_df = pd.merge(main_df, pivot_table.reset_index(), on=["dispositivo_id", "datetime_utc"]).drop(columns=["entrada_id", "num_strings", "amp_dc"]).drop_duplicates().rename(columns={0: "amp_dc"})

    for inv_id in np.sort(complete_df["dispositivo_id"].unique()):
        # Separación de los datos de entrenamiento y validación
        print(f"Dispositivo {inv_id}")
        disp_df = complete_df[complete_df["dispositivo_id"] == inv_id].copy()
        disp_df.dropna()
        validation_df = disp_df[(disp_df["datetime_utc"].dt.year == 2023) &
                                    ((disp_df["datetime_utc"].dt.month == 9) |
                                    (disp_df["datetime_utc"].dt.month == 10) |
                                    (disp_df["datetime_utc"].dt.month == 11) |
                                    (disp_df["datetime_utc"].dt.month == 12))]
        train_df = disp_df[(disp_df["datetime_utc"].dt.year != 2023) &
                                    ((disp_df["datetime_utc"].dt.month != 9) |
                                    (disp_df["datetime_utc"].dt.month != 10) |
                                    (disp_df["datetime_utc"].dt.month != 11) |
                                    (disp_df["datetime_utc"].dt.month != 12))]
        print(f"\tRegistros de entrenamiento: {disp_df.shape[0]}")
        print(f"\tRegistros de validación: {validation_df.shape[0]}")


        # Descarte de variables que no se usarán en el entrenamiento y separación de input y target
        processed_df = train_df.drop(columns = ["id",
                                        "dia_año",
                                        "hora_seg",
                                        "outlier_1"
                                        ]).set_index(["datetime_utc",
                                                      "parque_id",
                                                      "dispositivo_id"])
        processed_val_df = validation_df.drop(columns = ["id",
                                        "dia_año",
                                        "hora_seg",
                                        "outlier_1"
                                        ]).set_index(["datetime_utc",
                                                        "parque_id",
                                                        "dispositivo_id"])
        y = processed_df[["potencia_act"]]
        y_val = processed_val_df[["potencia_act"]]
        X = processed_df.drop(columns = ["potencia_act"])
        X_val = processed_val_df.drop(columns = ["potencia_act"])

        # Estandarización/normalización de variables numéricas y codificación de variables categóricas
        perc_attr = ['cloud_impact', 'consigna_pot_act_planta']
        std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'rad_diff', 'temp_panel', 'presion_atm', 'amp_dc']
        cat_attr = ['motivo']

        transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])                         # Introducir manualmente catergorías?
        transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 

        preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                    ('std', transformador_numerico_std, std_attr)],
                                    remainder='passthrough')
        
        X_prep = preprocessor.fit_transform(X)
        dtrain = xgb.DMatrix(X_prep, label=y)
        params = {
                    'objective': 'reg:squarederror',
                    'tree_method': "hist",
                    'random_state': 42,
                    'n_jobs': -1
                }
        model = xgb.train(params, dtrain) 
        
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
    

