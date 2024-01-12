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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from functools import partial
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, \
                            r2_score as r2


class degree_scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_scale_90 = None
        self.columns_to_scale_180 = None
        self.columns_to_scale_360 = None
        
    def fit(self, X, y = None):
        self.columns_to_scale_90, self.columns_to_scale_180,  self.columns_to_scale_360 = self.determine_columns_to_scale(X)
        return self
    
    def transform(self, X, y = None):
        scaled_data = X.copy()
        if self.columns_to_scale_90 or self.columns_to_scale_180 or self.columns_to_scale_360:  
            for column in self.columns_to_scale_90:
                scaled_data[column] = X[column] / 90
            for column in self.columns_to_scale_180:
                min_val = X[column].min()
                if min_val < 0:
                    scaled_data[column] = (X[column] + 90) / 180
                else:
                    scaled_data[column] = X[column] / 180
            for column in self.columns_to_scale_360:
                min_val = X[column].min()
                if min_val < 0:
                    scaled_data[column] = (X[column] + 180) / 360
                else:
                    scaled_data[column] = X[column] / 360
        return scaled_data
    
    def determine_columns_to_scale(self, X):
        columns_90 = []
        columns_180 = []
        columns_360 = []
        for column in X.columns:
            min_val = X[column].min()
            max_val = X[column].max()
            if (min_val >= 0 and max_val <= 90):
                columns_90.append(column)
            elif (min_val >= -90 and max_val <= 90) or (min_val >= 0 and max_val <= 180):
                columns_180.append(column)
            elif (min_val >= -180 and max_val <= 180) or (min_val >= 0 and max_val <= 360):
                columns_360.append(column)
        return columns_90, columns_180, columns_360

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            output_features = []
            for feature in input_features:
                output_features.append("scaled__" + feature)
            return output_features
        else:
            return None

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

def objective(space, model_name, train_set, test_set, use_cv = False, device = "cpu"):
    if model_name == "XGBRegressor" and device == "cuda":
        model = MultiOutputRegressor(XGBRegressor(device = 'cuda:0',
                        objective='reg:squarederror', 
                        tree_method="hist_gpu", 
                        n_jobs=-1, 
                        n_estimators =int(space['n_estimators']), 
                        max_depth = int(space['max_depth']), 
                        learning_rate = space['learning_rate'],
                        gamma = space['gamma'],
                        min_child_weight = space['min_child_weight'],
                        subsample = space['subsample'],
                        reg_alpha = space['reg_alpha'],
                        reg_lambda = space['reg_lambda'],
                        eval_metric='rmse'
                        ))
        if use_cv:
            dtrain = xgb.DMatrix(train_set[0], label=train_set[1])
            cv_results = xgb.cv(model.get_params(), dtrain, nfold=5, metrics={"rmse"}, seed=42)
            loss = cv_results['test-rmse-mean'].iloc[-1]
        else:
            model.fit(train_set[0], train_set[1])
            y_pred = model.predict(test_set[0])
            loss = mse(test_set[1], y_pred)

    if model_name == "XGBRegressor" and device == "cpu":
        model = XGBRegressor(device = 'cpu',
                        objective='reg:squarederror', 
                        tree_method="hist", 
                        multi_strategy="multi_output_tree", 
                        n_jobs=-1, 
                        n_estimators =int(space['n_estimators']), 
                        max_depth = int(space['max_depth']), 
                        learning_rate = space['learning_rate'],
                        gamma = space['gamma'],
                        min_child_weight = space['min_child_weight'],
                        subsample = space['subsample'],
                        reg_alpha = space['reg_alpha'],
                        reg_lambda = space['reg_lambda'],
                        eval_metric='rmse'
                        )
        if use_cv:
            scores = cross_val_score(model, train_set[0], train_set[1], cv=5, scoring='neg_mean_squared_error')
            loss = -np.mean(scores)
        else:
            model.fit(train_set[0], train_set[1])
            y_pred = model.predict(test_set[0])
            loss = mse(test_set[1], y_pred)

    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(n_jobs=-1, 
                                    n_estimators =int(space['n_estimators']), 
                                    max_depth = int(space['max_depth']), 
                                    max_features = space['max_features'], 
                                    min_samples_leaf = int(space['min_samples_leaf']), 
                                    min_samples_split = int(space['min_samples_split']))
        if use_cv:
            scores = cross_val_score(model, train_set[0], train_set[1], cv=5, scoring='neg_mean_squared_error')
            loss = -np.mean(scores)
        else:
            model.fit(train_set[0], train_set[1])
            y_pred = model.predict(test_set[0])
            loss = mse(test_set[1], y_pred)

    else:
        print("Modelo no reconocido")
        sys.exit()
    
    return {'loss':loss, 'status': STATUS_OK}

valid_models = {"1": "XGBRegressor", "2": "RandomForestRegressor"}
model_name = ""
while model_name not in valid_models.keys():
    model_name = input("Ingrese el valor numérico para el tipo de modelo que desea utilizar (XGBRegressor[1], RandomForestRegressor[2]): ")
model_name = valid_models[model_name]

valid_responses = ["y", "n"]
normalizacion = ""
while normalizacion not in valid_responses:
    normalizacion = input("¿Desea normalizar el target? (Y/N): ").lower()
normalizacion = normalizacion == "y"

optimizacion = ""
while optimizacion not in valid_responses:
    optimizacion = input("¿Desea optimizar el modelo? (Y/N): ").lower()
optimizacion = optimizacion == "y"

use_cv = False
if optimizacion:
    while use_cv not in valid_responses:
        use_cv = input("\t¿Desea utilizar validación cruzada? (Y/N): ").lower()
    use_cv = use_cv == "y"

print(f"\nLas opciones seleccionadas son: \nModelo: {model_name} \nNormalización: {normalizacion} \nOptimización: {optimizacion} \nValidación cruzada: {use_cv}", end="\n\n")

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
# Carga de los datos de entrenamiento
if schema_name == "galisteo":
    main_query = f"""
        WITH f AS (
            SELECT *
                FROM {schema_name}.inversores
                WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 5 = 0)
                    AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                    AND (EXTRACT(MONTH FROM datetime_utc) != 10)
                    AND (alarma = 0)
                    AND (estado = 6)
                ORDER BY datetime_utc)
        SELECT f.id, f.dispositivo_id, det.entrada_id, f.datetime_utc, potencia_act, lim_act, num_strings, 
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
elif schema_name == "bonete":   
        main_query = f"""
            WITH inv AS (
                SELECT *
                    FROM {schema_name}.inversores
                    WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 5 = 0)
                        AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                        AND (alarma = 0)
                        AND (estado = 6)),
                met AS (
                    SELECT datetime_utc, AVG(rad_poa) AS rad_poa, AVG(rad_hor) AS rad_hor, AVG(rad_celda1) AS rad_celda1,
                            AVG(rad_celda2) AS rad_celda2, AVG(temp_amb) AS temp_amb, AVG(temp_panel1) AS temp_panel1,
                            AVG(temp_panel2) AS temp_panel2, AVG(cloud_impact) AS cloud_impact, BOOL_OR(daylight) AS daylight
                        FROM {schema_name}.meteo
                        GROUP BY datetime_utc)
            SELECT inv.id, inv.dispositivo_id, det.entrada_id, inv.datetime_utc, potencia_act, lim_act, num_strings, 
                    rad_poa, rad_hor, rad_celda1, rad_celda2, temp_amb, temp_panel1, temp_panel2, cloud_impact, 
                    motivo, consigna_pot_act_ree, amp_dc
                FROM inv
                JOIN {schema_name}.inversores_detalle AS det
                    ON inv.id = det.id
                JOIN {schema_name}.distrib_inversores dist
                    ON  dist.dispositivo_id = inv.dispositivo_id
                        AND dist.entrada_id = det.entrada_id
                JOIN met
                    ON met.datetime_utc = inv.datetime_utc
                JOIN {schema_name}.ree AS ree
                    ON ree.datetime_utc = inv.datetime_utc
                WHERE daylight = true
                ORDER BY 4, 2, 3;"""
else:
    print("Esquema no reconocido")
    sys.exit()
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
main_df["motivo"] = main_df["motivo"].apply(lambda x: 0 if x == 0 else (2 if x == 7 else 1))
main_query = f"""
    SELECT MAX(consigna_pot_act_ree)
        FROM {schema_name}.ree AS ree;"""
max_pot_act = pd.read_sql_query(main_query, engine).values[0][0]
if schema_name == "galisteo":
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_planta"] / max_pot_act 
elif schema_name == "bonete":
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_ree"] / max_pot_act 
    main_df = main_df.drop(columns=["consigna_pot_act_ree"])

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
main_df[['lim_act', 'cloud_impact']] = main_df[['lim_act', 'cloud_impact']].apply(lambda x: x/100)

for inv_id in np.sort(main_df["dispositivo_id"].unique()):
    # Separación de los datos de entrenamiento y validación
    print(f"Dispositivo {inv_id}")
    disp_df = main_df[main_df["dispositivo_id"] == inv_id].copy()
    disp_df.dropna()
    validation_df = disp_df[disp_df["datetime_utc"].dt.month == 9]
    train_df = disp_df[disp_df["datetime_utc"].dt.month != 9]
    print(f"\tRegistros de entrenamiento: {disp_df.shape[0]}")
    print(f"\tRegistros de validación: {validation_df.shape[0]}")

    # Agrupación de las entradas de corriente continua por inversor y por hora en una tabla dinámica
    pivot_table = train_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
    pivot_table.columns = ["amp_dc_" + str(col) for col in pivot_table.columns]
    y = pivot_table

    pivot_table_val = validation_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
    pivot_table_val.columns = ["amp_dc_" + str(col) for col in pivot_table_val.columns]
    y_val = pivot_table_val

    # Descarte de variables que no se usarán en el entrenamiento
    processed_df = train_df.drop(columns = ["id",
                                    "entrada_id",
                                    "dia_año",
                                    "hora_seg",
                                    "potencia_act",
                                    "outlier_1",
                                    "num_strings"
                                    ]).set_index(["datetime_utc",
                                                    "dispositivo_id"])
    processed_val_df = validation_df.drop(columns = ["id",
                                    "entrada_id",
                                    "dia_año",
                                    "hora_seg",
                                    "potencia_act",
                                    "outlier_1",
                                    "num_strings"
                                    ]).set_index(["datetime_utc",
                                                    "dispositivo_id"])
    
    # Separación del target y descarte de filas duplicadas (una única fila por inversor y hora)
    X = processed_df.drop(columns = ["amp_dc"]).drop_duplicates()
    X_val = processed_val_df.drop(columns = ["amp_dc"]).drop_duplicates()
    print(f"\tNúmero de registros para entrenamiento tras colapsar en dataframe multi-output: {X.shape[0]}")

    # Estandarización/normalización de variables numéricas y codificación de variables categóricas
    perc_attr = ['lim_act', 'cloud_impact']
    std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'rad_diff', 'temp_panel']
    cat_attr = ['motivo', 'consigna_pot_act_planta']

    transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])                         # Introducir manualmente catergorías?
    transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 

    preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                ('std', transformador_numerico_std, std_attr)],
                                remainder='passthrough')
    
    X_prep = preprocessor.fit_transform(X)

    if optimizacion:
        # Split de los datos de entrenamiento para optimización de hiperparámetros en caso de no usar validación cruzada
        if use_cv:
            print("\tOptimización de hiperparámetros con cross-validation")
            train_set = [X_prep, y]
            test_set = [None, None]
        else:
            print("\tOptimización de hiperparámetros con validacion sobre conjunto de testeo")
            X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size = 0.2)
            train_set = [X_train, y_train]
            test_set = [X_test, y_test]

        # Comprueba si CUDA está disponible
        flag = False
        if torch.cuda.is_available() and model_name == "XGBRegressor" and flag:
            import cupy as cp
            device = 'cuda'
            train_set = [cp.asarray(train_set[0]), cp.asarray(train_set[1].values)]
            test_set = [cp.asarray(test_set[0]) if test_set[0] is not None else None,
                        cp.asarray(test_set[1].values) if test_set[1] is not None else None]
        else:
            device = 'cpu'
        # Inicialización  y primera fase de la optimización de hiperparámetros con gamma = 1 y espacio de búsqueda general
        trials = Trials()
        best_loss = np.inf
        gamma = 1
        stall_counter = 0
        STALL_LIMIT = 3
        MAX_EVALS_PER_RUN = 250
        total_evals = 0
        run_counter = 0
        if model_name == "XGBRegressor":
            space ={
                'max_depth': hp.quniform("x_max_depth", 3, 15, 3),
                'n_estimators': hp.quniform('x_n_estimators', 100, 1500, 200),
                'learning_rate': hp.quniform('x_learning_rate', 0.01, 0.2, 0.01),
                'gamma': hp.quniform ('x_gamma', 0, 1, 0.2),
                'min_child_weight' : hp.quniform('x_min_child', 1, 19, 3),
                'subsample' : hp.quniform('x_subsample', 0.5, 1, 0.1),
                'reg_alpha' : hp.quniform('x_reg_alpha', 0, 20, 0.5),
                'reg_lambda' : hp.quniform('x_reg_lambda', 0, 20, 0.5)
            }
        elif model_name == "RandomForestRegressor":
            space = {
                'max_depth': hp.quniform("x_max_depth", 3, 15, 2),
                'n_estimators': hp.quniform("x_n_estimators", 100, 1500, 100),
                'min_samples_split': hp.quniform ('x_min_samples_split', 2, 20, 2),
                'min_samples_leaf': hp.quniform ('x_min_samples_leaf', 1, 19, 2),
                'max_features': hp.choice('x_max_features', [None, 'sqrt', 'log2']),     
            }
        upper_limit = (MAX_EVALS_PER_RUN/(STALL_LIMIT - 1)) * 10
        while stall_counter < STALL_LIMIT and total_evals < upper_limit:
            best = fmin(fn=lambda space: objective(space, model_name = model_name, train_set = train_set, test_set = test_set, use_cv=use_cv), 
                        space=space, 
                        algo=partial(tpe.suggest, gamma = gamma),
                        max_evals=total_evals + MAX_EVALS_PER_RUN, 
                        trials=trials)
            
            new_loss = trials.best_trial['result']['loss']
            if new_loss < best_loss:
                threshold = 0.001
                if abs(new_loss - best_loss) <= threshold:
                    stall_counter += 1
                else:
                    stall_counter = 0
                best_loss = new_loss
            else:
                stall_counter += 1

            total_evals += MAX_EVALS_PER_RUN
            run_counter += 1

            if total_evals % (MAX_EVALS_PER_RUN/(STALL_LIMIT - 1)) == 0:
                gamma = max(0, gamma - 0.05)
            print(f"Run {run_counter}: Best loss so far: {best_loss}")

        print(f"\tMejor combinación de hiperparámetros en la primera búsqueda:\n{best}")
        # Segunda fase de la optimización de hiperparámetros con gamma = 0.33 y espacio de búsqueda fino
        stall_counter = 0
        STALL_LIMIT = 5
        MAX_EVALS_PER_RUN = 250
        initial_total_evals = total_evals
        run_counter = 0
        if model_name == "XGBRegressor":
            space ={
                'max_depth': hp.quniform("x_max_depth", 3, 15, 1),
                'n_estimators': hp.quniform('x_n_estimators', 100, 1500, 1),
                'learning_rate': hp.quniform('x_learning_rate', 0.01, 0.2, 0.001),
                'gamma': hp.quniform ('x_gamma', 0, 1, 0.01),
                'min_child_weight' : hp.quniform('x_min_child', 1, 20, 1),
                'subsample' : hp.quniform('x_subsample', 0.5, 1, 0.05),
                'reg_alpha' : hp.quniform('x_reg_alpha', 0, 10, 0.01),
                'reg_lambda' : hp.quniform('x_reg_lambda', 0, 10, 0.01)
            }
        elif model_name == "RandomForestRegressor":
            space = {
                'max_depth': hp.quniform("x_max_depth", 3, 15, 1),
                'n_estimators': hp.quniform("x_n_estimators", 100, 1500, 10),
                'min_samples_split': hp.quniform ('x_min_samples_split', 2, 21, 1),
                'min_samples_leaf': hp.quniform ('x_min_samples_leaf', 1, 20, 1),
                'max_features': hp.choice('x_max_features', ['auto', 'sqrt', 'log2']),     
            }
        while stall_counter < STALL_LIMIT and total_evals < initial_total_evals + 10000:
            best = fmin(fn=lambda space: objective(space, model_name = model_name, train_set = train_set, test_set = test_set, use_cv=use_cv), 
                        space=space, 
                        algo=partial(tpe.suggest, gamma=0.33), 
                        max_evals=total_evals + MAX_EVALS_PER_RUN, 
                        trials=trials)
            
            new_loss = trials.best_trial['result']['loss']
            if new_loss < best_loss:
                best_loss = new_loss
                stall_counter = 0
            else:
                stall_counter += 1

            total_evals += MAX_EVALS_PER_RUN
            run_counter += 1
            print(f"Run {run_counter}: Best loss so far: {best_loss}")

        print(f"\tMejor combinación de hiperparámetros:\n{best}")
    
        # Entrenamiento del modelo con los mejores hiperparámetros
        if model_name == "XGBRegressor" and device == "cuda":
            model = MultiOutputRegressor(XGBRegressor(device = 'cuda:0',
                        objective='reg:squarederror', 
                        tree_method="hist_gpu", 
                        n_jobs=-1, 
                        gamma = best["x_gamma"],
                        learning_rate = best["x_learning_rate"],
                        max_depth = int(best["x_max_depth"]),
                        min_child_weight = int(best["x_min_child"]),
                        n_estimators = int(best["x_n_estimators"]),
                        reg_alpha = best["x_reg_alpha"],
                        reg_lambda = best["x_reg_lambda"],
                        subsample = best["x_subsample"]))
        elif model_name == "XGBRegressor" and device == "cpu":
            model = XGBRegressor(device = 'cpu',
                        objective='reg:squarederror', 
                        tree_method="hist", 
                        multi_strategy="multi_output_tree", 
                        n_jobs=-1, 
                        gamma = best["x_gamma"],
                        learning_rate = best["x_learning_rate"],
                        max_depth = int(best["x_max_depth"]),
                        min_child_weight = int(best["x_min_child"]),
                        n_estimators = int(best["x_n_estimators"]),
                        reg_alpha = best["x_reg_alpha"],
                        reg_lambda = best["x_reg_lambda"],
                        subsample = best["x_subsample"])
            
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(n_jobs=-1, 
                                            n_estimators = int(best["x_n_estimators"]), 
                                            max_depth = int(best["x_max_depth"]), 
                                            max_features = best["x_max_features"], 
                                            min_samples_leaf = int(best["x_min_samples_leaf"]), 
                                            min_samples_split = int(best["x_min_samples_split"]))
        model.fit(X_prep, y)
    else:
        if model_name == "XGBRegressor":
            model = XGBRegressor(objective='reg:squarederror', 
                                    tree_method="hist", 
                                    multi_strategy="multi_output_tree", 
                                    n_jobs=-1)
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(n_jobs=-1)
        model.fit(X_prep, y)

    pipeline_model = Pipeline([('preprocessor', preprocessor),
                            ('regressor', model)])
    
    # Evaluación del modelo sobre conjunto de validación
    consulta_sql = f"""SELECT num_strings
                    FROM {schema_name}.distrib_inversores
                    WHERE dispositivo_id = {inv_id};"""
    num_strings = pd.read_sql_query(consulta_sql, engine).values.reshape(1, -1)

    y_pred_val = pd.DataFrame(pipeline_model.predict(X_val)).rename(columns={i: "y_pred_"+str(i+1) for i in pd.DataFrame(pipeline_model.predict(X_val)).columns})
    if normalizacion:
        y_pred_val = y_pred_val * num_strings
        y_val_reesc = y_val * num_strings
    else:
        y_pred_val = y_pred_val
        y_val_reesc = y_val
    
    target_pred_df = pd.concat([X_val.reset_index(), y_pred_val], axis=1)[["datetime_utc", "dispositivo_id"]+y_pred_val.columns.to_list()] \
                            .melt(id_vars=["datetime_utc", "dispositivo_id"], var_name="entrada_id", value_name="y_pred")
    target_pred_df["entrada_id"] = target_pred_df["entrada_id"].str.split("_").str[2].astype(int)

    y_val_reesc = y_val_reesc.reset_index().melt(id_vars=["datetime_utc", "dispositivo_id"], var_name="entrada_id", value_name="amp_dc")
    y_val_reesc["entrada_id"] = y_val_reesc["entrada_id"].str.split("_").str[2].astype(int)

    prediction_df = target_pred_df.merge(y_val_reesc, on=["datetime_utc", "dispositivo_id", "entrada_id"])

    # Cálculo de las métricas de error
    rmse_score = round(mse(prediction_df["amp_dc"], prediction_df["y_pred"], squared=False), 2)
    mae_score = round(mae(prediction_df["amp_dc"], prediction_df["y_pred"]), 2)
    r2_score = round(r2(prediction_df["amp_dc"], prediction_df["y_pred"]), 3)
    metricas = {"RMSE": rmse_score, "MAE": mae_score, "R2": r2_score}
    print(f"\tMétricas de error:\n\t\tRMSE: {rmse_score}\n\t\tMAE: {mae_score}\n\t\tR2: {r2_score}")

    # Guardado del modelo y de las métricas
    algoritmo = pipeline_model.named_steps["regressor"].__class__.__name__
    path = os.path.join(root_path, f"Modelos/entrada_amperaje_multi/Inversor_{inv_id - 20}/Repositorio/{algoritmo}-{pd.Timestamp.now()}/")
    os.makedirs(path)
    with open(path+'model.model', "wb") as archivo_salida:
        pickle.dump(pipeline_model, archivo_salida)
    with open(path+'informe_modelo.json', 'w') as archivo_json:
        informe = {"normalizacion": normalizacion,
                   "optimizacion": optimizacion,
                   "cross_validation": use_cv,
                    "metricas": metricas,
                    "hiperparametros": {k:v for k,v in pipeline_model.named_steps["regressor"].get_params().items() if v != None},
                    "input_description": processed_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                    "target": (y * num_strings).reset_index().melt(id_vars=["datetime_utc", "dispositivo_id"], var_name="entrada_id", value_name="amp_dc")[["amp_dc"]].describe().to_dict()}
        json.dump(informe, archivo_json)

    # Generación de gráficos: comparativa de valores reales y predichos, histograma de diferencias y matriz de correlación
    y_test_sampled, _,y_pred_sampled, _ = train_test_split(prediction_df["amp_dc"], prediction_df["y_pred"], train_size = 0.25)
    plt.figure()
    plt.tight_layout()
    plt.scatter(y_test_sampled, y_pred_sampled, marker = ".")
    plt.plot([min(y_test_sampled), max(y_test_sampled)], [min(y_test_sampled), max(y_test_sampled)], color='black', linestyle='-', linewidth=1)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("Comparación de valores reales y predichos")
    plt.savefig(path + "scatter_validacion.png")

    plt.figure()
    plt.tight_layout()
    ax = sns.histplot(prediction_df["amp_dc"] - prediction_df["y_pred"], kde=True, stat='percent')
    ax.axvline(x=0, color='black', linestyle='--', linewidth = 0.35, label='x=0')
    plt.title('Histograma de las diferencias entre valores reales y predichos')
    plt.xlabel('Diferencia')
    plt.ylabel('Porcentaje')
    plt.savefig(path + "histogram_validacion.png")

    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    sns.heatmap(processed_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.savefig(path + "correlacion.png")

    # Generación de gráficos: comparativa de RMSE y RMSE relativo por entrada
    rmse_list = []
    rmse_r_list = []
    entrada_list = []
    for group in prediction_df.groupby(["dispositivo_id", "entrada_id"]):
        entrada_list.append(group[0][1])
        rmse_score = round(mse(group[1]["amp_dc"], group[1]["y_pred"], squared=False), 2)
        rmse_r_score = round((mse(group[1]["amp_dc"], group[1]["y_pred"], squared=False)*100/group[1]['amp_dc'].mean()), 2)
        mae_score = round(mae(group[1]["amp_dc"], group[1]["y_pred"]), 2)
        r2_score = round(r2(group[1]["amp_dc"], group[1]["y_pred"]), 3)
        rmse_list.append(rmse_score)
        rmse_r_list.append(rmse_r_score)

        metricas_entrada = {"RMSE": rmse_score, "RMSE %": rmse_r_score, "MAE": mae_score, "R2": r2_score}
        
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
    if rmse_r_score.max() - rmse_r_score.min() > 0.25:
        ax2.set_yscale('log')
    ax2.plot(entrada_list, rmse_r_list, color=color, linewidth=1)
    sns.scatterplot(x=entrada_list, y=rmse_r_list, color=color, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Comparativa de RMSE y RMSE Relativo por entrada para el inversor {inv_id - 20}')
    plt.xticks(entrada_list)
    plt.tight_layout()
    ax1.grid(True, which='major', color='gray', linewidth=0.5)
    ax2.grid(True, which='minor', color='gray', linewidth=0.5)
    plt.savefig(path + "rmse_entrada.png")

    # Generación de gráficos: comparativa de RMSE y RMSE relativo por hora
    rmse_list = []
    rmse_r_list = []
    hora_list = []
    for group in prediction_df.groupby(["dispositivo_id", prediction_df["datetime_utc"].dt.hour]):
        hora_list.append(group[0][1])
        rmse_score = round(mse(group[1]["amp_dc"], group[1]["y_pred"], squared=False), 2)
        rmse_r_score = round((mse(group[1]["amp_dc"], group[1]["y_pred"], squared=False)*100/group[1]['amp_dc'].mean()), 2)
        mae_score = round(mae(group[1]["amp_dc"], group[1]["y_pred"]), 2)
        r2_score = round(r2(group[1]["amp_dc"], group[1]["y_pred"]), 3)
        rmse_list.append(rmse_score)
        rmse_r_list.append(rmse_r_score)

        metricas_entrada = {"RMSE": rmse_score, "RMSE %": rmse_r_score, "MAE": mae_score, "R2": r2_score}
        
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
    if rmse_r_score.max() - rmse_r_score.min() > 0.25:
        ax2.set_yscale('log')
    ax2.plot(hora_list, rmse_r_list, color=color, linewidth=1)
    sns.scatterplot(x=hora_list, y=rmse_r_list, color=color, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Comparativa de RMSE y RMSE Relativo por hora para el inversor {inv_id - 20}')
    plt.xticks(hora_list)
    plt.tight_layout()
    ax1.grid(True, which='major', color='gray', linewidth=0.5)
    ax2.grid(True, which='minor', color='gray', linewidth=0.5)
    plt.savefig(path + "rmse_hora.png")
    plt.close("all")