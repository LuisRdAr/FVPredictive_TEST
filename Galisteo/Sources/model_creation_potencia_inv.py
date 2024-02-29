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
import gc

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

def discriminador(row):
    mean = np.mean(row)
    std = np.std(row)
    threshold = 3 * std
    outlier = np.abs(row - mean) > threshold
    return outlier

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

    print(f"\tNúmero de evaluaciones realizadas: {num_evals}")
    print(f"\tBest params: {best_params}")
    print(f"\tBest loss: {best_loss}", end="\n\n")

    return best_params

print("Configuración para el entrenamiento de modelos de predicción de potencia activa de salida en inversores fotovoltaicos:", end="\n\n")
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

stage = True
valid_devices = ["cpu", "cuda"]
device = ""
if optimizacion:
    while stage not in valid_responses:
        stage = input("\t¿Desea optimizar por fases? (Y/N): ").lower()
    stage = stage == "y"
    # Comprobación de la disponibilidad de GPU para el entrenamiento
    if torch.cuda.is_available():
        while device not in valid_devices:
            device_bool = input("¿Desea utilizar GPU para el entrenamiento? (Y/N): ").lower()
            if device_bool == "y":
                device = "cuda"
            else:
                device = "cpu"
else:
    stage = False
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

intervalo_min = 30
num_mod_string = 30
sup_mod = 2
# Carga de los datos de entrenamiento
main_query = f"""
            WITH inv AS (
                SELECT 
                    date_trunc('hour', datetime_utc) + 
                    INTERVAL '{intervalo_min} min' * floor(date_part('minute', datetime_utc) / {intervalo_min}) as datetime_utc_rounded,
                    dispositivo_id,
                    MIN(inv.id) AS id, 
                    AVG(potencia_act) as potencia_act, 
                    AVG(amp_dc) as amp_dc,
                    det.entrada_id
                FROM {schema_name}.inversores AS inv
                JOIN {schema_name}.inversores_detalle AS det
                    ON inv.id = det.id
                WHERE (dispositivo_id != 26)
                    AND (alarma = 0)
                    AND (estado = 6)
                GROUP BY datetime_utc_rounded, dispositivo_id, det.entrada_id 
            ),
            met AS (
                SELECT 
                    date_trunc('hour', datetime_utc) + 
                    INTERVAL '{intervalo_min} min' * floor(date_part('minute', datetime_utc) / {intervalo_min}) as datetime_utc_rounded, 
                    dispositivo_id,
                    AVG(rad_poa) AS rad_poa, 
                    AVG(rad_hor) AS rad_hor, 
                    AVG(rad_celda1) AS rad_celda1,
                    AVG(rad_celda2) AS rad_celda2, 
                    AVG(temp_amb) AS temp_amb, 
                    AVG(temp_panel1) AS temp_panel1,
                    AVG(temp_panel2) AS temp_panel2, 
                    AVG(cloud_impact) AS cloud_impact,
                    BOOL_OR(daylight) AS daylight
                FROM {schema_name}.meteo
                    WHERE daylight = true
                GROUP BY dispositivo_id, datetime_utc_rounded
            )
            SELECT 
                inv.id,
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
                cloud_impact,
                motivo,
                consigna_pot_act_planta,
                amp_dc
            FROM inv
            JOIN {schema_name}.distrib_inversores dist
                ON dist.dispositivo_id = inv.dispositivo_id
                    AND dist.entrada_id = inv.entrada_id
            JOIN {schema_name}.dispositivos AS disp
                ON disp.dispositivo_id = inv.dispositivo_id
            JOIN met
                ON met.datetime_utc_rounded = inv.datetime_utc_rounded
                    AND met.dispositivo_id = disp.meteo_cercana_id
            JOIN {schema_name}.ree AS ree
                ON ree.datetime_utc = inv.datetime_utc_rounded
            ORDER BY 5, 2, 3, 4;"""

chunksize = 100000
chunks = pd.read_sql_query(main_query, engine, chunksize=chunksize)
main_df = pd.DataFrame()
for chunk in chunks:
    main_df = pd.concat([main_df, chunk], ignore_index = True)

# Normalización de la entrada de corriente continua, formateo de fechas y escalado de potencia
print(f"Carga inicial de {main_df.shape[0]} registros")
if normalizacion:
    print("Entrada de corriente continua normalizada según el número de strings")
    main_df["amp_dc"] = main_df["amp_dc"]/main_df["num_strings"]
else:
    print("Entrada de corriente continua sin normalizar")
main_df["datetime_utc"] = pd.to_datetime(main_df["datetime_utc"], utc = True)
main_df["potencia_act"] = main_df["potencia_act"] * 1000

# Pivotado de las entradas de corriente continua
target_df = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
target_df.columns = ["amp_dc_" + str(col) for col in target_df.columns]
print(f"Número de registros del dataframe tras pivotar: {target_df.shape[0]}")

# Descarte de registros con corriente anómala
target_df["outlier"] = target_df.apply(discriminador, axis=1).any(axis=1)
n_corriente_outlier = target_df[target_df["outlier"]].shape[0]
target_df = target_df[~target_df["outlier"]].drop(columns="outlier")
print(f"Registros descartados por corrientes anómalas ingresando en el inversor: {n_corriente_outlier}")

# Rellenado de valores faltantes por desconexión de entradas
consulta_sql = f"""SELECT MAX(entrada_id)
            FROM {schema_name}.distrib_inversores;"""
max_entradas = pd.read_sql_query(consulta_sql, engine).values[0][0]
entradas = [i for i in range(1, max_entradas + 1)]
for dispositivo in target_df.index.get_level_values(0).unique():
    consulta_sql = f"""SELECT entrada_id
                FROM {schema_name}.distrib_inversores
                WHERE dispositivo_id = {dispositivo};"""
    entradas_inv = pd.read_sql_query(consulta_sql, engine).values.reshape(1,-1)[0]
    entradas_off = list(set(entradas) - set(entradas_inv))
    for entrada in entradas_off:
        target_df.loc[dispositivo, "amp_dc_" + str(entrada)] = 0

# Descarte de registros con corriente desconocida
n_nan_values = target_df[target_df.isna().any(axis=1)].shape[0]
target_df = target_df.dropna()
print(f"Registros descartados por corriente desconocida: {n_nan_values}")

# Descarte de registros con corriente negativa
q1 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.25)
q3 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
n_corriente_neg = target_df[target_df < lower_bound].dropna(how='all').shape[0]
target_df = target_df[target_df >= lower_bound].dropna(how='any')
target_df[(target_df >= lower_bound) & (target_df <= 0)] = 0
main_df = main_df.drop(columns=["entrada_id", "amp_dc"]).drop_duplicates(subset=["id", "datetime_utc"]).set_index(["dispositivo_id", "datetime_utc"])
main_df = main_df.merge(target_df, left_index=True, right_index=True, how="inner")
del target_df
gc.collect()
print(f"Registros descartados por corriente negativa: {n_corriente_neg}")

# Búsqueda de outliers basándose en la potencia activa y la potencia solar
num_strings_inv = f"""SELECT dispositivo_id, SUM(num_strings) as num_strings 
                        FROM {schema_name}.distrib_inversores
                        GROUP BY dispositivo_id;"""
num_strings_inv = pd.read_sql_query(num_strings_inv, engine).sort_values(by="dispositivo_id")
potencia_df = pd.merge(main_df.reset_index()[["dispositivo_id", "datetime_utc", "potencia_act", "rad_poa"]], num_strings_inv, on="dispositivo_id").set_index(["dispositivo_id", "datetime_utc"])
potencia_df["potencia_solar"] = potencia_df["rad_poa"] * potencia_df["num_strings"] * num_mod_string * sup_mod
potencia_df["outlier_solar"] = np.where(potencia_df["potencia_act"] > 0.20 * potencia_df["potencia_solar"], True, False)
main_df = main_df.merge(potencia_df[["outlier_solar"]], left_index=True, right_index=True, how="inner")
print(f"Registros descartados por outlier de potencia: {main_df[main_df['outlier_solar'] == True].shape[0]}")
main_df = main_df[main_df["outlier_solar"] == False].drop(columns = ["outlier_solar"])
del potencia_df, num_strings_inv
gc.collect()

# Descarte de registros con potencia activa negativa
n_potencia_neg = main_df[main_df["potencia_act"] < 0].shape[0]
main_df = main_df[main_df["potencia_act"] >= 0]
print(f"Registros descartados por potencia activa negativa: {n_potencia_neg}")

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
main_df["dia_año"] = main_df.index.get_level_values("datetime_utc").dayofyear
main_df["dia_año_sen"] = np.sin(main_df["dia_año"] * (2*np.pi/365))
main_df["hora_seg"] = main_df.index.get_level_values("datetime_utc").hour * 3600 + \
                        main_df.index.get_level_values("datetime_utc").minute * 60 + \
                        main_df.index.get_level_values("datetime_utc").second
main_df["hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))

# Escalado de variables porcentuales
main_df[['cloud_impact']] = main_df[['cloud_impact']].apply(lambda x: x/100)

for inv_id in np.sort(main_df.index.get_level_values("dispositivo_id").unique()):
    # Separación de los datos de entrenamiento y validación
    print(f"Dispositivo {inv_id}")
    disp_df = main_df[main_df.index.get_level_values("dispositivo_id") == inv_id].copy()
    disp_df = disp_df.dropna()

    train_df, validation_df = train_test_split(disp_df, test_size = 0.2, random_state = 42)
    
    print(f"\tRegistros de entrenamiento: {train_df.shape[0]}")
    print(f"\tRegistros de validación: {validation_df.shape[0]}")

    # Separación de input y target
    y = train_df[["potencia_act"]].copy()
    y_val = validation_df[["potencia_act"]].copy()
    amp_columns = train_df.filter(like="amp_dc").columns.tolist()
    X = train_df.drop(columns = amp_columns+["id",
                                            "num_strings",
                                            "dia_año",
                                            "hora_seg",
                                            "potencia_act"
                                            ])
    X_val = validation_df.drop(columns = amp_columns+["id",
                                                    "num_strings",
                                                    "dia_año",
                                                    "hora_seg",
                                                    "potencia_act"
                                                    ])

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

    if optimizacion:
        start_time = time.time()
        if stage:
            print("\tOptimización del número de estimadores, learning rate, max_depth y min_child_weight")
            space = {'n_estimators': hp.quniform('n_estimators', 100, 1000, 5),
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                    'max_depth': hp.quniform("max_depth", 3, 15, 1),
                    'min_child_weight' : hp.quniform('min_child_weight', 1, 30, 1),
                    'gamma': 0,
                    'subsample': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 0}
            best_params = optimize_hyperparameters(space, dtrain, STALL_LIMIT = 10, MAX_EVALS_PER_RUN = 250, cv_folds = 5)

            print("\tOptimización de gamma y subsample")
            space = {'max_depth': best_params['max_depth'],
                    'min_child_weight' : best_params['min_child_weight'],
                    'n_estimators': best_params['n_estimators'],
                    'learning_rate': best_params['learning_rate'],
                    'gamma': hp.uniform ('gamma', 0, 10),
                    'subsample': hp.uniform('subsample', 0.4, 1),
                    'reg_alpha': 0,
                    'reg_lambda': 0}
            best_params = optimize_hyperparameters(space, dtrain, STALL_LIMIT = 10, MAX_EVALS_PER_RUN = 50, cv_folds = 5)
                    
            print("\tRegularización")
            space = {'reg_alpha' : hp.uniform('reg_alpha', 0, 10),
                    'reg_lambda' : hp.uniform('reg_lambda', 0, 10),
                    'n_estimators': best_params['n_estimators'],
                    'learning_rate': best_params['learning_rate'],
                    'max_depth': best_params['max_depth'],
                    'gamma': best_params['gamma'],
                    'min_child_weight': best_params['min_child_weight'],
                    'subsample': best_params['subsample']}
            best_params = optimize_hyperparameters(space, dtrain, STALL_LIMIT = 10, MAX_EVALS_PER_RUN = 50, cv_folds = 5)

            print(f"\tReoptimización de n_estimators y learning_rate con los parámetros finales")
            space = {'n_estimators': hp.quniform('n_estimators', 100, 2000, 5),
                    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
                    'max_depth': best_params['max_depth'],
                    'gamma': best_params['gamma'],
                    'min_child_weight': best_params['min_child_weight'],
                    'subsample': best_params['subsample'],
                    'reg_alpha': best_params['reg_alpha'],
                    'reg_lambda': best_params['reg_lambda']}
            best_params = optimize_hyperparameters(space, dtrain, STALL_LIMIT = 10, MAX_EVALS_PER_RUN = 75, cv_folds = 5)
        
        else:
            print("\tOptimización de los hiperparámetros")
            space = {'n_estimators': hp.quniform('n_estimators', 100, 1500, 5),
                    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
                    'max_depth': hp.quniform("max_depth", 3, 15, 1),
                    'gamma': hp.uniform ('gamma', 0, 5),
                    'min_child_weight' : hp.quniform('min_child_weight', 1, 30, 1),
                    'subsample' : hp.uniform('subsample', 0.4, 1),
                    'reg_alpha' : hp.uniform('reg_alpha', 0, 10),
                    'reg_lambda' : hp.uniform('reg_lambda', 0, 10)}
            best_params = optimize_hyperparameters(space, dtrain, STALL_LIMIT = 10, MAX_EVALS_PER_RUN = 250, cv_folds = 10)
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
    path = os.path.join(root_path, f"Modelos/potencia_inversor/Inversor_{inv_id - 20}/Repositorio/{algoritmo}-{pd.Timestamp.now()}/")
    os.makedirs(path)
    with open(path+'model.model', "wb") as archivo_salida:
        pickle.dump(pipeline_model, archivo_salida)

    # Generación de gráficos: comparativa de valores reales y predichos, histograma de diferencias y matriz de correlación
    plt.figure()
    plt.tight_layout()
    plt.scatter(prediction_df["potencia_act"], prediction_df["y_pred"], marker = ".")
    plt.plot([min(prediction_df["potencia_act"]), max(prediction_df["potencia_act"])], [min(prediction_df["potencia_act"]), max(prediction_df["potencia_act"])], color='black', linestyle='-', linewidth=1)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("Comparación de valores reales y predichos")
    plt.savefig(path + "scatter_validacion.png")

    plt.figure()
    plt.tight_layout()
    ax = sns.histplot(prediction_df["potencia_act"] - prediction_df["y_pred"], kde=True, stat='percent')
    ax.axvline(x=0, color='black', linestyle='--', linewidth = 0.35, label='x=0')
    plt.title('Histograma de las diferencias entre valores reales y predichos')
    plt.xlabel('Diferencia')
    plt.ylabel('Porcentaje')
    plt.savefig(path + "histogram_validacion.png")

    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    sns.heatmap(train_df[perc_attr + std_attr + y.columns.tolist()].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Matriz de correlación")
    plt.savefig(path + "correlacion.png")

    # Generación de gráficos: comparativa de RMSE y RMSE relativo por hora
    rmse_list = []
    rrmse_list = []
    hora_list = []
    for group in prediction_df.reset_index().groupby(["dispositivo_id", prediction_df.reset_index()["datetime_utc"].dt.hour]):
        hora_list.append(group[0][1])
        rmse_score = round(mse(group[1]["potencia_act"], group[1]["y_pred"], squared=False), 2)
        rmse_list.append(rmse_score)
        mean_potencia_act = group[1]['potencia_act'].mean()
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
        informe = {"normalizacion": normalizacion,
                    "optimizacion": optimizacion,
                    "por_fases": stage,
                    "intervalo_min": intervalo_min,
                    "metricas": metricas,
                    "metricas_hora": {k:v for k,v in zip(hora_list, rmse_list)},
                    "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia.values())}.items(), key=lambda item: item[1], reverse=True)),
                    "hiperparametros": {k:v for k,v in train_params.items() if v != None},
                    "training_input_description": train_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                    "training_target_description": train_df["potencia_act"].describe().to_dict(),
                    }
        json.dump(informe, archivo_json)