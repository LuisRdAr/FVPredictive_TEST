import numpy as np
import pandas as pd
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
import xgboost as xgb
import torch
from hyperopt import fmin, tpe, hp, Trials
from functools import partial
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, \
                            r2_score as r2
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(space_params, dtrain, device):
    params = {'device': device,
            'objective': 'reg:squarederror', 
            'tree_method': "hist", 
            'random_state': 42,
            'n_jobs': -1,
            'max_depth': int(space_params['max_depth']),
            'gamma': space_params['gamma'],
            'learning_rate': space_params['learning_rate'],
            'min_child_weight': int(space_params['min_child_weight']),
            'subsample': space_params['subsample'],
            'reg_alpha': space_params['reg_alpha'],
            'reg_lambda': space_params['reg_lambda'],
             }
    
    cv_result = xgb.cv(params, dtrain, nfold = 5, num_boost_round = int(space_params["n_estimators"]), early_stopping_rounds = 100, metrics = 'rmse', as_pandas = True)
    score = cv_result['test-rmse-mean'].min()
    return {'loss': score, 'status': STATUS_OK}

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

valid_models = {"1": "XGBRegressor", "2": "RandomForestRegressor"}
model_name = ""
while model_name not in valid_models.keys():
    model_name = input("Ingrese el valor numérico para el tipo de modelo que desea utilizar (XGBRegressor[1], RandomForestRegressor[2]): ")
model_name = valid_models[model_name]
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

use_cv = False
if optimizacion:
    while use_cv not in valid_responses:
        use_cv = input("\t¿Desea utilizar validación cruzada? (Y/N): ").lower()
    use_cv = use_cv == "y"

# Comprobación de la disponibilidad de GPU para el entrenamiento
valid_devices = ["cpu", "cuda"]
device = ""
if torch.cuda.is_available() and model_name == "XGBRegressor":
    while device not in valid_devices:
        device = input("¿Desea utilizar GPU para el entrenamiento? (CPU/GPU): ").lower()
else:
    device = 'cpu'

print(f"\nLas opciones seleccionadas son: \nModelo: {model_name} \nNormalización: {normalizacion} \nOptimización: {optimizacion} \nValidación cruzada: {use_cv} \nEntrenamiento: {device}", end="\n\n")

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
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_planta"] * 100 / max_pot_act 
elif schema_name == "bonete":
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_ree"] * 100 / max_pot_act 
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
main_df[['lim_act', 'cloud_impact', 'consigna_pot_act_planta']] = main_df[['lim_act', 'cloud_impact', 'consigna_pot_act_planta']].apply(lambda x: x/100)

# Agrupación de las entradas de corriente continua por inversor y por hora en una tabla dinámica
pivot_table = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc').sum(axis = 1)
complete_df = pd.merge(main_df, pivot_table.reset_index(), on=["dispositivo_id", "datetime_utc"]).drop(columns=["entrada_id", "num_strings", "amp_dc"]).drop_duplicates().rename(columns={0: "amp_dc"})

for inv_id in np.sort(main_df["dispositivo_id"].unique()):
    # Separación de los datos de entrenamiento y validación
    print(f"Dispositivo {inv_id}")
    disp_df = main_df[main_df["dispositivo_id"] == inv_id].copy()
    disp_df.dropna()
    validation_df = disp_df[disp_df["datetime_utc"].dt.month == 9]
    train_df = disp_df[disp_df["datetime_utc"].dt.month != 9]
    print(f"\tRegistros de entrenamiento: {disp_df.shape[0]}")
    print(f"\tRegistros de validación: {validation_df.shape[0]}")

    # Descarte de variables que no se usarán en el entrenamiento y separación de input y target
    processed_df = train_df.drop(columns = ["id",
                                    "dia_año",
                                    "hora_seg",
                                    "outlier_1",
                                    "num_strings"
                                    ]).set_index(["datetime_utc",
                                                    "dispositivo_id",
                                                    "entrada_id"])
    processed_val_df = validation_df.drop(columns = ["id",
                                    "dia_año",
                                    "hora_seg",
                                    "outlier_1",
                                    "num_strings"
                                    ]).set_index(["datetime_utc",
                                                    "dispositivo_id",
                                                    "entrada_id"])
        
    y = processed_df[["potencia_act"]]
    y_val = processed_val_df[["potencia_act"]]
    X = processed_df.drop(columns = ["potencia_act"])
    X_val = processed_val_df.drop(columns = ["potencia_act"])
    print(f"\tNúmero de registros para entrenamiento: {X.shape[0]}")

    # Estandarización/normalización de variables numéricas y codificación de variables categóricas
    perc_attr = ['lim_act', 'cloud_impact', 'consigna_pot_act_planta']
    std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'rad_diff', 'temp_panel', 'amp_dc']
    cat_attr = ['motivo']

    transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])     # Introducir manualmente catergorías?
    transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 

    preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                ('std', transformador_numerico_std, std_attr)],
                                remainder='passthrough')

    X_prep = preprocessor.fit_transform(X)
    dtrain = xgb.DMatrix(X_prep, label=y)

    # Entrenamiento del modelo y construcción del dataframe para validación
    if optimizacion:
        # Inicialización  y primera fase de la optimización de hiperparámetros con gamma = 1 y espacio de búsqueda general
        trials = Trials()
        best_loss = np.inf
        threshold = 0.001
        gamma = 1
        stall_counter = 0
        STALL_LIMIT = 3
        MAX_EVALS_PER_RUN = 250
        total_evals = 0
        run_counter = 0
        if use_cv:
            if model_name == "XGBRegressor":
                space ={'max_depth': hp.quniform("max_depth", 3, 15, 3),
                    'n_estimators': hp.quniform('n_estimators', 100, 1500, 200),
                    'learning_rate': hp.quniform('learning_rate', 0.01, 0.2, 0.01),
                    'gamma': hp.quniform ('gamma', 0, 1, 0.2),
                    'min_child_weight' : hp.quniform('min_child_weight', 1, 19, 3),
                    'subsample' : hp.quniform('subsample', 0.5, 1, 0.1),
                    'reg_alpha' : hp.quniform('reg_alpha', 0, 10, 1),
                    'reg_lambda' : hp.quniform('reg_lambda', 0, 10, 1)
                }
                upper_limit = (MAX_EVALS_PER_RUN * (STALL_LIMIT - 1)) * 10
                while stall_counter < STALL_LIMIT and total_evals < upper_limit:
                    best = fmin(fn = partial(objective, dtrain = dtrain, device = device), 
                                space = space,
                                algo = partial(tpe.suggest, gamma = gamma),
                                max_evals = total_evals + MAX_EVALS_PER_RUN,
                                trials = trials)
                    new_loss = trials.best_trial['result']['loss']
                    if new_loss < best_loss:
                        if abs(new_loss - best_loss) <= threshold:
                            stall_counter += 1
                        else:
                            stall_counter = 0
                        best_loss = new_loss
                    else:
                        stall_counter += 1

                    total_evals += MAX_EVALS_PER_RUN
                    run_counter += 1

                    if total_evals % (MAX_EVALS_PER_RUN * (STALL_LIMIT - 1)) == 0:
                        gamma = max(0, gamma - 0.05)
                    print(f"Run {run_counter}: Best loss so far: {best_loss}")

                print(f"\tMejor combinación de hiperparámetros en la primera búsqueda:\n{best}")

                # Segunda fase de la optimización de hiperparámetros con gamma = 0.33 y espacio de búsqueda fino
                stall_counter = 0
                STALL_LIMIT = 5
                MAX_EVALS_PER_RUN = 250
                initial_total_evals = total_evals
                run_counter = 0
                space ={'max_depth': hp.quniform("max_depth", 3, 15, 1),
                        'n_estimators': hp.quniform('n_estimators', 100, 1500, 25),
                        'learning_rate': hp.quniform('learning_rate', 0.01, 0.2, 0.01),
                        'gamma': hp.quniform ('gamma', 0, 1, 0.01),
                        'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1),
                        'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                        'reg_alpha' : hp.quniform('reg_alpha', 0, 10, 0.1),
                        'reg_lambda' : hp.quniform('reg_lambda', 0, 10, 0.1)
                    }
                while stall_counter < STALL_LIMIT and total_evals < initial_total_evals + 10000:
                    best = fmin(fn = partial(objective, dtrain=dtrain, device = device), 
                                space = space, 
                                algo = partial(tpe.suggest, gamma = 0.33), 
                                max_evals = total_evals + MAX_EVALS_PER_RUN, 
                                trials = trials)
                    
                    new_loss = trials.best_trial['result']['loss']
                    if new_loss < best_loss:
                        if abs(new_loss - best_loss) <= threshold:
                            stall_counter += 1
                        else:
                            stall_counter = 0
                        best_loss = new_loss
                    else:
                        stall_counter += 1

                    total_evals += MAX_EVALS_PER_RUN
                    run_counter += 1
                    print(f"Run {run_counter}: Best loss so far: {best_loss}")

                print(f"\tMejor combinación de hiperparámetros:\n{best}")

                params = {'max_depth': int(best['max_depth']),
                                'gamma': best['gamma'],
                                'learning_rate': best['learning_rate'],
                                'min_child_weight': best['min_child_weight'],
                                'subsample': best['subsample'],
                                'reg_alpha': best['reg_alpha'],
                                'reg_lambda': best['reg_lambda'],
                                'device': 'cuda',
                                'tree_method': 'hist',
                                'objective': 'reg:squarederror',
                                'random_state': 0,
                                'n_jobs': -1
                            }
                
                model = xgb.train(params, dtrain, num_boost_round = int(best['n_estimators']))
            else:
                print("Modelo no implementado o no reconocido")
                sys.exit()

    else:
        if model_name == "XGBRegressor":
            params = {
                'random_state': 0,
                'n_jobs': -1
            }
            model = xgb.train(params, dtrain)
        else:
            print("Modelo no implementado o no reconocido")
            sys.exit()

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
    with open(path+'informe_modelo.json', 'w') as archivo_json:
        informe = {"normalizacion": normalizacion,
                    "optimizacion": optimizacion,
                    "cross_validation": use_cv,
                    "metricas": metricas,
                    "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia.values())}.items(), key=lambda item: item[1], reverse=True)),
                    "validation_range": {"start": validation_df["datetime_utc"].min().strftime("%Y-%m-%d %H:%M:%S"), 
                                            "end": validation_df["datetime_utc"].max().strftime("%Y-%m-%d %H:%M:%S")},
                    "hiperparametros": {k:v for k,v in params.items() if v != None},
                    "training_input_description": train_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                    "training_target_description": train_df["potencia_act"].describe().to_dict(),
                    }
        json.dump(informe, archivo_json)

    # Generación de gráficos: comparativa de valores reales y predichos, histograma de diferencias y matriz de correlación
    y_test_sampled, _,y_pred_sampled, _ = train_test_split(prediction_df["potencia_act"], prediction_df["y_pred"], train_size = 0.25)
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
    ax = sns.histplot(prediction_df["potencia_act"] - prediction_df["y_pred"], kde=True, stat='percent')
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