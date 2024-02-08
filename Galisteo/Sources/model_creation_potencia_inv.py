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
    results["outlier_2"] = outlier
    results = results.reset_index(drop=True)
    return results


cuda_flag = torch.cuda.is_available()
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

intervalo_min = 15
num_mod_string = 30
sup_mod = 2
parque_id_query = f"""SELECT parque_id FROM {schema_name}.parques;"""
ids_parques = pd.read_sql_query(parque_id_query, engine)["parque_id"].values
for id_parque in ids_parques:
    print(f"Entrenamiento para el parque fotovoltaico {id_parque}")
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
                    WHERE dispositivo_id != 26
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
        validation_df = disp_df[((disp_df["datetime_utc"].dt.month == 9) |
                                    (disp_df["datetime_utc"].dt.month == 10) |
                                    (disp_df["datetime_utc"].dt.month == 11) |
                                    (disp_df["datetime_utc"].dt.month == 12))]
        train_df = disp_df[~disp_df.index.isin(validation_df.index)]
        
        print(f"\tRegistros de entrenamiento: {train_df.shape[0]}")
        print(f"\tRegistros de validación: {validation_df.shape[0]}")


        # Descarte de variables que no se usarán en el entrenamiento y separación de input y target
        processed_df = train_df.drop(columns = ["id",
                                        "dia_año",
                                        "hora_seg",
                                        "outlier_1",
                                        "amp_dc"
                                        ]).set_index(["datetime_utc",
                                                      "dispositivo_id"])
        processed_val_df = validation_df.drop(columns = ["id",
                                        "dia_año",
                                        "hora_seg",
                                        "outlier_1",
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

        if optimizacion:
            start_time = time.time()
            print("\tOptimización del número de estimadores y learning rate")
            # space = {'n_estimators': hp.quniform('n_estimators', 100, 1500, 5),
            #         'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            #         'max_depth': 5,
            #         'gamma': 0,
            #         'min_child_weight': 1,
            #         'subsample': 1,
            #         'reg_alpha': 0,
            #         'reg_lambda': 0}
            space = {'n_estimators': hp.quniform('n_estimators', 100, 1500, 5),
                    'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
                    'max_depth': hp.quniform("max_depth", 3, 15, 1),
                    'gamma': hp.uniform ('gamma', 0, 1),
                    'min_child_weight' : hp.quniform('min_child_weight', 1, 30, 1),
                    'subsample' : hp.uniform('subsample', 0.25, 1),
                    'reg_alpha' : hp.uniform('reg_alpha', 0, 100),
                    'reg_lambda' : hp.uniform('reg_lambda', 0, 100)}
            STALL_LIMIT = 5
            MAX_EVALS_PER_RUN = 250
            best_loss = np.inf
            trials = Trials()
            gamma_algo = 0.75
            stall_counter = 0
            num_evals = 0
            upper_limit = (MAX_EVALS_PER_RUN * (STALL_LIMIT - 1)) * 10
            while stall_counter < STALL_LIMIT and num_evals < upper_limit:
                best = fmin(fn=lambda space: objective(space, train_set = dtrain, cv_folds = 5), 
                        space = space, 
                        algo = partial(tpe.suggest, gamma = gamma_algo),
                        max_evals = num_evals + MAX_EVALS_PER_RUN, 
                        trials = trials,
                        verbose = True)
                best_params = space_eval(space, best)
                new_loss = trials.best_trial['result']['loss']
                if new_loss < best_loss:
                    # threshold = 0.001
                    # if abs(new_loss - best_loss) <= threshold:
                    #     stall_counter += 1
                    # else:
                    #     stall_counter = 0
                    best_loss = new_loss
                else:
                    stall_counter += 1
                num_evals += MAX_EVALS_PER_RUN
                gamma_algo -= 0.5 / ((STALL_LIMIT - 1) * 10)
            print(f"\tNúmero de evaluaciones realizadas: {num_evals}")
            print(f"\tBest params: {best_params}")
            print(f"\tBest loss: {best_loss}")

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
        with open(path+'informe_modelo.json', 'w') as archivo_json:
            informe = {"normalizacion": normalizacion,
                        "optimizacion": optimizacion,
                        "cross_validation": use_cv,
                        "intervalo_min": intervalo_min,
                        "metricas": metricas,
                        "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia.values())}.items(), key=lambda item: item[1], reverse=True)),
                        "training_range": {"start": train_df["datetime_utc"].min().strftime("%Y-%m-%d %H:%M:%S"),
                                            "end": train_df["datetime_utc"].max().strftime("%Y-%m-%d %H:%M:%S")},
                        "validation_range": {"start": validation_df["datetime_utc"].min().strftime("%Y-%m-%d %H:%M:%S"), 
                                                "end": validation_df["datetime_utc"].max().strftime("%Y-%m-%d %H:%M:%S")},
                        "hiperparametros": {k:v for k,v in train_params.items() if v != None},
                        "training_input_description": train_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                        "training_target_description": train_df["potencia_act"].describe().to_dict(),
                        }
            json.dump(informe, archivo_json)

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
        sns.heatmap(processed_df[perc_attr + std_attr].corr(), annot=True, fmt=".2f", cmap="coolwarm")
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
