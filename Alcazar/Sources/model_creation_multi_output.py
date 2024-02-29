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
import gc

def discriminador(row):
    mean = np.mean(row)
    std = np.std(row)
    threshold = 3 * std
    outlier = np.abs(row - mean) > threshold
    return outlier

class MultiOutputOpt(BaseEstimator, TransformerMixin):
    def __init__(self, model_class=XGBRegressor, device="cpu", trials=None):
        self.model_class = model_class
        self.device = device
        self.models = {}  
        self.trials = trials if trials is not None else {}
        try:
            print(len(self.trials[0].trials))
        except:
            pass
        self.params = {}
        
    def objective(self, space_params, train_set, cv_folds):
        n_estimators = int(space_params.pop('n_estimators'))
        params = {'device': self.device,
            'objective': 'reg:squarederror', 
            'tree_method': "hist", 
            'random_state': 42,
            'n_jobs': -1}
        params = {**params, **space_params}
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'min_child_weight' in params:
            params['min_child_weight'] = max(1, int(params['min_child_weight']))
        print
        cv_result = xgb.cv(params, train_set, nfold = cv_folds, num_boost_round = n_estimators, early_stopping_rounds = 100, metrics = 'rmse', as_pandas = True)
        score = cv_result['test-rmse-mean'].min()
        return {'loss': score, 'status': STATUS_OK}
        
    def optimize(self, X, y, space, cv_folds, gamma_algo = 1, STALL_LIMIT = 5, MAX_EVALS_PER_RUN = 250):
        for col in tqdm(range(y.shape[1]), total = y.shape[1]):
            if self.params.get(col) is not None:
                space.update(self.params[col])
            train_set = xgb.DMatrix(X, label=y[:, col])
            trials = Trials()
            TOTAL_EVALS = len(trials.trials)
            STALL_LIMIT = STALL_LIMIT
            MAX_EVALS_PER_RUN = MAX_EVALS_PER_RUN
            best_loss = np.inf
            stall_counter = 0
            num_evals = TOTAL_EVALS
            run_counter = 0
            upper_limit = (MAX_EVALS_PER_RUN * (STALL_LIMIT - 1)) * 10
            while stall_counter < STALL_LIMIT and num_evals < TOTAL_EVALS + upper_limit:
                best = fmin(fn=lambda space: self.objective(space, train_set = train_set, cv_folds = cv_folds), 
                        space = space, 
                        algo = partial(tpe.suggest, gamma = gamma_algo),
                        max_evals = num_evals + MAX_EVALS_PER_RUN, 
                        trials = trials,
                        verbose = False)  # Cambiado a True para que devuelva los parámetros óptimos
                best_params = space_eval(space, best)  # Obtener los parámetros óptimos en su forma original
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
                num_evals += MAX_EVALS_PER_RUN
                run_counter += 1
                gamma_algo -= 0.05
            print(f"\tEntrenamiento para entrada {col+1} finalizado")
            print(f"\tNúmero de evaluaciones realizadas: {num_evals}")
            print(f"\tBest params: {best_params}")
            print(f"\tBest loss: {best_loss}")

            final_params = {"device": self.device,
                        "objective": 'reg:squarederror', 
                        "tree_method": "hist",
                        "n_jobs": -1}
            final_params = {**final_params, **best_params}
            final_params.pop('n_estimators')
            if 'max_depth' in final_params:
                final_params['max_depth'] = int(final_params['max_depth'])
            if 'min_child_weight' in final_params:
                final_params['min_child_weight'] = max(1, int(final_params['min_child_weight']))
            model = xgb.train(final_params, train_set, num_boost_round = int(best_params['n_estimators']))
            self.models[col] = model
            self.trials[col] = trials
            self.params[col] = best_params  # Guardar los parámetros óptimos en su forma original

    def predict(self, X):
        predictions = []
        dmatrix = xgb.DMatrix(X)
        for col, model in self.models.items():
            pred = model.predict(dmatrix)
            predictions.append(pred)
        return np.column_stack(predictions)
    
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
if torch.cuda.is_available() and model_name == "XGBRegressor":
    while device not in valid_devices:
        device_bool = input("¿Desea utilizar GPU para el entrenamiento? (Y/N): ").lower()
        if device_bool == "y":
            device = "cuda"
        else:
            device = "cpu"
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

num_mod_string = 28
sup_mod = 2
ids_parques = [1,2,3]
for id_parque in ids_parques:
    print(f"Entrenamiento para el parque fotovoltaico {id_parque}", )
    # Carga de los datos de entrenamiento
    main_query = f"""
        WITH inv AS (
            SELECT id, parque_id, dispositivo_id, datetime_utc, potencia_act
                FROM {schema_name}.inversores
                WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 15 = 0)
                        AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                        AND parque_id = {id_parque}
                ORDER BY datetime_utc)
        SELECT inv.id, inv.dispositivo_id, det.entrada_id, inv.datetime_utc, potencia_act, num_strings, 
                rad_poa, rad_hor, rad_celda1, rad_celda2, temp_amb, temp_panel1, temp_panel2, presion_atm, cloud_impact, 
                motivo, consigna_pot_act_planta, amp_dc
            FROM inv
            JOIN {schema_name}.inversores_detalle AS det
                ON inv.id = det.id
             JOIN {schema_name}.distrib_inversores dist
                ON dist.parque_id = inv.parque_id
                    AND dist.dispositivo_id = inv.dispositivo_id
                    AND dist.entrada_id = det.entrada_id
            JOIN {schema_name}.dispositivos AS disp
                ON disp.dispositivo_id = inv.dispositivo_id
                    AND disp.parque_id = inv.parque_id
            JOIN {schema_name}.meteo AS met
                ON met.dispositivo_id = disp.meteo_cercana_id
                    AND met.datetime_utc = inv.datetime_utc
                    AND met.parque_id = inv.parque_id
            JOIN {schema_name}.ree AS ree
                ON ree.datetime_utc = inv.datetime_utc
                    AND ree.parque_id = inv.parque_id
            WHERE daylight = true
            ORDER BY 4, 2, 3;"""

    chunksize = 100000
    chunks = pd.read_sql_query(main_query, engine, chunksize=chunksize)
    main_df = pd.DataFrame()
    for chunk in chunks:
        main_df = pd.concat([main_df, chunk], ignore_index = True)
    del chunks, chunk
    gc.collect()

    # Normalización de la entrada de corriente continua, formateo de fechas y escalado de potencia
    print(f"\tCarga inicial de {main_df.shape[0]} registros")
    if normalizacion:
        print("\tEntrada de corriente continua normalizada según el número de strings")
        main_df["amp_dc"] = main_df["amp_dc"]/main_df["num_strings"]
    else:
        print("\tEntrada de corriente continua sin normalizar")
    main_df["datetime_utc"] = pd.to_datetime(main_df["datetime_utc"], utc = True)
    main_df["potencia_act"] = main_df["potencia_act"] * 1000

    # Pivotado de las entradas de corriente continua
    target_df = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
    target_df.columns = ["amp_dc_" + str(col) for col in target_df.columns]
    print(f"\tNúmero de registros del dataframe tras pivotar: {target_df.shape[0]}")

    # Descarte de registros con corriente anómala
    target_df["outlier"] = target_df.apply(discriminador, axis=1).any(axis=1)
    n_corriente_outlier = target_df[target_df["outlier"]].shape[0]
    target_df = target_df[~target_df["outlier"]].drop(columns="outlier")
    print(f"\tRegistros descartados por outlier de corriente: {n_corriente_outlier}")

    # Rellenado de valores faltantes por desconexión de entradas
    consulta_sql = f"""SELECT MAX(entrada_id)
                FROM {schema_name}.distrib_inversores
                WHERE parque_id = {id_parque};"""
    max_entradas = pd.read_sql_query(consulta_sql, engine).values[0][0]
    entradas = [i for i in range(1, max_entradas + 1)]
    for dispositivo in target_df.index.get_level_values(0).unique():
        consulta_sql = f"""SELECT entrada_id
                    FROM {schema_name}.distrib_inversores
                    WHERE parque_id = {id_parque}
                        AND dispositivo_id = {dispositivo};"""
        entradas_inv = pd.read_sql_query(consulta_sql, engine).values.reshape(1,-1)[0]
        entradas_off = list(set(entradas) - set(entradas_inv))
        for entrada in entradas_off:
            target_df.loc[dispositivo, "amp_dc_" + str(entrada)] = 0
    
    # Descarte de registros con corriente desconocida
    n_nan_values = target_df[target_df.isna().any(axis=1)].shape[0]
    target_df = target_df.dropna()
    print(f"\tRegistros descartados por corriente desconocida: {n_nan_values}")

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
    print(f"\tRegistros descartados por corriente negativa: {n_corriente_neg}")

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
    print(f"\tRegistros descartados por potencia activa negativa: {n_potencia_neg}")

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
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_planta"] / max_pot_act 

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
        print(f"\n\tDispositivo {inv_id}")
        disp_df = main_df[main_df.index.get_level_values("dispositivo_id") == inv_id].copy()
        disp_df = disp_df.dropna()
        validation_ratio = 0.2
        train_df, validation_df = train_test_split(disp_df, test_size = validation_ratio, random_state = 42)
        print(f"\t\tRegistros de entrenamiento: {disp_df.shape[0]}")
        print(f"\t\tRegistros de validación: {validation_df.shape[0]}")

        # Separación de input y target
        target_columns = train_df.filter(like="amp_dc").columns.tolist()
        y = train_df.filter(like="amp_dc")
        y_val = validation_df.filter(like="amp_dc")
        X = train_df.drop(columns = target_columns+["id",
                                    "dia_año",
                                    "hora_seg",
                                    "potencia_act",
                                    "num_strings"
                                    ])
        X_val = validation_df.drop(columns = target_columns+["id",
                                    "dia_año",
                                    "hora_seg",
                                    "potencia_act",
                                    "num_strings"
                                    ])

        # Estandarización/normalización de variables numéricas y codificación de variables categóricas
        perc_attr = ['cloud_impact', 'consigna_pot_act_planta']
        std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'rad_diff', 'temp_panel', 'presion_atm']
        cat_attr = ['motivo']

        transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])                         # Introducir manualmente catergorías?
        transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 

        preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                    ('std', transformador_numerico_std, std_attr)],
                                    remainder='passthrough')
        
        X_prep = preprocessor.fit_transform(X)

        if optimizacion:
            print("\tOptimización del número de estimadores")
            space = {'n_estimators': hp.quniform('n_estimators', 100, 1500, 5),
                    'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.005)}
            multioutput_model = MultiOutputOpt(model_class=XGBRegressor, device=device)
            multioutput_model.optimize(X_prep, y.values, space = space, cv_folds = 5, gamma_algo = 0.75, STALL_LIMIT = 5, MAX_EVALS_PER_RUN = 100)

            print("\n\tOptimización de profundidad, gamma, min_child_weight, subsample")
            space = {'max_depth': hp.quniform("max_depth", 3, 15, 1),
                    'gamma': hp.quniform ('gamma', 0, 1, 0.01),
                    'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1),
                    'subsample' : hp.quniform('subsample', 0.5, 1, 0.05)}
            multioutput_model.optimize(X_prep, y.values, space = space, cv_folds = 5, gamma_algo = 0.75, STALL_LIMIT = 5, MAX_EVALS_PER_RUN = 250)
        
            print("\n\tOptimización de parámetros de regularización")
            space = {'reg_alpha' : hp.quniform('reg_alpha', 0, 20, 0.1),
                    'reg_lambda' : hp.quniform('reg_lambda', 0, 20, 0.1)}
            multioutput_model.optimize(X_prep, y.values, space = space, cv_folds = 5, gamma_algo = 0.75, STALL_LIMIT = 5, MAX_EVALS_PER_RUN = 100)
        
        else:
            if model_name == "XGBRegressor":
                dtrain = xgb.DMatrix(X_prep, label=y)
                params = {
                    'objective': 'reg:squarederror',
                    'tree_method': "hist",
                    'multi_strategy': "multi_output_tree",
                    'random_state': 42,
                    'n_jobs': -1
                }
                multioutput_model = xgb.train(params, dtrain) 

            else:
                print("Modelo no reconocido")
                sys.exit()

        pipeline_model = Pipeline([('preprocessor', preprocessor),
                                ('regressor', multioutput_model)])

        consulta_sql = f"""SELECT dispositivo_id, entrada_id, num_strings
                        FROM {schema_name}.distrib_inversores
                        WHERE parque_id = {id_parque}
                            AND dispositivo_id = {inv_id};"""
        num_strings = pd.read_sql_query(consulta_sql, engine)

        if optimizacion:
            X_val_prep = pipeline_model.named_steps['preprocessor'].transform(X_val)
        else:
            X_val_prep = xgb.DMatrix(pipeline_model.named_steps['preprocessor'].transform(X_val))
        y_pred_val = pd.DataFrame(pipeline_model.named_steps['regressor'].predict(X_val_prep)).rename(columns={i: "y_pred_"+str(i+1) for i in pd.DataFrame(pipeline_model.named_steps['regressor'].predict(X_val_prep)).columns})
        y_val_reesc = y_val.copy()
        y_reesc = y.copy()
        if normalizacion:
            for entrada in num_strings["entrada_id"]:
                y_pred_val["y_pred_"+str(entrada)] = y_pred_val["y_pred_"+str(entrada)] * num_strings[num_strings["entrada_id"] == entrada]["num_strings"].values[0]
                y_val_reesc.loc[y_val.index, "amp_dc_"+str(entrada)] = y_val["amp_dc_"+str(entrada)] * num_strings[num_strings["entrada_id"] == entrada]["num_strings"].values[0]
                y_reesc.loc[y.index, "amp_dc_"+str(entrada)] = y["amp_dc_"+str(entrada)] * num_strings[num_strings["entrada_id"] == entrada]["num_strings"].values[0]
        
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
        print(f"\t\tMétricas de error:\n\t\t\tRMSE: {rmse_score}\n\t\t\tMAE: {mae_score}\n\t\t\tR2: {r2_score}")

        # Guardado del modelo y de las métricas
        algoritmo = pipeline_model.named_steps["regressor"].__class__.__name__
        # columnas = [col_name.split("__")[1] for col_name in preprocessor.get_feature_names_out()]
        # importance_scores = model.get_score(importance_type='gain')
        # total_gain = np.array([v for k,v in importance_scores.items()]).sum()
        # importancia = {k: v/total_gain for k, v in importance_scores.items()}
        path = os.path.join(root_path, f"Modelos/entrada_amperaje_multi/Parque_{id_parque}/Inversor_{inv_id - 20}/Repositorio/{algoritmo}-{pd.Timestamp.now()}/")
        os.makedirs(path)
        with open(path+'model.model', "wb") as archivo_salida:
            pickle.dump(pipeline_model, archivo_salida)
        with open(path+'informe_modelo.json', 'w') as archivo_json:
            informe = {"normalizacion": normalizacion,
                        "optimizacion": optimizacion,
                        "cross_validation": use_cv,
                        "metricas": metricas,
                        # "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia.values())}.items(), key=lambda item: item[1], reverse=True)),
                        "validation_split": validation_ratio,
                        # "hiperparametros": {k:v for k,v in params.items() if v != None},
                        "training_input_description": train_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                        "training_target_description": y_reesc.describe().to_dict(),
                        }
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
        sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de correlación")
        plt.savefig(path + "correlacion.png")

        # Generación de gráficos: comparativa de RMSE y RMSE relativo por entrada
        rmse_list = []
        rmse_r_list = []
        entrada_list = []
        for group in prediction_df.groupby(["dispositivo_id", "entrada_id"]):
            entrada_list.append(group[0][1])
            rmse_score = round(mse(group[1]["amp_dc"], group[1]["y_pred"], squared=False), 2)
            epsilon = 1e-7  # A small constant
            rmse_r_score = round((mse(group[1]["amp_dc"], group[1]["y_pred"], squared=False)*100/(group[1]['amp_dc'].mean() + epsilon)), 2)
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
        if np.isfinite(rmse_r_score).all():
            if rmse_r_score.max() - rmse_r_score.min() > 0.25:
                ax2.set_yscale('log')
        else:
            ax2.set_yscale('log')
        ax2.plot(entrada_list, rmse_r_list, color=color, linewidth=1)
        sns.scatterplot(x=entrada_list, y=rmse_r_list, color=color, ax=ax2)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'RMSE por entrada para el inversor {inv_id - 20}')
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
