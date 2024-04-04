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

def get_power_model(initial_path):
    models = {}
    for inv_dir in os.listdir(initial_path):
        if ("inversor" in inv_dir.lower()) & (os.path.isdir(os.path.join(initial_path, inv_dir))):
            explotacion_path = os.path.join(initial_path, inv_dir, "Explotacion")
            for model_dir in os.listdir(explotacion_path):
                model_path = os.path.join(explotacion_path, model_dir)
                if os.path.isdir(model_path):
                    model_dict = {}
                    with open(os.path.join(model_path, "model.model"), "rb") as f:
                        model_dict["model"] = pickle.load(f)
                    with open(os.path.join(model_path, "informe_modelo.json"), "r") as f:
                        informe_modelo = json.load(f)
                        if "intervalo_min" in informe_modelo:
                            model_dict["intervalo"] = informe_modelo["intervalo_min"]
                        model_dict["normalizacion"] = informe_modelo["normalizacion"]
                        model_dict["RMSE"] = informe_modelo["metricas"]["RMSE"]
                        model_dict["RMSE_rad"] = informe_modelo["metricas_rad"]
                    models[inv_dir] = model_dict
    return models    

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
model_name = "XGBRegressor"
valid_responses = ["y", "n"]

promediado = ""
while promediado not in valid_responses:
    promediado = input("¿Desea promediar los datos de entrada? (Y/N): ").lower()
promediado = promediado == "y"

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
        print("No se ha detectado una GPU disponible para el entrenamiento")
        device = "cpu"
else:
    stage = False
    device = 'cpu'

root_path = os.getcwd()
if promediado:
    power_models = get_power_model(os.path.join(root_path, "Modelos", "potencia_inversor_promedio"))
    intervalo = power_models["Inversor_1"]["intervalo"]
    normalizacion = power_models["Inversor_1"]["normalizacion"]
    print(f"\nLas opciones seleccionadas son: \nModelo: {model_name} \
      \nPromediado: {promediado} \
      \n\tIntervalo: {intervalo} \
      \nNormalización: {normalizacion} \
      \nOptimización: {optimizacion} \
      \n\tPor fases: {stage} \
      \nEntrenamiento: {device}", end="\n\n")
else:
    power_models = get_power_model(os.path.join(root_path, "Modelos", "potencia_inversor"))
    normalizacion = power_models["Inversor_1"]["normalizacion"]
    print(f"\nLas opciones seleccionadas son: \nModelo: {model_name} \
      \nPromediado: {promediado} \
      \nNormalización: {normalizacion} \
      \nOptimización: {optimizacion} \
      \n\tPor fases: {stage} \
      \nEntrenamiento: {device}", end="\n\n")

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
# Carga de los datos de entrenamiento y modelos de potencia
if promediado:
    main_query = f"""
            WITH inv AS (
                SELECT 
                    date_trunc('hour', datetime_utc) + 
                    INTERVAL '{intervalo} min' * floor(date_part('minute', datetime_utc) / {intervalo}) as datetime_utc_rounded,
                    dispositivo_id,
                    MIN(inv.id) AS id, 
                    AVG(potencia_act) as potencia_act, 
                    AVG(amp_dc) as amp_dc,
                    det.entrada_id
                FROM {schema_name}.inversores AS inv
                JOIN {schema_name}.inversores_detalle AS det
                    ON inv.id = det.id
                WHERE EXTRACT(YEAR FROM datetime_utc) != 2024
                GROUP BY datetime_utc_rounded, dispositivo_id, det.entrada_id 
            ),
            stats AS (
                SELECT
                    datetime_utc,
                    dispositivo_id,
                    COALESCE((cloud_impact - LAG(cloud_impact) OVER (PARTITION BY dispositivo_id ORDER BY datetime_utc)) / NULLIF(cloud_impact, 0), 0) * 100 AS diff_porc_cloud_impact,
                    COALESCE(cloud_impact - LAG(cloud_impact) OVER (PARTITION BY dispositivo_id ORDER BY datetime_utc), 0) AS diff_cloud_impact,
                    COALESCE((rad_poa - LAG(rad_poa) OVER (PARTITION BY dispositivo_id ORDER BY datetime_utc)) / NULLIF(rad_poa, 0), 0) * 100 AS diff_porc_rad_poa,
                    COALESCE(rad_poa - LAG(rad_poa) OVER (PARTITION BY dispositivo_id ORDER BY datetime_utc), 0) AS diff_rad_poa
                FROM galisteo.meteo
                ORDER BY 1,2
            ),
            met AS (
                SELECT 
                    date_trunc('hour', meteo.datetime_utc) + 
                    INTERVAL '{intervalo} min' * floor(date_part('minute', meteo.datetime_utc) / {intervalo}) as datetime_utc_rounded, 
                    meteo.dispositivo_id,
                    AVG(rad_poa) AS rad_poa,
                    STDDEV(rad_poa) AS std_rad_poa,
                    MAX(rad_poa) - MIN(rad_poa) AS max_diff_rad_poa,
                    AVG(diff_porc_rad_poa) AS diff_porc_rad_poa,
                    STDDEV(diff_porc_rad_poa) AS std_diff_porc_rad_poa,
                    AVG(diff_rad_poa) AS diff_rad_poa,
                    STDDEV(diff_rad_poa) AS std_diff_rad_poa,
                    AVG(rad_celda1) AS rad_celda1,
                    AVG(rad_celda2) AS rad_celda2, 
                    AVG(temp_amb) AS temp_amb, 
                    AVG(temp_panel1) AS temp_panel1,
                    AVG(temp_panel2) AS temp_panel2, 
                    AVG(cloud_impact) AS cloud_impact,
                    STDDEV(cloud_impact) AS std_cloud_impact,
                    MAX(cloud_impact) - MIN(cloud_impact) AS max_diff_cloud_impact,
                    AVG(diff_porc_cloud_impact) AS diff_porc_cloud_impact,
                    STDDEV(diff_porc_cloud_impact) AS std_diff_porc_cloud_impact,
                    AVG(diff_cloud_impact) AS diff_cloud_impact,
                    STDDEV(diff_cloud_impact) AS std_diff_cloud_impact,
                    BOOL_OR(daylight) AS daylight
                FROM {schema_name}.meteo
                JOIN stats
                    ON stats.datetime_utc = meteo.datetime_utc
                        AND stats.dispositivo_id = meteo.dispositivo_id
                    WHERE daylight = true
                GROUP BY meteo.dispositivo_id, datetime_utc_rounded
            )
            SELECT 
                inv.id,
                inv.dispositivo_id,
                inv.entrada_id,
                inv.datetime_utc_rounded as datetime_utc, 
                potencia_act,  
                num_strings, 
                rad_poa,
                std_rad_poa,
                max_diff_rad_poa,
                diff_porc_rad_poa,
                std_diff_porc_rad_poa,
                diff_rad_poa,
                std_diff_rad_poa,
                rad_celda1, 
                rad_celda2, 
                temp_amb, 
                temp_panel1, 
                temp_panel2,
                cloud_impact,
                std_cloud_impact,
                max_diff_cloud_impact,
                diff_porc_cloud_impact,
                std_diff_porc_cloud_impact,
                diff_cloud_impact,
                std_diff_cloud_impact,
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
            ORDER BY 4, 2, 3;"""
else:
    main_query = f"""
        WITH f AS (
            SELECT *
                FROM {schema_name}.inversores
                WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 5 = 0)
                    AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                    AND (EXTRACT(MONTH FROM datetime_utc) != 10)
                ORDER BY datetime_utc)
        SELECT f.id, f.dispositivo_id, det.entrada_id, f.datetime_utc, potencia_act, num_strings, 
                rad_poa, rad_celda1, rad_celda2, temp_amb, temp_panel1, temp_panel2, cloud_impact, 
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
del chunks, chunk
gc.collect()

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

# # Búsqueda de outliers basándose en la potencia activa y la potencia solar
# num_strings_inv = f"""SELECT dispositivo_id, SUM(num_strings) as num_strings 
#                         FROM {schema_name}.distrib_inversores
#                         GROUP BY dispositivo_id;"""
# num_strings_inv = pd.read_sql_query(num_strings_inv, engine).sort_values(by="dispositivo_id")
# potencia_df = pd.merge(main_df.reset_index()[["dispositivo_id", "datetime_utc", "potencia_act", "rad_poa"]], num_strings_inv, on="dispositivo_id").set_index(["dispositivo_id", "datetime_utc"])
# potencia_df["potencia_solar"] = potencia_df["rad_poa"] * potencia_df["num_strings"] * num_mod_string * sup_mod
# potencia_df["outlier_solar"] = np.where(potencia_df["potencia_act"] > 0.20 * potencia_df["potencia_solar"], True, False)
# main_df = main_df.merge(potencia_df[["outlier_solar"]], left_index=True, right_index=True, how="inner")
# print(f"Registros descartados por outlier de potencia: {main_df[main_df['outlier_solar'] == True].shape[0]}")
# main_df = main_df[main_df["outlier_solar"] == False].drop(columns = ["outlier_solar"])
# del potencia_df, num_strings_inv
# gc.collect()

# Manipulación de variables meteorológicas
main_df["rad_diff"] = (main_df["rad_celda1"] - main_df["rad_celda2"]) / main_df["rad_celda1"]
main_df["temp_panel"] = (((main_df["temp_panel1"] + main_df["temp_panel2"]) / 2)) - main_df["temp_amb"]
main_df = main_df.drop(columns = ["rad_celda1", "rad_celda2", "temp_panel1", "temp_panel2"])
# Descarte de registros con diferencias de radiación anómalas
rad_diff_mean = main_df["rad_diff"].mean()
rad_diff_std = main_df["rad_diff"].std()
outliers_rad_diff = main_df[(main_df["rad_diff"] < rad_diff_mean - rad_diff_std) | (main_df["rad_diff"] > rad_diff_mean + 6 * rad_diff_std)].shape[0]
main_df = main_df[(main_df["rad_diff"] > rad_diff_mean - rad_diff_std) & (main_df["rad_diff"] < rad_diff_mean + 6 * rad_diff_std)]
print(f"Registros descartados por diferencias de radiación anómalas: {outliers_rad_diff}")

# Manipulación de variables de consigna
main_df["motivo"] = main_df["motivo"].apply(lambda x: 0 if x == 0 else (2 if x == 7 else 1))  # Clusterización?
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
    # Carga del modelo de potencia y umbrales de radiación
    print(f"Dispositivo {inv_id}")
    power_model = power_models[f"Inversor_{inv_id-20}"]["model"]
    metricas_rad = power_models[f"Inversor_{inv_id-20}"]["RMSE_rad"]
    
    # Separación del dataframe propio del inversor
    disp_df = main_df[main_df.index.get_level_values("dispositivo_id") == inv_id].copy()
    disp_df = disp_df.dropna()
    
    # Predicción de potencia en función de las variables meteorológicas para descartar registros anómalos
    power_df = disp_df[["rad_poa", "rad_diff", "temp_amb", "temp_panel", "cloud_impact", "motivo", "consigna_pot_act_planta", "dia_año_sen", "hora_seg_sen"]]
    power_prep_df = power_model.named_steps["preprocessor"].transform(power_df)
    dtrain = xgb.DMatrix(power_prep_df)
    disp_df.loc[:, "potencia_act_pred"] = power_model.named_steps["regressor"].predict(dtrain)
    
    # Descarte de registros con potencia anómala
    rad_bins = [0, 100, 250, 500, np.inf]
    rad_labels = ["0-100", "100-250", "250-500", "500+"]
    disp_df["rad_bin"] = pd.cut(disp_df["rad_poa"], bins = rad_bins, labels = rad_labels)
    disp_df["metrica_rad"] = disp_df["rad_bin"].map(metricas_rad)
    disp_df["outlier_potencia"] = np.abs(disp_df["potencia_act"] - disp_df["potencia_act_pred"]) > 1 * np.asarray(disp_df["metrica_rad"])
    print(f"Registros descartados por potencia anómala utilizando modelo regresor: {disp_df[disp_df['outlier_potencia']].shape[0]}")
    disp_df = disp_df[~disp_df["outlier_potencia"]]
    disp_df = disp_df.drop(columns = ["rad_bin", "metrica_rad", "outlier_potencia", "potencia_act_pred"])

    # Separación de registros de entrenamiento y validación
    train_df, validation_df = train_test_split(disp_df, test_size = 0.2, random_state = 42)
    print(f"\tRegistros de entrenamiento: {train_df.shape[0]}")
    print(f"\tRegistros de validación: {validation_df.shape[0]}")

    # Separación de input y target
    target_columns = train_df.filter(like="amp_dc").columns.tolist()
    y = train_df[target_columns].copy()
    y_val = validation_df[target_columns].copy()
    X = train_df.drop(columns = target_columns+["id",
                                    "potencia_act",
                                    "num_strings",
                                    "dia_año",
                                    "hora_seg"
                                    ])
    X_val = validation_df.drop(columns = target_columns+["id",
                                    "potencia_act",
                                    "num_strings",
                                    "dia_año",
                                    "hora_seg"
                                    ])

    # Estandarización/normalización de variables numéricas y codificación de variables categóricas
    perc_attr = ['cloud_impact', 'consigna_pot_act_planta']
    std_attr = ['std_rad_poa', 'max_diff_rad_poa', 'diff_porc_rad_poa', 'std_diff_porc_rad_poa', 'diff_rad_poa', 'std_diff_rad_poa', 'std_cloud_impact', 'max_diff_cloud_impact', 'diff_porc_cloud_impact', 'std_diff_porc_cloud_impact', 'diff_cloud_impact', 'std_diff_cloud_impact']
    norm_attr = ['rad_poa', 'temp_amb', 'rad_diff', 'temp_panel']

    cat_attr = ['motivo']

    transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])
    transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 
    transformador_numerico_norm = Pipeline([('minmax_scaler', MinMaxScaler())])

    preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                   ('norm', transformador_numerico_norm, norm_attr),
                                                ('std', transformador_numerico_std, std_attr)],
                                remainder='passthrough')
    X_prep = preprocessor.fit_transform(X)

    dtrain = xgb.DMatrix(X_prep, label=y)
    params = {
        'objective': 'reg:squarederror',
        'tree_method': "hist",
        'multi_strategy': "multi_output_tree",
        'random_state': 42,
        'n_jobs': -1
    }
    multioutput_model = xgb.train(params, dtrain) 

    pipeline_model = Pipeline([('preprocessor', preprocessor),
                            ('regressor', multioutput_model)])

    consulta_sql = f"""SELECT num_strings
                    FROM {schema_name}.distrib_inversores
                    WHERE dispositivo_id = {inv_id};"""
    num_strings = pd.read_sql_query(consulta_sql, engine).values.reshape(1, -1)

    X_val_prep = xgb.DMatrix(pipeline_model.named_steps['preprocessor'].transform(X_val))
    y_pred_val = pd.DataFrame(pipeline_model.named_steps['regressor'].predict(X_val_prep)).rename(columns={i: "y_pred_"+str(i+1) for i in pd.DataFrame(pipeline_model.named_steps['regressor'].predict(X_val_prep)).columns})
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
    prediction_df = pd.merge(prediction_df, X_val["rad_poa"], left_on=["dispositivo_id","datetime_utc"], right_index=True ,how="inner")

    # Cálculo de las métricas de error
    rmse_score = round(mse(prediction_df["amp_dc"], prediction_df["y_pred"], squared=False), 2)
    mae_score = round(mae(prediction_df["amp_dc"], prediction_df["y_pred"]), 2)
    r2_score = round(r2(prediction_df["amp_dc"], prediction_df["y_pred"]), 3)
    nrmse_score = round(rmse_score / (prediction_df["amp_dc"].mean()), 3)
    metricas = {"RMSE": rmse_score, "MAE": mae_score, "R2": r2_score}
    print(f"\tMétricas de error:\n\t\tRMSE: {rmse_score}\n\t\tMAE: {mae_score}\n\t\tR2: {r2_score}\n\t\tNRMSE: {nrmse_score}")

    # Guardado del modelo
    algoritmo = pipeline_model.named_steps["regressor"].__class__.__name__
    if promediado:
        path = os.path.join(root_path, f"Modelos/entrada_amperaje_multi_promediado/Inversor_{inv_id - 20}/Nuevos/{algoritmo}-{pd.Timestamp.now()}/")
    else:
        path = os.path.join(root_path, f"Modelos/entrada_amperaje_multi/Inversor_{inv_id - 20}/Nuevos/{algoritmo}-{pd.Timestamp.now()}/")
    os.makedirs(path)
    print(path)
    with open(path+'model.model', "wb") as archivo_salida:
        pickle.dump(pipeline_model, archivo_salida)

    # Cálculo de RMSE por rangos de irradiancia
    rad_bins = [0, 100, 250, 500, np.inf]
    rad_labels = ["0-100", "100-250", "250-500", "500+"]
    prediction_df["rad_bin"] = pd.cut(prediction_df["rad_poa"], bins = rad_bins, labels = rad_labels)
    rmse_rad = prediction_df.groupby("rad_bin").apply(lambda x: mse(x["amp_dc"], x["y_pred"], squared=False)).to_dict()

    with open(path+'informe_modelo.json', 'w') as archivo_json:
        informe = {"promediado": promediado,
                    "normalizacion": normalizacion,
                    "optimizacion": optimizacion,
                    "por_fases": stage,
                    "metricas": metricas,
                    "metricas_rad": rmse_rad,
                    # "hiperparametros": pipeline_model.named_steps['regressor'].save_config(),
                    "training_input_description": train_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                    "training_target_description": (train_df.filter(like="amp_dc") * num_strings).describe().to_dict(),
                    }
        if promediado:
            informe["intervalo_min"] = intervalo
        json.dump(informe, archivo_json)

    # Generación de gráficos: comparativa de valores reales y predichos, histograma de diferencias
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
    y_reset = y.reset_index()
    y_melted = y_reset.melt(id_vars=["dispositivo_id", "datetime_utc"], var_name="entrada_id", value_name="amp_dc").sort_values(by=["dispositivo_id", "datetime_utc"])
    y_melted["entrada_id"] = y_melted["entrada_id"].str.split("_").str[2].astype(int)
    y_melted = y_melted.merge(X, on=["dispositivo_id", "datetime_utc"], how="inner")
    sns.heatmap(y_melted[perc_attr + std_attr + norm_attr + ["amp_dc"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Matriz de correlación")
    plt.savefig(path + "correlacion.png")
    del y_reset, y_melted
    gc.collect()

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
