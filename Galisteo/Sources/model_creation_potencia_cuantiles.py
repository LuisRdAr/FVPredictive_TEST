# -*- coding: utf-8 -*-
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
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from functools import partial
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, \
                            r2_score as r2
from tqdm import tqdm
import time
import gc


def objective(space, train_set, cv_folds, device='gpu', quantiles=[0.05, 0.5, 0.95]):
    n_estimators = int(space.pop('n_estimators'))
    params = {'device': device,
            'objective': 'reg:quantileerror',
            'quantile_alpha': np.array([0.01, 0.5, 0.99]),
            'tree_method': "hist",
            'random_state': 42,
            'n_jobs': -1}
    params = {**params, **space}
    if 'max_depth' in params:
        params['max_depth'] = int(params['max_depth'])
    if 'min_child_weight' in params:
        params['min_child_weight'] = max(1, int(params['min_child_weight']))

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=21)
    cv_results = []
    for train_index, test_index in kf.split(train_set[0]):
        X_train, X_test = train_set[0][train_index], train_set[0][test_index]
        y_train, y_test = train_set[1].iloc[train_index], train_set[1].iloc[test_index]

        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
        dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref = dtrain)

        evals_result = {}
        model = xgb.train(params, dtrain, num_boost_round=n_estimators,
                          evals=[(dtrain, 'train'), (dtest, 'test')],
                          early_stopping_rounds=100, evals_result=evals_result, verbose_eval=False)
        best_iteration = model.best_iteration
        loss = evals_result['test']['quantile'][best_iteration]  # Obtenemos el último valor de la pérdida de cuantil

        cv_results.append(loss)

    return {'loss': np.mean(cv_results), 'status': STATUS_OK, 'n_rounds': best_iteration}

def optimize_hyperparameters(space, train_set, STALL_LIMIT = 10, MAX_EVALS_PER_RUN = 250, gamma_algo = 1, cv_folds = 5, verbose = False):
    best_loss = np.inf
    trials = Trials()
    stall_counter = 0
    num_evals = 0
    upper_limit = (MAX_EVALS_PER_RUN * (STALL_LIMIT - 1)) * 10

    while stall_counter < STALL_LIMIT and num_evals < upper_limit:
        best = fmin(fn=lambda space: objective(space, train_set = train_set, cv_folds = cv_folds), 
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
        gamma_algo = max(0.25, gamma_algo - (0.5/(2 * STALL_LIMIT)))

    best_params["n_estimators"] = trials.best_trial['result']['n_rounds']
    print(f"\tNúmero de evaluaciones realizadas: {num_evals}")
    print(f"\tBest params: {best_params}")
    print(f"\tBest trial: {trials.best_trial}")
    print(f"\tBest loss: {best_loss}", end="\n\n")

    return best_params


def discriminador(row):
    mean = np.mean(row)
    std = np.std(row)
    threshold = 3 * std
    outlier = np.abs(row - mean) > threshold
    return outlier


if __name__ == '__main__':

    ### Configuración de la ejecución
    print("Configuración para el entrenamiento de modelos de predicción de potencia activa de salida en inversores fotovoltaicos:", end="\n\n")
    model_name = "XGBRegressor"

    valid_responses = ["y", "n"]
    normalizacion = ""
    while normalizacion not in valid_responses:
        normalizacion = input("¿Desea normalizar el target? (Y/N): ").lower()
    normalizacion = normalizacion == "y"

    valid_intervals = [5, 10, 15, 20, 30]
    intervalo = 0
    while intervalo <= 0:
        intervalo = int(input("Introduzca el intervalo de tiempo para el promediado (5, 10, 15, 20, 30): "))
        if intervalo not in valid_intervals:
            print("El intervalo de tiempo introducido no es válido")
            intervalo = 0

    optimizacion = ""
    while optimizacion not in valid_responses:
        optimizacion = input("¿Desea optimizar el modelo? (Y/N): ").lower()
    optimizacion = optimizacion == "y"

    valid_devices = ["cpu", "cuda"]
    device = ""
    if optimizacion & torch.cuda.is_available():
        while device not in valid_devices:
            device_bool = input("¿Desea utilizar GPU para el entrenamiento? (Y/N): ").lower()
            if device_bool == "y":
                device = "cuda"
            else:
                device = "cpu"
    else:
        device = 'cpu'

    print(f"\nLas opciones seleccionadas son: \nModelo: {model_name} \nNormalización: {normalizacion} \nOptimización: {optimizacion} \nEntrenamiento: {device}", end="\n\n")

    ### Carga de parámetros de conexión a la base de datos
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

    ### Carga de datos:
    ### La tabla inv contiene los datos de los inversores promediados por intervalos de tiempo gracias a date_trunc y floor
    ### La tabla stats devuelve la diferencia porcentual entre registros consecutivos de la irradiancia inclinada y el impacto de nubes
    ###     además de limitar el impacto de nubes a 100
    ### La tabla meteo contiene los datos meteorológicos promediados por intervalos de tiempo. Contiene el valor medio y desviación estandar
    ###     de la irradiancia inclinada, de la diferencia porcentual entre registros consecutivos de la irradiancia inclinada, el promedio del
    ###     impacto del soiling (diferencia porcentual entre la irradiancia de la celda 1 y la celda 2), la temperatura ambiente, la temperatura
    ###     media de los paneles, el impacto de nubes (valos medio invertido y desviación estandar), la diferencia porcentual entre registros 
    ###     consecutivos del impacto de nubes.
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
                WHERE (alarma = 0)
                    AND (estado = 6)
                GROUP BY datetime_utc_rounded, dispositivo_id, det.entrada_id 
            ),
            stats AS (
                SELECT
                    datetime_utc,
                    dispositivo_id,
                    COALESCE((rad_poa - LAG(rad_poa) OVER (PARTITION BY dispositivo_id ORDER BY datetime_utc)) / NULLIF(rad_poa, 0), 0) AS diff_rad_poa,
                    LEAST(cloud_impact, 100) AS cloud_impact,
                    COALESCE((LEAST(cloud_impact, 100) - LAG(LEAST(cloud_impact, 100)) OVER (PARTITION BY dispositivo_id ORDER BY datetime_utc)) / NULLIF(LEAST(cloud_impact, 100), 0), 0) AS diff_cloud_impact
                FROM {schema_name}.meteo
                ORDER BY 1,2
            ),
            met AS (
                SELECT 
                    date_trunc('hour', meteo.datetime_utc) + 
                    INTERVAL '{intervalo} min' * floor(date_part('minute', meteo.datetime_utc) / {intervalo}) as datetime_utc_rounded, 
                    meteo.dispositivo_id,
                    AVG(rad_poa) AS rad_poa, 
                    STDDEV(rad_poa)/NULLIF(AVG(rad_poa), 0) AS std_rad_poa,
                    AVG(diff_rad_poa) AS diff_rad_poa,
                    STDDEV(diff_rad_poa)/NULLIF(AVG(diff_rad_poa), 0) AS std_diff_rad_poa,
                    AVG((rad_celda1 - rad_celda2))/NULLIF(AVG(rad_celda1), 0) AS rad_soiling,
                    AVG(temp_amb) AS temp_amb, 
                    AVG((temp_panel1 + temp_panel2)/2)  AS temp_panel,
                    AVG(stats.cloud_impact/100) AS cloud_impact,
                    STDDEV(stats.cloud_impact/100)/NULLIF(AVG(stats.cloud_impact/100), 0) AS std_cloud_impact,
                    AVG(diff_cloud_impact) AS diff_cloud_impact,
                    STDDEV(diff_cloud_impact)/NULLIF(AVG(diff_cloud_impact), 0) AS std_diff_cloud_impact,
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
                diff_rad_poa,
                std_diff_rad_poa,
                rad_soiling, 
                temp_amb, 
                temp_panel, 
                cloud_impact,
                std_cloud_impact,
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
            ORDER BY 5, 2, 3, 4;"""
    chunksize = 100000
    chunks = pd.read_sql_query(main_query, engine, chunksize=chunksize)
    main_df = pd.DataFrame()
    for chunk in chunks:
        main_df = pd.concat([main_df, chunk], ignore_index = True)

    ### Normalización de la entrada de corriente continua, formateo de fechas y escalado de potencia
    print(f"Carga inicial de {main_df.shape[0]} registros")
    if normalizacion:
        print("Entrada de corriente continua normalizada según el número de strings")
        main_df["amp_dc"] = main_df["amp_dc"]/main_df["num_strings"]
    else:
        print("Entrada de corriente continua sin normalizar")
    main_df["datetime_utc"] = pd.to_datetime(main_df["datetime_utc"], utc = True)
    main_df["potencia_act"] = main_df["potencia_act"] * 1000

    ### Pivotado de las entradas de corriente continua. El dato se recibe con una fila por cada entrada del inversor
    ### y se pivotan las columnas para tener una fila por cada inversor y momento
    amp_df = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
    amp_df.columns = ["amp_dc_" + str(col) for col in amp_df.columns]
    print(f"Número de registros del dataframe tras pivotar: {amp_df.shape[0]}")

    ### Descarte de registros con corriente anómala aplicando un discriminador basado en la desviación estándar
    ### de las corrientes de entrada para cada instante e inversor
    amp_df["outlier"] = amp_df.apply(discriminador, axis=1).any(axis=1)
    n_corriente_outlier = amp_df[amp_df["outlier"]].shape[0]
    amp_df = amp_df[~amp_df["outlier"]].drop(columns="outlier")
    print(f"Registros descartados por corrientes anómalas ingresando en el inversor: {n_corriente_outlier}")

    ### Descarte de registros con corriente muy negativa aplicando el criterio del IQR e asignación de corriente 0 a los valores negativos restantes
    q1 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.25)
    q3 = main_df[main_df['amp_dc'] < 0]['amp_dc'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    n_corriente_neg = amp_df[amp_df < lower_bound].dropna(how='all').shape[0]
    amp_df = amp_df[amp_df >= lower_bound].dropna(how='any')
    amp_df[(amp_df >= lower_bound) & (amp_df <= 0)] = 0
    main_df = main_df.drop(columns=["entrada_id", "amp_dc"]).drop_duplicates(subset=["id", "datetime_utc"]).set_index(["dispositivo_id", "datetime_utc"])
    main_df = main_df.merge(amp_df, left_index=True, right_index=True, how="inner")
    del amp_df
    gc.collect()
    print(f"Registros descartados por corriente negativa: {n_corriente_neg}")

    ### Descarte de registros con potencia activa negativa
    n_potencia_neg = main_df[main_df["potencia_act"] < 0].shape[0]
    main_df = main_df[main_df["potencia_act"] >= 0]
    print(f"Registros descartados por potencia activa negativa: {n_potencia_neg}")

    ### Búsqueda de outliers basándose en la potencia activa y la potencia solar
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

    ### Descarte de registros con diferencias de radiación anómalas para el soiling. Se descartan los registros donde el panel sucio recibe mucha más radiación que el limpio
    ### y los registros donde la diferencia de radiación es muy anómala
    rad_diff_mean = main_df["rad_soiling"].mean()
    rad_diff_std = main_df["rad_soiling"].std()
    outliers_rad_diff = main_df[(main_df["rad_soiling"] < rad_diff_mean - rad_diff_std) | (main_df["rad_soiling"] > rad_diff_mean + 6 * rad_diff_std)].shape[0]
    main_df = main_df[(main_df["rad_soiling"] > rad_diff_mean - rad_diff_std) & (main_df["rad_soiling"] < rad_diff_mean + 6 * rad_diff_std)]
    print(f"Registros descartados por medidas de soiling anómalas: {outliers_rad_diff}")

    ### Manipulación de variables de consigna
    # main_df["motivo"] = main_df["motivo"].apply(lambda x: 0 if x == 0 else (2 if x == 7 else 1))
    main_query = f"""
        SELECT MAX(consigna_pot_act_planta)
            FROM {schema_name}.ree AS ree;"""
    max_pot_act = pd.read_sql_query(main_query, engine).values[0][0]
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_planta"] / max_pot_act   # Valor normalizado

    ### Variables de temperatura normalizadas según la irradiancia
    main_df["temp_amb"] = main_df["temp_amb"] + 273.15
    main_df["temp_panel"] = main_df["temp_panel"] + 273.15
    main_df["temp_panel_des"] = main_df["temp_panel"]/main_df["temp_amb"]

    ### Asignación de variables temporales
    main_df.loc[:, "dia_año"] = main_df.index.get_level_values("datetime_utc").dayofyear
    main_df.loc[:, "dia_año_sen"] = np.sin(main_df["dia_año"] * (2*np.pi/365))
    main_df.loc[:, "hora_seg"] = main_df.index.get_level_values("datetime_utc").hour * 3600 + \
                            main_df.index.get_level_values("datetime_utc").minute * 60 + \
                            main_df.index.get_level_values("datetime_utc").second
    main_df.loc[:, "hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))

    # Calcula la media móvil para 'temp_amb_des' y 'temp_panel_des'
    main_df = main_df.sort_index()
    main_df['rad_poa_mm_1h'] = main_df.groupby([main_df.index.get_level_values(0), main_df.index.get_level_values(1).date])['rad_poa'].rolling(4, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    main_df['rad_poa_avg_change_1h'] = main_df.groupby([main_df.index.get_level_values(0), main_df.index.get_level_values(1).date])['rad_poa'].diff().abs().rolling(4, min_periods=1).mean().fillna(0)
    main_df['rad_poa_avg_change_1h'] = main_df['rad_poa_avg_change_1h'] / main_df['rad_poa_mm_1h']

    ### Descarte de registros con valores anómalos en variables estadísticas
    rad_stat_attr = ["std_rad_poa", "diff_rad_poa", "std_diff_rad_poa", "rad_poa_avg_change_1h", "rad_poa_mm_1h"]
    cloud_stat_attr = ["std_cloud_impact", "diff_cloud_impact", "std_diff_cloud_impact"] 
    other_attr = ["temp_panel", "temp_panel_des"] 
    stat_attr = rad_stat_attr + cloud_stat_attr + other_attr
    ids = []
    n_rows_total = main_df.shape[0]
    for attr in stat_attr:
        mean = main_df[attr].mean()
        std = main_df[attr].std()
        ids.append(main_df[(main_df[attr] <= mean - 3 * std) | (main_df[attr] >= mean + 3 * std)]['id'].values)
        print(f"Registros para descartar por outlier en {attr}: {len(main_df[(main_df[attr] <= mean - 3 * std) | (main_df[attr] >= mean + 3 * std)]['id'].values)}")
    ids = list(set.union(*map(set, ids)))
    print(f"Registros descartados por outlier en variables estadísticas: {len(ids)}")
    main_df = main_df[~main_df["id"].isin(ids)]

    for inv_id in [21]:
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
        X = train_df.drop(columns = amp_columns+rad_stat_attr+cloud_stat_attr+other_attr+["id",
                                                                            "num_strings",
                                                                            "dia_año",
                                                                            "hora_seg",
                                                                            "potencia_act",
                                                                            "temp_amb",
                                                                            "motivo",
                                                                            "cloud_impact"
                                                                            ])
        X_val = validation_df.drop(columns = amp_columns+rad_stat_attr+cloud_stat_attr+other_attr+["id",
                                                                            "num_strings",
                                                                            "dia_año",
                                                                            "hora_seg",
                                                                            "potencia_act",
                                                                            "temp_amb",
                                                                            "motivo",
                                                                            "cloud_impact"
                                                                            ])

        # Estandarización/normalización de variables numéricas y codificación de variables categóricas
        perc_attr = ['consigna_pot_act_planta']
        std_attr = []#rad_stat_attr+cloud_stat_attr
        norm_attr = ['rad_poa','rad_soiling']#+other_attr
        cat_attr = []

        transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))]) # Introducir manualmente catergorías?
        transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 
        transformador_numerico_norm = Pipeline([('minmax_scaler', MinMaxScaler())])

        preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                    ('norm', transformador_numerico_norm, norm_attr),
                                                    ('std', transformador_numerico_std, std_attr)],
                                    remainder='passthrough')
        X_prep = preprocessor.fit_transform(X)
        train_set = [X_prep, y]

        if optimizacion:
            print("\tOptimización de los hiperparámetros")
            space = {'n_estimators': hp.quniform('n_estimators', 100, 5000, 5),
                    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
                    'max_depth': hp.quniform("max_depth", 3, 30, 1),
                    'gamma': hp.uniform ('gamma', 0, 25),
                    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
                    'subsample' : hp.uniform('subsample', 0.4, 1),
                    'reg_alpha' : hp.uniform('reg_alpha', 0, 50),
                    'reg_lambda' : hp.uniform('reg_lambda', 0, 50)}
            start_time = time.time()
            best_params = optimize_hyperparameters(space, train_set, STALL_LIMIT = 5, MAX_EVALS_PER_RUN = 50, cv_folds = 5, verbose = True)
            end_time = time.time()
            execution_time = end_time - start_time
            execution_hours = int(execution_time // 3600)
            execution_minutes = int((execution_time % 3600) // 60)
            execution_seconds = int(execution_time % 60)
            print(f"Tiempo de ejecución: {execution_hours} horas, {execution_minutes} minutos y {execution_seconds} segundos")

            final_params = {"device": device,
                        "objective": 'reg:quantileerror',
                        "quantile_alpha": np.array([0.05, 0.5, 0.95]),
                        "tree_method": "hist",
                        "random_state": 42,
                        "n_jobs": -1}
            final_params = {**final_params, **best_params}
            train_params = final_params.copy()
            final_params.pop('n_estimators')
            final_params['max_depth'] = int(final_params['max_depth'])
            final_params['min_child_weight'] = max(1, int(final_params['min_child_weight']))
            train_set = xgb.QuantileDMatrix(X_prep, label=y)
            prediction_model = xgb.train(final_params, train_set, num_boost_round = int(best_params['n_estimators']))

        else:
            train_set = xgb.QuantileDMatrix(X_prep, label=y)
            train_params = {'device': device,
                    'objective': 'reg:quantileerror',
                    'quantile_alpha': np.array([0.05, 0.5, 0.95]),
                    'tree_method': "hist",
                    'random_state': 42,
                    'n_jobs': -1}
            prediction_model = xgb.train(train_params, train_set) 
        
        pipeline_model = Pipeline([('preprocessor', preprocessor),
                            ('regressor', prediction_model)])

        prediction_df = y_val.copy()

        val_set = pipeline_model.named_steps['preprocessor'].transform(X_val)
        val_set = xgb.QuantileDMatrix(val_set, ref=train_set)
        
        prediction_df["y_pred"] = pipeline_model.named_steps['regressor'].predict(val_set)[:, 1]
        prediction_df["y_pred_inf"] = pipeline_model.named_steps['regressor'].predict(val_set)[:, 0]
        prediction_df["y_pred_sup"] = pipeline_model.named_steps['regressor'].predict(val_set)[:, 2]
        prediction_df["y_diff"] = prediction_df["potencia_act"] - prediction_df["y_pred"]
        prediction_df["intervalo"] = prediction_df["y_pred_sup"] - prediction_df["y_pred_inf"]

        rmse_score = round(mse(prediction_df["potencia_act"], prediction_df["y_pred"], squared = False),3)
        mae_score = round(mae(prediction_df["potencia_act"], prediction_df["y_pred"]),3)
        r2_score = round(r2(prediction_df["potencia_act"], prediction_df["y_pred"]),3)
        metricas = {"RMSE": rmse_score, "MAE": mae_score, "R2": r2_score}
        print(f"RMSE: {rmse_score}", 
                f"MAE: {mae_score}",
                f"R2: {r2_score}",
                sep = "\n")

        ### Creación del directorio para guardado del modelo, informe y figuras de apoyo
        algoritmo = pipeline_model.named_steps["regressor"].__class__.__name__
        columnas = [col_name.split("__")[1] for col_name in preprocessor.get_feature_names_out()]
        importance_scores = prediction_model.get_score(importance_type='gain')
        total_gain = np.array([v for k,v in importance_scores.items()]).sum()
        importancia = {k: v/total_gain for k, v in importance_scores.items()}
        path = os.path.join(root_path, f"Modelos/potencia_inversor_cuantiles/Inversor_{inv_id - 20}/Nuevos/{algoritmo}-{pd.Timestamp.now()}/")
        os.makedirs(path)
        with open(path+'model.model', "wb") as archivo_salida:
            pickle.dump(pipeline_model, archivo_salida)
        
        ### Gráfico de la distribución de los intervalos y márgenes de confianza
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.histplot(prediction_df["intervalo"], bins=100, kde=True, ax=ax, stat="percent")
        plt.title("Distribución de los intervalos de confianza")
        ax.text(0.05, 0.975, f"RMSE: {rmse_score}\nMAE: {mae_score}\nR2: {r2_score}", transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        ax.text(0.95, 0.975, f"Media: {prediction_df['intervalo'].mean()}\nMediana: {prediction_df['intervalo'].quantile([0.5]).values[0]}", 
                transform=ax.transAxes, fontsize=10,verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5));

        sns.histplot(prediction_df["y_pred_sup"] - prediction_df["y_pred"], bins=100, kde=True, ax=ax, stat="percent")
        upper_margin = prediction_df["y_pred_sup"] - prediction_df["y_pred"]
        ax.text(0.95, 0.90, f"Media: {upper_margin.mean()}\nMediana: {upper_margin.quantile([0.5]).values[0]}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
        plt.title("Distribución de los márgenes superiores de confianza")

        sns.histplot(prediction_df["y_pred"] - prediction_df["y_pred_inf"], bins=100, kde=True, ax=ax, stat="percent")
        lower_margin = prediction_df["y_pred"] - prediction_df["y_pred_inf"]
        ax.text(0.95, 0.825, f"Media: {lower_margin.mean()}\nMediana: {lower_margin.quantile([0.5]).values[0]}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
        plt.title("Distribución de los márgenes e intervalo de confianza")
        plt.legend(["Intervalo de confianza", "Márgen superior", "Márgen inferior"], loc = "upper center")
        plt.savefig(path + "distribucion_intervalos.png")

        ### Generación de gráficos: comparativa de valores reales y predichos, histograma de diferencias y matriz de correlación
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
        ax = sns.histplot(prediction_df["y_diff"], kde=True, stat='percent')
        ax.axvline(x=0, color='black', linestyle='--', linewidth = 0.35, label='x=0')
        ax.text(0.95, 0.95, f"Media: {prediction_df['y_diff']}\nDesviación estándar: {prediction_df['y_diff'].std()}", 
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

        ### Generación de gráficos: comparativa de RMSE y RMSE relativo por hora
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
        if max(rrmse_list) - min(rrmse_list) > 0.25:                    ### REPLANTEAR ESTE EJE
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

        ### Guardado de los resultados en un archivo JSON
        with open(path+'informe_modelo.json', 'w') as archivo_json:
            informe = {"normalizacion": normalizacion,
                        "optimizacion": optimizacion,
                        "intervalo_min": intervalo,
                        "metricas": metricas,
                        "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia.values())}.items(), key=lambda item: item[1], reverse=True)),
                        "hiperparametros": {k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in train_params.items() if v is not None},
                        "training_input_description": train_df[perc_attr + std_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                        "training_target_description": train_df["potencia_act"].describe().to_dict(),
                        }
            json.dump(informe, archivo_json)