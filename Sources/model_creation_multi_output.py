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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
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

def discriminador(x):

    amp_dc_norm = x["amp_dc"]
    amp_dc_sum = amp_dc_norm.sum() - amp_dc_norm
    amp_dc_std = 3 * amp_dc_sum.std()
    amp_dc_mean = amp_dc_sum / (x["amp_dc"].count() - 1)
    outlier = []
    for pair in zip(amp_dc_norm, amp_dc_mean):
        if (pair[0] > pair[1] + amp_dc_std) or (pair[0] < pair[1] - amp_dc_std):
            outlier.append(True)
        else:
            outlier.append(False)
    results = x[["id", "datetime_utc", "dispositivo_id", "entrada_id"]]
    results["outlier_2"] = outlier
    results = results.reset_index(drop=True)
    return results

if __name__ == "__main__":
    # Carga de los parámetros para conexión a la base de datos
    root_path = "/home/luis/Desktop/venv_test/test_env/FVPREDICTIVE/Galisteo/"
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
    inv_id = 22
    # Carga de los datos de entrenamiento
    print(f"Entrenamiento para el inversor {inv_id - 20}")
    if schema_name == "galisteo":
        main_query = f"""
            WITH f AS (
                SELECT *
                    FROM {schema_name}.inversores
                    WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 5 = 0)
                        AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                        AND (EXTRACT(MONTH FROM datetime_utc) != 10)
                        AND (alarma = 0)
                        AND (dispositivo_id = {inv_id})
                        AND (estado = 6)
                    ORDER BY datetime_utc)
            SELECT f.id, f.dispositivo_id, det.entrada_id, f.datetime_utc, potencia_act, lim_act, num_strings, 
                    rad_poa, rad_hor, rad_celda1, rad_celda2, temp_amb, temp_panel1, temp_panel2, vel_viento,
                    elevation, azimuth, cloud_impact, motivo, consigna_pot_act_planta, amp_dc
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
                ORDER BY 3, 1, 2;"""
    elif schema_name == "bonete":   
        main_query = f"""
            WITH inv AS (
                SELECT *
                    FROM {schema_name}.inversores
                    WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 15 = 0)
                        AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                        AND (alarma = 0)
                        AND (dispositivo_id = {inv_id})
                        AND (estado = 6)),
                met AS (
                    SELECT datetime_utc, AVG(rad_poa) AS rad_poa, AVG(rad_hor) AS rad_hor, AVG(rad_celda1) AS rad_celda1,
                            AVG(rad_celda2) AS rad_celda2, AVG(temp_amb) AS temp_amb, AVG(temp_panel1) AS temp_panel1,
                            AVG(temp_panel2) AS temp_panel2, AVG(vel_viento) AS vel_viento, AVG(elevation) AS elevation,
                            AVG(azimuth) AS azimuth, AVG(cloud_impact) AS cloud_impact, BOOL_OR(daylight) AS daylight
                        FROM {schema_name}.meteo
                        GROUP BY datetime_utc)
            SELECT inv.id, inv.dispositivo_id, det.entrada_id, inv.datetime_utc, potencia_act, lim_act, num_strings, 
                    rad_poa, rad_hor, rad_celda1, rad_celda2, temp_amb, temp_panel1, temp_panel2, vel_viento,
                    elevation, azimuth, cloud_impact, motivo, consigna_pot_act_ree, amp_dc
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
                ORDER BY 3, 1, 2;"""
    chunksize = 100000
    chunks = pd.read_sql_query(main_query, engine, chunksize=chunksize)
    main_df = pd.DataFrame()
    for chunk in chunks:
        main_df = pd.concat([main_df, chunk], ignore_index = True)

    # Normalización de la entrada de corriente continua, formateo de fechas y escalado de potencia
    print(f"\tCarga inicial de {main_df.shape[0]} registros")
    normalizacion = True
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
    main_df = main_df[~(main_df["id"].isin(main_df[(main_df["amp_dc"] < -0.1)]["id"].unique()))]
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
    main_df["dia_año_cos"] = np.cos(main_df["dia_año"] * (2*np.pi/365))
    main_df["hora_seg"] = main_df["datetime_utc"].dt.hour * 3600 + \
                            main_df["datetime_utc"].dt.minute * 60 + \
                            main_df["datetime_utc"].dt.second
    main_df["hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))
    main_df["hora_seg_cos"] = np.cos(main_df["hora_seg"] * (2*np.pi/86400))
    main_df = main_df.dropna()

    # Búsqueda de outliers en la corriente continua basándose en la desviación dentro del comportamiento del inversor en un instante dado
    outliers = main_df.groupby(['datetime_utc', 'dispositivo_id']).apply(discriminador).reset_index(drop=True)
    print(f"\tRegistros descartados por outlier de corriente: {main_df[(main_df['id'].isin(outliers[outliers['outlier_2'] == True]['id'].unique()))].shape[0]}")
    main_df = main_df[(main_df["id"].isin(outliers[outliers["outlier_2"] == False]["id"].unique()))]

    # Escalado de variables porcentuales
    main_df[['lim_act', 'cloud_impact']] = main_df[['lim_act', 'cloud_impact']].apply(lambda x: x/100)

    # Separación del conjunto de validación
    validation_df = main_df[main_df["datetime_utc"].dt.month == 9]
    main_df = main_df[main_df["datetime_utc"].dt.month != 9]
    print(f"\tRegistros de entrenamiento: {main_df.shape[0]}")
    print(f"\tRegistros de validación: {validation_df.shape[0]}")

    # Agrupación de las entradas de corriente continua por inversor y por hora en una tabla dinámica
    pivot_table = main_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
    pivot_table.columns = ["amp_dc_" + str(col) for col in pivot_table.columns]
    y = pivot_table

    pivot_table_val = validation_df.pivot(index=["dispositivo_id", "datetime_utc"], columns='entrada_id', values='amp_dc')
    pivot_table_val.columns = ["amp_dc_" + str(col) for col in pivot_table_val.columns]
    y_val = pivot_table_val

    # Descarte de variables que no se usarán en el entrenamiento
    processed_df = main_df.drop(columns = ["id",
                                    "datetime_utc",
                                    "dispositivo_id", 
                                    "entrada_id",
                                    "dia_año",
                                    "hora_seg",
                                    "potencia_act",
                                    "outlier_1",
                                    "num_strings"
                                    ])
    processed_val_df = validation_df.drop(columns = ["id",
                                    "datetime_utc",
                                    "dispositivo_id", 
                                    "entrada_id",
                                    "dia_año",
                                    "hora_seg",
                                    "potencia_act",
                                    "outlier_1",
                                    "num_strings"
                                    ])

    # Descarte del target y de filas duplicadas (una única fila por inversor y hora)
    X = processed_df.drop(columns = ["amp_dc"]).drop_duplicates()
    X_val = processed_val_df.drop(columns = ["amp_dc"]).drop_duplicates()
    print(f"Número de registros tras colapsar en dataframe multi-output: {X.shape[0]}")

    # Estandarización/normalización de variables numéricas y codificación de variables categóricas
    perc_attr = ['lim_act', 'cloud_impact']
    std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'vel_viento', 'rad_diff', 'temp_panel']
    degree_attr = ['azimuth', 'elevation']
    cat_attr = ['motivo', 'consigna_pot_act_planta']

    transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])                         # Introducir manualmente catergorías?
    transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 
    transformador_numerico_deg = Pipeline([('degree_scaler', degree_scaler())])

    preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                    ('std', transformador_numerico_std, std_attr),
                                                    ('deg', transformador_numerico_deg, degree_attr)],
                                    remainder='passthrough')
    
    # Spliteo, preprocesado y entrenamiento del modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    X_val_prep = preprocessor.transform(X_val)

    def objective(space, use_cv = False):
        model = XGBRegressor(objective='reg:squarederror', 
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
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            loss = -np.mean(scores)
        else:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            y_val_pred = model.predict(X_val)
            loss = mse(y_val, y_val_pred)
        
        return {'loss':loss, 'status': STATUS_OK}

    space ={
        'max_depth': hp.quniform("x_max_depth", 3, 20, 1),
        'n_estimators': hp.quniform('x_n_estimators', 100, 2000, 1),
        'learning_rate': hp.uniform('x_learning_rate', 0.01, 0.2),
        'gamma': hp.uniform ('x_gamma', 0, 1),
        'min_child_weight' : hp.quniform('x_min_child', 1, 30, 1),
        'subsample' : hp.uniform('x_subsample', 0.5, 1),
        'reg_alpha' : hp.uniform('x_reg_alpha', 0, 20),
        'reg_lambda' : hp.uniform('x_reg_lambda', 0, 10)
    }

    trials = Trials()
    use_cv = True
    if use_cv:
        print("Optimización de hiperparámetros con cross-validation")
    else:
        print("Optimización de hiperparámetros sobre conjunto de validación")
    best = fmin(fn=lambda space: objective(space, use_cv=use_cv), space=space, algo=tpe.suggest, max_evals=1000, trials=trials)
    print(f"Mejor combinación de hiperparámetros:\n{best}")
    
    # Entrenamiento del modelo con los mejores hiperparámetros
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            #('regressor', MultiOutputRegressor(RandomForestRegressor(n_jobs=-1)))
                            ('regressor', XGBRegressor(objective='reg:squarederror', 
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
                                                       subsample = best["x_subsample"]))
                            ])
    model.fit(X_train, y_train)

    # Evaluación del modelo sobre conjunto de validación
    consulta_sql = f"""SELECT num_strings
                    FROM {schema_name}.distrib_inversores
                    WHERE dispositivo_id = {inv_id};"""
    num_strings = pd.read_sql_query(consulta_sql, engine).values.reshape(1, -1)
    y_pred_val = pd.DataFrame(model.predict(X_val)).rename(columns={i: "y_pred_"+str(i+1) for i in pd.DataFrame(model.predict(X_val)).columns})
    if normalizacion:
        y_pred_val = y_pred_val * num_strings
        y_val_reesc = y_val * num_strings
    else:
        y_pred_val = y_pred_val
        y_val_reesc = y_val

    target_pred_df = pd.concat([X_val.reset_index().rename(columns={"index": "id"})[["id"]], y_pred_val], axis=1) \
                    .merge(validation_df[["datetime_utc", "dispositivo_id"]], left_on="id", right_index = True) \
                        .set_index(["id"])
    target_pred_df = target_pred_df.melt(id_vars=["datetime_utc", "dispositivo_id"], var_name="entrada_id", value_name="amp_dc")
    target_pred_df["entrada_id"] = target_pred_df["entrada_id"].str.split("_").str[2].astype(int)

    y_val_reesc = y_val * num_strings
    y_val_reesc = y_val_reesc.reset_index().melt(id_vars=["datetime_utc", "dispositivo_id"], var_name="entrada_id", value_name="amp_dc")
    y_val_reesc["entrada_id"] = y_val_reesc["entrada_id"].str.split("_").str[2].astype(int)
    prediction_df = target_pred_df.merge(y_val_reesc, on=["datetime_utc", "dispositivo_id", "entrada_id"], suffixes=("_pred", "_real"))

    # Cálculo de las métricas de error
    rmse_score = round(mse(prediction_df["amp_dc_real"], prediction_df["amp_dc_pred"], squared=False), 2)
    mae_score = round(mae(prediction_df["amp_dc_real"], prediction_df["amp_dc_pred"]), 2)
    r2_score = round(r2(prediction_df["amp_dc_real"], prediction_df["amp_dc_pred"]), 3)
    metricas = {"RMSE": rmse_score, "MAE": mae_score, "R2": r2_score}

    # Guardado del modelo y de las métricas
    algoritmo = model.named_steps["regressor"].__class__.__name__
    path = os.path.join(root_path, f"Modelos/entrada_amperaje_multi/Inversor_{inv_id - 20}/Repositorio/{algoritmo}-{pd.Timestamp.now()}/")
    os.makedirs(path)
    with open(path+'model.model', "wb") as archivo_salida:
        pickle.dump(model, archivo_salida)
    with open(path+'informe_modelo.json', 'w') as archivo_json:
        informe = {"metricas": metricas,
                    "hiperparametros": {k:v for k,v in model.named_steps["regressor"].get_params().items() if v != None},
                    "input_description": processed_df[perc_attr + std_attr + degree_attr].describe().loc[["mean", "std", "min", "max"]].to_dict(),
                    "target": (y_train*num_strings).reset_index().melt(id_vars=["datetime_utc", "dispositivo_id"], var_name="entrada_id", value_name="amp_dc")[["amp_dc"]].describe().to_dict()}
        json.dump(informe, archivo_json)
    
    # Generación de gráficos: comparativa de valores reales y predichos, histograma de diferencias y matriz de correlación
    y_test_sampled, _,y_pred_sampled, _ = train_test_split(prediction_df["amp_dc_real"], prediction_df["amp_dc_pred"], train_size = 0.25)
    plt.figure()
    plt.tight_layout()
    plt.scatter(y_test_sampled, y_pred_sampled, marker = ".")
    plt.plot([min(y_test_sampled), max(y_test_sampled)], [min(y_test_sampled), max(y_test_sampled)], color='black', linestyle='-', linewidth=1)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("Comparación de valores reales y predichos")
    plt.savefig(path + "scatter_validacion.png")
    plt.close()

    plt.figure()
    plt.tight_layout()
    ax = sns.histplot(prediction_df["amp_dc_real"] - prediction_df["amp_dc_pred"], kde=True, stat='percent')
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
        rmse_score = round(mse(group[1]["amp_dc_real"], group[1]["amp_dc_pred"], squared=False), 2)
        rmse_r_score = round((mse(group[1]["amp_dc_real"], group[1]["amp_dc_pred"], squared=False)*100/group[1]['amp_dc_real'].mean()), 2)
        mae_score = round(mae(group[1]["amp_dc_real"], group[1]["amp_dc_pred"]), 2)
        r2_score = round(r2(group[1]["amp_dc_real"], group[1]["amp_dc_pred"]), 3)
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
        rmse_score = round(mse(group[1]["amp_dc_real"], group[1]["amp_dc_pred"], squared=False), 2)
        rmse_r_score = round((mse(group[1]["amp_dc_real"], group[1]["amp_dc_pred"], squared=False)*100/group[1]['amp_dc_real'].mean()), 2)
        mae_score = round(mae(group[1]["amp_dc_real"], group[1]["amp_dc_pred"]), 2)
        r2_score = round(r2(group[1]["amp_dc_real"], group[1]["amp_dc_pred"]), 3)
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
