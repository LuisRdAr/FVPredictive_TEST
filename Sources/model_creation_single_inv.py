import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
import json
import pickle
import os
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

class degree_scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_scale_90 = None
        self.columns_to_scale_180 = None
        self.columns_to_scale_360 = None
        
    def fit(self, X, y = None):
        self.columns_to_scale_90, self.columns_to_scale_180, self.columns_to_scale_360 = self.determine_columns_to_scale(X)
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
                output_features.append("scaled_" + feature)
            return output_features
        else:
            return None
    
class percentage_scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_scale = None
        
    def fit(self, X, y = None):
        self.columns_to_scale_ = X.columns
        return self
    
    def transform(self, X, y=None):
        scaled_data = X.copy()
        if self.columns_to_scale is not None:
            for column in self.columns_to_scale:
                scaled_column = "scaled_" + column
                scaled_data[scaled_column] = X[column] / 100
                scaled_data.drop(column, axis=1, inplace=True)
        return scaled_data

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            output_features = []
            for feature in input_features:
                output_features.append("scaled_" + feature)
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
    results = x[["datetime_utc", "dispositivo_id", "entrada_id"]]
    results["outlier_2"] = outlier
    results = results.reset_index(drop=True)
    return results

print("Comienzo de la ejecución del programa")
print(f"Hora: {pd.Timestamp.now()}")

# Carga de los parámetros en función del sistema operativo usado
root_path = os.getcwd()
with open(os.path.join(root_path,'galisteodb_params.json')) as f:
    params = json.load(f)
schema_name = params["schema"]

password = params['password'].replace('@', '%40')
engine = create_engine(f'postgresql://{params["user"]}:{password}@{params["host"]}:{params["port"]}/{params["dbname"]}')

inv_ids_query = """SELECT DISTINCT(dispositivo_id) FROM galisteo.inversores"""
inv_ids = np.sort([id[0] for id in pd.read_sql_query(inv_ids_query, engine).values])

num_mod_string = 30
sup_mod = 2

for inv_id in tqdm(inv_ids, total = len(inv_ids)):
    main_query = f"""
        WITH f AS (
            SELECT *
                FROM {schema_name}.inversores
                WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 15 = 0)
                    AND (EXTRACT(SECOND FROM datetime_utc) = 0)
                    AND (EXTRACT(MONTH FROM datetime_utc) != 9
                        OR EXTRACT(MONTH FROM datetime_utc) != 10)
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
    chunksize = 100000
    chunks = pd.read_sql_query(main_query, engine, chunksize=chunksize)
    main_df = pd.DataFrame()
    for chunk in chunks:
        main_df = pd.concat([main_df, chunk], ignore_index = True)
    
    main_df["amp_dc"] = main_df["amp_dc"]/main_df["num_strings"]
    main_df["datetime_utc"] = pd.to_datetime(main_df["datetime_utc"], utc = True)
    main_df["potencia_act"] = main_df["potencia_act"] * 1000
    main_df["rad_diff"] = (main_df["rad_celda1"] - main_df["rad_celda2"])
    main_df["temp_panel"] = (main_df["temp_panel1"] + main_df["temp_panel2"]) / 2
    main_df = main_df.drop(columns = ["rad_celda1", "rad_celda2", "temp_panel1", "temp_panel2"])
    potencia_df = main_df.groupby(["dispositivo_id", "datetime_utc"]).agg({"num_strings": "sum",
                                                        "rad_poa": "mean",
                                                        "potencia_act": "mean"
                                                        }).reset_index()
    potencia_df["potencia_solar"] = potencia_df["rad_poa"] * potencia_df["num_strings"] * num_mod_string * sup_mod 
    potencia_df["outlier_1"] = np.where(potencia_df["potencia_act"] > 0.20 * potencia_df["potencia_solar"], True, False)
    main_df = main_df.merge(potencia_df[["dispositivo_id", "datetime_utc", "outlier_1"]], on = ["dispositivo_id", "datetime_utc"])
    main_df = main_df[main_df["outlier_1"] == False]
    main_df = main_df[~(main_df["id"].isin(main_df[(main_df["amp_dc"] > 12.5)|(main_df["amp_dc"] < -0.1)]["id"].unique()))]
    main_df["motivo"] = main_df["motivo"].apply(lambda x: 0 if x == 0 else (2 if x == 7 else 1))
    main_query = f"""
        SELECT MAX(consigna_pot_act_ree)
            FROM {schema_name}.ree AS ree;"""
    max_pot_act = pd.read_sql_query(main_query, engine).values[0][0]
    main_df["consigna_pot_act_planta"] = main_df["consigna_pot_act_planta"] / max_pot_act 
    main_df["dia_año"] = main_df["datetime_utc"].dt.dayofyear
    main_df["dia_año_sen"] = np.sin(main_df["dia_año"] * (2*np.pi/365))
    main_df["dia_año_cos"] = np.cos(main_df["dia_año"] * (2*np.pi/365))
    main_df["hora_seg"] = main_df["datetime_utc"].dt.hour * 3600 + \
                            main_df["datetime_utc"].dt.minute * 60 + \
                            main_df["datetime_utc"].dt.second
    main_df["hora_seg_sen"] = np.sin(main_df["hora_seg"] * (2*np.pi/86400))
    main_df["hora_seg_cos"] = np.cos(main_df["hora_seg"] * (2*np.pi/86400))
    main_df = main_df.dropna()
    outliers = main_df.groupby(['datetime_utc', 'dispositivo_id']).apply(discriminador).reset_index(drop=True)
    main_df = pd.merge(main_df, outliers[outliers["outlier_2"] == False], on = ["datetime_utc", "dispositivo_id", "entrada_id"])
    processed_df = main_df.drop(columns = ["id",
                                        "datetime_utc",
                                        "dispositivo_id", 
                                        "entrada_id",
                                        "dia_año",
                                        "hora_seg",
                                        "potencia_act",
                                        "outlier_1",
                                        "outlier_2",
                                        "num_strings"
                                        ])

    y = processed_df["amp_dc"]
    X = processed_df.drop(columns = ["amp_dc"])
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2, 
                                                        random_state=42)

    perc_attr = ['lim_act', 'cloud_impact']
    std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'vel_viento', 'rad_diff', 'temp_panel']
    degree_attr = ['azimuth', 'elevation']
    cat_attr = ['motivo']

    transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))])
    transformador_numerico_perc = Pipeline([('perc_scaler', percentage_scaler())]) 
    transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 
    transformador_numerico_deg = Pipeline([('degree_scaler', degree_scaler())])

    preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
                                                    ('perc', transformador_numerico_perc, perc_attr),
                                                    ('std', transformador_numerico_std, std_attr),
                                                    ('deg', transformador_numerico_deg, degree_attr)],
                                    remainder='passthrough')
                                    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', XGBRegressor())
                    ])
                    
    param_grid = {'regressor__max_depth': list(range(3, 14, 2)),
                'regressor__n_estimators': list(range(100, 1600, 200)),
                'regressor__min_child_weight': list(range(2, 11, 2)),
                'regressor__learning_rate': [0.1, 0.05, 0.01]
                }
    print("Comienzo de la primera fase de optimización del modelo (max_depth, n_estimators, min_child_weight)")
    print(f"Hora: {pd.Timestamp.now()}")
    print(f"Número de combinaciones: {len(param_grid['regressor__max_depth']) * \
                                    len(param_grid['regressor__n_estimators']) * \
                                    len(param_grid['regressor__min_child_weight']) * \
                                    len(param_grid['regressor__learning_rate'])}")
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    dict_hiperparam = {k:v for k,v in grid_search.best_estimator_.named_steps["regressor"].get_params().items() if v != None}
    print("Mejor combinación de hiperparámetros: ", dict_hiperparam)
    print("Score: ", grid_search.best_score_)

    param_grid = {'regressor__max_depth': list(range(dict_hiperparam["max_depth"] - 1, dict_hiperparam["max_depth"] + 2, 1)),
                'regressor__n_estimators': list(range(dict_hiperparam["n_estimators"] - 100, dict_hiperparam["n_estimators"] + 150, 50)),
                'regressor__min_child_weight': list(range(dict_hiperparam["min_child_weight"] - 1, dict_hiperparam["min_child_weight"] + 2, 1)),
                'regressor__learning_rate': [dict_hiperparam["learning_rate"]],
                'regressor__subsample': [0.8, 0.9, 1],
                'regressor__gamma': [i/10.0 for i in range(0,5)]
                }
    print("Comienzo de la segunda fase de optimización del modelo (subsample, gamma)")
    print(f"Hora: {pd.Timestamp.now()}")
    print(f"Número de combinaciones: {len(param_grid['regressor__max_depth']) * \
                                    len(param_grid['regressor__n_estimators']) * \
                                    len(param_grid['regressor__min_child_weight']) * \
                                    len(param_grid['regressor__subsample']) * \
                                    len(param_grid['regressor__gamma'])}")
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    dict_hiperparam = {k:v for k,v in grid_search.best_estimator_.named_steps["regressor"].get_params().items() if v != None}
    print("Mejor combinación de hiperparámetros: ", dict_hiperparam)
    print("Score: ", grid_search.best_score_)

    param_grid = {'regressor__max_depth': list(range(dict_hiperparam["max_depth"] - 1, dict_hiperparam["max_depth"] + 2, 1)),
                'regressor__n_estimators': list(range(dict_hiperparam["n_estimators"] - 50, dict_hiperparam["n_estimators"] + 100, 50)),
                'regressor__min_child_weight': list(range(dict_hiperparam["min_child_weight"] - 1, dict_hiperparam["min_child_weight"] + 2, 1)),
                'regressor__learning_rate': [dict_hiperparam["learning_rate"]],
                'regressor__subsample': [dict_hiperparam["subsample"]],
                'regressor__gamma': [dict_hiperparam["gamma"]],
                'regressor__reg_alpha': list(range(0,5)/4),
                'regressor__reg_lambda': list(range(0,5)/4),
                }
    print("Comienzo de la tercera fase de optimización del modelo (reg_alpha, reg_lambda)")
    print(f"Hora: {pd.Timestamp.now()}")
    print(f"Número de combinaciones: {len(param_grid['regressor__max_depth']) * \
                                    len(param_grid['regressor__n_estimators']) * \
                                    len(param_grid['regressor__min_child_weight']) * \
                                    len(param_grid['regressor__reg_alpha']) * \
                                    len(param_grid['regressor__reg_lambda'])}")
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    dict_hiperparam = {k:v for k,v in grid_search.best_estimator_.named_steps["regressor"].get_params().items() if v != None}
    print("Mejor combinación de hiperparámetros: ", dict_hiperparam)
    print("Score: ", grid_search.best_score_)

    # param_grid = {'regressor__max_depth': list(range(dict_hiperparam["max_depth"] - 1, dict_hiperparam["max_depth"] + 2, 1)),
    #             'regressor__n_estimators': list(range(dict_hiperparam["n_estimators"] - 100, dict_hiperparam["n_estimators"] * 5, 200)),
    #             'regressor__min_child_weight': list(range(dict_hiperparam["min_child_weight"] - 1, dict_hiperparam["min_child_weight"] + 2, 1)),
    #             'regressor__subsample': [dict_hiperparam["subsample"]],
    #             'regressor__gamma': [dict_hiperparam["gamma"]],
    #             'regressor__reg_alpha': [dict_hiperparam["reg_alpha"]],
    #             'regressor__reg_lambda': [dict_hiperparam["reg_lambda"]],
    #             'regressor__learning_rate': [0.1, 0.05, 0.01]
    #             }
    # print("Comienzo de la cuarta fase de optimización del modelo (n_estimators, learning_rate)")
    # print(f"Hora: {pd.Timestamp.now()}")
    # print(f"Número de combinaciones: {len(param_grid['regressor__max_depth']) * \
    #                                 len(param_grid['regressor__n_estimators']) * \
    #                                 len(param_grid['regressor__min_child_weight']) * \
    #                                 len(param_grid['regressor__reg_alpha']) * \
    #                                 len(param_grid['regressor__reg_lambda'])}")
    # grid_search = GridSearchCV(model, param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    # dict_hiperparam = {k:v for k,v in grid_search.best_estimator_.named_steps["regressor"].get_params().items() if v != None}
    # print("Mejor combinación de hiperparámetros: ", dict_hiperparam)
    # print("Score: ", grid_search.best_score_)

    param_grid = {'regressor__max_depth': list(range(dict_hiperparam["max_depth"] - 1, dict_hiperparam["max_depth"] + 2, 1)),
                'regressor__n_estimators': list(range(dict_hiperparam["n_estimators"] - 150, dict_hiperparam["n_estimators"] + 250, 50)),
                'regressor__min_child_weight': list(range(dict_hiperparam["min_child_weight"] - 1, dict_hiperparam["min_child_weight"] + 2, 1)),
                'regressor__subsample': [dict_hiperparam["subsample"]],
                'regressor__gamma': [dict_hiperparam["gamma"]],
                'regressor__reg_alpha': [dict_hiperparam["reg_alpha"]],
                'regressor__reg_lambda': [dict_hiperparam["reg_lambda"]],
                'regressor__learning_rate': [dict_hiperparam["learning_rate"]]
                }
    print("Entrenamiento del modelo final")
    print(f"Hora: {pd.Timestamp.now()}")
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    dict_hiperparam = {k:v for k,v in grid_search.best_estimator_.named_steps["regressor"].get_params().items() if v != None}
    print("Hiperparámetros finales: ", dict_hiperparam)
    print("Score: ", grid_search.best_score_)

    from sklearn.metrics import mean_squared_error as mse, \
                                mean_absolute_error as mae, \
                                r2_score as r2

    y_pred = grid_search.best_estimator_.predict(X_test)
    prediction_df = pd.DataFrame(y_test).join(main_df[["id", "dispositivo_id", "entrada_id", "datetime_utc", "num_strings"]])
    prediction_df["amp_dc"] = prediction_df["amp_dc"] * prediction_df["num_strings"]
    prediction_df["y_pred"] = y_pred * prediction_df["num_strings"]
    prediction_df["y_diff"] = prediction_df["amp_dc"] - prediction_df["y_pred"]

    rmse_score = round(mse(prediction_df["amp_dc"], prediction_df["y_pred"], squared = False),3)
    mae_score = round(mae(prediction_df["amp_dc"], prediction_df["y_pred"]),3)
    r2_score = round(r2(prediction_df["amp_dc"], prediction_df["y_pred"]),3)
    rmse_p_score = round(mse(prediction_df["amp_dc"], prediction_df["y_pred"], squared = False)/prediction_df["amp_dc"].describe()["std"], 3)
    metricas = {"RMSE": rmse_score, "MAE": mae_score, "NRMSE": rmse_p_score, "R2": r2_score}
    print(f"RMSE: {rmse_score}", 
        f"MAE: {mae_score}",
        f"NRMSE: {rmse_p_score}",
        f"R2: {r2_score}",
        sep = "\n")

    columnas = [col_name.split("__")[1] for col_name in grid_search.best_estimator_.named_steps["preprocessor"].get_feature_names_out()]
    importancia = grid_search.best_estimator_.named_steps["regressor"].feature_importances_.tolist()
    algoritmo = type(grid_search.best_estimator_.named_steps["regressor"]).__name__
    path = os.path.join(root_path, f"Modelos/entrada_amperaje/Inversor_{inv_id - 20}/Repositorio/{algoritmo}-{pd.Timestamp.now()}/")
    os.makedirs(path)
    with open(path+'model.model', "wb") as archivo_salida:
        pickle.dump(grid_search.best_estimator_, archivo_salida)
    with open(path+'informe_modelo.json', 'w') as archivo_json:
        informe = {"metricas": metricas,
                    "feature_importance": dict(sorted({k:v for k,v in zip(columnas,importancia)}.items(), key=lambda item: item[1], reverse=True)),
                    "hiperparametros": {k:v for k,v in grid_search.best_estimator_.named_steps["regressor"].get_params().items() if v != None},
                    "param_grid": param_grid,
                    "input_description": processed_df[perc_attr + std_attr + degree_attr].describe().loc[["mean", "std"]].to_dict(),
                    "target": prediction_df["amp_dc"].describe().to_dict()}
        json.dump(informe, archivo_json)

    y_test_sampled, _,y_pred_sampled, _ = train_test_split(prediction_df["amp_dc"], prediction_df["y_pred"], train_size = 0.25)
    plt.scatter(y_test_sampled, y_pred_sampled, marker = ".")
    plt.plot([min(y_test_sampled), max(y_test_sampled)], [min(y_test_sampled), max(y_test_sampled)], color='black', linestyle='-', linewidth=1)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.savefig(path+'dispersion_test_pred')

    plt.figure()
    ax = sns.histplot(y_test_sampled - y_pred_sampled, kde=True, stat='percent')
    ax.axvline(x=0, color='black', linestyle='--', linewidth = 0.35, label='x=0')
    plt.title('Histograma de Diferencias entre y_test y y_pred')
    plt.xlabel('Diferencia')
    plt.ylabel('Porcentaje')
    plt.savefig(path+'histograma_diferencia_test_pred')