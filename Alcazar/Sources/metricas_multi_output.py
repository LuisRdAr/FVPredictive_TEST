import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
import warnings
import json
import os
import sys
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, \
                            r2_score as r2
from tqdm import tqdm
import gc

def discriminador(row):
    mean = np.mean(row)
    std = np.std(row)
    threshold = 3 * std
    outlier = np.abs(row - mean) > threshold
    return outlier


cuda_flag = torch.cuda.is_available()
print("Configuración para el entrenamiento de modelos de predicción de potencia activa de salida en inversores fotovoltaicos:", end="\n\n")
model_name = "XGBRegressor"

valid_responses = ["y", "n"]
normalizacion = True
optimizacion = False

# Comprobación de la disponibilidad de GPU para el entrenamiento
# valid_devices = ["cpu", "cuda"]
# device = ""
# if cuda_flag and model_name == "XGBRegressor":
#     while device not in valid_devices:
#         device_bool = input("¿Desea utilizar GPU para el entrenamiento? (Y/N): ").lower()
#         if device_bool == "y":
#             device = "cuda"
#         else:
#             device = "cpu"
# else:
#     device = 'cpu'
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

seeds = np.random.randint(0, 10000, size=1000)
num_mod_string = 30
sup_mod = 2
ids_parques = [1,2,3]
for id_parque in ids_parques:
    print(f"Ictios Alcazar {id_parque}")
    # Carga de los datos de entrenamiento    
    # main_query = f"""
    #     WITH inv AS (
    #         SELECT id, parque_id, dispositivo_id, datetime_utc, potencia_act
    #             FROM {schema_name}.inversores
    #             WHERE (EXTRACT(MINUTE FROM datetime_utc) %% 15 = 0)
    #                 AND (EXTRACT(SECOND FROM datetime_utc) = 0)
    #                 AND parque_id = {id_parque}
    #                 ),
    #         met AS (
    #             SELECT parque_id, datetime_utc, AVG(rad_poa) AS rad_poa, AVG(rad_hor) AS rad_hor, AVG(rad_celda1) AS rad_celda1,
    #                     AVG(rad_celda2) AS rad_celda2, AVG(temp_amb) AS temp_amb, AVG(temp_panel1) AS temp_panel1,
    #                     AVG(temp_panel2) AS temp_panel2, AVG(presion_atm) AS presion_atm, AVG(cloud_impact) AS cloud_impact, 
    #                     BOOL_OR(daylight) AS daylight
    #                 FROM {schema_name}.meteo
    #                 WHERE parque_id = {id_parque}
    #                     AND daylight = true
    #                 GROUP BY parque_id, datetime_utc)
    #     SELECT inv.id, inv.dispositivo_id, det.entrada_id, inv.datetime_utc, potencia_act, num_strings, 
    #             rad_poa, rad_hor, rad_celda1, rad_celda2, temp_amb, temp_panel1, temp_panel2, presion_atm, cloud_impact, 
    #             motivo, consigna_pot_act_planta, amp_dc
    #         FROM inv
    #         JOIN {schema_name}.inversores_detalle AS det
    #             ON inv.id = det.id
    #         JOIN {schema_name}.distrib_inversores dist
    #             ON dist.parque_id = inv.parque_id
    #                 AND dist.dispositivo_id = inv.dispositivo_id
    #                 AND dist.entrada_id = det.entrada_id
    #         JOIN met
    #             ON met.parque_id = inv.parque_id
    #                 AND met.datetime_utc = inv.datetime_utc
    #         JOIN {schema_name}.ree AS ree
    #             ON ree.datetime_utc = inv.datetime_utc
    #                 AND ree.parque_id = inv.parque_id
    #         ORDER BY 4, 2, 3;"""

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
    print(f"Registros descartados por outlier de corriente: {n_corriente_outlier}")

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

    rmse_seeds = []
    mae_seeds = []
    for seed in tqdm(seeds, total = len(seeds)):
        rmse_inv = []
        mae_inv = []
        for inv_id in np.sort(main_df.index.get_level_values("dispositivo_id").unique()):
        # Separación de los datos de entrenamiento y validación
            disp_df = main_df[main_df.index.get_level_values("dispositivo_id") == inv_id].copy()
            disp_df = disp_df.dropna()

            train_df, validation_df = train_test_split(disp_df, test_size = 0.2, random_state = seed)

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
            std_attr = ['rad_poa', 'rad_hor', 'temp_amb', 'rad_diff', 'temp_panel', "presion_atm"]
            cat_attr = ['motivo']

            transformador_categorico = Pipeline([('onehot', OneHotEncoder(handle_unknown = 'ignore'))]) # Introducir manualmente catergorías?
            transformador_numerico_std = Pipeline([('std_scaler', StandardScaler())]) 

            preprocessor = ColumnTransformer(transformers=[('cat', transformador_categorico, cat_attr),
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

            consulta_sql = f"""SELECT dispositivo_id, entrada_id, num_strings
                        FROM {schema_name}.distrib_inversores
                        WHERE parque_id = {id_parque}
                            AND dispositivo_id = {inv_id};"""
            num_strings = pd.read_sql_query(consulta_sql, engine)
            X_val_prep = xgb.DMatrix(pipeline_model.named_steps['preprocessor'].transform(X_val))
            y_pred_val = pd.DataFrame(pipeline_model.named_steps['regressor'].predict(X_val_prep)).rename(columns={i: "y_pred_"+str(i+1) for i in pd.DataFrame(pipeline_model.named_steps['regressor'].predict(X_val_prep)).columns})
            y_val_reesc = y_val.copy()
            for entrada in num_strings["entrada_id"]:
                y_pred_val["y_pred_"+str(entrada)] = y_pred_val["y_pred_"+str(entrada)] * num_strings[num_strings["entrada_id"] == entrada]["num_strings"].values[0]
                y_val_reesc.loc[y_val.index, "amp_dc_"+str(entrada)] = y_val["amp_dc_"+str(entrada)] * num_strings[num_strings["entrada_id"] == entrada]["num_strings"].values[0]

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
            
            rmse_inv.append(rmse_score)
            mae_inv.append(mae_score)
        rmse_seeds.append(np.mean(rmse_inv))
        mae_seeds.append(np.mean(mae_inv))
    print()
    print(f"RMSE: {np.mean(rmse_seeds)}")
    print(f"MAE: {np.mean(mae_seeds)}", end="\n\n")