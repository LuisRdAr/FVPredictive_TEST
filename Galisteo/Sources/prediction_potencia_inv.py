import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
import warnings
import json
import pickle
import os
import sys

warnings.filterwarnings('ignore')

def get_models(initial_path):
    models = {}
    for inv_dir in os.listdir(initial_path):
        if ("inversor" in inv_dir.lower()) & (os.path.isdir(os.path.join(initial_path, inv_dir))):
            explotacion_path = os.path.join(initial_path, inv_dir, "Explotacion")
            for model_dir in os.listdir(explotacion_path):
                model_path = os.path.join(explotacion_path, model_dir)
                if os.path.isdir(model_path):
                    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
                        models[model_dir] = pickle.load(f)

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

models_dir = os.path.join(root_path, "Modelos", "potencia_inversor")
models = get_models(models_dir)

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