import numpy as np
import pandas as pd
from sqlalchemy import create_engine, types
import json
import os
import sys
import pvlib
from pvlib import location
import requests


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

# Consulta de coordenadas del parque
consulta_sql = f"SELECT latitud, longitud FROM {schema_name}.parques;"
gps = pd.read_sql_query(consulta_sql, engine).values[0]
lat, lon = gps
parque = location.Location(latitude=lat, longitude=lon, tz="Europe/Madrid")

fechas = pd.date_range(start=pd.to_datetime("2021-01-01").date(), end=pd.to_datetime("2023-12-31").date(), freq='1D')
quantiles = np.linspace(0.3, 0.4, 6) 
quantiles = np.concatenate((quantiles[::-1], quantiles))

# Inicialización de listas para almacenar resultados
sunrise = []
sunset = []
plateau_start = []
plateau_end = []

# Cálculo de las horas de salida y puesta del sol, y los periodos de plateau
for fecha in fechas:
    r = requests.get('https://api.sunrise-sunset.org/json', params={'lat': lat, 'lng': lon, 'date': fecha.strftime("%Y-%m-%d")}).json()['results']
    hora_salida_sol = pd.to_datetime(f"{fecha} {r['sunrise']}", utc=True)
    hora_puesta_sol = pd.to_datetime(f"{fecha} {r['sunset']}", utc=True)
    sunrise.append(hora_salida_sol)
    sunset.append(hora_puesta_sol)
    
    times = pd.date_range(start=hora_salida_sol, end=hora_puesta_sol, freq='30s')
    clear_sky = parque.get_clearsky(times)
    clear_sky["sum"] = clear_sky["dni"] + clear_sky["dhi"]
    threshold = clear_sky["sum"].quantile(quantiles[fecha.month - 1])
    plateau_times = clear_sky[clear_sky["sum"] > threshold].index
    plateau_start.append(plateau_times[0])
    plateau_end.append(plateau_times[-1])

# Creación del DataFrame con los resultados
daily_df = pd.DataFrame({
    "date": fechas,
    "sunrise": sunrise,
    "sunset": sunset,
    "plateau_start": plateau_start,
    "plateau_end": plateau_end
})

daily_df["plateau_start_seconds"] = daily_df["plateau_start"].dt.time.apply(lambda x: (x.hour * 60 + x.minute) * 60 + x.second)
daily_df["plateau_end_seconds"] = daily_df["plateau_end"].dt.time.apply(lambda x: (x.hour * 60 + x.minute) * 60 + x.second)
daily_df["month"] = daily_df["date"].dt.month  

# Cálculo de los promedios mensuales
plateau_limits_df = daily_df.groupby("month")[["plateau_start_seconds", "plateau_end_seconds"]].mean()
plateau_limits_df["plateau_start"] = plateau_limits_df["plateau_start_seconds"].apply(lambda x: pd.to_datetime(x, unit="s", utc=True).strftime('%Y-%m-%d %H:%M'))
plateau_limits_df["plateau_end"] = plateau_limits_df["plateau_end_seconds"].apply(lambda x: pd.to_datetime(x, unit="s", utc=True).strftime('%Y-%m-%d %H:%M'))
plateau_limits_df["plateau_start"] = pd.to_datetime(plateau_limits_df["plateau_start"], utc=True)
plateau_limits_df["plateau_end"] = pd.to_datetime(plateau_limits_df["plateau_end"], utc=True)

# Escritura de resultados en la base de datos
try:
    dtypes = {
        "month": types.SMALLINT(),
        "plateau_start": types.TIMESTAMP(timezone=True),
        "plateau_end": types.TIMESTAMP(timezone=True)
    }
    plateau_limits_df[["plateau_start", "plateau_end"]].reset_index().to_sql("plateau", engine, schema=schema_name, if_exists="append", index=False, dtype=dtypes)
    print("Datos escritos correctamente en la base de datos")
except Exception as error:
    print(f"Error en la escritura de datos en la tabla de estadística histórica: \n\t{error}")
    sys.exit()