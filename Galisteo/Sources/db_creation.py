import psycopg2
import json
import sys
import os


if __name__ == "__main__":
    # Carga, desde la ruta de ejecución, de los parámetros para conexión a la base de datos  
    root_path = os.getcwd()
    params = None
    for filename in os.listdir(root_path):
        if "params.json" in filename:
            with open(os.path.join(root_path, filename)) as f:
                params = json.load(f)
    if params is None:
        print("No se ha encontrado el archivo de parámetros para la conexión a la base de datos")
        sys.exit()
    schema_name = params["schema"].lower()

    try:
        conn = psycopg2.connect(
            database = params["dbname"],
            user = params["user"],
            password = params["password"],
            host = params["host"],  
            port = params["port"])
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        open = True
        print("Conexión exitosa a PostgreSQL")

    except psycopg2.Error as e:
        if f'FATAL:  database "{params["dbname"]}" does not exist' in str(e):
            print("Error al conectarse a PostgreSQL:", e, sep = "\n")
            print("Creando nueva base de datos")
            try:
                conn = psycopg2.connect(
                    database = 'postgres',
                    user = params["user"],
                    password = params["password"],
                    host = params["host"],  
                    port = params["port"])
                conn.set_session(autocommit=True)
                cur = conn.cursor()
            except psycopg2.Error as e:
                print("Error al conectarse a PostgreSQL:", e, sep = "\n")
                sys.exit()

            cur.execute(f'CREATE DATABASE {params["dbname"]};')

            cur.close()
            conn.close()

            conn = psycopg2.connect(
                    database = params["dbname"],
                    user = params["user"],
                    password = params["password"],
                    host = params["host"],  
                    port = params["port"])

            conn.set_session(autocommit=True)
            cur = conn.cursor()
            open = True
            print(f'Conexión exitosa a nueva database {params["dbname"]} de PostgreSQL')
        else:
            print("Error al conectarse a PostgreSQL:", e, sep = "\n")
            open = False
            sys.exit()

    if open:
        ###################################
        # CREACIÓN TABLA PARA LOS PARQUES #
        ###################################
        cur.execute(f"""CREATE SCHEMA IF NOT EXISTS {schema_name};""")

        ###################################
        # CREACIÓN TABLA PARA LOS PARQUES #
        ###################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.parques(
                parque_id SMALLINT PRIMARY KEY,
                descripcion VARCHAR(25),
                localizacion VARCHAR(25),
                potencia_max REAL,
                num_paneles INTEGER,
                latitud NUMERIC(16,14),
                longitud NUMERIC(16,15) );""")
        if schema_name == "galisteo":
            try:
                cur.execute(
                    f"""INSERT INTO {schema_name}.parques VALUES(
                    1, 
                    'Galisteo',
                    'Extremadura',
                    49.87575,
                    147240,
                    40.06559972131205,
                    -6.273260952948442);""")
            except psycopg2.errors.UniqueViolation:
                pass
        else:
	        print("Esquema inválido")
	        sys.exit()
        
        ########################################
        # CREACIÓN TABLA PARA LOS DISPOSITIVOS #
        ########################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.dispositivos(
                parque_id SMALLINT,
                dispositivo_id SMALLINT,
                nombre VARCHAR(25),
                referencia VARCHAR(25),
                ubicacion VARCHAR(25),
                descripcion VARCHAR(25),
                meteo_cercana_id SMALLINT,
                PRIMARY KEY (dispositivo_id, parque_id),
                FOREIGN KEY (parque_id) REFERENCES {schema_name}.parques(parque_id));""")
        
        ######################################################################
        # CREACIÓN TABLA PARA LOS DISTRIBUCIÓN DE ENTRADAS EN LOS INVERSORES #
        ######################################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.distrib_inversores(
                parque_id SMALLINT,
                subcampo SMALLINT,
                dispositivo_id SMALLINT,
                entrada_id SMALLINT,
                tipo VARCHAR(10),
                num_strings SMALLINT,
                num_modulos SMALLINT,
                potencia_mod VARCHAR(10),
                potencia INTEGER,
                PRIMARY KEY (parque_id, dispositivo_id, entrada_id),
                FOREIGN KEY (parque_id, dispositivo_id) 
                        REFERENCES {schema_name}.dispositivos(parque_id, dispositivo_id));""")
        
        ##############################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS EN BRUTO DE LOS INVERSORES #
        ##############################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.inversores_raw(
                id SERIAL PRIMARY KEY,
                parque_id SMALLINT,
                descripcion_parque VARCHAR(25),
                localizacion_parque VARCHAR(25),
                potencia_max REAL,
                num_paneles REAL,
                dispositivo_id SMALLINT NOT NULL,
                nombre_dispositivo VARCHAR(25),
                ref VARCHAR(25),
                ubicacion VARCHAR(25),
                descripcion_dispositivo VARCHAR(25),
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma SMALLINT,
                estado SMALLINT,
                potencia_act SMALLINT,
                potencia_rea SMALLINT,
                cos_phi REAL,
                vol_12 REAL,
                vol_13 REAL,
                vol_23 REAL,
                amp_1 REAL,
                amp_2 REAL,
                amp_3 REAL,
                frec REAL,
                lim_act REAL,
                lim_rea REAL,
                vol_dc_bus SMALLINT,
                amp_dc_in1 REAL,
                amp_dc_in2 REAL,
                amp_dc_in3 REAL,
                amp_dc_in4 REAL,
                amp_dc_in5 REAL,
                amp_dc_in6 REAL,
                amp_dc_in7 REAL,
                amp_dc_in8 REAL,
                amp_dc_in9 REAL,
                amp_dc_in10 REAL,
                amp_dc_in11 REAL,
                amp_dc_in12 REAL,
                amp_dc_in13 REAL,
                amp_dc_in14 REAL,
                amp_dc_in15 REAL,
                amp_dc_in16 REAL,
                amp_dc_in17 REAL,
                amp_dc_in18 REAL,
                amp_dc_in19 REAL,
                amp_dc_in20 REAL,
                amp_dc_in21 REAL,
                amp_dc_in22 REAL,
                amp_dc_in23 REAL,
                amp_dc_in24 REAL,
                aisl_dc INTEGER,
                energia_dia INTEGER,
                procesado BOOLEAN DEFAULT FALSE,
                datetime_procesado TIMESTAMP WITH TIME ZONE,
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        ################################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS PROCESADAS DE LOS INVERSORES #
        ################################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.inversores(
                id INTEGER PRIMARY KEY,
                parque_id SMALLINT,
                dispositivo_id SMALLINT NOT NULL,
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma SMALLINT,
                estado SMALLINT,
                potencia_act SMALLINT,
                potencia_rea SMALLINT,
                cos_phi REAL,
                vol_12 REAL,
                vol_13 REAL,
                vol_23 REAL,
                amp_1 REAL,
                amp_2 REAL,
                amp_3 REAL,
                frec REAL,
                lim_act REAL,
                lim_rea REAL,
                vol_dc_bus REAL,
                aisl_dc INTEGER,
                energia_dia INTEGER,
                FOREIGN KEY (parque_id, dispositivo_id) 
                        REFERENCES {schema_name}.dispositivos(parque_id, dispositivo_id),
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        ###################################################################
        # CREACIÓN TABLA PARA EL DETALLE DE LAS MEDIDAS DE LOS INVERSORES #
        ###################################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.inversores_detalle(
                id INTEGER,
                entrada_id SMALLINT,
                amp_dc REAL,
                PRIMARY KEY (id, entrada_id),
                FOREIGN KEY (id) REFERENCES {schema_name}.inversores(id));""")
        cur.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_inversores_detalle_id
                ON {schema_name}.inversores_detalle(id);""")
        
        #####################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS EN BRUTO DE METEO #
        #####################################################

        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.meteo_raw(
                id SERIAL PRIMARY KEY,
                parque_id SMALLINT,
                descripcion_parque VARCHAR(25),
                localizacion_parque VARCHAR(25),
                potencia_max REAL,
                num_paneles REAL,
                dispositivo_id SMALLINT NOT NULL,
                nombre_dispositivo VARCHAR(25),
                ref VARCHAR(25),
                ubicacion VARCHAR(25),
                descripcion_dispositivo VARCHAR(25),
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma REAL, 
                rad_poa REAL, 
                rad_hor REAL, 
                rad_celda1 REAL, 
                rad_celda2 REAL, 
                temp_amb REAL, 
                temp_panel1 REAL, 
                temp_panel2 REAL, 
                vel_viento REAL, 
                dir_viento REAL, 
                hum_rel REAL,
                presion_atm REAL,
                procesado BOOLEAN DEFAULT FALSE,
                datetime_procesado TIMESTAMP WITH TIME ZONE,
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        #######################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS PROCESADAS DE METEO #
        #######################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.meteo(
                id INTEGER PRIMARY KEY,
                parque_id SMALLINT,
                dispositivo_id SMALLINT NOT NULL,
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma REAL, 
                rad_poa REAL, 
                rad_hor REAL, 
                rad_celda1 REAL, 
                rad_celda2 REAL, 
                temp_amb REAL, 
                temp_panel1 REAL, 
                temp_panel2 REAL, 
                vel_viento REAL, 
                dir_viento REAL, 
                hum_rel REAL,
                elevation REAL,
                azimuth REAL,
                cloud_impact REAL,
                daylight BOOLEAN,
                status_srl SMALLINT,
                FOREIGN KEY (parque_id, dispositivo_id) 
                        REFERENCES {schema_name}.dispositivos(parque_id, dispositivo_id),
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        ##################################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS EN BRUTO DEL ANALIZADOR DE RED #
        ##################################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.analizador_red_raw(
                id SERIAL PRIMARY KEY,
                parque_id SMALLINT,
                descripcion_parque VARCHAR(25),
                localizacion_parque VARCHAR(25),
                potencia_max REAL,
                num_paneles REAL,
                dispositivo_id SMALLINT NOT NULL,
                nombre_dispositivo VARCHAR(25),
                ref VARCHAR(25),
                ubicacion VARCHAR(25),
                descripcion_dispositivo VARCHAR(25),
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma REAL,
                potencia_act REAL,
                potencia_rea REAL,
                cos_phi REAL,
                volt_12 REAL,
                volt_13 REAL,
                volt_23 REAL,
                amp_1 REAL,
                amp_2 REAL,
                amp_3 REAL,
                frec REAL,
                procesado BOOLEAN DEFAULT false,
                datetime_procesado TIMESTAMP WITH TIME ZONE,
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        ####################################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS PROCESADAS DEL ANALIZADOR DE RED #
        ####################################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.analizador_red(
                id INTEGER PRIMARY KEY,
                parque_id SMALLINT,
                dispositivo_id SMALLINT NOT NULL,
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma REAL,
                potencia_act REAL,
                potencia_rea REAL,
                cos_phi REAL,
                volt_12 REAL,
                volt_13 REAL,
                volt_23 REAL,
                amp_1 REAL,
                amp_2 REAL,
                amp_3 REAL,
                frec REAL,
                FOREIGN KEY (parque_id, dispositivo_id) 
                        REFERENCES {schema_name}.dispositivos(parque_id, dispositivo_id),
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        #########################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS EN BRUTO DEL CONTADOR #
        #########################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.contador_raw(
                id SERIAL PRIMARY KEY,
                parque_id SMALLINT,
                descripcion_parque VARCHAR(25),
                localizacion_parque VARCHAR(25),
                potencia_max REAL,
                num_paneles REAL,
                dispositivo_id SMALLINT NOT NULL,
                nombre_dispositivo VARCHAR(25),
                ref VARCHAR(25),
                ubicacion VARCHAR(25),
                descripcion_dispositivo VARCHAR(25),
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                energia_act_imp REAL,
                energia_act_exp REAL,
                energia_rea_q1 REAL,
                energia_rea_q2 REAL,
                energia_rea_q3 REAL,
                energia_rea_q4 REAL,
                procesado BOOLEAN DEFAULT false,
                datetime_procesado TIMESTAMP WITH TIME ZONE,
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        ###########################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS PROCESADAS DEL CONTADOR #
        ###########################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.contador(
                id INTEGER PRIMARY KEY,
                parque_id SMALLINT,
                dispositivo_id SMALLINT NOT NULL,
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                energia_act_imp REAL,
                energia_act_exp REAL,
                energia_rea_q1 REAL,
                energia_rea_q2 REAL,
                energia_rea_q3 REAL,
                energia_rea_q4 REAL,
                FOREIGN KEY (parque_id, dispositivo_id) 
                        REFERENCES {schema_name}.dispositivos(parque_id, dispositivo_id),
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        ###################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS EN BRUTO DE REE #
        ###################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.ree_raw(
                id SERIAL PRIMARY KEY,
                parque_id SMALLINT,
                descripcion_parque VARCHAR(25),
                localizacion_parque VARCHAR(25),
                potencia_max REAL,
                num_paneles REAL,
                dispositivo_id SMALLINT NOT NULL,
                nombre_dispositivo VARCHAR(25),
                ref VARCHAR(25),
                ubicacion VARCHAR(25),
                descripcion_dispositivo VARCHAR(25),
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma SMALLINT,
                motivo SMALLINT,
                consigna_pot_act_ree REAL,
                consigna_pot_act_planta REAL,
                consigna_fdp_ree REAL,
                consigna_fdp_planta REAL,
                procesado BOOLEAN DEFAULT false,
                datetime_procesado TIMESTAMP WITH TIME ZONE,
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")
        
        ######################################################
        # CREACIÓN TABLA PARA LAS MEDIDAS PROCESADAS DEL REE #
        ######################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.ree(
                id INTEGER PRIMARY KEY,
                parque_id SMALLINT,
                dispositivo_id SMALLINT NOT NULL,
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                med_id INTEGER,
                status REAL,
                alarma SMALLINT,
                motivo SMALLINT,
                consigna_pot_act_ree REAL,
                consigna_pot_act_planta REAL,
                consigna_fdp_ree REAL,
                consigna_fdp_planta REAL,
                FOREIGN KEY (parque_id, dispositivo_id) 
                        REFERENCES {schema_name}.dispositivos(parque_id, dispositivo_id),
                UNIQUE(datetime_utc, parque_id, dispositivo_id));""")

        ################################
        # CREACIÓN TABLA PARA DAYLIGHT #
        ################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.daylight(
                id SERIAL PRIMARY KEY,
                date DATE UNIQUE,
                sunrise TIMESTAMP WITH TIME ZONE,
                sunset TIMESTAMP WITH TIME ZONE);""")

        ###############################################
        # CREACIÓN TABLA PARA VOLCADO DE PREDICCIONES #
        ###############################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.pred_entrada_corriente(
                id INTEGER NOT NULL,
                dispositivo_id SMALLINT NOT NULL,
                entrada_id SMALLINT NOT NULL,
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                y_pred REAL,
                PRIMARY KEY (id, entrada_id),
                FOREIGN KEY (id) REFERENCES {schema_name}.inversores(id));""")
        
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.pred_salida_potencia(
                id INTEGER PRIMARY KEY,
                dispositivo_id SMALLINT NOT NULL,
                datetime_utc TIMESTAMP WITH TIME ZONE NOT NULL,
                y_pred REAL,
                FOREIGN KEY (id) REFERENCES {schema_name}.inversores(id));""")

        #####################################################
        # CREACIÓN TABLA ESTADÍSTICA HISTÓRICA DE AMPERAJES #
        #####################################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.historico_stats(
                        id SERIAL PRIMARY KEY,
                        rad_bins VARCHAR(50) NOT NULL,
                        cloud_bins VARCHAR(50) NOT NULL,
                        consigna_cat VARCHAR(50) NOT NULL,
                        dispositivo_id VARCHAR(50) NOT NULL,
                        mes INT NOT NULL CHECK (mes BETWEEN 1 AND 12),
                        lower_amp NUMERIC NOT NULL,
                        upper_amp NUMERIC NOT NULL,
                        lower_pot NUMERIC NOT NULL,
                        upper_pot NUMERIC NOT NULL,
                        UNIQUE (rad_bins, cloud_bins, consigna_cat, dispositivo_id, mes)
                        );""")
        cur.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_historico_stats_rad_bins
                        ON {schema_name}.historico_stats(rad_bins);
                CREATE INDEX IF NOT EXISTS idx_historico_stats_cloud_bins
                        ON {schema_name}.historico_stats(cloud_bins);
                CREATE INDEX IF NOT EXISTS idx_historico_stats_consigna_cat
                        ON {schema_name}.historico_stats(consigna_cat);
                CREATE INDEX IF NOT EXISTS idx_historico_stats_dispositivo_id
                        ON {schema_name}.historico_stats(dispositivo_id);
                CREATE INDEX IF NOT EXISTS idx_historico_stats_mes
                        ON {schema_name}.historico_stats(mes);
                CREATE INDEX IF NOT EXISTS idx_categorias_mes 
                        ON {schema_name}.historico_stats(rad_bins, 
                                                        cloud_bins, 
                                                        consigna_cat, 
                                                        dispositivo_id, 
                                                        mes);""")
        
        ###########################################
        # CREACIÓN TABLA PARA LIMITES DEL PLATEAU #
        ###########################################
        cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {schema_name}.plateau(
                        month INT PRIMARY KEY,
                        plateau_start TIMESTAMP WITH TIME ZONE,
                        plateau_end TIMESTAMP WITH TIME ZONE
                        );""")
        
        ##############################
        # CREACIÓN TABLA PARA FALLOS #
        ##############################
        cur.execute(
             f"""CREATE TABLE IF NOT EXISTS {schema_name}.incidencias(
                        id_fallo SERIAL PRIMARY KEY,
                        parque_id SMALLINT,
                        dispositivo_id SMALLINT,
                        codigo_error VARCHAR(20),
                        inicio TIMESTAMP WITH TIME ZONE,
                        ultima_actualizacion TIMESTAMP WITH TIME ZONE,
                        fin TIMESTAMP WITH TIME ZONE DEFAULT NULL,
                        abierta BOOLEAN DEFAULT TRUE,
                        FOREIGN KEY (parque_id, dispositivo_id) 
                                REFERENCES {schema_name}.dispositivos(parque_id, dispositivo_id)
                );

                CREATE INDEX IF NOT EXISTS idx_abierta ON {schema_name}.incidencias(abierta);
                """
                )
