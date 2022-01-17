# -*- coding: utf-8 -*-
import sqlite3 # psycopg2
import datetime
from datetime import timedelta
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import json
import requests

# carga parametros conexión DB
#with open("C:/HIDRODELTA/dbConnectionParams.json") as f:
#	dbConnParams = json.load(f)

# working_dir
working_dir = "C:/HIDRODELTA" # "/home/leyden/HIDRODELTA" # 

with open(working_dir + "/config.json") as f:
    config = json.load(f)
    apiLoginParams = config["api"]

if config["use_proxy"]:
    proxy_dict = config["proxy_dict"]
else:
    proxy_dict = None


def consultaAPI(f_inicio,f_fin,list_idSerie):
    dias = (f_fin - f_inicio).days
    filter = {
        "series_id": ",".join([str(i) for i in list_idSerie]),
        "timestart": f_inicio.isoformat(),
        "timeend": f_fin.isoformat()
    }
    print("filter: %s" % json.dumps(filter))
    response = requests.get(
        apiLoginParams["url"] + '/obs/puntual/observaciones',
        params = filter,
        headers = {'Authorization': 'Bearer ' + apiLoginParams["token"]},
        proxies = proxy_dict
    )
    if response.status_code != 200:
        raise Exception("request failed: %s" % response.text)
    json_response = response.json()
    df_todo = pd.DataFrame.from_dict(json_response,orient='columns')
    print('')
    print('Cantidad de registros: ',len(df_todo))
    if len(df_todo):
        print('Cantidad por dia: ',round(len(df_todo)/dias,1), '  Cantidad de estaciones: ', len(df_todo.groupby('series_id').nunique()))
        print('Cantidad por Estacion:')
        print(df_todo['series_id'].value_counts())
        arg = pytz.timezone('America/Argentina/Buenos_Aires')
        df_todo['timestart'] = [datetime.datetime.strptime(i,'%Y-%m-%dT%H:%M:%S.%f%z').astimezone(arg).replace(tzinfo=None) for i in df_todo['timestart']]
    df_renamed = df_todo[['series_id','timestart','valor']].rename(columns={'series_id':'id','timestart':'fecha','valor':'valor'},inplace=False)
    return df_renamed

def consultaAPImareas(f_inicio,f_fin,list_idEstacion):
    filter = {
        "estacion_id": ",".join([str(i) for i in list_idEstacion]),
        "timestart": f_inicio.isoformat(),
        "timeend": f_fin.isoformat()
    }
    print("filter: %s" % json.dumps(filter))
    response = requests.get(
        apiLoginParams["url"] + '/obs/mareas_rdp',
        params = filter,
        headers = {'Authorization': 'Bearer ' + apiLoginParams["token"]},
        proxies = proxy_dict
    )
    if response.status_code != 200:
        raise Exception("request failed: %s" % response.text)
    json_response = response.json()
    df_todo = pd.DataFrame.from_dict(json_response,orient='columns')
    arg = pytz.timezone('America/Argentina/Buenos_Aires')
    df_todo['timestart'] = [datetime.datetime.strptime(i,'%Y-%m-%dT%H:%M:%S.%f%z').astimezone(arg).replace(tzinfo=None) for i in df_todo['timestart']]
    df_todo['fecha_emision'] = [datetime.datetime.strptime(i,'%Y-%m-%dT%H:%M:%S.%f%z').astimezone(arg).replace(tzinfo=None) for i in df_todo['fecha_emision']]
    return df_todo


def pronofrente(ahora,bbdd_loc):
    # Conecta BBDD Local
    connLoc = sqlite3.connect(bbdd_loc)
    
    '''Conecta con BBDD'''
    #Conecta BBDD Alertas
    #try:
    #   connIna = psycopg2.connect("dbname='" + dbConnParams["dbname"] + "' user='" + dbConnParams["user"] + "' host='" + dbConnParams["host"] + "'")
       #cur = conn.cursor()
    #except:
    #   print( "No se ha podido establecer conexion.")

    #ahora = datetime.datetime.now()
    f_inicio = (ahora - timedelta(days=15)).replace(hour=0, minute=0, second=0)
    f_fin = (ahora + timedelta(days=5))

    plotea = False

    # OBSERVADOS
    #Parametros para la consulta SQL a la BBDD
    """ paramH0 = (f_inicio, f_fin,) 
    sql_query = ('''SELECT timestart as fecha, valor as h_obs
                  FROM alturas_all
                  WHERE  timestart BETWEEN %s AND %s AND unid = 52;''')       """   #Consulta SQL
    #df_sferObs = pd.read_sql_query(sql_query, connIna, params=paramH0)             #Toma los datos de la BBDD
    df_sferObs = consultaAPI(f_inicio,f_fin,[52])
    df_sferObs = df_sferObs[["fecha","valor"]].rename(columns={"valor":"h_obs"})
    keys =  pd.to_datetime(df_sferObs['fecha'])
    df_sferObs.set_index(keys, inplace=True)

    # SIMULADO
    """ paramH1 = (f_inicio, f_fin)
    sql_query = ('''SELECT unid as id, timestart as fecha,
                    altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met
                    FROM alturas_marea_suma 
                    WHERE  timestart BETWEEN %s AND %s AND unid = 1838;''')              #Consulta SQL """

    #df_sferSim = pd.read_sql_query(sql_query, connIna, params=paramH1)								#Toma los datos de la BBDD	
    df_sferSim = consultaAPI(f_inicio,f_fin,[6041,6054])
    df_sferAst = df_sferSim.loc[df_sferSim["id"]==6041]
    df_sferMet = df_sferSim.loc[df_sferSim["id"]==6054]
    keys =  pd.to_datetime(df_sferAst['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
    df_sferAst.set_index(keys, inplace=True)
    keys =  pd.to_datetime(df_sferMet['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
    df_sferMet.set_index(keys, inplace=True)
    df_sferAst['valor'] = df_sferAst['valor'] + 0.53
    df_merge = pd.merge(df_sferAst,df_sferMet,left_index=True,right_index=True)
    df_merge.rename(columns={"valor_x":"h_ast_ign","valor_y":"h_met"},inplace=True)
    df_merge['h_suma'] = df_merge['h_ast_ign'] + df_merge['h_met']
    df_sferSim = df_merge[["h_ast_ign","h_met","h_suma"]]

    indexUnico = pd.date_range(start=df_sferSim.index.min(), end=df_sferSim.index.max(), freq='15min')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
    df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    df_base.index = df_base.index.round("15min")

    df_base = df_base.join(df_sferSim[['h_ast_ign','h_met','h_suma']], how = 'left')
    df_base = df_base.join(df_sferObs['h_obs'], how = 'left')
    df_base['h_obs'] = df_base['h_obs'].interpolate(limit=2)

    df_base_0 = df_base.dropna().copy()

    if plotea:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df_base_0.index, df_base_0['h_obs'],'o',label='h_obs',linewidth=1)
        ax.plot(df_base_0.index, df_base_0['h_obs'],'-',color='k',linewidth=1)
        
        ax.plot(df_base_0.index, df_base_0['h_suma'],'-',label='h_sim: solo Suma',linewidth=2)
        ax.plot(df_base_0.index, df_base_0['h_ast_ign'],'-',label='h_ast_ign',linewidth=0.8)
        ax.plot(df_base_0.index, df_base_0['h_met'],'-',label='h_met',linewidth=0.8)
        
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.7)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()

    ## Modelo
    train = df_base_0[:].copy()
    var_obj = 'h_obs'
    covariav = ['h_ast_ign','h_met']
    lr = linear_model.LinearRegression()
    X_train = train[covariav]
    Y_train = train[var_obj]
    lr.fit(X_train,Y_train)

    # Create the test features dataset (X_test) which will be used to make the predictions.
    X_test = train[covariav].values 
    # The labels of the model
    Y_test = train[var_obj].values
    Y_predictions = lr.predict(X_test)
    train['Y_predictions'] = Y_predictions

    # The coefficients
    print('Coefficients: \n', lr.coef_)
    # The mean squared error
    mse = mean_squared_error(Y_test, Y_predictions)
    print('Mean squared error: %.5f' % mse)
    # The coefficient of determination: 1 is perfect prediction
    coefDet = r2_score(Y_test, Y_predictions)
    print('r2_score: %.5f' % coefDet)
    train['Error_pred'] =  train['Y_predictions']  - train[var_obj]

    if plotea:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        # Plot outputs
        plt.scatter(Y_predictions, Y_test,  label='A01')
        plt.xlabel('h_sim', size=12)
        plt.ylabel('h_obs', size=12)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        # Plot outputs
        plt.scatter(train['Y_predictions'], train['Error_pred'], label='Error')
        plt.xlabel('H Sim', size=12)
        plt.ylabel('Error', size=12)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()

    # Plot final
    f_guarda = (ahora - timedelta(days=1)).replace(hour=0, minute=0, second=0)

    # Pronostico
    covariav = ['h_ast_ign','h_met']
    prediccion = lr.predict(df_sferSim[covariav].values)
    df_sferSim['h_sim'] = prediccion

    # Guarda en BBDD LOCAL

    df_sferSim['fecha'] = df_sferSim.index
    df_sim_guarda = df_sferSim[df_sferSim['fecha']>f_guarda].copy()
    df_sim_guarda = df_sim_guarda[['fecha','h_sim']]

    mes = str(ahora.month)
    dia = str(ahora.day)
    hora = str(ahora.hour)
    cod = ''.join([mes, dia, hora])

    df_sim_guarda['emision'] = cod
    df_sim_guarda['Id'] = 2

    print(df_sim_guarda.head())
    print(df_sim_guarda.columns)
    
    df_sim_guarda['h_sim'] =  df_sim_guarda['h_sim'] - 0.53
    df_sim_guarda.to_sql('PronoFrente', con = connLoc, if_exists='replace',index=False)

    # Plotea
    if False:
        quant_Err = train['Error_pred'].quantile([0.05,.25, .75,0.95])
        df_sferSim['e_pred_05'] = df_sferSim['h_sim'] + quant_Err[0.05]
        df_sferSim['e_pred_25'] = df_sferSim['h_sim'] + quant_Err[0.25]
        df_sferSim['e_pred_75'] = df_sferSim['h_sim'] + quant_Err[0.75]
        df_sferSim['e_pred_95'] = df_sferSim['h_sim'] + quant_Err[0.95]

        # PLOT
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(df_sferSim['fecha'], df_sferSim['h_sim'],label='Nivel Pronosticado',linewidth=3)

        # ax.plot(df_sferSim['fecha'], df_sferSim['h_ast_ign'],label='Astronomica',linewidth=1)
        # ax.plot(df_sferSim['fecha'], df_sferSim['h_met'],label='Meteorologica',linewidth=1)
        # ax.plot(df_sferSim['fecha'], df_sferSim['altura'],label='Pronóstico 0',linewidth=3)
        ax.plot(df_sferObs['fecha'], df_sferObs['altura'],'o',color='k',label='Nivel Observado',linewidth=3)
        ax.plot(df_sferObs['fecha'], df_sferObs['altura'],'-',color='k',linewidth=1,label="_altura")

        ax.plot(df_sferSim['fecha'], df_sferSim['e_pred_05'],'-',color='k',linewidth=0.5,alpha=0.75,label="_p95")
        #ax.plot(df_sferSim['fecha'], df_sferSim['e_pred_25'],'-',color='k',linewidth=0.5,alpha=0.75)#label='0.25',
        #ax.plot(df_sferSim['fecha'], df_sferSim['e_pred_75'],'-',color='k',linewidth=0.5,alpha=0.75)#label='0.75',
        ax.plot(df_sferSim['fecha'], df_sferSim['e_pred_95'],'-',color='k',linewidth=0.5,alpha=0.75,label="_p05")

        ax.fill_between(df_sferSim['fecha'],df_sferSim['e_pred_05'], df_sferSim['e_pred_95'],alpha=0.1,label='Banda de error')
        #ax.plot(df_sferSim['fecha'], df_sferSim['e_pred_95'],color='k',label='0.95 ',linewidth=0.5)

        # Lineas: 1 , 1.5 y 2 mts
        xmin=df_sferSim['fecha'].min()
        xmax=df_sferSim['fecha'].max()

        plt.hlines(3.5, xmin, xmax, colors='r', linestyles='-.', label='Evacuación',linewidth=1.5)
        plt.hlines(3, xmin, xmax, colors='y', linestyles='-.', label='Alerta',linewidth=1.5)

        # fecha emision
        plt.axvline(x=ahora,color="black", linestyle="--",linewidth=2)#,label='Fecha de emisión')

        bbox = dict(boxstyle="round", fc="0.7")
        arrowprops = dict(
            arrowstyle="->",
            connectionstyle="angle,angleA=0,angleB=90,rad=10")
        offset = 10

        #xycoords='figure pixels',
        xdisplay = ahora + timedelta(days=1)
        ax.annotate('Pronóstico a 4 días',
            xy=(xdisplay, 3.4), xytext=(0.5*offset, -offset), textcoords='offset points',
            bbox=bbox, fontsize=18)#arrowprops=arrowprops
            
        xdisplay = ahora - timedelta(days=1.2)
        ax.annotate('Días pasados',
            xy=(xdisplay, 3.4), xytext=(0.5*offset, -offset), textcoords='offset points',
            bbox=bbox, fontsize=18)

        ax.annotate('Fecha de emisión',
            xy=(ahora, -0.35),fontsize=15, xytext=(ahora+timedelta(days=0.3), -0.30), arrowprops=dict(facecolor='black',shrink=0.05))

        ax.set_ylim(-0.5,3.5)
        ax.set_xlim(xmin,xmax)

        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=16)
        plt.ylabel('Nivel [m]', size=20)
        plt.legend(prop={'size':18},loc=1,ncol=1 )
        # plt.title('nombre')

        date_form = DateFormatter("%H hrs \n %d-%b")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_minor_locator(mdates.HourLocator((0,6,12,18,)))

        plt.tight_layout()
        # plt.show()
        # plt.close()

        nameout = 'Prono_SanFernando.png'
        plt.savefig(nameout, format='png')# , dpi=200, facecolor='w', edgecolor='w',bbox_inches = 'tight', pad_inches = 0
        plt.close()


    ##################   Nueva Palmira
    f_inicio_m = datetime.datetime.now() - timedelta(days=180)
    f_inicio_m =  f_inicio_m.replace(day=1,hour=0,minute=0,second=0,microsecond=0) # datetime.datetime(2020,1, 1, 00, 0, 0, 0)
    f_fin = (ahora + timedelta(days=5))

    iDest = tuple([1699,2233])
    Est_Uruguay = {1699:'NuevaPalmira',2233:'ConcepciUru'}
    
    # OBSERVADOS
    #Parametros para la consulta SQL a la BBDD
    """ paramH0 = (f_inicio_m, f_fin,iDest)  
    sql_query = ('''SELECT unid as id, timestart as fecha, valor as h_obs
                  FROM alturas_all
                  WHERE  timestart BETWEEN %s AND %s AND unid IN %s;''')  #Consulta SQL """
    #df_Obs0 = pd.read_sql_query(sql_query, connIna, params=paramH0)              #Toma los datos de la BBDD
    iDserie = [3280,7061]
    df_Obs0 = consultaAPI(f_inicio_m,f_fin,iDserie)
    df_Obs0.rename(columns={"valor":"h_obs"},inplace=True)
    #keys =  pd.to_datetime(df_Obs0['fecha'])
    #df_Obs0.set_index(keys, inplace=True)

    #df_Obs0['fecha'] = pd.to_datetime(df_Obs0['fecha'])#.round('min')
    # Estaciones en cada columnas
    df_Obs = pd.pivot_table(df_Obs0, values='h_obs', index=['fecha'], columns=['id'])
    columns = dict()
    for i in range(len(iDserie)):
        columns[iDserie[i]] = iDest[i]
    
    df_Obs.rename(columns=columns,inplace=True)
    df_Obs_H = df_Obs.resample('H').mean()

    # Nombres a los calumnas
    df_Obs_H = df_Obs_H.rename(columns=Est_Uruguay)
    if plotea:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        for col in df_Obs_H.columns:
          ax.plot(df_Obs_H.index, df_Obs_H[col], linestyle='-',label=col)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=16)
        plt.ylabel('Nivel [m]', size=20)
        plt.legend(prop={'size':18},loc=1,ncol=1 )
        plt.show()

    ###############################
    paramH1 = (f_inicio, f_fin)
    # sql_query = ('''SELECT  timestart as fecha, fecha_emision, 
    #                         altura_meteo, altura_astro, altura_suma, altura_suma_corregida
    #                 FROM alturas_marea_full 
    #                 WHERE  timestart BETWEEN %s AND %s AND estacion_id = 1843;''')#WHERE  
    # df_Sim0 = pd.read_sql_query(sql_query, connIna, params=paramH1)	
    df_Sim0 = consultaAPImareas(f_inicio,f_fin,[1843])
    #df_Sim0['timestart'] =  pd.to_datetime(df_Sim0['timestart'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
    #df_Sim0['fecha_emision'] =  pd.to_datetime(df_Sim0['fecha_emision'])

    # Calcula la cantidad de horas de pronóstico (anticipo): fecha del dato menos fecha de emision
    df_Sim0['horas_prono'] = (df_Sim0['timestart'] - df_Sim0['fecha_emision']).astype('timedelta64[h]')
    # Hora del pronostico
    df_Sim0['hora'] = df_Sim0.apply(lambda row: row['timestart'].hour,axis=1)

    # Lleva las salidas del modelo en cota IGN al cero local
    # df_Sim0['altura_astro'] = df_Sim0['altura_astro'].add(0.53)
    df_Sim0['Cat_anticipo'] = np.nan
    df_Sim0.loc[df_Sim0['horas_prono'].isin([0,1,2,3,4,5]), 'Cat_anticipo'] = 'A01'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([6,7,8,9,10,11]), 'Cat_anticipo'] = 'A02'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([12,13,14,15,16,17]), 'Cat_anticipo'] = 'A03'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([18,19,20,21,22,23]), 'Cat_anticipo'] = 'A04'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([24,25,26,27,28,29]), 'Cat_anticipo'] = 'A05'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([30,31,32,33,34,35]), 'Cat_anticipo'] = 'A06'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([36,37,38,39,40,41]), 'Cat_anticipo'] = 'A07'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([42,43,44,45,46,47]), 'Cat_anticipo'] = 'A08'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([48,49,50,51,52,53]), 'Cat_anticipo'] = 'A09'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([54,55,56,57,58,59]), 'Cat_anticipo'] = 'A10'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([60,61,62,63,64,65]), 'Cat_anticipo'] = 'A11'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([66,67,68,69,70,71]), 'Cat_anticipo'] = 'A12'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([72,73,74,75,76,77]), 'Cat_anticipo'] = 'A13'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([78,79,80,81,82,83]), 'Cat_anticipo'] = 'A14'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([84,85,86,87,88,89]), 'Cat_anticipo'] = 'A15'
    df_Sim0.loc[df_Sim0['horas_prono'].isin([90,91,92,93,94,95]), 'Cat_anticipo'] = 'A16'

    df_Sim0.rename(columns={"timestart":"fecha"},inplace=True)
    #print(df_Sim0["Cat_anticipo"].value_counts(ascending=False))
    df_NP_PronoCat = pd.pivot_table(df_Sim0, 
                            values=['altura_meteo'],#, 'altura_suma', 'altura_suma_corregida'], 'fecha_emision','altura_astro'
                            index=['fecha','altura_astro'],
                            columns=['Cat_anticipo'], aggfunc=np.sum)

    df_NP_PronoCat.columns = df_NP_PronoCat.columns.get_level_values(1)
    df_NP_PronoCat = df_NP_PronoCat.reset_index(level=[1,])

    l_cat = ['A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','A11','A12','A13','A14','A15','A16']

    if plotea:
        df_NP_PronoCat = df_NP_PronoCat.fillna(-2)
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        for columas in l_cat:
          ax.plot(df_NP_PronoCat.index, df_NP_PronoCat[columas], linestyle='-',label=columas)
        ax.plot(df_NP_PronoCat.index, df_NP_PronoCat['altura_astro'], linestyle='-',label='Astro')
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=16)
        plt.ylabel('Nivel [m]', size=20)
        plt.legend(prop={'size':18},loc=1,ncol=1 )

        plt.show()
        df_NP_PronoCat = df_NP_PronoCat.replace(-2,np.nan)

    if plotea:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        for col in df_Obs_H.columns:
          ax.plot(df_Obs_H.index, df_Obs_H[col], linestyle='-',label=col)
        ax.plot(df_NP_PronoCat.index, df_NP_PronoCat['A01'], linestyle='-',label='h Meteorologica')
        ax.plot(df_NP_PronoCat.index, df_NP_PronoCat['altura_astro'], linestyle='-',label='h Astronomica')
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=16)
        plt.ylabel('Nivel [m]', size=20)
        plt.legend(prop={'size':18},loc=1,ncol=1 )
        plt.show()

    ### PRIMERA CORRECCION
    df_NP_PronoCat.index = df_NP_PronoCat.index + timedelta(hours=1)
    #df_NP_PronoCat['altura_astro'] = df_NP_PronoCat['altura_astro'] + 0.41

    ## Union #######################################################
    #Crea DF base:
    indexUnico = pd.date_range(start=df_Sim0['fecha'].min(), end=df_Sim0['fecha'].max(), freq='H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
    df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    df_base.index = df_base.index.round("H")

    # Une obs y sim
    df_base = df_base.join(df_NP_PronoCat[['A01','altura_astro']], how = 'left')
    df_base = df_base.join(df_Obs_H[['ConcepciUru','NuevaPalmira']], how = 'left')

    if plotea:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        #for col in df_base.columns:
        ax.plot(df_base.index, df_base['NuevaPalmira'], linestyle='-',label='Nueva Palmira')
        ax.plot(df_base.index, df_base['ConcepciUru'], linestyle='-',label='Concep. del Uruguay')
        ax.plot(df_base.index, df_base['altura_astro'], linestyle='-',label='h Astronomica')
        ax.plot(df_base.index, df_base['A01'], linestyle='-',label='h Meteorologica')
        
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=16)
        plt.ylabel('Nivel [m]', size=20)
        plt.legend(prop={'size':18},loc=1,ncol=1 )
        plt.show()

    df_base = df_base.dropna().copy()
    n_datos = len(df_base)
    print (n_datos)
    print ('Dias: :', round(n_datos/24,0))

    ###########################
    df_base['CUru_dly'] = df_base['ConcepciUru'].shift(24*4)
    df_base['CUru_dly'] = df_base['CUru_dly'].interpolate(limit_direction="forward")
    df_base['h_met_Mavg'] = df_base['A01'].rolling(24, min_periods=1).mean()
    del df_base['ConcepciUru']
    df_base = df_base.dropna()

    print('Cantidad de datos de entrenamiento:',len(df_base))
    ###########################

    ## Modelo
    train = df_base[:].copy()
    var_obj = 'NuevaPalmira'
    covariav = ['altura_astro','A01','h_met_Mavg','CUru_dly']
    lr = linear_model.LinearRegression()
    X_train = train[covariav]
    Y_train = train[var_obj]
    lr.fit(X_train,Y_train)

    # Create the test features dataset (X_test) which will be used to make the predictions.
    X_test = train[covariav].values 
    # The labels of the model
    Y_test = train[var_obj].values
    Y_predictions = lr.predict(X_test)
    train['Y_predictions'] = Y_predictions

    # The coefficients
    print('Coefficients B0: \n', lr.intercept_)
    print('Coefficients: \n', lr.coef_)

    # The mean squared error
    mse = mean_squared_error(Y_test, Y_predictions)
    print('Mean squared error: %.5f' % mse)
    # The coefficient of determination: 1 is perfect prediction
    coefDet = r2_score(Y_test, Y_predictions)
    print('r2_score: %.5f' % coefDet)
    train['Error_pred'] =  train['Y_predictions']  - train[var_obj]

    if plotea:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        # Plot outputs
        plt.scatter(Y_predictions, Y_test)
        plt.plot([-0.5, 2.5], [-0.5, 2.5], color = 'black', linewidth = 2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.xlabel('h_sim', size=12)
        plt.ylabel('h_obs', size=12)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        # Plot outputs
        plt.scatter(train['Y_predictions'], train['Error_pred'])#, label='Error')
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.xlabel('H Sim', size=12)
        plt.ylabel('Error', size=12)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()

    quant_Err = train['Error_pred'].quantile([.02,.05,.95,.98])

    if plotea:
        test_mPlot = train[:]
        yplot = Y_predictions[:]

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(test_mPlot.index, test_mPlot['NuevaPalmira'],'-.',label='NuevaPalmira',linewidth=2)
        ax.plot(test_mPlot.index, yplot,'-',label='Simulado',linewidth=2)

        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()

    ####################################################################################################
    DaysMod = 10
    f_fin = ahora
    f_inicio = (f_fin - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)
    f_fin = (f_fin + timedelta(days=5))

    iDest = tuple([1699,2233])
    Est_Uruguay = {1699:'NuevaPalmira',2233:'ConcepciUru'}

    # OBSERVADOS
    #Parametros para la consulta SQL a la BBDD
    # paramH0 = (f_inicio, f_fin,iDest)  
    # sql_query = ('''SELECT unid as id, timestart as fecha, valor as h_obs
    #               FROM alturas_all
    #               WHERE  timestart BETWEEN %s AND %s AND unid IN %s;''')  #Consulta SQL
    # df_Obs0 = pd.read_sql_query(sql_query, connIna, params=paramH0)              #Toma los datos de la BBDD
    iDserie = [3280,7061]
    df_Obs0 = consultaAPI(f_inicio,f_fin,iDserie)
    df_Obs0.rename(columns={"valor":"h_obs"},inplace=True)
    df_Obs0['fecha'] = pd.to_datetime(df_Obs0['fecha']).round('min')

    # Estaciones en cada columnas
    df_Obs = pd.pivot_table(df_Obs0, values='h_obs', index=['fecha'], columns=['id'])
    print("Resample horario")
    columns = dict()
    for i in range(len(iDserie)):
        columns[iDserie[i]] = iDest[i]
    
    df_Obs.rename(columns=columns,inplace=True)
    df_Obs_H = df_Obs.resample('H').mean()
    # Nombres a los calumnas
    df_Obs_H = df_Obs_H.rename(columns=Est_Uruguay)

    # SIMULADO
    paramH1 = (f_inicio, f_fin)         #Parametros para la consulta SQL a la BBDD
    # sql_query = ('''SELECT unid as id, timestart as fecha,
    #                 altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met
    #                 FROM alturas_marea_suma_corregida 
    #                 WHERE  timestart BETWEEN %s AND %s AND unid = 1843;''')              #Consulta SQL
    # df_Sim = pd.read_sql_query(sql_query, connIna, params=paramH1)								#Toma los datos de la BBDD	
    df_Sim = consultaAPI(f_inicio,f_fin,[6041,6054])
    df_Ast = df_Sim.loc[df_Sim["id"]==6041]
    df_Met = df_Sim.loc[df_Sim["id"]==6054]
    keys =  pd.to_datetime(df_Ast['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
    df_Ast.set_index(keys, inplace=True)
    keys =  pd.to_datetime(df_Met['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
    df_Met.set_index(keys, inplace=True)
    # df_Ast['valor'] = df_Ast['valor'] + 0.53
    df_merge = pd.merge(df_Ast,df_Met,left_index=True,right_index=True)
    df_merge.rename(columns={"valor_x":"h_ast_ign","valor_y":"h_met"},inplace=True)
    df_merge['h_suma'] = df_merge['h_ast_ign'] + df_merge['h_met']
    df_Sim = df_merge[["h_ast_ign","h_met","h_suma"]]

    df_Sim.index = df_Sim.index + timedelta(hours=1)

    ## Union
    indexUnico = pd.date_range(start=df_Sim.index.min(), end=df_Sim.index.max(), freq='H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
    df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    df_base.index = df_base.index.round("H")

    df_base = df_base.join(df_Sim[['h_met','h_ast_ign']], how = 'left')
    df_base = df_base.join(df_Obs_H['ConcepciUru'], how = 'left')

    df_base['CUru_dly'] = df_base['ConcepciUru'].shift(24*4)
    df_base['CUru_dly'] = df_base['CUru_dly'].interpolate(limit_direction="forward")

    df_base['h_met_Mavg'] = df_base['h_met'].rolling(24, min_periods=1).mean()

    del df_base['ConcepciUru']

    df_base = df_base.dropna()
    #print(df_base.tail(5))

    covariav = ['h_ast_ign','h_met','h_met_Mavg','CUru_dly']
    prediccion = lr.predict(df_base[covariav].values)
    df_base['h_sim'] = prediccion

    # Guarda en BBDD LOCAL
    df_base['fecha'] = df_base.index
    df_sim_guarda = df_base[df_base['fecha']>f_guarda].copy()

    df_sim_guarda = df_sim_guarda[['fecha','h_sim']]

    mes = str(ahora.month)
    dia = str(ahora.day)
    hora = str(ahora.hour)
    cod = ''.join([mes, dia, hora])

    df_sim_guarda['emision'] = cod
    df_sim_guarda['Id'] = 3

    df_sim_guarda['h_sim'] =  df_sim_guarda['h_sim'] + 0.0275
    df_sim_guarda.to_sql('PronoFrente', con = connLoc, if_exists='append',index=False)


    df_base['e_pred_05'] = df_base['h_sim'] + quant_Err[0.05]
    df_base['e_pred_02'] = df_base['h_sim'] + quant_Err[0.02]
    df_base['e_pred_98'] = df_base['h_sim'] + quant_Err[0.98]
    df_base['e_pred_95'] = df_base['h_sim'] + quant_Err[0.95]



    if plotea:
        # PLOT
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(df_base.index, df_base['h_sim'],label='Nivel Pronosticado',linewidth=3)

        # ax.plot(df_base.index, df_base['h_ast_ign'],label='Astronomica',linewidth=1)
        # ax.plot(df_base.index, df_base['h_met'],label='Meteorologica',linewidth=1)
        #ax.plot(df_simulado['fecha'], df_simulado['altura'],label='Pronóstico 0',linewidth=3)
        ax.plot(df_Obs_H.index, df_Obs_H['NuevaPalmira'],'o',color='k',label='Nivel Observado',linewidth=3)
        ax.plot(df_Obs_H.index, df_Obs_H['NuevaPalmira'],'-',color='k',linewidth=1)

        #ax.plot(df_base.index, df_base['e_pred_02'],'-',color='k',linewidth=0.5,alpha=0.75)#label='0.05',
        ax.plot(df_base.index, df_base['e_pred_05'],'-',color='k',linewidth=0.5,alpha=0.75)#label='0.25',
        ax.plot(df_base.index, df_base['e_pred_95'],'-',color='k',linewidth=0.5,alpha=0.75)#label='0.75',
        #ax.plot(df_base.index, df_base['e_pred_98'],'-',color='k',linewidth=0.5,alpha=0.75)#label='0.95',
        ax.fill_between(df_base.index,df_base['e_pred_05'], df_base['e_pred_95'],alpha=0.1,label='Banda de error')
        #ax.plot(df_simulado['fecha'], df_simulado['e_pred_95'],color='k',label='0.95 ',linewidth=0.5)

        # Lineas: 1 , 1.5 y 2 mts
        xmin=df_base.index.min()
        xmax=df_base.index.max()

        # En cero escala Paranacito Prefectura
        plt.hlines(5, xmin, xmax, colors='r', linestyles='-.', label='Evacuación',linewidth=1.5)
        #plt.hlines(2.3, xmin, xmax, colors='y', linestyles='-.', label='Alerta',linewidth=1.5)
        #plt.hlines(1.1, xmin, xmax, colors='y', linestyles='-.', label='Aguas Bajas',linewidth=1.5)

        # fecha emision
        plt.axvline(x=ahora,color="black", linestyle="--",linewidth=2)#,label='Fecha de emisión')

        bbox = dict(boxstyle="round", fc="0.7")
        arrowprops = dict(
            arrowstyle="->",
            connectionstyle="angle,angleA=0,angleB=90,rad=10")
        offset = 10

        #xycoords='figure pixels',
        xdisplay = ahora + timedelta(days=1)
        ax.annotate('Pronóstico a 4 días',
            xy=(xdisplay, 3.4), xytext=(0.5*offset, -offset), textcoords='offset points',
            bbox=bbox, fontsize=18)#arrowprops=arrowprops

        xdisplay = ahora - timedelta(days=1.5)
        ax.annotate('Días pasados',
            xy=(xdisplay, 3.4), xytext=(0.5*offset, -offset), textcoords='offset points',
            bbox=bbox, fontsize=18)

        ax.annotate('Fecha de emisión',
            xy=(ahora, -0.35),fontsize=15, xytext=(ahora+timedelta(days=0.3), -0.30), arrowprops=dict(facecolor='black',shrink=0.05))

        ax.set_ylim(-0.5,3.5)
        ax.set_xlim(xmin,xmax)

        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=16)
        plt.ylabel('Nivel [m]', size=20)
        plt.legend(prop={'size':18},loc=2,ncol=1 )
        # plt.title('nombre')

        date_form = DateFormatter("%H hrs \n %d-%b")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_minor_locator(mdates.HourLocator((0,6,12,18,)))

        plt.tight_layout()
        # plt.show()
        # plt.close()

        nameout = working_dir + '/Prono_NPalmiraV3.png'    
        plt.savefig(nameout, format='jpg')# , dpi=200, facecolor='w', edgecolor='w',bbox_inches = 'tight', pad_inches = 0

        plt.close()
