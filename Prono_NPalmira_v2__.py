# -*- coding: utf-8 -*-
import psycopg2
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib as matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import requests
import json

with open("dbConnectionParams.json") as f:
	connParams = json.load(f)

dbConnectionParams = "dbname='" + connParams["dbname"] + "' user='" + connParams["user"] + "' host='" + connParams["host"] + "' port='" + connParams["port"] + "'"

with open("apiLoginParams.json") as f:
	apiLoginParams = json.load(f)

apiUrl = apiLoginParams["url"] 
del apiLoginParams["url"]
upsertSimulado = 1
outputFile = 'productos/Prono_NPalmira.png'

'''Conecta con BBDD'''
try:
   conn = psycopg2.connect(dbConnectionParams)
   cur = conn.cursor()
except:
   print( "No se ha podido establecer conexion.")

plotea = False


ahora = datetime.datetime.now()

f_inicio = datetime.datetime(2020,1, 1, 00, 0, 0, 0)
f_fin = ahora

# DaysMod = 20   
# f_inicio = (f_fin - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)

iDest = tuple([1699,2233])
Est_Uruguay = {1699:'NuevaPalmira',2233:'ConcepciUru'}

# OBSERVADOS
#Parametros para la consulta SQL a la BBDD
paramH0 = (f_inicio, f_fin,iDest)  
sql_query = ('''SELECT unid as id, timestart as fecha, valor as h_obs
              FROM alturas_all
              WHERE  timestart BETWEEN %s AND %s AND unid IN %s;''')  #Consulta SQL
df_Obs0 = pd.read_sql_query(sql_query, conn, params=paramH0)              #Toma los datos de la BBDD
df_Obs0['fecha'] = pd.to_datetime(df_Obs0['fecha'])#.round('min')

# Estaciones en cada columnas
df_Obs = pd.pivot_table(df_Obs0, values='h_obs', index=['fecha'], columns=['id'])
print("Resample horario")
df_Obs_H = df_Obs.resample('H').mean()

# Nombres a los calumnas
df_Obs_H = df_Obs_H.rename(columns=Est_Uruguay)

#print(df_Obs_H.tail(3))
#df_Obs_H = df_Obs_H.interpolate()
for coli in df_Obs_H.columns:
  print(coli)
  print('NaNs: ',df_Obs_H[coli].isna().sum())

if plotea:
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    for col in df_Obs_H.columns:
      ax.plot(df_Obs_H.index, df_Obs_H[col], linestyle='-',label=col)
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.7)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Fecha', size=16)
    plt.ylabel('Nivel [m]', size=20)
    plt.legend(prop={'size':18},loc=1,ncol=1, framealpha=0.3)

    plt.show()


###############################

paramH1 = (f_inicio, f_fin)

sql_query = ('''SELECT  timestart as fecha, fecha_emision, 
                        altura_meteo, altura_astro, altura_suma, altura_suma_corregida
                FROM alturas_marea_full 
                WHERE  timestart BETWEEN %s AND %s AND estacion_id = 1843;''')#WHERE  
df_Sim0 = pd.read_sql_query(sql_query, conn, params=paramH1)	
df_Sim0['fecha'] =  pd.to_datetime(df_Sim0['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
df_Sim0['fecha_emision'] =  pd.to_datetime(df_Sim0['fecha_emision'])

# Calcula la cantidad de horas de pronóstico (anticipo): fecha del dato menos fecha de emision
df_Sim0['horas_prono'] = (df_Sim0['fecha'] - df_Sim0['fecha_emision']).astype('timedelta64[h]')
# Hora del pronostico
df_Sim0['hora'] = df_Sim0.apply(lambda row: row['fecha'].hour,axis=1)

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

print(df_Sim0["Cat_anticipo"].value_counts(ascending=False))

df_NP_PronoCat = pd.pivot_table(df_Sim0, 
                       values=['altura_meteo'],#, 'altura_suma', 'altura_suma_corregida'], 'fecha_emision','altura_astro'
                       index=['fecha','altura_astro'],
                       columns=['Cat_anticipo'], aggfunc=np.sum)

df_NP_PronoCat.columns = df_NP_PronoCat.columns.get_level_values(1)
df_NP_PronoCat = df_NP_PronoCat.reset_index(level=[1,])

l_cat = ['A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','A11','A12','A13','A14','A15','A16']
for cat in l_cat:
  print('NaN : '+cat+' '+str(df_NP_PronoCat[cat].isna().sum()))

df_NP_PronoCat = df_NP_PronoCat.fillna(-2)

if plotea:
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    for columas in l_cat:
      ax.plot(df_NP_PronoCat.index, df_NP_PronoCat[columas], linestyle='-',label=columas)
    ax.plot(df_NP_PronoCat.index, df_NP_PronoCat['altura_astro'], linestyle='-',label='Astro')
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Fecha', size=16)
    plt.ylabel('Nivel [m]', size=20)
    plt.legend(prop={'size':18},loc=1,ncol=1, framealpha=0.3 )

    plt.show()
    df_NP_PronoCat = df_NP_PronoCat.replace(-2,np.nan)


if False:
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
    plt.legend(prop={'size':18},loc=1,ncol=1,framealpha=0.3 )

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

print(df_base.head())

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
    plt.legend(prop={'size':18},loc=1,ncol=1, framealpha=0.3 )

    plt.show()
    
    

for col in df_base.columns:
  print(col+' NaN : '+str(df_base[col].isna().sum()))

# Elimina filas con faltantes
print (len(df_base))

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
print(df_base.tail(5))


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
    plt.legend(prop={'size':16},loc=2,ncol=2, framealpha=0.3 )
    plt.show()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    # Plot outputs
    plt.scatter(train['Y_predictions'], train['Error_pred'])#, label='Error')
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.xlabel('H Sim', size=12)
    plt.ylabel('Error', size=12)
    plt.legend(prop={'size':16},loc=2,ncol=2, framealpha=0.3 )
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
    plt.legend(prop={'size':16},loc=2,ncol=2, framealpha=0.3 )
    plt.show()
    plt.close()


cut_labels = ['L1: < 0','L2: 0 - 0.5' ,'L3: 0.5 - 1.0','L4: 1.0 - 1.5','L5: 1.5 - 2.5']
cut_bins = [-0.5,0.0,0.5,1.0,1.5,2.5]
train['h_cat'] = pd.cut(train['NuevaPalmira'], bins=cut_bins, labels=cut_labels)
print(train['h_cat'].value_counts())

if plotea:  
    ax = sns.boxplot(x="h_cat", y="Error_pred", data=train,orient='v',order=cut_labels)
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    #plt.title('Respecto a h Obs') 
    plt.xlabel('h Obs', size=15)
    plt.ylabel('Residuos ', size=15)
    plt.show()
    plt.close()

    ax = sns.boxplot(x="Error_pred", data=train,orient='h',)
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.xlabel('Residuos ', size=15)
    plt.show()
    plt.close()






## Pronóstico 
'''Conecta con BBDD'''
#Conecta BBDD Alertas
try:
   conn = psycopg2.connect("dbname='meteorology' user='sololectura' host='correo.ina.gob.ar' port='9049'")
   cur = conn.cursor()
except:
   print( "No se ha podido establecer conexion.")

# Plot final
ahora = datetime.datetime.now()
DaysMod = 10
f_fin = ahora
f_inicio = (f_fin - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)
f_fin = (f_fin + timedelta(days=5))

iDest = tuple([1699,2233])
Est_Uruguay = {1699:'NuevaPalmira',2233:'ConcepciUru'}

# OBSERVADOS
#Parametros para la consulta SQL a la BBDD
paramH0 = (f_inicio, f_fin,iDest)  
sql_query = ('''SELECT unid as id, timestart as fecha, valor as h_obs
              FROM alturas_all
              WHERE  timestart BETWEEN %s AND %s AND unid IN %s;''')  #Consulta SQL
df_Obs0 = pd.read_sql_query(sql_query, conn, params=paramH0)              #Toma los datos de la BBDD
df_Obs0['fecha'] = df_Obs0['fecha'].dt.round('min') # pd.to_datetime(df_Obs0['fecha']).round('min')

# Estaciones en cada columnas
df_Obs = pd.pivot_table(df_Obs0, values='h_obs', index=['fecha'], columns=['id'])
print("Resample horario")
df_Obs_H = df_Obs.resample('H').mean()

# Nombres a los calumnas
df_Obs_H = df_Obs_H.rename(columns=Est_Uruguay)

#print(df_Obs_H.tail(3))
#df_Obs_H = df_Obs_H.interpolate()
for coli in df_Obs_H.columns:
  print(coli)
  print('NaNs: ',df_Obs_H[coli].isna().sum())

# SIMULADO
paramH1 = (f_inicio, f_fin)         #Parametros para la consulta SQL a la BBDD
# ~ sql_query = ('''SELECT unid as id, timestart as fecha,
                # ~ altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met,
                # ~ timeupdate as fecha_emision
                # ~ FROM alturas_marea_suma_corregida 
                # ~ WHERE  timestart BETWEEN %s AND %s AND unid = 1843;''')              #Consulta SQL
sql_query = ('''SELECT unid as id, timestart as fecha,
                altura_astronomica_ign as h_ast_ign, altura_meteorologica as h_met,
                timeupdate as fecha_emision
                FROM alturas_marea_suma
                WHERE  timestart BETWEEN %s AND %s AND unid = 1843;''')              #Consulta SQL
df_Sim = pd.read_sql_query(sql_query, conn, params=paramH1)								#Toma los datos de la BBDD	
df_Sim['fecha'] =  pd.to_datetime(df_Sim['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]

keys =  pd.to_datetime(df_Sim['fecha'])#, format='%Y-%m-%d')                     #Convierte a formato fecha la columna [fecha]
df_Sim.set_index(keys, inplace=True)

df_Sim.index = df_Sim.index + timedelta(hours=1)


## Union
indexUnico = pd.date_range(start=df_Sim['fecha'].min(), end=df_Sim['fecha'].max(), freq='H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
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
print(df_base.tail(5))

covariav = ['h_ast_ign','h_met','h_met_Mavg','CUru_dly']
prediccion = lr.predict(df_base[covariav].values)
df_base['h_sim'] = prediccion

df_base['e_pred_05'] = df_base['h_sim'] + quant_Err[0.05]
df_base['e_pred_02'] = df_base['h_sim'] + quant_Err[0.02]
df_base['e_pred_98'] = df_base['h_sim'] + quant_Err[0.98]
df_base['e_pred_95'] = df_base['h_sim'] + quant_Err[0.95]


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
ax.plot(df_base.index, df_base['e_pred_05'],'-',color='k',linewidth=0.5,alpha=0.75,label="")#label='0.25',
ax.plot(df_base.index, df_base['e_pred_95'],'-',color='k',linewidth=0.5,alpha=0.75,label="")#label='0.75',
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
plt.legend(prop={'size':18},loc=2,ncol=1, framealpha=0.3 )
# plt.title('nombre')

date_form = DateFormatter("%H hrs \n %d-%b")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_minor_locator(mdates.HourLocator((0,6,12,18,)))
###
class GridShader():
    def __init__(self, ax, first=True, **kwargs):
        self.spans = []
        self.sf = first
        self.ax = ax
        self.kw = kwargs
        self.ax.autoscale(False, axis="x")
        self.cid = self.ax.callbacks.connect('xlim_changed', self.shade)
        self.shade()
    def clear(self):
        for span in self.spans:
            try:
                span.remove()
            except:
                pass
    def shade(self, evt=None):
        self.clear()
        xticks = self.ax.get_xticks()
        xlim = self.ax.get_xlim()
        xticks = xticks[(xticks > xlim[0]) & (xticks < xlim[-1])]
        locs = np.concatenate(([[xlim[0]], xticks, [xlim[-1]]]))

        start = locs[1-int(self.sf)::2]  
        end = locs[2-int(self.sf)::2]

        for s, e in zip(start, end):
            self.spans.append(self.ax.axvspan(s, e, zorder=0, **self.kw))

gs = GridShader(ax, facecolor="lightgrey", first=False, alpha=0.7)
###
plt.tight_layout()
# plt.show()
# plt.close()

nameout = outputFile    
plt.savefig(nameout, format='png')# , dpi=200, facecolor='w', edgecolor='w',bbox_inches = 'tight', pad_inches = 0

plt.close()

df_base.reset_index(inplace=True)

# pone timezone a fecha
for i in df_base.index:
	df_base.at[i,"fecha"] = df_base.at[i,"fecha"].tz_localize("America/Argentina/Buenos_Aires")

if upsertSimulado:
	# df para UPSERT
	df_base = df_base.reset_index()
	df_para_upsert = df_base[['fecha','h_sim']].rename(columns = {'fecha':'timestart', 'h_sim':'valor'},inplace = False)
	df_para_upsert['qualifier'] = 'main'
	df_para_upsert = df_para_upsert.append(df_base[['fecha','e_pred_02']].rename(columns = {'fecha':'timestart', 'e_pred_02': 'valor'}), ignore_index=True)
	df_para_upsert['qualifier'].fillna(value='p02',inplace=True)
	df_para_upsert = df_para_upsert.append(df_base[['fecha','e_pred_05']].rename(columns = {'fecha':'timestart', 'e_pred_05': 'valor'}), ignore_index=True)
	df_para_upsert['qualifier'].fillna(value='p05',inplace=True)
	df_para_upsert = df_para_upsert.append(df_base[['fecha','e_pred_95']].rename(columns = {'fecha':'timestart', 'e_pred_95': 'valor'}), ignore_index=True)
	df_para_upsert['qualifier'].fillna(value='p95',inplace=True)
	df_para_upsert = df_para_upsert.append(df_base[['fecha','e_pred_98']].rename(columns = {'fecha':'timestart', 'e_pred_98': 'valor'}), ignore_index=True)
	df_para_upsert['qualifier'].fillna(value='p98',inplace=True)
	df_para_upsert['timeend'] = df_para_upsert['timestart']  # .map(lambda a : a.isoformat())
	para_upsert = {'forecast_date':df_Sim['fecha_emision'].max().isoformat(),
				 'series': [
					{
						'series_table': 'series',
						'series_id': 26203,
						'pronosticos': json.loads(df_para_upsert.to_json(orient='records',date_format='iso'))
					}
				]}

	# UPSERT Simulado
	with requests.Session() as session:
		try:
			login_response = session.post(apiUrl + "/login", data = json.dumps(apiLoginParams), headers={'Content-type':'application/json'})
			print(login_response.headers)
		except:
			print("login error. Reason:" + requests.reason +", text:"+requests.text)
		else:
			upsert_response = session.post(apiUrl + '/sim/calibrados/433/corridas',json=para_upsert)
			print(upsert_response.headers)
