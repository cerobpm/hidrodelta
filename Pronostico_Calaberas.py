import requests, psycopg2
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import json
import pytz


#matplotlib.use('Agg')

with open("apiLoginParams.json") as f:
	apiLoginParams = json.load(f)

## Carga Simulados
response = requests.get(
    apiLoginParams["url"] + '/sim/calibrados/288/corridas/last',
    params={'estacion_id': '5876'},
    headers={'Authorization': 'Bearer ' + apiLoginParams["token"]},
)

json_response = response.json()
forecast_date = json_response['forecast_date']
df_simulado = pd.DataFrame.from_dict(json_response['series'][0]['pronosticos'],orient='columns')
df_simulado = df_simulado.rename(columns={0:'fecha',1:'fecha2',2:'h_sim',3:'main'})
df_simulado = df_simulado[['fecha','h_sim']]
df_simulado['fecha'] = pd.to_datetime(df_simulado['fecha'])
df_simulado['h_sim'] = df_simulado['h_sim'].astype(float)


## Carga Observados
'''Conecta con BBDD'''
#Conecta BBDD Alertas
# try:
#    conn = psycopg2.connect("dbname='meteorology' user='sololectura' host='correo.ina.gob.ar' port='9049'")
#    cur = conn.cursor()
# except:
#    print( "No se ha podido establecer conexion.")


# Fechas
# f_inicio = datetime.datetime(2020,11, 21, 00, 0, 0, 0) 
# f_fin = datetime.datetime(2021, 6, 16, 00, 0, 0, 0)
f_inicio = df_simulado['fecha'].min()
f_fin = df_simulado['fecha'].max()

Estaciones = {1696:'Martinez',
              1257:'Borches',
              52:'SFernando',
              1698:'Carapachay',
              1699:'NPalmira',
              5876:'Carabelas'}
              
Ceros = {'Martinez': 0,
         'Borches': 0.682,
         'SFernando': -0.53,
         'Carapachay': -1.459,
         'NPalmira':0.41,
         'Carabelas':0}

# Aca se eligen las estaciones a consultar
idDest = 5876
seriesIdDest = 26206   # id de serie de altura observada en Carabelas
# paramH0 = (f_inicio, f_fin,idDest) # Parametros para la consulta SQL a la BBDD
# sql_query = ('''SELECT timestart as fecha, valor as h_obs
#                 FROM alturas_all
#                 WHERE  timestart BETWEEN %s AND %s AND unid = %s;''')      # Consulta SQL
# api_query = "/obs/puntual/observaciones?var_id=2&estacion_id=%s&timestart=%s&timeend=%s"
# df_Obs = pd.read_sql_query(sql_query, conn, params=paramH0)                # Toma los datos de la BBDD
## Carga Observados
response = requests.get(
    apiLoginParams["url"] + '/obs/puntual/observaciones',
    params={
        'series_id': seriesIdDest,
        'timestart': f_inicio,
        'timeend': f_fin
    },
    headers={'Authorization': 'Bearer ' + apiLoginParams["token"]},
)
json_response = response.json()
df_Obs = pd.DataFrame.from_dict(json_response) # ['series'][0]['pronosticos'],orient='columns')
df_Obs = df_Obs.rename(columns={'timestart':'fecha','valor':'h_obs'})
df_Obs = df_Obs[['fecha','h_obs']]
df_Obs['fecha'] = pd.to_datetime(df_Obs['fecha']).dt.round('min')            # Fecha a formato fecha -- CAMBIADO PARA QUE CORRA EN PYTHON 3.5
df_Obs['h_obs'] = df_Obs['h_obs'].astype(float)

## Union
df_simulado.set_index(df_simulado['fecha'], inplace=True)
df_simulado.index = df_simulado.index.tz_convert(None)
del df_simulado['fecha']

###### Correccion 1
df_simulado.index = df_simulado.index + timedelta(hours=1) # - timedelta(hours=2)


df_Obs.set_index(df_Obs['fecha'], inplace=True)
df_Obs.index = df_Obs.index.tz_convert(None)
del df_Obs['fecha']

df_union = df_simulado.join(df_Obs, how = 'outer')
df_union['h_sim'] = df_union['h_sim'].interpolate(method='linear',limit=4)
#df_union['h_sim_Mavg'] = df_union['h_sim'].rolling(4, min_periods=1).mean()

df_union = df_union.dropna()

## Plot
if False:
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_union.index, df_union['h_sim'], 'b-',label='Simulado')
    ax.plot(df_union.index, df_union['h_obs'], 'r-',label='Observado')
    plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=18)
    plt.xlabel('Fecha', size=18)
    plt.ylabel('Nivel [m]', size=18)
    plt.legend(prop={'size':20},loc=0)
    plt.tight_layout()
    plt.show()




from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

## Modelo
train = df_union[:].copy()
var_obj = 'h_obs'
covariav = ['h_sim',]

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
quant_Err = train['Error_pred'].quantile([.001,.05,.95,.999])
if False:
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_union.index, df_union['h_sim'], 'b-',label='Simulado')
    ax.plot(df_union.index, df_union['h_obs'], 'r-',label='Observado')

    ax.plot(train.index, train['Y_predictions'], 'k-',label='Ajuste RL')

    plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=18)
    plt.xlabel('Fecha', size=18)
    plt.ylabel('Nivel [m]', size=18)
    plt.legend(prop={'size':20},loc=0)
    plt.tight_layout()
    plt.show()


X_input = df_simulado[['h_sim',]].values
df_simulado['Y_predic'] = lr.predict(X_input)

horas_plot = 24*7
df_simulado = df_simulado[-horas_plot:]

if False:
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_simulado.index, df_simulado['Y_predic'], 'r-',label='Prediccion')
    ax.plot(df_Obs.index, df_Obs['h_obs'], 'b-.',label='Observado')

    plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=18)
    plt.xlim(df_simulado.index.min(),df_simulado.index.max())
    plt.xlabel('Fecha', size=18)
    plt.ylabel('Nivel [m]', size=18)
    plt.legend(prop={'size':20},loc=0)
    plt.tight_layout()
    plt.show()

# df_simulado['e_pred_05'] = df_simulado['Y_predic'] + quant_Err[0.05]
# df_simulado['e_pred_95'] = df_simulado['Y_predic'] + quant_Err[0.95]
df_simulado['e_pred_01'] = df_simulado['Y_predic'] + quant_Err[0.001]
df_simulado['e_pred_99'] = df_simulado['Y_predic'] + quant_Err[0.999]

# PLOT
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1)

ax.plot(df_simulado.index, df_simulado['Y_predic'], '-',color='b',label='Nivel Pronosticado',linewidth=3)
ax.plot(df_Obs.index, df_Obs['h_obs'],'o',color='k',label='Nivel Observado',linewidth=3)
ax.plot(df_Obs.index, df_Obs['h_obs'],'-',color='k',linewidth=1)

ax.plot(df_simulado.index, df_simulado['e_pred_01'],'-',color='k',linewidth=0.5,alpha=0.75)
ax.plot(df_simulado.index, df_simulado['e_pred_99'],'-',color='k',linewidth=0.5,alpha=0.75)
ax.fill_between(df_simulado.index,df_simulado['e_pred_01'], df_simulado['e_pred_99'],alpha=0.1,label='Banda de error')

# Lineas: 1 , 1.5 y 2 mts
xmin=df_simulado.index.min()
xmax=df_simulado.index.max()

# En cero escala Paranacito Prefectura
#plt.hlines(5, xmin, xmax, colors='r', linestyles='-.', label='Evacuación',linewidth=1.5)
#plt.hlines(2.3, xmin, xmax, colors='y', linestyles='-.', label='Alerta',linewidth=1.5)
#plt.hlines(1.1, xmin, xmax, colors='y', linestyles='-.', label='Aguas Bajas',linewidth=1.5)

# fecha emision
ahora = df_Obs.index.max()
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

#ax.set_ylim(-0.5,3.5)
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

#plt.tight_layout()
# plt.show()
# plt.close()

nameout = 'productos/Prono_Carabelas.png'    
plt.savefig(nameout, format='png')# , dpi=200, facecolor='w', edgecolor='w',bbox_inches = 'tight', pad_inches = 0

plt.close()

csvoutput = open("productos/salida_prono.csv","w")
csvoutput.write(df_simulado.to_csv())
csvoutput.close()

## GUARDA EN A5DB
# timezone = pytz.timezone("America/Argentina/Buenos_Aires")
df_simulado = df_simulado.reset_index()
df_simulado['fecha'] = df_simulado['fecha'].dt.tz_localize("America/Argentina/Buenos_Aires") # timezone.localize(df_simulado['fecha'])
df_para_upsert = df_simulado[['fecha','h_sim']].rename(columns = {'fecha':'timestart', 'h_sim':'valor'},inplace = False)
df_para_upsert['qualifier'] = 'main'
df_para_upsert = df_para_upsert.append(df_simulado[['fecha','e_pred_01']].rename(columns = {'fecha':'timestart', 'e_pred_01': 'valor'}), ignore_index=True)
df_para_upsert['qualifier'].fillna(value='p01',inplace=True)
df_para_upsert = df_para_upsert.append(df_simulado[['fecha','e_pred_99']].rename(columns = {'fecha':'timestart', 'e_pred_99': 'valor'}), ignore_index=True)
df_para_upsert['qualifier'].fillna(value='p99',inplace=True)
df_para_upsert['timeend'] = df_para_upsert['timestart']  # .map(lambda a : a.isoformat())
para_upsert = {
    'forecast_date': forecast_date, # df_Sim['fecha_emision'].max().isoformat(),
        'series': [
        {
            'series_table': 'series',
            'series_id': seriesIdDest,
            'pronosticos': json.loads(df_para_upsert.to_json(orient='records',date_format='iso'))
        }
    ]}

output = open('productos/salida_prono.json','w')
output.write(json.dumps(para_upsert))
output.close()

response = requests.post(
    apiLoginParams["url"] + '/sim/calibrados/439/corridas',
    data=json.dumps(para_upsert),
    headers={'Authorization': 'Bearer ' + apiLoginParams["token"], 'Content-type': 'application/json'},
)
print("prono upload, response code: " + str(response.status_code))
print("prono upload, reason: " + response.reason)
if(response.status_code == 200):
    outresponse = open("productos/upsert_response.json","w")
    outresponse.write(json.dumps(response.json()))
    outresponse.close()