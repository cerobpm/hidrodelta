# -*- coding: utf-8 -*-
### Control de datos de entrada al modelo hidrodinamico ###

### importa librerias
# !pip install psycopg2
import os, psycopg2, sqlite3 
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import geopandas
import json

print('\n ---------------------------------------------------------------------------------------')

# carga parametros conexión DB
with open("dbConnectionParams.json") as f:
	dbConnParams = json.load(f)

# Conecta BBDD Local
bbdd_loc = 'BD_Delta_001.sqlite'
connLoc = sqlite3.connect(bbdd_loc)

# carpetaFiguras = 'Figuras/'

Id_EstLocal = { 'Parana':1,
                'SanFernando':2,
                'NuevaPalmira':3}

# Carga los datos de las estaciones
#Df_Estaciones = pd.read_csv('EstacionesFull.csv', encoding = "ISO-8859-1",index_col='Id')

#Df_Estaciones =  geopandas.read_file('Estaciones/EstDelta_5347.shp')
Df_Estaciones =  pd.read_csv('Estaciones/EstDelta_5347.csv')

DaysMod = 31*4
DaysMod2 = 365*1
ahora = datetime.datetime.now()
f_fin = ahora
f_inicio = (f_fin - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)
#f_fin = (f_fin.replace(hour=0, minute=0, second=0)-timedelta(days=DaysMod2))

#Conexión con la BBDD de Alert
try:
    conn = psycopg2.connect("dbname='" + dbConnParams["dbname"] + "' user='" + dbConnParams["user"] + "' host='" + dbConnParams["host"] + "'")
except:
    print( "No se ha podido establecer conexion.")

# Guarda un Id de la corrida tomando Año / mes / dia y hora de la corrida
nom_clv = str(ahora.year)[2:]+str(ahora.month)+str(ahora.day)+str(ahora.hour)
print('\n Id Corrida: ',nom_clv)

print ('Fecha de corrida: Desde: '+f_inicio.strftime('%d%b%Y')+ ' Hasta: '+f_fin.strftime('%d%b%Y')+'\n')

### Funciones
def consultaBBDD(f_inicio,f_fin,l_idEst):
  #Parametros para la consulta SQL a la BBDD
  dias = (f_fin - f_inicio).days
  t_idEst = tuple(l_idEst)
  paramH = (f_inicio, f_fin, t_idEst,) 
  sql_query = ('''SELECT unid as id, timestart as fecha, valor as altura
                  FROM alturas_all
                  WHERE  timestart BETWEEN %s AND %s AND unid IN %s;''')  #Consulta SQL
  df_todo = pd.read_sql_query(sql_query, conn, params=paramH)             #Toma los datos de la BBDD
  df_todo['fecha'] =  pd.to_datetime(df_todo['fecha'])
  # print('')
  # print(df_todo.head())
  print('')
  print('Cantidad de registros: ',len(df_todo))
  print('Cantidad por dia: ',round(len(df_todo)/dias,1), '  Cantidad de estaciones: ', len(df_todo.groupby('id').nunique()))
  print('Cantidad por Estacion:')
  print(df_todo['id'].value_counts())
  return df_todo

### CB Aguas Arriba ########################################################################################
if True:
    print ('CB Aguas Arriba  ------------------')
    ### Consluta la BBDD
    # Selecciona Estaciones
    l_idEst = [29,30,31] # CB Aguas Arriba
    Df_EstacionesAA = Df_Estaciones[Df_Estaciones['unid'].isin(l_idEst)]
    print(Df_EstacionesAA[['unid','nombre']])

    # Consulta BBDD / Ver En Funciones
    df_todo = consultaBBDD(f_inicio, f_fin,l_idEst)

    ''' Paso 1:
    Une las series en un DF Base:
    Cada serie en una columna. Todas con la misma frecuencia, en este caso diaria.

    También:
    *   Calcula la frecuencia horaria de los datos.
    *   Reemplaza Ceros por NAN.
    *   Calcula diferencias entre valores concecutivos.
    '''
    # Crea DF con una frecuencia constante para unir las series
    indexUnico = pd.date_range(start=f_inicio, end=f_fin, freq='1D')	    #Fechas desde f_inicio a f_fin con un paso de 1 Dia
    df_base = pd.DataFrame(index = indexUnico)							      	      #Crea el Df con indexUnico
    df_base.index.rename('fecha', inplace=True)							              
    df_base.index = df_base.index.round("1D")

    df_FrecD = pd.DataFrame()

    for index,row in Df_EstacionesAA.iterrows():
        nombre = (row['nombre'])
        #print(nombre)
        # Toma cada serie del dataframe todo
        df_var = df_todo[(df_todo['id']==row['unid'])].copy()

        # Valores unicos de Horas
        df_var['Horas'] = df_var['fecha'].apply(lambda x: x.hour)
        df_FrecD[nombre] = pd.Series(df_var['Horas'].value_counts())
        # print(df_var['Horas'].value_counts())
        del df_var['Horas']

        #Acomoda DF para unir
        df_var.set_index(pd.DatetimeIndex(df_var['fecha']), inplace=True)   #Pasa la fecha al indice del dataframe (DatetimeIndex)
        del df_var['fecha']    
        del df_var['id'] 
        df_var = df_var.resample('D').mean()
        df_var.columns = [nombre,]
        
        # Une al DF Base
        df_base = df_base.join(df_var, how = 'left')
        
        # Reemplaza Ceros por NAN
        df_base[nombre] = df_base[nombre].replace(0, np.nan)
        
        # Calcula diferencias entre valores concecutivos
        VecDif = np.diff(df_base[nombre].values)
        VecDif = np.append([0,],VecDif)
        coldiff = 'Diff_'+nombre[:4]
        df_base[coldiff] = VecDif
    # print(df_base.head())

    # Frecuencias de los datos por hora
    # ax = df_FrecD.plot.bar(rot=0)
    # plt.title('Frecuencias de los datos por horas del dia')
    # plt.tight_layout()
    # f_nameFAArriba = carpetaFiguras+'01_1AArriba_Frec.jpg'
    # plt.savefig(f_nameFAArriba, format='jpg')
    # # plt.show()
    # plt.close()


    # Reemplaza nan por -1. Son vacíos de la Base.
    df_base[nombre] = df_base[nombre].replace(np.nan,-1)

    # fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # for index,row in Df_EstacionesAA.iterrows():
        # nombre = (row['nombre'])
        # ax.plot(df_base.index, df_base[nombre],'-',label=nombre)
    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('AArriba Serie Base')
    # plt.tight_layout()
    # f_nameS0 = carpetaFiguras+'01_2AArriba_Serie0.jpg'
    # plt.savefig(f_nameS0, format='jpg')
    # #plt.show()
    # plt.close()

    # Vuelve a poner los -1 como nan
    df_base[nombre] = df_base[nombre].replace(-1,np.nan)

    ''' ### Paso 2:
    Elimina saltos:

    Se establece un umbral_1: si la diferencia entre datos consecutivos supera este umbral_1, se fija si en el resto de las series tambien se produce el salto (se supera el umbral_1).
     Si en todas las series se observa un salto se toma el dato como valido.
    Si el salto no se produce en las tres o si es mayo al segundo umbral_2 (> que el 1ero) se elimina el dato. '''

    # Datos faltante
    print('\nDatos Faltantes')
    for index,row in Df_EstacionesAA.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base[nombre].isna().sum()))

    # print (df_base.head())

    # Elimina Saltos en la serie
    umbral_1 = 0.3
    umbral_2 = 1.0

    def EliminaSaltos(df_base):
        # Parana
        for index,row in df_base.iterrows():
            if abs(row['Diff_Para']) > umbral_1:
                if (abs(row['Diff_Sant']) > umbral_1) and (abs(row['Diff_Diam']) > umbral_1):
                    if abs(row['Diff_Para']) > umbral_2:
                        df_base.loc[index,'Parana']= np.nan
                    else:
                        # print('Los 3 presentan un salto. Se supone que esta ok.')
                        continue
                else:
                    # print('Salto en Parana y StFe o Diamante')
                    df_base.loc[index,'Parana']= np.nan
            else:
                continue

        # Santa Fe
        for index,row in df_base.iterrows():
            if abs(row['Diff_Sant']) > umbral_1:
                if (abs(row['Diff_Para']) > umbral_1) and (abs(row['Diff_Diam']) > umbral_1):
                    if abs(row['Diff_Sant']) > umbral_2:
                        df_base.loc[index,'SantaFe']= np.nan
                    else:
                        # print('Los 3 presentan un salto. Se supone que esta ok.')
                        continue
                else:
                    # print('Salto en StFe y Parana o Diamante')
                    df_base.loc[index,'SantaFe']= np.nan
            else:
                continue

        # Diamante
        for index,row in df_base.iterrows():
            if abs(row['Diff_Diam']) > umbral_1:
                if (abs(row['Diff_Para']) > umbral_1) and (abs(row['Diff_Sant']) > umbral_1):
                    if abs(row['Diff_Diam']) > umbral_2:
                        df_base.loc[index,'Diamante']= np.nan
                    else:
                        # print('Los 3 presentan un salto. Se supone que esta ok.')
                        continue
                else:
                    # print('Salto en Diamante y Parana o StFe')
                    df_base.loc[index,'Diamante']= np.nan
            else:
                continue
        return df_base

    # Elimina Saltos
    df_base = EliminaSaltos(df_base)

    print('\nDatos Faltantes Luego de limpiar saltos')
    for index,row in Df_EstacionesAA.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base[nombre].isna().sum()))

    # fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(1, 1, 1)

    # ax.plot(df_base.index, df_base['Parana'],'-',label='Parana')
    # ax.plot(df_base.index, df_base['SantaFe'],'-',label='SantaFe')
    # ax.plot(df_base.index, df_base['Diamante'],'-',label='Diamante')
    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('CB AArriba')
    # plt.tight_layout()
    # f_nameS1 = carpetaFiguras+'01_3AArriba_Serie1.jpg'
    # plt.savefig(f_nameS1, format='jpg')
    # # plt.show()
    # plt.close()

    ''' Paso 3:
    Completa Faltantes en base a los datos en las otras series.
    1. Lleva las saries al mismo plano.
    2. Calcula meidas de a pares. Parana-Santa Fe , Parana-Diamanate, Parana-Diamante.
    3. Si no hay datos toma el de la media de las otras dos.
    4. Si la diferencia entre el dato y la media es mayor al umbral_3 elimina el dato. '''
    from scipy import stats
    from scipy.stats import linregress
    from sklearn.metrics import mean_squared_error

    # df_Niveles = df_base.copy().dropna()
    # df_Niveles = df_Niveles.drop(['Diff_Para', 'Diff_Sant','Diff_Diam'], axis=1)

    # gradient_SP, intercept_SP, r_value_SP, p_value_SP, std_err_SP = linregress(df_Niveles.SantaFe,df_Niveles.Parana)
    # y1=gradient_SP*df_Niveles.SantaFe+intercept_SP
    # df_Niveles['Para_reg'] = y1
    # y_error=np.sqrt(((y1 - datos.ProfMed) ** 2))
    # y_error=datos.ProfMed-y1 
    # y_rmse = np.sqrt(mean_squared_error(datos.ProfMed, y1))
    # print ("Error max. ", max(y_error),"\n","Error min. ", min(y_error))
    # gradient_DP, intercept_DP, r_value_DP, p_value_DP, std_err_DP = linregress(df_Niveles.Diamante,df_Niveles.Parana)

    # fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(1, 1, 1)

    # ax.plot(df_Niveles.index, df_Niveles['Parana'],'-',label='Parana')
    # ax.plot(df_Niveles.index, df_Niveles['Para_reg'],'-',label='Para_reg')
    # ax.plot(df_Niveles.index, df_Niveles['SantaFe'],'-',label='SantaFe')
    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('CB Aguas Arriba')
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    df_Niveles = df_base.copy()
    df_Niveles = df_Niveles.drop(['Diff_Para', 'Diff_Sant','Diff_Diam'], axis=1)

    # Llevo a la misma face y plano de referencia
    corim_SantaFe = -0.30
    corim_Diamante = -0.30
    df_Niveles['SantaFe'] = df_Niveles['SantaFe'].add(corim_SantaFe)
    df_Niveles['Diamante'] = df_Niveles['Diamante'].add(corim_Diamante)

    # Calcula media de a pares
    df_Niveles['mediaPS'] = df_Niveles[['Parana','SantaFe']].mean(axis = 1,)
    df_Niveles['mediaPD'] = df_Niveles[['Parana','Diamante']].mean(axis = 1,)
    df_Niveles['mediaSD'] = df_Niveles[['SantaFe','Diamante']].mean(axis = 1,)

    print('\nFaltantes de la media de a pares:')
    for mediapar in ['mediaPS','mediaPD','mediaSD']:
        print('NaN '+mediapar+': '+str(df_Niveles[mediapar].isna().sum()))

    # Completa Faltantes
    umbral_3 = 0.3
    for index,row in df_Niveles.iterrows():
        # Parana
        if np.isnan(row['Parana']):
            # print ('Parana Nan')
            df_Niveles.loc[index,'Parana']= row['mediaSD']
        elif    abs(row['Parana']-row['mediaSD']) > umbral_3:
            # print ('Parana Dif Media')
            df_Niveles.loc[index,'Parana']= np.nan
        
        # Santa Fe
        if np.isnan(row['SantaFe']):
            # print ('SantaFe Nan')
            df_Niveles.loc[index,'SantaFe']= row['mediaPD']
        elif    abs(row['SantaFe']-row['mediaPD']) > umbral_3:
            # print ('SantaFe Dif Media')
            df_Niveles.loc[index,'SantaFe']= np.nan

        # Diamante
        if np.isnan(row['Diamante']):
            # print ('Diamante Nan')
            df_Niveles.loc[index,'Diamante']= row['mediaSD']
        elif    abs(row['Diamante']-row['mediaPS']) > umbral_3:
            # print ('Diamante Dif Media')
            df_Niveles.loc[index,'Diamante']= np.nan

    # Faltantes luego de completar con la media de las otras dos series y eliminar cuando hay difrencias mayores a umbral_3
    print('\nFaltentes luego de completar y filtrar:')
    print('NaN Parana: '+str(df_Niveles['Parana'].isna().sum()))
    print('NaN SantaFe: '+str(df_Niveles['SantaFe'].isna().sum()))
    print('NaN Diamante: '+str(df_Niveles['Diamante'].isna().sum()))

    # Interpola de forma Linal
    # Interpola para completa todos los fltantes
    df_Niveles = df_Niveles.interpolate(method='linear',limit_direction='both')
    print('\n Faltentes luego de interpolar:')
    print('NaN Parana: '+str(df_Niveles['Parana'].isna().sum()))
    print('NaN SantaFe: '+str(df_Niveles['SantaFe'].isna().sum()))
    print('NaN Diamante: '+str(df_Niveles['Diamante'].isna().sum()))

    # Vuelve las series a su nivel original
    df_Niveles['SantaFe'] = df_Niveles['SantaFe'].add(-corim_SantaFe)
    df_Niveles['Diamante'] = df_Niveles['Diamante'].add(-corim_Diamante)

    # Series final
    # fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(1, 1, 1)

    # ax.plot(df_Niveles.index, df_Niveles['Parana'],'-',label='Parana')
    # ax.plot(df_Niveles.index, df_Niveles['SantaFe'],'-',label='SantaFe')
    # ax.plot(df_Niveles.index, df_Niveles['Diamante'],'-',label='Diamante')
    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('CB Aguas Arriba')
    # plt.tight_layout()
    # f_nameS2 = carpetaFiguras+'01_4AArriba_Serie2.jpg'
    # plt.savefig(f_nameS2, format='jpg')
    # #plt.show()
    # plt.close()

    df_aux_i = pd.DataFrame()                        # Pasa lista a DF
    
    cero_parana = Df_Estaciones[Df_Estaciones['nombre']=='Parana']['ceroEscala'].values[0]
    df_aux_i['Nivel'] = df_Niveles['Parana'] + cero_parana
    df_aux_i['Fecha'] = df_Niveles.index
    df_aux_i['Caudal'] = np.nan
    df_aux_i['Id_CB'] = Id_EstLocal['Parana']
    
    fecha_0 = df_Niveles.index.max() + timedelta(days=1)
    fecha_f = (fecha_0 + timedelta(days=4))
    index1dia = pd.date_range(start=fecha_0, end=fecha_f, freq='D')
    df_aux_i_prono = pd.DataFrame(index = index1dia,columns=['Nivel','Fecha','Caudal','Id_CB'])
    df_aux_i_prono['Nivel'] = df_aux_i['Nivel'][-1]
    df_aux_i_prono['Fecha'] = df_aux_i_prono.index
    df_aux_i_prono['Caudal'] = np.nan   
    df_aux_i_prono['Id_CB'] = Id_EstLocal['Parana']
    
    df_aux_i =  pd.concat([df_aux_i, df_aux_i_prono], ignore_index=False)
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='replace',index=False)

## CB Frente Margen Derecha ################################################################################
if True:
    print ('\nCB Frente Margen Derecha  ------------------')
    ### Consluta la BBDD
    l_idE_MDer = [52,1702,85]

    # MArgen Derecha
    Df_EstacionesMD = Df_Estaciones[Df_Estaciones['unid'].isin(l_idE_MDer)]
    print(Df_EstacionesMD[['unid','nombre']])

    # Consulta BBDD / Ver En Funciones
    df_todo = consultaBBDD(f_inicio, f_fin,l_idE_MDer)

    ''' Paso 1:
    Une las series en un DF Base:
    Cada serie en una columna. Todas con la misma frecuencia, en este caso diaria.

    También:
    *   Calcula la frecuencia horaria de los datos.
    *   Calcula diferencias entre valores concecutivos.'''

    indexUnico = pd.date_range(start=f_inicio, end=f_fin, freq='1H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
    df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    df_base.index = df_base.index.round("1H")

    df_FrecD = pd.DataFrame()
    for index,row in Df_EstacionesMD.iterrows():
        nombre = (row['nombre'])
        # print(nombre)
        df_var = df_todo[(df_todo['id']==row['unid'])].copy()
        
        # Valores Unicos horarios
        df_var['Horas'] = df_var['fecha'].apply(lambda x: x.hour)
        df_FrecD[nombre] = pd.Series(df_var['Horas'].value_counts())
        del df_var['Horas'] 

        #Acomoda DF para unir
        df_var.set_index(pd.DatetimeIndex(df_var['fecha']), inplace=True)   #Pasa la fecha al indice del dataframe (DatetimeIndex)
        del df_var['fecha']    
        del df_var['id'] 
        df_var = df_var.resample('H').mean()
        df_var.columns = [nombre,]
        
        # Une al DF Base.
        df_base = df_base.join(df_var, how = 'left')
        
        # Reemplaza Ceros por NAN
        #df_base[nombre] = df_base[nombre].replace(0, np.nan)

        # Calcula diferencias entre valores concecutivos
        VecDif = np.diff(df_base[nombre].values)
        VecDif = np.append([0,],VecDif)
        coldiff = 'Diff_'+nombre[:4]
        df_base[coldiff] = VecDif
    #print(df_base.head())

    # Frecuencias de los datos por hora
    # ax = df_FrecD.plot.bar(rot=0)
    # plt.title('Frecuencias de los datos por horas del dia')
    # plt.tight_layout()
    # f_nameFrecFMD = carpetaFiguras+'02_1FMDerecha_Frec.jpg'
    # plt.savefig(f_nameFrecFMD, format='jpg')
    # # plt.show()
    # plt.close()

    # Reemplaza nan por -1. Son vacíos de la Base
    df_base[nombre] = df_base[nombre].replace(np.nan,-1)

    # fig = plt.figure(figsize=(15, 8))
    # ax1 = fig.add_subplot(1, 1, 1)
    # for index,row in Df_EstacionesMD.iterrows():
        # nombre = (row['nombre'])
        # ax1.plot(df_base.index, df_base[nombre],'-',label=nombre)

    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('FMDerecha Serie Base')
    # plt.tight_layout()
    # f_nameS1 = carpetaFiguras+'02_2FMDerecha_Serie0.jpg'
    # plt.savefig(f_nameS1, format='jpg')
    # #plt.show()
    # plt.close()

    df_base[nombre] = df_base[nombre].replace(-1,np.nan)


    ''' Paso 2:
    Elimina saltos:

    Se establece un umbral_1: si la diferencia entre datos consecutivos supera este umbral_1, se fija si en el resto de las series tambien se produce el salto (se supera el umbral_1).
     Si en todas las series se observa un salto se toma el dato como valido.
    Si el salto no se produce en las tres o si es mayo al segundo umbral_2 (> que el 1ero) se elimina el dato.'''

    # Datos faltante
    print('\nDatos Faltantes')
    for index,row in Df_EstacionesMD.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base[nombre].isna().sum()))

    # Elimina Saltos
    umbral_1 = 0.5
    umbral_2 = 1.0

    def EliminaSaltos(df_base):# Elimina Saltos en la serie
        # SFernando
        for index,row in df_base.iterrows():
            if abs(row['Diff_SanF']) > umbral_1:
                if (abs(row['Diff_BsAs']) > umbral_1) and (abs(row['Diff_Brag']) > umbral_1):
                    if abs(row['Diff_SanF']) > umbral_2:
                        df_base.loc[index,'SanFernando']= np.nan
                    else:
                        continue
                        #print('Los 3 presentan un salto. Se supone que esta ok.')
                else:
                    # print('Salto en SanFer y BsAs o Brag')
                    df_base.loc[index,'SanFernando']= np.nan
            else:
                continue
        # BsAs
        for index,row in df_base.iterrows():
            if abs(row['Diff_BsAs']) > umbral_1:
                if (abs(row['Diff_SanF']) > umbral_1) and (abs(row['Diff_Brag']) > umbral_1):
                    if abs(row['Diff_BsAs']) > umbral_2:
                        df_base.loc[index,'BsAs']= np.nan
                    else:
                        continue
                        # print('Los 3 presentan un salto. Se supone que esta ok.')
                else:
                    # print('Salto en BsAs y SFer o Braga')
                    df_base.loc[index,'BsAs']= np.nan
            else:
                continue
        # Braga
        for index,row in df_base.iterrows():
            if abs(row['Diff_Brag']) > umbral_1:
                if (abs(row['Diff_SanF']) > umbral_1) and (abs(row['Diff_BsAs']) > umbral_1):
                    if abs(row['Diff_Brag']) > umbral_2:
                        df_base.loc[index,'Braga']= np.nan
                    else:
                        print('Los 3 presentan un salto. Se supone que esta ok.')
                else:
                    # print('Salto en Braga y SFer o BsAs')
                    df_base.loc[index,'Braga']= np.nan
            else:
                continue
        return df_base

    #  Elimina Saltos
    df_base = EliminaSaltos(df_base)

    # Datos Faltantes Luego de limpiar saltos
    print('\nDatos Faltantes Luego de limpiar saltos')
    for index,row in Df_EstacionesMD.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base[nombre].isna().sum()))

    # fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(1, 1, 1)
    
    # ax.plot(df_base.index, df_base['SanFernando'],'-',label='SFernando')
    # ax.plot(df_base.index, df_base['BsAs'],'-',label='BsAs')
    # ax.plot(df_base.index, df_base['Braga'],'-',label='Braga')
    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('Frente Delta Margen Derecha')
    # plt.tight_layout()
    # f_nameS1 = carpetaFiguras+'02_3FMDerecha_Serie1.jpg'
    # plt.savefig(f_nameS1, format='jpg')
    # #plt.show()
    # plt.close()

    ''' Paso 3:
    Completa Faltantes en base a los datos en las otras series.

    1.   Lleva las saries al mismo plano.
    2.   Calcula meidas de a pares. Parana-Santa Fe , Parana-Diamanate, Parana-Diamante.
    3.   Si no hay datos toma el de la media de las otras dos.
    4.   Si la diferencia entre el dato y la media es mayor al umbral_3 elimina el dato.'''

    df_Niveles = df_base.copy()
    df_Niveles = df_Niveles.drop(['Diff_SanF', 'Diff_BsAs','Diff_Brag'], axis=1)

    # Copia cada serie en un DF distinto
    df_SFer = df_Niveles[['SanFernando']].copy()
    df_BsAs = df_Niveles[['BsAs']].copy()
    df_Brag = df_Niveles[['Braga']].copy()

    # Corrimiento Vertical
    corim_BsAs = 0.2
    corim_Braga = 0.05
    df_BsAs['BsAs'] = df_BsAs['BsAs'].add(corim_BsAs)
    df_Brag['Braga'] = df_Brag['Braga'].add(corim_Braga)
    # Corrimiento Temporal
    df_BsAs.index = df_BsAs.index + pd.DateOffset(minutes=50)
    df_Brag.index = df_Brag.index - pd.DateOffset(minutes=45)

    # Crea DF para unir todas las series/ frec 5 minutos
    index5m = pd.date_range(start=f_inicio, end=f_fin, freq='5min')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_5m = pd.DataFrame(index = index5m)                               #Crea el Df con indexUnico
    df_5m.index.rename('fecha', inplace=True)							              #Cambia nombre incide por Fecha
    df_5m.index = df_5m.index.round("5min")

    # Une en df_5m
    df_5m = df_5m.join(df_SFer, how = 'left')
    df_5m = df_5m.join(df_BsAs, how = 'left')
    df_5m = df_5m.join(df_Brag, how = 'left')

    # Calcula la media de las tres series. E interpola para completar todos los vacios
    df_5m['medio'] = df_5m[['SanFernando','BsAs','Braga']].mean(axis = 1,)
    df_5m = df_5m.interpolate(method='linear',limit_direction='both')
    #print('\nNaN medio: '+str(df_5m['medio'].isna().sum()))

    # A cada DF de las series les une la serie media de las 3.
    # Siempre se une por Izquierda
    df_SFer = df_SFer.join(df_5m['medio'], how = 'left')
    df_BsAs = df_BsAs.join(df_5m['medio'], how = 'left')
    df_Brag = df_Brag.join(df_5m['medio'], how = 'left')
    
    # print('\n')
    # print('NaN SFernando: '+str(df_SFer['SanFernando'].isna().sum()))
    # print('NaN BsAs: '+str(df_BsAs['BsAs'].isna().sum()))
    # print('NaN Braga: '+str(df_Brag['Braga'].isna().sum()))
    
    # Completa falrastes usando la serie media
    def CompletaFaltanes(df_ConMedia):
        ncol = df_ConMedia.columns[0]
        for index,row in df_ConMedia.iterrows():
            if np.isnan(row[ncol]):
                df_ConMedia.loc[index,ncol]= row['medio']
        return df_ConMedia
    df_SFer = CompletaFaltanes(df_SFer)
    df_BsAs = CompletaFaltanes(df_BsAs)
    df_Brag = CompletaFaltanes(df_Brag)

    print('\nFaltentes luego de completar con serie media:')
    print('NaN SFernando: '+str(df_SFer['SanFernando'].isna().sum()))
    print('NaN BsAs: '+str(df_BsAs['BsAs'].isna().sum()))
    print('NaN Braga: '+str(df_Brag['Braga'].isna().sum()))


    # Vuelve a llevar las series a su lugar original
    df_BsAs['BsAs'] = df_BsAs['BsAs'].add(-corim_BsAs)
    df_Brag['Braga'] = df_Brag['Braga'].add(-corim_Braga)
    df_BsAs.index = df_BsAs.index - pd.DateOffset(minutes=50)
    df_Brag.index = df_Brag.index + pd.DateOffset(minutes=45)

    # Une en df_
    df_SFer.columns = ['SanFernando_f','medio']
    df_BsAs.columns = ['BsAs_f','medio']
    df_Brag.columns = ['Braga_f','medio']
    df_Niveles = df_Niveles.join(df_SFer['SanFernando_f'], how = 'left')
    df_Niveles = df_Niveles.join(df_BsAs['BsAs_f'], how = 'left')
    df_Niveles = df_Niveles.join(df_Brag['Braga_f'], how = 'left')

    # Interpola de forma Linal. Maximo 3 dias

    # df_Niveles[nombre] = df_Niveles[nombre].replace(np.nan,-1)
    print('\n Interpola para que no queden faltantes')
    df_Niveles = df_Niveles.interpolate(method='linear',limit_direction='both')
    print('NaN SFernando: '+str(df_Niveles['SanFernando_f'].isna().sum()))
    print('NaN BsAs: '+str(df_Niveles['BsAs_f'].isna().sum()))
    print('NaN Braga: '+str(df_Niveles['Braga_f'].isna().sum()))

    # for index,row in Df_EstacionesMD.iterrows():
        # nombre = (row['nombre'])
        # fig = plt.figure(figsize=(15, 8))
        # ax = fig.add_subplot(1, 1, 1)

        # nombref = nombre+'_f'
        # ax.plot(df_Niveles.index, df_Niveles[nombre],'.',label=nombre)
        # # ax.plot(df_Niveles.index, df_Niveles['BsAs'],'.',label='BsAs')
        # # ax.plot(df_Niveles.index, df_Niveles['Braga'],'.',label='Braga')
        
        # ax.plot(df_Niveles.index, df_Niveles[nombref],'-',label=nombref)
        # # ax.plot(df_Niveles.index, df_Niveles['BsAs_f'],'-',label='BsAsF')   
        # # ax.plot(df_Niveles.index, df_Niveles['Braga_f'],'-',label='BragaF')
        
        # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        # plt.tick_params(axis='both', labelsize=16)
        # plt.xlabel('Fecha', size=18)
        # plt.ylabel('Nivel [m]', size=18)
        # plt.legend(prop={'size':16},loc=0)
        # plt.title('Frente Delta Margen Derecha')
        # plt.tight_layout()
        # f_nameS1 = carpetaFiguras+'02_4FMDerecha_'+nombre+'.jpg'
        # plt.savefig(f_nameS1, format='jpg')
        # #plt.show()
        # plt.close()
        
    df_aux_i = pd.DataFrame()                        # Pasa lista a DF
    cero_sanfer = Df_Estaciones[Df_Estaciones['nombre']=='SanFernando']['ceroEscala'].values[0]
    df_aux_i['Nivel'] = df_Niveles['SanFernando_f'] + cero_sanfer
    df_aux_i['Fecha'] = df_Niveles.index
    df_aux_i['Id_CB'] = Id_EstLocal['SanFernando']
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

## CB Frente Margen Izquierda ##############################################################################
if True:
    ### Consluta la BBDD
    l_idE_MIzq = [1696,1699]
    # Margen Derecha
    Df_EstacionesMI = Df_Estaciones[Df_Estaciones['unid'].isin(l_idE_MIzq)]
    print(Df_EstacionesMI[['unid','nombre']])

    # Consulta BBDD / Ver En Funciones
    df_todo = consultaBBDD(f_inicio, f_fin,l_idE_MIzq)

    ''' Paso 1:
    Une las series en un DF Base:
    Cada serie en una columna. Todas con la misma frecuencia, en este caso diaria.

    También:
    *   Calcula la frecuencia horaria de los datos.
    *   Calcula diferencias entre valores concecutivos.'''

    indexUnico = pd.date_range(start=f_inicio, end=f_fin, freq='1H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base = pd.DataFrame(index = indexUnico)								#Crea el Df con indexUnico
    df_base.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    df_base.index = df_base.index.round("1H")

    df_FrecD = pd.DataFrame()
    for index,row in Df_EstacionesMI.iterrows():
        nombre = (row['nombre'])
        print(nombre)
        df_var = df_todo[(df_todo['id']==row['unid'])].copy()
        
        # Valores Unicos horarios
        df_var['Horas'] = df_var['fecha'].apply(lambda x: x.hour)
        df_FrecD[nombre] = pd.Series(df_var['Horas'].value_counts())
        del df_var['Horas'] 

        #Acomoda DF para unir
        df_var.set_index(pd.DatetimeIndex(df_var['fecha']), inplace=True)   #Pasa la fecha al indice del dataframe (DatetimeIndex)
        del df_var['fecha']    
        del df_var['id'] 
        df_var = df_var.resample('H').mean()
        df_var.columns = [nombre,]
        
        # Une al DF Base.
        df_base = df_base.join(df_var, how = 'left')
        
        # Reemplaza Ceros por NAN
        #df_base[nombre] = df_base[nombre].replace(0, np.nan)

        # Calcula diferencias entre valores concecutivos
        VecDif = abs(np.diff(df_base[nombre].values))
        VecDif = np.append([0,],VecDif)
        coldiff = 'Diff_'+nombre[:4]
        df_base[coldiff] = VecDif
    #print(df_base.head())

    # Frecuencias de los datos por hora
    # ax = df_FrecD.plot.bar(rot=0)
    # plt.title('Frecuencias de los datos por horas del dia')
    # plt.tight_layout()
    # f_nameFrecFMI = carpetaFiguras+'03_1FMIzq_Frec.jpg'
    # plt.savefig(f_nameFrecFMI, format='jpg')
    # # plt.show()
    # plt.close()

    # Reemplaza nan por -1. Son vacíos de la Base
    df_base[nombre] = df_base[nombre].replace(np.nan,-1)

    # fig = plt.figure(figsize=(15, 8))
    # ax1 = fig.add_subplot(1, 1, 1)

    # for index,row in Df_EstacionesMI.iterrows():
        # nombre = (row['nombre'])
        # ax1.plot(df_base.index, df_base[nombre],'-',label=nombre)
        # #coldiff = 'Diff_'+nombre[:4]
    
    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('FMIzq Serie Base')
    # plt.tight_layout()
    # f_nameS1 = carpetaFiguras+'03_2FMIzq_Serie0.jpg'
    # plt.savefig(f_nameS1, format='jpg')
    # #plt.show()
    # plt.close()

    df_base[nombre] = df_base[nombre].replace(-1,np.nan)

    ### Diferencias entre consecutivos"""
    # for index,row in Df_EstacionesMI.iterrows():
        # nombre = (row['nombre'])
        # print('NaN '+nombre+': '+str(df_base[nombre].isna().sum()))

        # coldiff = 'Diff_'+nombre[:4]
        # print(df_base[coldiff].describe())

        # plt.figure()
        # df_base[coldiff].plot.hist(bins=6)
        # df_base[coldiff].plot.density(secondary_y=True)
        # plt.show()
    # plt.close()
    
    ''' Paso 2:

    Elimina saltos:

    Se establece un umbral_1: si la diferencia entre datos consecutivos supera este umbral_1, se fija si en el resto de las series tambien se produce el salto (se supera el umbral_1).
     Si en todas las series se observa un salto se toma el dato como valido.
    Si el salto no se produce en las tres o si es mayo al segundo umbral_2 (> que el 1ero) se elimina el dato.'''

    # Datos faltante
    print('\nDatos Faltantes')
    for index,row in Df_EstacionesMI.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base[nombre].isna().sum()))

    # Elimina Saltos
    umbral_1 = 0.3
    umbral_2 = 0.7

    def EliminaSaltos(df_base):# Elimina Saltos en la serie
        # SFernando
        for index,row in df_base.iterrows():
            if abs(row['Diff_Nuev']) > umbral_1:
                if abs(row['Diff_Mart']) > umbral_1:
                    if abs(row['Diff_Nuev']) > umbral_2:
                        df_base.loc[index,'Nueva Palmira']= np.nan
                    else:
                        continue
                        #print('Los 2 presentan un salto. Se supone que esta ok.')
                else:
                    # print('Salto en NPalmira')
                    df_base.loc[index,'Nueva Palmira']= np.nan
            else:
                continue
        # BsAs
        for index,row in df_base.iterrows():
            if abs(row['Diff_Mart']) > umbral_1:
                if abs(row['Diff_Nuev']) > umbral_1:
                    if abs(row['Diff_Mart']) > umbral_2:
                        df_base.loc[index,'Martinez']= np.nan
                    else:
                        continue
                        # print('Los 2 presentan un salto. Se supone que esta ok.')
                else:
                    # print('Salto en Martinez')
                    df_base.loc[index,'Martinez']= np.nan
            else:
                continue
        return df_base  

    # Elimina Saltos
    df_base = EliminaSaltos(df_base)


    # Datos Faltantes Luego de limpiar saltos
    print('\nDatos Faltantes Luego de limpiar saltos')
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    # for index,row in Df_EstacionesMI.iterrows():
        # nombre = (row['nombre'])
        # #print (nombre)
        # print('NaN '+nombre+': '+str(df_base[nombre].isna().sum()))
        # ax.plot(df_base.index, df_base[nombre],'-',label=nombre)

    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('Frente Delta Margen Izquierda')
    # plt.tight_layout()
    # f_nameS2 = carpetaFiguras+'03_3FMIzq_Serie1.jpg'
    # plt.savefig(f_nameS2, format='jpg')
    # #plt.show()
    # plt.close()

    ''' Paso 3:

    Completa Faltantes en base a los datos en las otras series.

    1.   Lleva las saries al mismo plano.
    2.   Calcula meidas de a pares. Parana-Santa Fe , Parana-Diamanate, Parana-Diamante.
    3.   Si no hay datos toma el de la media de las otras dos.
    4.   Si la diferencia entre el dato y la media es mayor al umbral_3 elimina el dato.'''

    df_Niveles = df_base.copy()
    df_Niveles = df_Niveles.drop(['Diff_Mart', 'Diff_Nuev'], axis=1)

    # Copia cada serie en un DF distinto
    df_NPal = df_Niveles[['Nueva Palmira']].copy()
    df_Mart = df_Niveles[['Martinez']].copy()

    # Corrimiento Vertical y Temporal
    corim_Mart = 0.24
    df_Mart['Martinez'] = df_Mart['Martinez'].add(corim_Mart)
    df_Mart.index = df_Mart.index - pd.DateOffset(minutes=60)

    # Crea DF para unir todas las series/ frec 1 hora
    index1H = pd.date_range(start=f_inicio, end=f_fin, freq='1H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_1H = pd.DataFrame(index = index1H)								#Crea el Df con indexUnico
    df_1H.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    df_1H.index = df_1H.index.round("H")

    # Une en df_1H
    df_1H = df_1H.join(df_NPal, how = 'left')
    df_1H = df_1H.join(df_Mart, how = 'left')
    
    df_1H['Diff'] = df_1H['Nueva Palmira']-df_1H['Martinez']
    #print(df_1H['Diff'].describe())

    boxplot = df_1H.boxplot(column=['Diff'])


    df_1H['Nueva Palmira'] = df_1H['Nueva Palmira'].interpolate(method='linear',limit_direction='both')
    
    # fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(1, 1, 1)
    
    # ax.plot(df_1H.index, df_1H['Nueva Palmira'],'.',label='Nueva Palmira')
    
    # plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
    # plt.tick_params(axis='both', labelsize=16)
    # plt.xlabel('Fecha', size=18)
    # plt.ylabel('Nivel [m]', size=18)
    # plt.legend(prop={'size':16},loc=0)
    # plt.title('Frente Delta Margen Izquierda')
    # plt.tight_layout()
    # f_nameS1 = carpetaFiguras+'03_4FMIzq.jpg'
    # plt.savefig(f_nameS1, format='jpg')
    # #plt.show()
    # plt.close()
    
    ################# Guarda para modelo
    df_aux_i = pd.DataFrame()                        # Pasa lista a DF
    cero_NPalmira = Df_Estaciones[Df_Estaciones['nombre']=='Nueva Palmira']['ceroEscala'].values[0]
    df_aux_i['Nivel'] = df_1H['Nueva Palmira'] + cero_NPalmira
    df_aux_i['Fecha'] = df_1H.index
    df_aux_i['Id_CB'] = Id_EstLocal['NuevaPalmira']
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

    connLoc.commit()

from Prono_Frente import pronofrente
print('\n Pronostico  ---------------------------------')
pronofrente(ahora,bbdd_loc)

## Condiciones de Borde Frente  ############################################################################
Estaciones = ['SanFernando', 'NuevaPalmira']
df_CB_F0 = pd.DataFrame()
for Estac in Estaciones:
    paramCBF = [Id_EstLocal[Estac],]
    sql_query = ('''SELECT Nivel, Fecha FROM DataEntrada WHERE Id_CB = ?''')
    I_CB_Fte = pd.read_sql_query(sql_query, connLoc,params=paramCBF)
    keys =  pd.to_datetime(I_CB_Fte['Fecha'])#, format='%Y-%m-%d')								#Convierte a formato fecha la columna [fecha]
    I_CB_Fte.set_index(pd.DatetimeIndex(keys), inplace=True)									#Pasa la fecha al indice del dataframe (DatetimeIndex)
    del I_CB_Fte['Fecha']																#Elimina el campo fecha que ya es index
    I_CB_Fte.index.rename('fecha', inplace=True)    #Cambia el nombre del indice
    I_CB_Fte[Estac] = I_CB_Fte['Nivel']  
    del I_CB_Fte['Nivel']
    df_CB_F0 = df_CB_F0.join(I_CB_Fte, how = 'outer')

df_CB_F_prono = pd.DataFrame()
for Estac in Estaciones:
    paramCBF = [Id_EstLocal[Estac],]
    sql_query = ('''SELECT Fecha, h_sim as Nivel FROM PronoFrente WHERE Id = ?''')
    Prono_Fte = pd.read_sql_query(sql_query, connLoc,params=paramCBF)
    keys =  pd.to_datetime(Prono_Fte['fecha'])#, format='%Y-%m-%d')								#Convierte a formato fecha la columna [fecha]
    Prono_Fte.set_index(pd.DatetimeIndex(keys), inplace=True)									#Pasa la fecha al indice del dataframe (DatetimeIndex)
    del Prono_Fte['fecha']																#Elimina el campo fecha que ya es index
    Prono_Fte.index.rename('fecha', inplace=True)    #Cambia el nombre del indice
    Prono_Fte[Estac] = Prono_Fte['Nivel']  
    del Prono_Fte['Nivel']
    df_CB_F_prono = df_CB_F_prono.join(Prono_Fte, how = 'outer')

fecha0 = df_CB_F0.index.max()
#fecha1 = (fecha0 - timedelta(hours=6))
df_CB_F_prono = df_CB_F_prono[df_CB_F_prono.index>fecha0]
df_CB_F =  pd.concat([df_CB_F0, df_CB_F_prono], ignore_index=False)

# df_CB_F.plot()
# plt.xlabel('fecha')
# plt.ylabel('Altura')
# plt.legend()
# plt.title('Condiciones de Borde: Delta Frontal - Resultados')
# f_nameCBF = carpetaFiguras+'04_CBFrente.jpg'
# plt.savefig(f_nameCBF, format='jpg')
# #plt.show()

#Interpola Frente
df_CB_F['aux'] = df_CB_F['SanFernando'] - df_CB_F['NuevaPalmira']

df_CB_F['Lujan'] = df_CB_F['SanFernando']
df_CB_F['SanAntonio'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.024)
df_CB_F['CanaldelEste'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.077)
df_CB_F['Palmas'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.123)
df_CB_F['Palmas b'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.227)
df_CB_F['Mini'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.388)
df_CB_F['LaBarquita'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.427)
df_CB_F['BarcaGrande'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.493)
df_CB_F['Correntoso'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.598)
df_CB_F['Guazu'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.800)
df_CB_F['Sauce'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.900)
df_CB_F['Bravo'] = df_CB_F['NuevaPalmira']
df_CB_F['Gutierrez'] = df_CB_F['NuevaPalmira']


del df_CB_F['aux']

df_CB_F['Fecha'] = df_CB_F.index
df_CB_F2 = pd.melt(df_CB_F, id_vars=['Fecha'], value_vars=['Lujan','SanAntonio','CanaldelEste','Palmas','Palmas b','Mini','LaBarquita','BarcaGrande','Correntoso','Guazu','Sauce','Bravo','Gutierrez'],var_name='Estacion', value_name='Nivel')
df_CB_F2['Nivel'] = df_CB_F2['Nivel'].round(3)

df_CB_F2.to_sql('CB_FrenteDelta', con = connLoc, if_exists='replace',index=False)
connLoc.commit()

### Temporal Agrega condBorde Lujan, Gualeguay y Ibicuy
if True:
    print('\nTemporal:  ----------------------------------------')
    print('Agrega condBorde Lujan, Gualeguay y Ibicuy: Q cte')
    ## Lujan
    df_aux_i = pd.DataFrame()    
    df_aux_i['Fecha'] = df_CB_F.index
    df_aux_i['Nivel'] = np.nan
    df_aux_i['Caudal'] = 10
    df_aux_i['Id_CB'] = 10  # Lujan
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

    ## Gualeguay
    df_aux_i = pd.DataFrame()    
    df_aux_i['Fecha'] = df_CB_F.index
    df_aux_i['Nivel'] = np.nan
    df_aux_i['Caudal'] = 10
    df_aux_i['Id_CB'] = 11 # Gualeguay
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

    ## Ibicuy
    df_aux_i = pd.DataFrame()    
    df_aux_i['Fecha'] = df_CB_F.index
    df_aux_i['Nivel'] = np.nan
    df_aux_i['Caudal'] = 50
    df_aux_i['Id_CB'] = 12 # Ibicuy
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)
