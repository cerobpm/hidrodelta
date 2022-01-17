# -*- coding: utf-8 -*-
import datetime, sqlite3, os
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import glob
import h5py

from pyras.controllers import RAS41, kill_ras
from pyras.controllers.hecras import ras_constants as RC

'''
Nombre de las columnas
Nombre en NodeName
'''

working_dir = "C:/HIDRODELTA"

'''Carga Datos de Entrada'''
def carga_datos_entrada():
    DataEntrada = pd.read_excel(working_dir + '/Datos/CondBorde.xlsx')
    DataEntrada.to_sql('DataEntrada', con = conn, if_exists='replace',index=False)
    conn.commit()
    
    quit()
    #Lee lista de secciones donde saca resultados:
    #De .csv
    Estaciones = pd.read_csv(working_dir + '/Datos/ArchSalida.csv', sep=',')

    # #De Excel
    # Estaciones = pd.read_excel('Datos/ArchSalida.xlsx')

    # #Base de Datos
    # sql_q = ('''SELECT *  FROM  WHERE  = ?;''')
    # Estaciones = pd.read_sql(sql_q, conn, params=n_est)

    print (Estaciones)

    '''	Para agregar puntos donde informar la salida: 
    Seleccionar la sección deseada en la lista del HEC-RAS desde:
    panel del plan > Options > Stage and Flow Output Locations...

    Luego agregar a la lista del archivo de entrada con: river, reach y progresiva. 
    Ademas Nombre del punto, ID para alertas, nombre abreviado y cero de escala'''

# Abrel el modelo
def AbreModelo(rutaModelo,nombre_project):
    project = rutaModelo+'/'+nombre_project+'.prj'
    

    rc = RAS41()

    res = rc.HECRASVersion()
    print('\n	HECRASVersion: '+ res)
    rc.ShowRas()
    rc.Project_Open(project)
    return rc

# Imprime informacion del modelo
def InfoModelo(rc_f):
    res = rc_f.CurrentProjectTitle()
    ProjectTitle = res
    print('	CurrentProjectTitle: '+ ProjectTitle)
    res = rc_f.CurrentGeomFile()
    CurrentGeomFile = res.split('.')[-1]
    print('	CurrentGeomFile: '+ CurrentGeomFile)
    res1 = rc_f.CurrentPlanFile()
    CurrentPlanFile = res1.split('.')[-1]
    print('	CurrentPlanFile: '+ CurrentPlanFile)
    res = rc_f.CurrentUnSteadyFile()
    CurrentUnSteadyFile = res.split('.')[-1]
    print('	CurrentUnSteadyFile: '+ CurrentUnSteadyFile)
    return CurrentGeomFile, CurrentPlanFile, CurrentUnSteadyFile

# Consulta River, Reach, Node
def listToDict(lst):
    op = { i+1 : lst[i].strip() for i in range(0, len(lst) ) }
    return op

def InfoGeon(rc_g):
    # Geometry_GetRivers'
    C_Rivers_m = rc_g.Geometry_GetRivers()
    Rivers_m_l = C_Rivers_m[1]
    CantR = C_Rivers_m[0]
    # print(Rivers_m)
    geo = rc_g.Geometry()
    gres = geo.nRiver()
    print('nRiver')
    print(gres)
    print('')
    quit()


    Rivers_m = listToDict(Rivers_m_l)
    for River_m in Rivers_m:
        id_Ri = River_m
        nombreRio = Rivers_m[River_m]
        print(id_Ri,': ',nombreRio)
        # Geometry_GetReaches
        C_Reaches_m = rc_g.Geometry_GetReaches(id_Ri)
        Reaches_m_l = C_Reaches_m[1]
        CantRe = C_Reaches_m[0]
        
        Reaches_m = listToDict(Reaches_m_l)
        
        for Reache_m in Reaches_m:
            id_Re = Reache_m
            nombre_Reach = Rivers_m[River_m]
            print(id_Re,': ',nombre_Reach)
            # Geometry_GetNodes
            C_Nodes_m = rc_g.Geometry_GetNodes(id_Ri, id_Re)
            Nodes_m_l = C_Nodes_m[0]
            
            #res = rc_g.Geometry_GetNode(1, 1, '5.39')
            #print('Geometry_GetNode')
            #print(res)
            #print('')
            print (Nodes_m_l)
    InfoGeon(rc)

def LeeCB(rutaModelo,nombre_project,CurrentUnSteadyFile):
# Lee Condicion de Borde del .u
    print ('\n Condiciones de Borde: \n')
    ruta_ptU = rutaModelo+'/'+nombre_project+'.'+CurrentUnSteadyFile
    f_ptU = open(ruta_ptU,'r')							#Abre el plan para leerlo
    df_ListaCB =  pd.DataFrame(columns=['FID', 'River','Reach', 'River_Stat','Interval','CondBorde'])
    id = 0
    for line in f_ptU:									#Lee linea por linea el plan
        line = line.rstrip()
        if line.startswith('Boundary Location='):
            line_c = line.split('=')[1]
            line_c = line_c.split(',')
            River, Reach, River_Stat = line_c[0].strip(),line_c[1].strip(),line_c[2].strip()
        if line.startswith('Interval='):
            Interval = line.split('=')[1].strip()
        if line.startswith('Stage Hydrograph='):
            tipoCondicion = line.split('=')[0].strip()
        if line.startswith('Flow Hydrograph='):
            tipoCondicion = line.split('=')[0].strip()
        if line.startswith('Lateral Inflow Hydrograph='):
            tipoCondicion = line.split('=')[0].strip()
        if line.startswith('DSS Path='):
            id += 1
            df_ListaCB = df_ListaCB.append({'FID': int(id), 
                                                    'River': River,
                                                    'Reach': Reach,
                                                    'River_Stat': River_Stat,
                                                    'Interval' : Interval,
                                                    'CondBorde' : tipoCondicion }, ignore_index=True)
    f_ptU.close()   #Cierra el archivo .u
    df_ListaCB['Node_Name'] = df_ListaCB['River']
    return df_ListaCB

def EditConBorde(rutaModelo,nombre_project,CurrentUnSteadyFile,FlowTitle,HEC_Vers,f_inicio,f_fin,restNum,restNum2):  # EDITA ARCHIVO DE CONDICIONES DE BORDE (.u)
    print ('Modifica condicion de borde(.u)')
    ruta_ptU = rutaModelo+'/'+nombre_project+'.'+CurrentUnSteadyFile    #Ruta del archivo (.u) de condiciones de borde 
    f_ptU = open(ruta_ptU,"w")
	
    # Escribe el .u
    f_ptU.write('Flow Title='+FlowTitle+'\n')					#Escribe el titulo
    f_ptU.write('Program Version='+HEC_Vers+'\n')         #Version del programa
    
    #Condicion Inicial
    f_ptU.write('Use Restart=-1 \n')							#Usa una condicion inicial ya generada
	
    date_CB = f_inicio.strftime('%d%b%Y')								            #Fecha de la condicion de borde (LECTURA, generado hace 5 dias)
    RestFile = nombre_project+restNum+date_CB+' '+restNum2+'.rst'				#Archivo de condiciones de borde
    RestFile = 'DeltaParana_2016.p21.28Sep2018 2400.rst'# Borrar luego
    
    #Este bloque chequea que el archivo de condiciones de borde existas. Si no existe busca en dias anteriores hasta encontrar alguno.
    f_inicio_aux = f_inicio                                                         #Guarda la fecha de inicio.
    while os.path.isfile(rutaModelo+'/'+RestFile) == False:                         #Busca si el archivo existe.
        f_inicio_aux = f_inicio_aux - timedelta(days=1)                             #Resta un dia a la fecha de inicio para buscar archivos anteriores.
        date_CB = f_inicio_aux.strftime('%d%b%Y')						            #Nueva fecha de la condicion de borde.
        RestFile = nombre_project+restNum+date_CB+' '+restNum2+'.rst'			#Archivo de condiciones de borde
        print ('No encuentra archivo de CB. Busca dias anteriores.')
        break
    print ('	Fecha de condicion de borde: '+date_CB)
    f_ptU.write('Restart Filename='+RestFile+'\n')                         #Escribe el nombre del archivo de arranque utilizado
    
    f_ptU.write('Keep SA Constant=-1\n')
    
	#Funcion que esribe en el archivo .u (Unsteady Flow Data) cada condicion de borde
    def cargaDatos(rio, reach, id_progi, condborde, listQ, StartDate, Interval):
        f_ptU.write('Boundary Location='+rio+','+reach+','+str(id_progi)+',        ,                ,                ,                ,                \n')
        f_ptU.write('Interval='+Interval+'\n')
        f_ptU.write(condborde+'= '+str(len(listQ))+'\n')
        i2 = 1
        for i in listQ:
            if i2 == 10:
                f_ptU.write(str(i).rjust(8, ' ')+'\n')
                i2 = 1
            else:
                f_ptU.write(str(i).rjust(8, ' '))
                i2 += 1
        f_ptU.write('\n')
        
        if rio == 'Victoria':
            f_ptU.write('Flow Hydrograph Inital WS= 2\n')
        
        f_ptU.write('DSS Path=\n')
        f_ptU.write('Use DSS=False\n')
        f_ptU.write('Use Fixed Start Time=True\n')
        f_ptU.write('Fixed Start Date/Time='+StartDate+',\n')
        f_ptU.write('Is Critical Boundary=False\n')
        f_ptU.write('Critical Boundary Flow=\n')
    
    # Escribe para cada CB
    sql_query = ('''SELECT FID, River, Reach, River_Stat, Interval, CondBorde FROM Est_CB''')
    Est_CB = pd.read_sql_query(sql_query, conn)
    #print(Est_CB)
    
    # index_1H = pd.date_range(start=f_inicio, end=f_fin, freq='H')
    for idx,rios in Est_CB.iterrows():										#Loop para cada rio/reach con condicion de borde	
        # df_input = pd.DataFrame(index = index_1H)					#DataFrame que almacena los datos de la BBDD. Ya tiene las fechas como indice,
        # df_input.index = df_input.index.round("H")						#Redondea para quitar decimales
        # df_input.index.rename('fecha', inplace=True)					#Sirve para que todas las estaciones consultadas tengan la misma cantidad de registros. Si faltan interpola.
		
        rio = rios['River']												    	#Rio
        reach = rios['Reach']													#Reach
        print(rio, reach)
        id_progi = rios['River_Stat']												#Progresiva 
        condborde = rios['CondBorde']												#Tipo de condicion de borde
        f_inicio_Hdin = f_inicio.strftime('%d%b%Y')						#Fecha de inicion del modelo hidrodinamico
        id = rios['FID']													#Id para buscar en la BBDD
        Interval = rios['Interval']
        
        #Consulta BBDD Alertas
        if rio == 'Parana':
            paramH = [f_inicio, f_fin]
            sql_query = ('''SELECT Fecha, Nivel as Variable
                            FROM DataEntrada WHERE Fecha BETWEEN ? AND ? AND Id_CB=1
                            ORDER BY Fecha''')
        elif rio == 'Gualeguay':
            paramH = [f_inicio, f_fin]
            sql_query = ('''SELECT Fecha, Caudal as Variable
                            FROM DataEntrada WHERE Fecha BETWEEN ? AND ? AND Id_CB=11
                            ORDER BY Fecha''')         
        elif rio == 'Ibicuy':
            paramH = [f_inicio, f_fin]
            sql_query = ('''SELECT Fecha, Caudal as Variable
                            FROM DataEntrada WHERE Fecha BETWEEN ? AND ? AND Id_CB=12
                            ORDER BY Fecha''')
        elif reach == '1':#Lujan
            paramH = [f_inicio, f_fin]
            sql_query = ('''SELECT Fecha, Caudal as Variable
                            FROM DataEntrada WHERE Fecha BETWEEN ? AND ? AND Id_CB=10
                            ORDER BY Fecha''')
        else:
            paramH = [f_inicio, f_fin, rio]
            sql_query = ('''SELECT Fecha, Nivel as Variable
                FROM CB_FrenteDelta WHERE Fecha BETWEEN ? AND ? AND Estacion=?
                ORDER BY Fecha''')
        
        dfh = pd.read_sql_query(sql_query, conn, params=paramH)		#Toma los datos de la BBDD
        dfh['Fecha'] =  pd.to_datetime(dfh['Fecha'], format='%Y-%m-%d')		#Pasa a una lista la columna fecha y lo combierte en formato fecha					
        dfh.set_index(dfh['Fecha'], inplace=True)        #Pasa la lista de fechas al indice
        del dfh['Fecha']											#Elimino el campo fecha que ya pasamos a indice
        dfh.index.rename('Fecha', inplace=True)						#Renombre el indice
        # dfh.columns = [rios[5],]									#Cambia nombre de la columna por nombre del rio
        # # dfh[rios[5]] = dfh[rios[5]] + rios[7]						#Suma Cota IGN
        # dfh = dfh.resample('D').mean()								#Toma solo un dato diario, el prodio de los que tenga
        # #Completa en caso que falta algun dato para las fechas fijadas
        # df_input = df_input.join(dfh, how = 'left').resample('D').mean()
        # df_input = df_input.interpolate()
		
        
        listV = dfh['Variable'].values.tolist()					#Pasa la columna a una lista
        listV = [ '%.2f' % elem for elem in listV ]					#Elimina decimales OJO!!!
		
		#Funcion Carga Datos: Escribe en el .u
        cargaDatos(rio, reach, id_progi, condborde, listV, f_inicio_Hdin, Interval)

    f_ptU.close()		#Cierra el archivo .u
    print ('Guarda condicion de borde(.u) \n')

# Lee Puntos de salida en el plan.
def SalidasPaln(rutaModelo,nombre_project,CurrentPlanFile):
    print ('\n Lee puntos de Salidas')
    ruta_plan = rutaModelo+'/'+nombre_project+'.'+CurrentPlanFile
    f_plan = open(ruta_plan,'r')							#Abre el plan para leerlo
    df_SalidasPlan =  pd.DataFrame(columns=['FID', 'River','Reach', 'River_Stat'])
    id = 0
    for line in f_plan:										#Lee linea por linea el plan
        line = line.rstrip()
        if line.startswith('Stage Flow Hydrograph='):
            line = line.split('=')[1]
            line = line.split(',')
            River, Reach, River_Stat = line[0].strip(),line[1].strip(),line[2].strip()
            id += 1
            df_SalidasPlan = df_SalidasPlan.append({'FID': int(id), 'River': River,'Reach': Reach,'River_Stat': River_Stat}, ignore_index=True)
            df_SalidasPlan['Node_Name'] = df_SalidasPlan['River_Stat']
    f_plan.close()
    return df_SalidasPlan

# Edita el Plan de corrida
def EditaPlan(rutaModelo,nombre_project,CurrentPlanFile,f_inicio,f_fin,f_condBorde):
	print ('Modifica el plan de la corrida')
	#Cambia el formato de las fechas para escribirlas en el plan del HECRAS
	f_inicio_Hdin = f_inicio.strftime('%d%b%Y')
	f_fin_Hdin = f_fin.strftime('%d%b%Y')
	f_condBorde = f_condBorde.strftime('%d%b%Y')

	ruta_plan = rutaModelo+'/'+nombre_project+'.'+CurrentPlanFile
	ruta_temp = rutaModelo+'/temp.'+CurrentPlanFile
	f_plan = open(ruta_plan,'r')							#Abre el plan para leerlo
	temp = open(ruta_temp,'w')								#Crea un archivo temporal
	for line in f_plan:										#Lee linea por linea el plan
		line = line.rstrip()
		if line.startswith('Simulation Date'):				#Modifica la fecha de simulacion
			newL1 = ('Simulation Date='+f_inicio_Hdin+',0000,'+f_fin_Hdin+',0000')
			temp.write(newL1+'\n')							#Escribe en el archivo temporal la fecha cambiada
		elif line.startswith('IC Time'):
			newL2 = ('IC Time=,'+f_condBorde+',')			#Modifica la fecha de condicion de borde
			temp.write(newL2+'\n')							#Escribe en el archivo temporal la fecha de condicon de borde
		else:
			temp.write(line+'\n')							#Escribe en el archivo temporal la misma linea
	temp.close()
	f_plan.close()
	os.remove(ruta_plan)									#Elimina el plan viejo
	os.rename(ruta_temp,ruta_plan)							#Cambia el nombre del archivo temporal
	print ('Guarda el plan de la corrida\n')

# Corre el HECRAS
def correModelo(rc_m):
	print('Corre el modelo:')
	res = rc_m.Compute_CurrentPlan()				#Corre el modelo
	if res == False:
		print('	Error al correr el modelo.')
		rc_m.close()
		kill_ras()
		quit()
	if res == True:
		print('	Sin problemas')

# Loop sobre el archivo con las secciones de salida
def ExtraeSimulados(Estaciones,NomTabla):
    cur.execute('DROP TABLE IF EXISTS '+NomTabla+';')
    #query = 'SELECT * FROM {}'.format(table)
    for index, Estac in Estaciones.T.iteritems():
        Id = Estac['FID']
        river = Estac['River']
        reach = Estac['Reach']
        RS = Estac['River_Stat']
        nombre = Estac['Node_Name']
        
        print (Id,' ',river,' - ',reach,': ',RS)
        
        #CARGA SIMULADOS
        res = rc.OutputDSS_GetStageFlow(river, reach, RS)
        res = list(res)
        data = pd.DataFrame({'fecha': res[1], 'nivel_sim': res[2], 'caudal_sim': res[3]})
        data['fecha'] =  pd.to_datetime(data['fecha'])
        data['Id'] = Id
        data.to_sql(NomTabla, con = conn, if_exists='append',index=False)
    conn.commit()

def print_h5_structure(f, level=0):
    '''    prints structure of hdf5 file    '''
    for key in f.keys():
        if isinstance(f[key], h5py._hl.dataset.Dataset):
            print(f"{'  '*level} DATASET: {f[key].name}")
        elif isinstance(f[key], h5py._hl.group.Group):
            print(f"{'  '*level} GROUP: {key, f[key].name}")
            level += 1
            print_h5_structure(f[key], level)
            level -= 1

        if f[key].parent.name == "/":
            print("\n"*2)

def FechasCorrida(rutaModelo,nombre_project,CurrentPlanFile):
    print ('\n Lee fechas de corrida')
    ruta_plan = rutaModelo+'/'+nombre_project+'.'+CurrentPlanFile
    f_plan = open(ruta_plan,'r')							#Abre el plan para leerlo
    for line in f_plan:										#Lee linea por linea el plan
        line = line.rstrip()
        if line.startswith('Simulation Date='):
            line = line.split('=')[1]
            line = line.split(',')
            
            f_inici0 = line[0]+' '+line[1]
            f_fin0 = line[2]+' '+line[3]
            break
    
    f_inici0 = pd.to_datetime(f_inici0)#, format='%Y-%m-%d')
    f_fin0 = pd.to_datetime(f_fin0)
    f_plan.close()
    return f_inici0,f_fin0

##############################################################################
working_dir = "C:/HIDRODELTA"

#Arranca Programa
conn = sqlite3.connect(working_dir + '/BD_Delta_001.sqlite')
cur = conn.cursor()

# Ruta al Modelo y Nombre del projecto
rutaCodigo = working_dir
rutaModelo = rutaCodigo+"/Modelo-DeltaParana_2017_pre"		#Ruta al modelo
# rutaModelo = "Modelo-DeltaParana_2017_pre"
nombre_project='DeltaParana_2016'
CurrentUnSteadyFile = 'u10'
CurrentPlanFile = 'p21'

def run(editarCB = True,Plotear = False):
    df_Lista_CB = LeeCB(rutaModelo,nombre_project,CurrentUnSteadyFile)
    print (df_Lista_CB)
    df_Lista_CB.to_sql('Est_CB', con = conn, if_exists='replace',index=False)
    conn.commit()
    sql_query = ('''SELECT min(Fecha) as f_ini, max(Fecha) as f_fin
                FROM CB_FrenteDelta;''')
    fechas = pd.read_sql_query(sql_query, conn)
    f_inicio = datetime.datetime.strptime(fechas['f_ini'].values[0], '%Y-%m-%d %H:%M:%S')
    f_fin  = datetime.datetime.strptime(fechas['f_fin'].values[0], '%Y-%m-%d %H:%M:%S')
    f_condBorde = f_inicio + timedelta(days=5)
    if editarCB == True:
        FlowTitle = 'BOUNDARY_008'           #Titulo de la corrida
        HEC_Vers = '4.10'                   #Version del programa
        #Use Restart=-1
        restNum = '.p21.'
        restNum2 = '2400'
        EditConBorde(rutaModelo,nombre_project,CurrentUnSteadyFile,FlowTitle,HEC_Vers,f_inicio,f_fin,restNum,restNum2)
    EditaPlan(rutaModelo,nombre_project,CurrentPlanFile,f_inicio,f_fin,f_condBorde)
    print ('HEC-RAS:')
    rc = AbreModelo(rutaModelo,nombre_project)
    correModelo(rc)
    df_Sfinal = pd.DataFrame()
    Nom_Salidas = '/0_Salidas.xls'
    archivoSalida = rutaCodigo + Nom_Salidas
    lst_output = pd.read_excel(archivoSalida)
    for index, CBi in lst_output.T.iteritems():
        res = rc.OutputDSS_GetStageFlow(CBi['River'], CBi['Reach'], CBi['River Stat'])
        res = list(res)
        data = pd.DataFrame({'fecha': res[1], 'altura': res[2], 'caudal': res[3]})
        #print (CBi['River'], CBi['Reach'], CBi['River Stat'], data['caudal'].mean())
        keys =  pd.to_datetime(data['fecha'])
        data.set_index(pd.DatetimeIndex(keys), inplace=True)
        data.index.rename('fecha', inplace=True)
        #
        data['Id'] = CBi['IdBBDD']
        data['Nombre'] = CBi['NomSalida']
        data['DiaCorrida'] = f_fin
        data = data[['Id', 'Nombre', 'fecha', 'altura', 'caudal','DiaCorrida']]
        #        
        if Plotear==True:
            plt.plot(data['altura'],label='Altura_Sim')
            plt.xlabel('Fecha')
            plt.ylabel('Altura')
            plt.title(CBi['NomBBDD'])
            plt.legend()
            plt.show()
        #       
        frames = [df_Sfinal, data]
        df_Sfinal = pd.concat(frames)
    #
    rc.close()
    #            
    df_Sfinal = df_Sfinal.reset_index(drop=True)
    #Guarda en BBDD
    df_Sfinal.to_sql(working_dir + '/SalidasDelta', con = conn, if_exists='replace', index=False)
    conn.commit()
    #
    #Guarda en CSV
    df_Sfinal.to_csv(working_dir + '/SalidasDelta.csv', index=False, sep=';')
    print ('Guarda Salidas')
    #
    cur.close()				#Cierra la conexion con la BBDD
    conn.close()				#Cierra la conexion con la BBDD	
    #
    #Control de corridas
    ahora2 = datetime.datetime.now()
    date_time = ahora2.strftime("%m/%d/%Y, %H:%M:%S")
    #
    archivo_salida=open(rutaCodigo+'/Control.txt',"a")
    archivo_salida.write(date_time+'\n')
    archivo_salida.close()

#Exporta JSON e inserta a BD
# import salidaDelta2json

# with open("apiLoginParams.json") as f:
	# apiLoginParams = json.load(f)

# salidaDelta2json.run(apiurl=apiLoginParams["url"],apicredentials={"username":apiLoginParams["username"], "password": apiLoginParams["password"]})



