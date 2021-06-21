# -*- coding: utf-8 -*-
'''Implementación y automatización del modelo hidrodinámico del Delta del río Parana.'''

import os, psycopg2, datetime, sqlite3 
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

print ('\n Modelo Hidrodinamico del Delta del rio Parana. \n')

nomArchivo = os.path.splitext(__file__)[0]
rutaCodigo = os.path.abspath(__file__).split("\\"+nomArchivo[:5])[0].replace('\\', '/')
rutaCodigo = "C:/HIDRODELTA"


rutaModelo = rutaCodigo+"/Modelo-DeltaParana_2017_pre"		#Ruta al modelo

with open("dbConnParams.json") as f:
	dbConnParams = json.load(f)

'''Fechas'''
if True:
	semanasMod = 50
	ahora = datetime.datetime.now()
	#ahora = datetime.datetime(2018, 12, 20, 18, 00)
	if ahora.hour <= 12:												# Si alertas actualizo la BBDD  
		f_fin = ahora.replace(hour=8, minute=0,second=0, microsecond=0)	# Fija hora final de la corrida
	else:
		f_fin = ahora.replace(hour=12, minute=0,second=0, microsecond=0)
	f_inicio = (f_fin - timedelta(weeks=semanasMod)).replace(hour=0, minute=0, second=0)	# Hora inicial de la corrida. Final - semanasMOD
	f_condBorde = f_inicio + timedelta(days=5)							# Guarda las condiciones de borde 5 dias despues del arranque

	print ('Fecha de corrida: ')
	print ('Desde: '+ f_inicio.strftime('%d%b%Y')+ ' Hasta: '+f_fin.strftime('%d%b%Y')+'\n')

	#DataFrame con fechas fijas para almacenar los datos de entrada. 
	index5min = pd.date_range(start=f_inicio, end=f_fin, freq='5min')	#Fechas desde f_inicio a f_fin con un paso de 5 minutos
	index1hora = pd.date_range(start=f_inicio, end=f_fin, freq='H')
	index1dia = pd.date_range(start=f_inicio, end=f_fin, freq='D')

'''Conecta con BBDD'''
if True:
	print ('Conecta con BBDD.')
	#Conecta BBDD Alertas
	try:
		conn1 = psycopg2.connect("dbname='" + dbConnParams["dbname"] + "' user='" + dbConnParams["user"] + "' host='" + dbConnParams["host"] + "'")
		cur1 = conn1.cursor()
	except:
		print( "No se ha podido establecer conexion.")

#Crea BBDD local para guardar las salidas del modelo hidrodinamico.
Nom_BBDDSalida = '/Delta_Salidas.sqlite'
BBDDSalida = rutaCodigo+Nom_BBDDSalida
print (BBDDSalida)
conn2 = sqlite3.connect(BBDDSalida)								#BBDD guarda caudales simulados.
cur2 = conn2.cursor()											#cursor: BBDD2.	

'''Condiciones de Borde: Frente Delta '''
if True:	
					#IdAlerta / NombreEstacion / Abreviatura / CeroEscala
	Estac_frente = [[1696, 'Arroyo Martinez', 'Martinez', 0],			#
					[1699, 'Nueva Palmira  ', 'NPalmira', 0.0275],		#
					#[1710, 'Desembocadura', 'Desembocadura', 0.327],	# No hay datos
					[1257, 'Arroyo Borches  ', 'Borches', 0.682],		#
					#[1702, 'Braga', 'Braga', -0.337],					# No hay datos
					[52, 'San Fernando   ', 'SFernando', -0.53],]		# CHEQUEAR CERO		A las 4am hay un error
	
	df_5m = pd.DataFrame(index = index5min)								#Crea el Df con indexUnico
	df_5m.index.rename('Fecha', inplace=True)							#Cambia nombre incide por Fecha
	df_5m.index = df_5m.index.round("5min")								#Redondea para quitar decimales
	
	for Estac in Estac_frente:
		IdAlerta, NombreEstacion, Nomb_abrev, CeroEscala = Estac[0], Estac[1], Estac[2], Estac[3]
		paramH = [f_inicio, f_fin, IdAlerta]												#Parametros para la consulta SQL a la BBDD
		sql_query = ('''SELECT timestart as fecha, valor as altura FROM alturas_all 
						WHERE  timestart BETWEEN %s AND %s AND unid=%s ''')					#Consulta SQL		
		dfh = pd.read_sql_query(sql_query, conn1, params=paramH)							#Toma los datos de la BBDD	
		keys =  pd.to_datetime(dfh['fecha'], format='%Y-%m-%d')								#Convierte a formato fecha la columna [fecha]
		dfh.set_index(pd.DatetimeIndex(keys), inplace=True)									#Pasa la fecha al indice del dataframe (DatetimeIndex)
		del dfh['fecha']																	#Elimina el campo fecha que ya es index
		dfh.index.rename('Fecha', inplace=True)												#Cambia el nombre del indice
		dfh[Nomb_abrev] = dfh['altura'] + CeroEscala										#Nombre abreviado como nombre de columna: Suma cero de escala a la altura
		del dfh['altura']																	#Elimina el campo altura

		print ('\t'+NombreEstacion+' \t'+str(len(dfh))+' datos.')							#Nombre Estacion
		dfh.index = dfh.index.round("min")													#Redondea para quitar decimales

		df_5m = df_5m.join(dfh, how = 'outer')												#Une los datos al Df creado al principio (fechas fijas cada 5min)
		df_5m = df_5m.interpolate()															#Interpola linealmente para completrar datos faltantes

	#DataFrame2 con fechas fijas para filtar un dato por hora.
	df_1h = pd.DataFrame(index = index1hora)												#Crea el Df con un paso de 1 hora
	df_1h.index.rename('Fecha', inplace=True)												#Cambia nombre incide por Fecha
	df_1h = df_1h.join(df_5m, how = 'left')													#Une por izquierda: se queda con los datos que coincida con las fechas del Df2

	if 1>2:	#Grafica Input para condiciones de borde
		df_1h.plot()
		plt.xlabel('Fecha')
		plt.ylabel('Altura')
		plt.legend()
		plt.title('Cond de Borde: Delta Frontal - Input')
		plt.show()
	
	#Se guardan los datos en NPalmira y en Borches para comprar posteriormente
	if 3>2:	#Compara
		df_4 = pd.DataFrame(index = index1hora)
		df_4.index = df_1h.index.round("H")
		df_4['Borches_Obs'] = df_1h['Borches']
		del df_1h['Borches']
	
					#id	Desde			Hasta			Long	LongTot	LongRel
	tr1=0.050		#1	Lujan			SanAntonio		3549	3549	0.049883338
	tr2=0.114		#2	SanAntonio		CanalDelEste	4539	8088	0.113681725
	tr3=0.147		#3	CanalDelEste	Palmas			2344	10432	0.146628061
	tr4=0.402		#4	Palmas			Mini			18147	28579	0.401695106
	tr5=0.451		#5	Mini			LaBarquita		3502	32081	0.450917831
	tr6=0.525		#6	LaBarquita		BarcaGrande		5278	37359	0.525103309
	tr7=0.608		#7	BarcaGrande		Correntoso		5873	43232	0.607651871
	tr8=0.795		#8	Correntoso		Guazu			13332	56564	0.795041183
	tr9=0.887		#9	Guazu			Sauce			6558	63122	0.887217834
	tr10=0.956		#10	Sauce			Bravo			4901	68023	0.956104349
					#11	Bravo			NuevaPalmira	3123	71146	1
	tr11=0.192		#12	NuevaPalmira	Gutierrez		5607	5607	0.191803783
					#13	Gutierrez		Martinez		23626	29233	1

	df_1h['Lujan'] = df_1h['SFernando']
	del df_1h['SFernando']
	df_1h['aux'] = df_1h['Lujan'] - df_1h['NPalmira']
	
	df_1h['SanAntonio'] = df_1h['Lujan'] - (df_1h['aux']*tr1)
	df_1h['CanaldelEste'] = df_1h['Lujan'] - (df_1h['aux']*tr2)
	df_1h['Palmas'] = df_1h['Lujan'] - (df_1h['aux']*tr3)
	df_1h['Mini'] = df_1h['Lujan'] - (df_1h['aux']*tr4)
	df_1h['LaBarquita'] = df_1h['Lujan'] - (df_1h['aux']*tr5)
	df_1h['BarcaGrande'] = df_1h['Lujan'] - (df_1h['aux']*tr6)
	df_1h['Correntoso'] = df_1h['Lujan'] - (df_1h['aux']*tr7)
	df_1h['Guazu'] = df_1h['Lujan'] - (df_1h['aux']*tr8)
	df_1h['Sauce'] = df_1h['Lujan'] - (df_1h['aux']*tr9)
	df_1h['Bravo'] = df_1h['Lujan'] - (df_1h['aux']*tr10)
	
	df_1h['aux'] = df_1h['NPalmira'] - df_1h['Martinez']
	df_1h['Gutierrez'] = df_1h['NPalmira'] - (df_1h['aux']*tr11)
	del df_1h['aux']
	
	df_1h = df_1h.fillna(method='bfill')					#Completa un posible faltante
	
	#Guarda en BBDD Local
	df_1h.to_sql('CB_FrenteDelta', con = conn2, if_exists='replace', index=False)
	conn2.commit()
	
	imprime_CB_Frente = False
	if imprime_CB_Frente == True:
		df_1h.to_excel('CB_FrenteDelta_interpo.xlsx')			#Guarda las salidas en un archivo de excel

	if 1>2:	#Grafica los resultados
		df_1h.plot()
		plt.xlabel('Fecha')
		plt.ylabel('Altura')
		plt.legend()
		plt.title('Condiciones de Borde: Delta Frontal - Resultados')
		plt.show()
		
		df_4['Correntoso_Interp'] = df_1h['Correntoso']
		df_4.plot()
		plt.xlabel('Fecha')
		plt.ylabel('Altura')
		plt.legend()
		plt.title('Condiciones de Borde: Delta Frontal - Interpolado Vrs Obs')
		plt.show()

	del df_1h['Martinez']
	#df_1h['Uruguay'] = df_1h['NPalmira']
	del df_1h['NPalmira']
	
	print ('Carga datos de pronostico para el Frente del Delta')
	#Fechas Pronostico
	fecha_i_prom = df_1h.index[-1] + timedelta(hours=1)
	fecha_f_prom = fecha_i_prom + timedelta(days=4)
	
	#Consulta ultima fecha de pronostico en BBDD
	cur1.execute('''SELECT MAX(timestart)
					FROM alturas_marea_suma
					WHERE id=1;''')
	fecha_f_prom = cur1.fetchone()[0]
	print ('\t'+'Fecha fin del pronóstico: '+str(fecha_f_prom))
	
	index1hora_pron = pd.date_range(start=f_inicio, end=fecha_f_prom, freq='H')
	index1dia_pron = pd.date_range(start=f_inicio, end=fecha_f_prom, freq='D')
	
	#Carga datos de pronostico para el Frente del Delta
	ids = [1,2,3,4,5,7,8,9,10,11,12,13] #Falta Gutierrez numero 6 NO hay datos en la BBDD
	for idd in ids:
		param2 = [fecha_i_prom, fecha_f_prom,idd]
		sql_q2 = ('''	SELECT nombre, timestart, altura_suma_corregida as altura 
						FROM alturas_marea_suma_corregida 
						WHERE  timestart BETWEEN %s AND %s 
						AND id=%s''')
		df_Pron_i = pd.read_sql(sql_q2, conn1, params=param2)
		nomCB = df_Pron_i['nombre'][1]
		del df_Pron_i['nombre']
		
		keys =  pd.to_datetime(df_Pron_i['timestart'], format='%Y-%m-%d')
		df_Pron_i.set_index(pd.DatetimeIndex(keys), inplace=True)
		del df_Pron_i['timestart']
		df_Pron_i.index.rename('fecha', inplace=True)
		df_Pron_i[nomCB] = df_Pron_i['altura']
		del df_Pron_i['altura']
		
		if idd == 13:	
			df_Pron_i['Gutierrez'] = df_Pron_i['Uruguay']
			del df_Pron_i['Uruguay']
		
		if idd == 1:
			df_Pron = df_Pron_i
		else:
			df_Pron = df_Pron.join(df_Pron_i, how = 'outer')
			
	frames = [df_1h, df_Pron]	
	df_1h = pd.concat(frames,sort=True)
	
	if 1>2:
		df_1h.to_excel('control_CBFrente.xlsx',index=True)
		df_1h.plot()
		#plt.plot_date(x=df_1h.index, y=df_1h['BarcaGrande'],label='Marea', fmt='r-')
		plt.xlabel('Fecha')
		plt.ylabel('Nivel')
		plt.title('Marea Astronomica')
		plt.legend()
		plt.tight_layout()
		plt.show()

'''EDITA ARCHIVO DE CONDICIONES DE BORDE (.u)'''
'''Escribe primera parte del archivo .u'''
def EscreiveU(rutaModelo,nomAchivoU,FlowTitle,ProgVersion,FechaCBLect,RestFile1,RestFile2):
	archivo_salida.write('Flow Title='+FlowTitle+'\n')				#Escribe el titulo.
	archivo_salida.write('Program Version='+ProgVersion+'\n')		#Version del programa.
	archivo_salida.write('Use Restart=-1 \n')						#Usa una condicion inicial ya generada.
	#archivo_salida.write('Use Restart=0 \n')						#No Usa una condicion inicial ya generadda.

	date_CB = FechaCBLect.strftime('%d%b%Y')						#Fecha de la condicion de borde.
	RestFile = RestFile1+date_CB+RestFile2				#Archivo de condiciones de borde
	#Este bloque chequea que el archivo de condiciones de borde existas. Si no existe busca en dias anteriores hasta encontrar alguno.
	f_inicio_aux = FechaCBLect											#Guarda la fecha de inicio.
	print (RestFile)
	while os.path.isfile(rutaModelo+'/'+RestFile) == False:				#Busca si el archivo existe.
		print ('No encuentra archivo de CB. Busca dias anteriores.')
		f_inicio_aux = f_inicio_aux - timedelta(days=1) 				#Resta un dia a la fecha de inicio para buscar archivos anteriores.
		date_CB = f_inicio_aux.strftime('%d%b%Y')						#Nueva fecha de la condicion de borde.
		RestFile = RestFile1+date_CB+RestFile2							#Archivo de condiciones de borde
		
	'''Agregar en el informe cual tomo.'''
	print ('	Fecha de condicion de borde: '+date_CB)
	archivo_salida.write('Restart Filename='+RestFile+'\n')				#Escribe el nombre del archivo de arranque utilizado

'''Escribe condicinoes de borde del archivo .u'''	
class CondBorde:
	def __init__(self, rio, reach, progresiva, tipo_cb, fuente, tabla, id_bbdd, interval, ceroIGN, ceroIGN2, tabla_pron, id_pron, f_inicio, f_fin):
		'''Agregar control de datos ingresados'''
		self.rio = rio
		#print (isinstance(rio,str))
		#print(isinstance(fooInstance, (list, tuple, Foo)))
		try:
			reach = str(reach)
			self.reach = reach
		except:
			self.reach = reach
		self.progresiva = str(progresiva)
		self.tipo_cb = tipo_cb
		self.fuente = fuente
		self.tabla = tabla
		self.id_bbdd = id_bbdd
		self.interval = interval
		self.ceroIGN = ceroIGN
		self.ceroIGN2 = ceroIGN2
		self.fch_iHEC = f_inicio.strftime('%d%b%Y')
		self.tabla_pron = tabla_pron
		self.id_pron = id_pron
		self.f_inicio = f_inicio
		self.f_fin = f_fin

	def serieData(self):
		print (self.interval)
		if self.interval == '1HOUR':
			df_input = pd.DataFrame(index = index1hora_pron)			#DataFrame que almacena los datos de la BBDD. Ya tiene las fechas como indice,
			df_input.index = df_input.index.round("H")			#Redondea para quitar decimales
		if self.interval == '1DAY':
			df_input = pd.DataFrame(index = index1dia_pron)
			df_input.index = df_input.index.round("D")	
		df_input.index.rename('fecha', inplace=True)			#para que todas las estaciones tengan la misma cantidad de registros. Si faltan interpola.
		
		#Frentre Delta
		if self.fuente == 'Interpolado':
			df_input['valor'] = df_1h[self.tabla]
		#Aporte del rio Lujan se fija como un caudal constante de 10m3/s
		if self.fuente == 'Constante':
			df_input['valor'] = 11
		#Parana e Ibicuy: distinta escala temporal
		if self.fuente == 'BBDDAlerta':
			paramH = [self.f_inicio, self.f_fin, self.id_bbdd]
			#Consulta BBDD Alertas
			sql_query = ('''SELECT to_char(timestart, 'YYYY-MM-DD HH24:MI') as fecha, valor FROM '''+self.tabla+''' WHERE timestart BETWEEN %s AND %s AND unid=%s ORDER BY fecha''')
			# sql_query = ('SELECT timestart as fecha, valor FROM '+self.tabla+' WHERE timestart BETWEEN ? AND ? AND unid=? ORDER BY fecha')
			dfh = pd.read_sql_query(sql_query, conn1, params=paramH)		#Toma los datos de la BBDD
			keys =  pd.to_datetime(dfh['fecha'], format='%Y-%m-%d')		#Pasa a una lista la columna fecha y lo combierte en formato fecha
			dfh.set_index(keys, inplace=True)							#Pasa la lista de fechas al indice
			dfh.set_index(dfh.index.round('H'), inplace=True)			#
			del dfh['fecha']											#Elimino el campo fecha que ya pasamos a indice
			dfh.index.rename('fecha', inplace=True)						#Renombre el indice
			dfh['valor'] = dfh['valor'] + self.ceroIGN					#Suma Cota IGN
			
			#Carga Pronostico
			#Fechas
			fecha_i_prom = dfh.index[-1] + timedelta(days=1)
			fecha_f_prom = fecha_i_prom + timedelta(days=4)
			paramH_Pron = [fecha_i_prom, fecha_f_prom, self.id_pron]
			#Consulta BBDD Alertas
			sql_query_Pron = ('''SELECT to_char(timestart, 'YYYY-MM-DD HH24:MI') as fecha, valor FROM '''+self.tabla_pron+''' WHERE timestart BETWEEN %s AND %s AND id=%s ORDER BY fecha''')
			dfh_Pron = pd.read_sql_query(sql_query_Pron, conn1, params=paramH_Pron)		#Toma los datos de la BBDD
			keys =  pd.to_datetime(dfh_Pron['fecha'], format='%Y-%m-%d')		#Pasa a una lista la columna fecha y lo combierte en formato fecha
			dfh_Pron.set_index(keys, inplace=True)							#Pasa la lista de fechas al indice
			dfh_Pron.set_index(dfh_Pron.index.round('H'), inplace=True)			#
			del dfh_Pron['fecha']											#Elimino el campo fecha que ya pasamos a indice
			dfh_Pron.index.rename('fecha', inplace=True)						#Renombre el indice
			dfh_Pron['valor'] = dfh_Pron['valor'] + self.ceroIGN2					#Suma Cota IGN
			
			frames_2 = [dfh, dfh_Pron]	
			dfh = pd.concat(frames_2,sort=True)
			
			dfh = dfh.resample('D').mean()								#Toma solo un dato diario, el prodio de los que tenga

			#Completa en caso que falta algun dato para las fechas fijadas
			df_input = df_input.join(dfh, how = 'left')#.resample('D').mean()
			df_input = df_input.interpolate()
			df_input = df_input.fillna(method='bfill')
			
		# nombControl = self.rio+'-'+self.tabla
		# df_input.to_excel(nombControl+'.xlsx',index=True)


		listQ = df_input['valor'].values.tolist()					#Pasa la columna a una lista
		self.listaDatos = [ '%.2f' % elem for elem in listQ ]		#Elimina decimales
		print ('\t'+self.rio+':  '+self.interval+' - Cantidad de datos: '+str(len(self.listaDatos)))

	def witreCBi(self):
		archivo_salida.write('Boundary Location='+self.rio+','+self.reach+','+self.progresiva+',        ,                ,                ,                ,                \n')
		archivo_salida.write('Interval='+self.interval+'\n')
		archivo_salida.write(self.tipo_cb+'= '+str(len(self.listaDatos))+'\n')
		i2 = 1
		for i in self.listaDatos:
			if i2 == 10:
				archivo_salida.write(str(i).rjust(8, ' ')+'\n')
				i2 = 1
			else:
				archivo_salida.write(str(i).rjust(8, ' '))
				i2 += 1
		archivo_salida.write('\n')
		archivo_salida.write('DSS Path=\n')
		archivo_salida.write('Use DSS=False\n')
		archivo_salida.write('Use Fixed Start Time=True\n')
		archivo_salida.write('Fixed Start Date/Time='+self.fch_iHEC+',\n')
		archivo_salida.write('Is Critical Boundary=False\n')
		archivo_salida.write('Critical Boundary Flow=\n')


Nom_CondBorde = '/0_CondDeBorde.xlsx'
archivoCB = rutaCodigo + Nom_CondBorde
L_CondBorde = pd.read_excel(archivoCB)

print ('Modifica condicion de borde(.u)')

nomAchivoU = '/DeltaParana_2016.u10'								#Nombre archivo .u
ruta_archivo_salida= rutaModelo+nomAchivoU							#Ruta del archivo (.u) de condiciones de borde.
archivo_salida=open(ruta_archivo_salida,"w")						#Crea el archivo y lo abre para escribir.

FlowTitle = 'BOUNDARY_008'									#Titulo de la corrida
ProgVersion = '4.10'												#Version del programa HECRAS
RestFile1 = 'DeltaParana_2016.p21.'
RestFile2 = ' 2400.rst'
FechaCBLect = f_inicio												#Fecha de la condicion de borde (LECTURA, generado hace 5 dias)

EscreiveU(rutaModelo,nomAchivoU,FlowTitle,ProgVersion,FechaCBLect,RestFile1,RestFile2)

for index, CBi in L_CondBorde.T.iteritems():											#Loop para cada rio/reach con condicion de borde
	rio_1, reach_2, progresiva_3, tipo_cb_4, fuente_5, tabla_6, id_bbdd_7, interval_8, ceroIGN_9, ceroIGN2_10, tabla_pron_11, id_pron_12 = CBi[1],CBi[2],CBi[3],CBi[4],CBi[5],CBi[6],CBi[7],CBi[8],CBi[9],CBi[10],CBi[11], CBi[12]
	
	CB = CondBorde(CBi['River'], CBi['Reach'], CBi['RS'], CBi['BoundCond'], CBi['Fuente'], CBi['Tabla'], CBi['id_bbdd'], CBi['intervalo'], CBi['ceroIGN'], CBi['ceroIGN2'], CBi['Tabla_pron'], CBi['id_pron'],f_inicio,f_fin)
	CB.serieData()
	CB.witreCBi()

archivo_salida.close()		#Cierra el archivo .u
print ('Guarda condicion de borde(.u) \n')

'''EDITA EL PLAN'''
if True:
	def EditaPlan(rutaModelo,archivo_plan,f_inicio,fecha_f_prom,f_condBorde):
		#Cambia el formato de las fechas para escribirlas en el plan del HECRAS
		f_inicio_Hdin = f_inicio.strftime('%d%b%Y')
		f_fin_Hdin = fecha_f_prom.strftime('%d%b%Y')
		f_condBorde = f_condBorde.strftime('%d%b%Y')

		ruta_plan = rutaModelo+'/'+archivo_plan
		ruta_temp = rutaModelo+'/temp.p21'
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
	
	print ('Modifica el plan de la corrida')
	archivo_plan='DeltaParana_2016.p21'
	EditaPlan(rutaModelo,archivo_plan,f_inicio,fecha_f_prom,f_condBorde)
	print ('Guarda el plan de la corrida\n')


if 1>2:	#Grafica Input para condiciones de borde
	df_1h.plot()
	plt.xlabel('Fecha')
	plt.ylabel('Altura')
	plt.legend()
	plt.title('Cond de Borde: Delta Frontal - Input')
	plt.show()	


'''CORRE EL HEC-RAS'''
if True:
	'''
	pip install pyras --upgrade
	pip install pywin32

	Step 2: Run makepy utilities
		- Go to the path where Python modules are sitting:
			It may look like this -> C:/Users\solo\Anaconda\Lib\site-packages\win32com\client
			or C:/Python27\ArcGIS10.2\Lib\site-packages\win32com\client
			or C:/Python27\Lib\site-packages\win32com\client
	- Open command line at the above (^) path and run $: python makepy.py
		select HECRAS River Analysis System (1.1) from the pop-up window
		this will build definitions and import modules of RAS-Controller for use'''

	print ('HEC-RAS:')
	from pyras.controllers import RAS500, RAS41, kill_ras
	from pyras.controllers.hecras import ras_constants as RC
	
	archivo_project='DeltaParana_2016.prj'
	project = rutaModelo+'/'+archivo_project
	rc = RAS41()

	res = rc.HECRASVersion()
	print('	HECRASVersion: '+ res)
	rc.ShowRas()
	rc.Project_Open(project)

	res = rc.CurrentProjectTitle()
	print('	CurrentProjectTitle: '+ res)
	res = rc.CurrentProjectFile()
	print('	CurrentProjectFile: '+ res)
	res = rc.CurrentGeomFile()
	print('	CurrentGeomFile: '+ res)
	res1 = rc.CurrentPlanFile()
	print('	CurrentPlanFile: '+ res1)
	res = rc.CurrentUnSteadyFile()
	print('	CurrentUnSteadyFile: '+ res)
	
	print('Corre el modelo:')
	res = rc.Compute_CurrentPlan()				#Corre el modelo
	if res == False:
		print('	Error al correr el modelo.')
		rc.close()
		#kill_ras()
		quit()
	if res == True:
		print('	Fin de la Corrida')

'''Gestión de Salidas'''
if True:
	'''	Para agregar puntos donde informar la salida: 
		Seleccionar la sección deseada en la lista del HEC-RAS desde el panel del plan > Options > Stage and Flow Output Locations...
		Luego agregar a la lista lst_output el El rio, reach y progresiva. 
		Ademas Nombre del punto, ID para alertas, nombre abreviado y cero de escala'''
	df_Sfinal = pd.DataFrame()
	Nom_Salidas = '/0_Salidas.xlsx'
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
		
		data['Id'] = CBi['IdBBDD']
		data['Nombre'] = CBi['NomSalida']
		data['DiaCorrida'] = f_fin
		data = data[['Id', 'Nombre', 'fecha', 'altura', 'caudal','DiaCorrida']]
		
		Plotear = False
		if Plotear==True:
			plt.plot(data['altura'],label='Altura_Sim')
			plt.xlabel('Fecha')
			plt.ylabel('Altura')
			plt.title(CBi['NomBBDD'])
			plt.legend()
			plt.show()
		
		frames = [df_Sfinal, data]
		df_Sfinal = pd.concat(frames)

rc.close()
		
df_Sfinal = df_Sfinal.reset_index(drop=True)
#Guarda en BBDD
df_Sfinal.to_sql('SalidasDelta', con = conn2, if_exists='replace', index=False)
conn2.commit()

#Guarda en CSV
df_Sfinal.to_csv(rutaCodigo+'/SalidasDelta.csv', index=False, sep=';')
print ('Guarda Salidas')

cur1.close()				#Cierra la conexion con la BBDD
conn1.close()				#Cierra la conexion con la BBDD	

#Control de corridas
ahora2 = datetime.datetime.now()
date_time = ahora2.strftime("%m/%d/%Y, %H:%M:%S")
archivo_salida=open(rutaCodigo+'/Control.txt',"a")
archivo_salida.write(date_time+'\n')
archivo_salida.close()

#Exporta JSON e inserta a BD
import salidaDelta2json

with open("apiLoginParams.json") as f:
	apiLoginParams = json.load(f)

salidaDelta2json.run(apiurl=apiLoginParams["url"],apicredentials={"username":apiLoginParams["username"], password: apiLoginParams["password"])
