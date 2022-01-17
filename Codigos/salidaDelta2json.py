# -*- coding: utf-8 -*-

import datetime
from datetime import timedelta
import pandas as pd
import json
import requests
import pytz

local = pytz.timezone("America/Argentina/Buenos_Aires")

working_dir = "C:/HIDRODELTA"

with open(working_dir + "/config.json") as f:
    config = json.load(f)
    apiLoginParams = config["api"]

if config["use_proxy"]:
    proxy_dict = config["proxy_dict"]
else:
    proxy_dict = None

def readSalidaCSV(inputfile= working_dir + '/SalidasDelta.csv'):
	return pd.read_csv(inputfile,sep=";") # , skiprows=1)

def toDate(fecha):
    naive = datetime.datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt

def salida2json(df_Sfinal,cal_id=288,days_before_forecast_date=14,fd=datetime.datetime.now()):
	#Crea Salida JSON
	map_file = open(working_dir + "/estaciones_map.json",)
	estaciones_map = json.load(map_file)
	
	fd = fd - datetime.timedelta(hours=fd.hour % 6) ## REDONDEA HORA A MÚLTIPLO DE 6
	fd = fd.strftime("%Y-%m-%d") + " %02d:00:00" % (fd.hour) 
	print("forecast_date: " + fd)

	corrida = {
		'forecast_date': fd, # datetime.datetime.now().strftime("%Y-%m-%d") + " 00:00:00",
		'cal_id': cal_id,
		'series': []
	}
	fdate = date_time_obj = local.localize(datetime.datetime.strptime(corrida['forecast_date'], '%Y-%m-%d %H:%M:%S'),is_dst=None)
	# ~ df_Sfinal = pd.read_csv(inputfile,sep=";") # , skiprows=1)
	# ~ print(df_Sfinal.head())
	df_Sfinal['fecha'] = df_Sfinal.apply(lambda row: toDate(row['fecha']), axis=1)
	initial_date = fdate - datetime.timedelta(days=days_before_forecast_date)
	isAfter = df_Sfinal['fecha']>=initial_date # fdate
	df_Sfinal = df_Sfinal[isAfter]
	# df_Sfinal['fecha'] = df_Sfinal['fecha'].dt.strftime('%Y-%m-%d %H:%M:%S') # .replace("\s+","T") + ".000Z"
	df_Sfinal['fecha'] = df_Sfinal.apply(lambda row: row['fecha'].isoformat(), axis=1)
	df_alturas = df_Sfinal[['Id','fecha','altura']].rename(columns = {'Id': 'estacion_id', 'fecha':'timestart', 'altura':'valor'},inplace = False).copy()
	df_alturas['timeend'] = df_alturas['timestart']
	# ~ df_alturas['series_id'] = df_alturas.apply(lambda row: getSeriesId(2,row['estacion_id']),axis=1)
	for key in estaciones_map["2"]:
		# print(key)
		pronos = df_alturas.loc[df_alturas['estacion_id'] == float(key)] # .drop(['estacion_id'],axis=1)
		del pronos['estacion_id']
		# print(pronos.head())
		if estaciones_map["2"][key]["cero_ign"] is not None:
			valores_0 = pronos['valor'] - estaciones_map["2"][key]["cero_ign"] 
			pronos.loc[:,'valor'] = valores_0
			# ~ print(valores_0.head())
		
		corrida['series'].append({
			'series_table': 'series',
			'series_id': estaciones_map["2"][key]['series_id'],
			'pronosticos': pronos.to_dict('records') #df_alturas.loc[df_alturas['estacion_id'] == float(key)][['timestart','timeend','valor']].to_dict('records')
		})
		# ~ corrida['series'].append(serie)

	df_caudal = df_Sfinal[['Id','fecha','caudal']].rename(columns = {'Id': 'estacion_id', 'fecha':'timestart', 'caudal':'valor'},inplace = False).copy()
	df_caudal['timeend'] = df_caudal['timestart']
	# ~ df_alturas['series_id'] = df_alturas.apply(lambda row: getSeriesId(4,row['estacion_id']),axis=1)
	for key in estaciones_map["4"]:
		pronos = df_caudal.loc[df_caudal['estacion_id'] == float(key)]
#		pronos = df_alturas.loc[df_alturas['estacion_id'] == float(key)]
		del pronos['estacion_id']
		# ~ print(pronos.head())
		corrida['series'].append({
			'series_table': 'series',
			'series_id': estaciones_map["4"][key]['series_id'],
			'pronosticos': pronos.to_dict('records') # df_alturas.loc[df_alturas['estacion_id'] == float(key)][['timestart','timeend','valor']].to_dict('records')
		})
		# ~ corrida['series'].append(serie)
	
	return corrida

def printFile(corrida,outputfile=working_dir + "/SalidasDelta.json"):
	json_salida = open(outputfile,"w")
	json_salida.write(json.dumps(corrida))
	json_salida.close()

def upsertCorrida(corrida):
	# UPSERT Corrida
	with requests.Session() as session:
		try:
		# 	login_response = session.post(apiurl + '/login', data = json.dumps(apicredentials), headers={'Content-type':'application/json'}, proxies = proxy_dict)
		# 	print(login_response.headers)
		# except:
		# 	print("login error")
		# else:
			upsert_response = session.post(
				apiLoginParams["url"] + '/sim/calibrados/' + str(corrida['cal_id']) + '/corridas',json=corrida, 
				headers = {'Authorization': 'Bearer ' + apiLoginParams["token"]},
        		proxies = proxy_dict)
			print(upsert_response.headers)
			print(upsert_response.status_code)
			if (upsert_response.status_code > 299):
				print(upsert_response.text)
			else:
				with open(working_dir + "/upsert_response.json","w") as f:
					f.write(upsert_response.text)
		except:
			print("https post error")

def run(inputfile=working_dir + '/SalidasDelta.csv',cal_id=288,days_before_forecast_date=14,outputfile=working_dir + "/SalidasDelta.json",apiurl='https://host:port/path',apicredentials={'username':'','password':''},print_file=True,update=True):
	df_Sfinal = readSalidaCSV(inputfile)
	corrida = salida2json(df_Sfinal,cal_id,days_before_forecast_date)
	if print_file:
		printFile(corrida,outputfile)
	if update:
		upsertCorrida(corrida)

# RUN

# salidaDelta2json() # apiurl='https://alerta.ina.gob.ar/a5')
# df_Sfinal = readSalidaCSV()
# corrida = salida2json(df_Sfinal)
# printFile(corrida,"salidaDelta.json")
