# -*- coding: utf-8 -*-

import datetime
from datetime import timedelta
import pandas as pd
import json
import requests
import pytz

local = pytz.timezone("America/Argentina/Buenos_Aires")

def readSalidaCSV(inputfile='SalidasDelta.csv'):
	return pd.read_csv(inputfile,sep=";") # , skiprows=1)

def toDate(fecha):
	naive = datetime.datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
	local_dt = local.localize(naive, is_dst=None)
	utc_dt = local_dt.astimezone(pytz.utc)
	return utc_dt

def salida2json(df_Sfinal,cal_id=288,days_before_forecast_date=14):
	#Crea Salida JSON
	estaciones_map = {
	  "2": {
		"151": {
			"estacion_id": 151,
			"nombre": "Atucha",
			"series_id": 3403,
			"var_id": 2,
			"cero_ign": -0.556
		},
		"1702": {
			"estacion_id": 1702,
			"nombre": "Braga",
			"series_id": 3407,
			"var_id": 2,
			"cero_ign": -0.337
		},
		"1710": {
			"estacion_id": 1710,
			"nombre": "Desembocadura",
			"series_id": 3395,
			"var_id": 2,
			"cero_ign": 0.327
		},
		"1711": {
			"estacion_id": 1711,
			"nombre": "Las Rosas",
			"series_id": 3406,
			"var_id": 2,
			"cero_ign": 0.03
		},
		"1712": {
			"estacion_id": 1712,
			"nombre": "Carabelitas",
			"series_id": 3396,
			"var_id": 2,
			"cero_ign": 0.209
		},
		"1713": {
			"estacion_id": 1713,
			"nombre": "Zárate",
			"series_id": 3404,
			"var_id": 2,
			"cero_ign": 0.388
		},
		"1714": {
			"estacion_id": 1714,
			"nombre": "Baradero",
			"series_id": 3402,
			"var_id": 2,
			"cero_ign": 1.156
		},
		"1715": {
			"estacion_id": 1715,
			"nombre": "Brazo Largo",
			"series_id": 3394,
			"var_id": 2,
			"cero_ign": 0.717
		},
		"1717": {
			"estacion_id": 1717,
			"nombre": "Bifurcaci\u00f3n",
			"series_id": 3417,
			"var_id": 2,
			"cero_ign": 1.37
		},
		"1770": {
			"estacion_id": 1770,
			"nombre": "Timb\u00faes",
			"series_id": 3389,
			"var_id": 2,
			"cero_ign": None
		},
		"29": {
			"estacion_id": 29,
			"nombre": "Paraná",
			"series_id": 3408,
			"var_id": 2,
			"cero_ign": 9.432
		},
		"31": {
			"estacion_id": 31,
			"nombre": "Diamante",
			"series_id": 3409,
			"var_id": 2,
			"cero_ign": 6.747
		},
		"32": {
			"estacion_id": 32,
			"nombre": "Victoria",
			"series_id": 3419,
			"var_id": 2,
			"cero_ign": 1.536
		},
		"33": {
			"estacion_id": 33,
			"nombre": "San Lorenzo (San Mart\u00edn)",
			"series_id": 3411,
			"var_id": 2,
			"cero_ign": 3.309
		},
		"34": {
			"estacion_id": 34,
			"nombre": "Rosario",
			"series_id": 3412,
			"var_id": 2,
			"cero_ign": 2.923
		},
		"35": {
			"estacion_id": 35,
			"nombre": "Villa Constituci\u00f3n",
			"series_id": 3413,
			"var_id": 2,
			"cero_ign": 1.981
		},
		"36": {
			"estacion_id": 36,
			"nombre": "San Nicolás",
			"series_id": 3414,
			"var_id": 2,
			"cero_ign": 1.91
		},
		"37": {
			"estacion_id": 37,
			"nombre": "Ramallo",
			"series_id": 3415,
			"var_id": 2,
			"cero_ign": 1.642
		},
		"38": {
			"estacion_id": 38,
			"nombre": "San Pedro",
			"series_id": 3416,
			"var_id": 2,
			"cero_ign": 0.714,
		},
		"39": {
			"estacion_id": 39,
			"nombre": "Baradero",
			"series_id": 3418,
			"var_id": 2,
			"cero_ign": 0.641
		},
		"41": {
			"estacion_id": 41,
			"nombre": "Campana",
			"series_id": 3405,
			"var_id": 2,
			"cero_ign": 0.418
		},
		"42": {
			"estacion_id": 42,
			"nombre": "Escobar",
			"series_id": 3398,
			"var_id": 2,
			"cero_ign": None
		},
		"44": {
			"estacion_id": 44,
			"nombre": "Canal Nuevo",
			"series_id": 3397,
			"var_id": 2,
			"cero_ign": 0.39
		},
		"45": {
			"estacion_id": 45,
			"nombre": "Ibicuy",
			"series_id": 3420,
			"var_id": 2,
			"cero_ign": 0.456
		},
		"49": {
			"estacion_id": 49,
			"nombre": "Tigre",
			"series_id": 3400,
			"var_id": 2,
			"cero_ign": -0.099
		},
		"50": {
			"estacion_id": 50,
			"nombre": "Dique Luján",
			"series_id": 3399,
			"var_id": 2,
			"cero_ign": -0.074
		},
		"51": {
			"estacion_id": 51,
			"nombre": "Chaná Miní",
			"series_id": 3401,
			"var_id": 2,
			"cero_ign": -0.286
		},
		"5876": {
			"estacion_id": 5876,
			"nombre": "Carabelas (ina_delta)",
			"series_id": 26206,
			"var_id": 2,
			"cero_ign": None
		},
		"5907": {
			"estacion_id": 5907,
			"nombre": "Paraná Las Palmas - Zárate (sat2)",
			"series_id": 29437,
			"var_id": 2,
			"cero_ign": 0.243
		},
		"5905": {
			"estacion_id": 5905,
			"nombre": "Paraná - Villa Constitución (sat2)",
			"series_id": 29436,
			"var_id": 2,
			"cero_ign": 1.981
		},
		"5901": {
			"estacion_id": 5901,
			"nombre": "Paraná - Túnel subfluvial (sat2)",
			"series_id": 29440,
			"var_id": 2,
			"cero_ign": 9.432
		},
		"5900": {
			"estacion_id": 5900,
			"nombre": "Paraná - Diamante (sat2)",
			"series_id": 29439,
			"var_id": 2,
			"cero_ign": 6.747
		},
		"5894": {
			"estacion_id": 5894,
			"nombre": "Riacho Victoria - Victoria (sat2)",
			"series_id": 29438,
			"var_id": 2,
			"cero_ign": 1.536
		},
		"5893": {
			"estacion_id": 5893,
			"nombre": "Paraná - Rosario (sat2)",
			"series_id": 29435,
			"var_id": 2,
			"cero_ign": 2.923
		}
	  },
	  "4": {
		"151": {
			"estacion_id": 151,
			"nombre": "Atucha",
			"series_id": 3457,
			"var_id": 4
		},
		"1702": {
			"estacion_id": 1702,
			"nombre": "Braga",
			"series_id": 3461,
			"var_id": 4
		},
		"1710": {
			"estacion_id": 1710,
			"nombre": "Desembocadura",
			"series_id": 3449,
			"var_id": 4
		},
		"1711": {
			"estacion_id": 1711,
			"nombre": "Las Rosas",
			"series_id": 3460,
			"var_id": 4
		},
		"1712": {
			"estacion_id": 1712,
			"nombre": "Carabelitas",
			"series_id": 3450,
			"var_id": 4
		},
		"1713": {
			"estacion_id": 1713,
			"nombre": "Z\u00e1rate",
			"series_id": 3458,
			"var_id": 4
		},
		"1714": {
			"estacion_id": 1714,
			"nombre": "Baradero",
			"series_id": 3456,
			"var_id": 4
		},
		"1715": {
			"estacion_id": 1715,
			"nombre": "Brazo Largo",
			"series_id": 3448,
			"var_id": 4
		},
		"1717": {
			"estacion_id": 1717,
			"nombre": "Bifurcaci\u00f3n",
			"series_id": 3471,
			"var_id": 4
		},
		"1770": {
			"estacion_id": 1770,
			"nombre": "Timb\u00faes",
			"series_id": 3392,
			"var_id": 4
		},
		"29": {
			"estacion_id": 29,
			"nombre": "Paran\u00e1",
			"series_id": 3462,
			"var_id": 4
		},
		"31": {
			"estacion_id": 31,
			"nombre": "Diamante",
			"series_id": 3463,
			"var_id": 4
		},
		"32": {
			"estacion_id": 32,
			"nombre": "Victoria",
			"series_id": 1459,
			"var_id": 4
		},
		"33": {
			"estacion_id": 33,
			"nombre": "San Lorenzo (San Mart\u00edn)",
			"series_id": 3465,
			"var_id": 4
		},
		"34": {
			"estacion_id": 34,
			"nombre": "Rosario",
			"series_id": 3466,
			"var_id": 4
		},
		"35": {
			"estacion_id": 35,
			"nombre": "Villa Constituci\u00f3n",
			"series_id": 3467,
			"var_id": 4
		},
		"36": {
			"estacion_id": 36,
			"nombre": "San Nicol\u00e1s",
			"series_id": 3468,
			"var_id": 4
		},
		"37": {
			"estacion_id": 37,
			"nombre": "Ramallo",
			"series_id": 3469,
			"var_id": 4
		},
		"38": {
			"estacion_id": 38,
			"nombre": "San Pedro",
			"series_id": 3470,
			"var_id": 4
		},
		"39": {
			"estacion_id": 39,
			"nombre": "Baradero",
			"series_id": 3472,
			"var_id": 4
		},
		"41": {
			"estacion_id": 41,
			"nombre": "Campana",
			"series_id": 3459,
			"var_id": 4
		},
		"42": {
			"estacion_id": 42,
			"nombre": "Escobar",
			"series_id": 3452,
			"var_id": 4
		},
		"44": {
			"estacion_id": 44,
			"nombre": "Canal Nuevo",
			"series_id": 3451,
			"var_id": 4
		},
		"45": {
			"estacion_id": 45,
			"nombre": "Ibicuy",
			"series_id": 3474,
			"var_id": 4
		},
		"49": {
			"estacion_id": 49,
			"nombre": "Tigre",
			"series_id": 3454,
			"var_id": 4
		},
		"50": {
			"estacion_id": 50,
			"nombre": "Dique Luj\u00e1n",
			"series_id": 3453,
			"var_id": 4
		},
		"51": {
			"estacion_id": 51,
			"nombre": "Chan\u00e1 Min\u00ed",
			"series_id": 3455,
			"var_id": 4
		}
	  }
	}
	corrida = {
		'forecast_date': datetime.datetime.now().strftime("%Y-%m-%d") + " 00:00:00",
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
	df_Sfinal['fecha'] = df_Sfinal['fecha'].dt.strftime('%Y-%m-%d %H:%M:%S') # .replace("\s+","T") + ".000Z"
	df_alturas = df_Sfinal[['Id','fecha','altura']].rename(columns = {'Id': 'estacion_id', 'fecha':'timestart', 'altura':'valor'},inplace = False)
	df_alturas['timeend'] = df_alturas['timestart']
	# ~ df_alturas['series_id'] = df_alturas.apply(lambda row: getSeriesId(2,row['estacion_id']),axis=1)
	for key in estaciones_map["2"]:
		pronos = df_alturas.loc[df_alturas['estacion_id'] == float(key)] # .drop(['estacion_id'],axis=1)
		del pronos['estacion_id']
		# ~ print(pronos.head())
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

	df_caudal = df_Sfinal[['Id','fecha','caudal']].rename(columns = {'Id': 'estacion_id', 'fecha':'timestart', 'caudal':'valor'},inplace = False)
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

def printFile(corrida,outputfile="SalidasDelta.json"):
	json_salida = open(outputfile,"w")
	json_salida.write(json.dumps(corrida))
	json_salida.close()

def upsertCorrida(corrida,apiurl='https://host:port/path',apicredentials={'username':'','password':''}):
	# UPSERT Corrida
	with requests.Session() as session:
		try:
			login_response = session.post(apiurl + '/login', data = json.dumps(apicredentials), headers={'Content-type':'application/json'})
			print(login_response.headers)
		except:
			print("login error")
		else:
			upsert_response = session.post(apiurl + '/sim/calibrados/' + str(corrida['cal_id']) + '/corridas',json=corrida)
			print(upsert_response.headers)
			print(upsert_response.status_code)
			if (upsert_response.status_code > 299):
				print(upsert_response.text)
			else:
				with open("upsert_response.json","w") as f:
					f.write(upsert_response.text)

def run(inputfile='SalidasDelta.csv',cal_id=288,days_before_forecast_date=14,outputfile="SalidasDelta.json",apiurl='https://host:port/path',apicredentials={'username':'','password':''}):
	df_Sfinal = readSalidaCSV(inputfile)
	corrida = salida2json(df_Sfinal,cal_id,days_before_forecast_date)
	printFile(corrida,outputfile)
	upsertCorrida(corrida,apiurl,apicredentials)

# RUN

# salidaDelta2json() # apiurl='https://alerta.ina.gob.ar/a5')
# df_Sfinal = readSalidaCSV()
# corrida = salida2json(df_Sfinal)
# printFile(corrida,"salidaDelta1.json")
