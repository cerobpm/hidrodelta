# -*- coding: utf-8 -*-
'''Implementación y automatización del modelo hidrodinámico del Delta del río de la Plata.'''

import os, psycopg2, datetime 
from datetime import timedelta
import pandas as pd
import numpy as np
import subprocess

nomArchivo = os.path.splitext(__file__)[0]
rutaCodigo = os.path.abspath(__file__).split("\\"+nomArchivo[:5])[0].replace('\\', '/')
rutaCodigo = "C:/HIDRODELTA"

with open(rutaCodigo+'/Control.txt', 'r') as f:
    lines = f.read().splitlines()
    last_line = lines[-1]
    print (last_line)

date_time_obj = datetime.datetime.strptime(last_line, '%m/%d/%Y, %H:%M:%S')
ahora = datetime.datetime.now()

if date_time_obj.date() != ahora.date(): 
	if ahora.hour < 12:
		process1 = subprocess.Popen(['python', rutaCodigo+'/00_Delta_BBDDaHEC_conPronostico.py'])
	else:
		print ('Actualizado')
else:
	print ('Actualizado')