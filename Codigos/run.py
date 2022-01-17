# -*- coding: utf-8 -*-

import sys, getopt
import control_entrada
import autohec
import json
import salidaDelta2json

working_dir = "C:/HIDRODELTA"

with open(working_dir + "/apiLoginParams.json") as f:
	apiLoginParams = json.load(f)

#python39 C:\HIDRODELTA\01_Delta_controlentrada.py
#python39 C:\HIDRODELTA\02_AutoHEC_01.py
#python39 C:\HIDRODELTA\salidaDelta2json_run.py

help_string = 'run.py -h -i bool -m bool -s bool\
            -i, --write_input Boolean <True>: Consulta API de a5 y genera archivo SQLite con las entradas del modelo\
            -m, --run_model Boolean <True>: a partir de archivo SQLite corre HEC-RAS y genera archivo CSV con las salidas del modelo\
            -s, --save Boolean <True>: convierte archivo CSV de salida de modelo en JSON para POSTear a la API a5\
            -u, --update Boolean <True>: POSTea el JSON de salida del modelo a la API a5'

def main(argv):
    write_input = True
    run_model = True
    save = True
    update = True
    try:
        opts, args = getopt.getopt(argv,"hi:m:s:u:",["help","write_input=","run_model=","save=","update="])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-i","--write_input"):
            write_input = arg.lower() == 'true' 
        elif opt in ("-m","--run_model"):
            run_model = arg.lower() == 'true' 
        elif opt in ("-s","--save"):
            save = arg.lower() == 'true' 
        elif opt in ("-u","--update"):
            update = arg.lower() == 'true' 
    print((write_input,run_model,save))
    if write_input:
        control_entrada.run()
        print("passed write_input")
    if run_model:
        autohec.run()
        print("passed run_model")
    if save:
        if update:
            salidaDelta2json.run(apiurl=apiLoginParams["url"],apicredentials={"username":apiLoginParams["username"], "password": apiLoginParams["password"]})
            print("passed save  & update")
        else:
            salidaDelta2json.run(apiurl=apiLoginParams["url"],apicredentials={"username":apiLoginParams["username"], "password": apiLoginParams["password"]},update=False)
            print("passed save")

if __name__ == "__main__":
   main(sys.argv[1:])
   print("passed run")
