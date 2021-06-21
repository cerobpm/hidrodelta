#Exporta JSON e inserta a BD
import json
import salidaDelta2json

with open("apiLoginParams.json") as f:
	apiLoginParams = json.load(f)

salidaDelta2json.run(apiurl=apiLoginParams["url"],apicredentials={"username":apiLoginParams["username"], "password": apiLoginParams["password"]})
