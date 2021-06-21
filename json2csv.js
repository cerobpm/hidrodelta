const fs = require("promise-fs")

var input = "series_map_altura.json"
var output = "series_map_altura.csv"

json2csv(input,output)


function json2csv(input,output) {
	fs.readFile(input,{encoding:"utf-8"})
	.then(content=>{
		var data=JSON.parse(content)
		var header, rows
		if(Array.isArray(data)) {
			if(data.length==0) {
				return ""
			}
			header = Object.keys(data[0])
			rows = data.map(item=>{
				var row = header.map(key=>{
					return (item[key]) ? item[key] : ""
				})
				return JSON.stringify(row).slice(1,-1)
			})
		} else {
			var rowKeys = Object.keys(data)
			header = Object.keys(data[rowKeys[0]])
			rows = rowKeys.map(rowKey=>{
				var row = header.map(propKey=>{
					return (data[rowKey][propKey]) ? data[rowKey][propKey] : ""
				})
				return JSON.stringify([rowKey,...row]).slice(1,-1)
			})
		}
		return fs.writeFile(output,[JSON.stringify(["index",...header]).slice(1,-1),...rows].join("\n"))
	})
	.catch(e=>{
		console.error(e)
		return false
	})
}
