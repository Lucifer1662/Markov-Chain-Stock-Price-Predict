import json 
import sys
#Comma separated Ticker, Date, Open, High, Low, Close, Volume.
from filesInPath import filesInPath
import pandas as pd

def toTuple(line: str):
    line = line.split(",")
    name = line[0]
    date = line[1]
    open = float(line[2])
    high = float(line[3])
    low = float(line[4])
    close = float(line[5])
    volume = int(line[6])
    return (name, date,(open, high, low, close, volume))


def getDataFromFile(pathToFiles, fileName, selectedListings, stockPrices: dict):
    f = open(pathToFiles + "/" + fileName, "r")
    line = ""
    lines = []
    while(True):
        line = f.readline()
        if(len(line) == 0):
            break
        lines.append(line.strip())
    f.close()

    for line in lines:
        [name, date, priceData] = toTuple(line)
        if(selectedListings == None or name in selectedListings):
            if(name in stockPrices):
                stockPrices[name][date] = priceData
            else:
                stockPrices[name] = {date: priceData}

    return stockPrices


def getDataFromFiles(pathToFiles, fileNames, selectedListings):
    stockPrices = {}
    for fileName in fileNames:
        getDataFromFile(pathToFiles, fileName, selectedListings, stockPrices)
    
    return stockPrices

def getDataOfListings(selectedListings, pathToFiles = "raw"):

    filenames = filesInPath(pathToFiles)

    stockPrices = getDataFromFiles(pathToFiles, filenames, selectedListings)

    for key in stockPrices:
        try:
            jsonFile = open("data/"+key+".json", "w+")
            jsonFile.write(json.dumps(stockPrices[key]))
        except:
            print("error with " + key)

    return stockPrices



def getDataFileNames():
    return map(lambda fileName:fileName.replace(".json",""),
        filesInPath("data"))


def getJsonDataFromFile(fileName):
    f = open("data/" + fileName + ".json", "r")    
    return json.load(f)



