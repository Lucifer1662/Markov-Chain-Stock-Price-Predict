# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
order = 15
states = 4


transitionProbabilities = np.zeros((states, states))

transitionCounts = np.zeros((states, states))

transitionStateCounts = np.zeros(states)


# %%
def jsonToDataFrame(stockData):
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    dates = []

    for key in stockData:
        (open_, high, low, close, volume) = stockData[key]
        opens.append(open_)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        volumes.append(volume)
        dates.append(key)

    data = {'Date':dates, 'Open':opens, 'High':highs, 'Low':lows, 'Close':closes, 'Volume': volumes}

    return pd.DataFrame(data=data)


# %%



def convertArrayToTimeSeries(array, duration):
    length = len(array) - duration
    if(length <= 0):
        return np.ndarray((0,duration))

    subTimeSeries = np.ndarray((length, duration))
    for i in range(0, length):
        subTimeSeries[i] = array[i:i+duration]
    return subTimeSeries

def normalizeTimeSerie(timeSerie):
    avg = np.mean(timeSerie)
    var = np.var(timeSerie)
    if(var == 0):
        return np.zeros(timeSerie.shape)
    else:
        return (timeSerie - avg)/var

def normalizeTimeSeries(timeSeries):
    avg = np.mean(timeSeries, axis=1)
    var = np.var(timeSeries, axis=1)
    

    timeSeriesNorm = np.ndarray(timeSeries.shape)
    for i in range(len(timeSeries)):
        if(var[i] == 0):
            timeSeriesNorm[i] = np.zeros(timeSeries[i].shape)
        else:
            timeSeriesNorm[i] = (timeSeries[i] - avg[i])/var[i]
    return timeSeriesNorm



# %%
from convertStockToJson import getJsonDataFromFile

def loadTimeSeriesNormalized(name):
    stockData = getJsonDataFromFile(name)
    stockDatadf = jsonToDataFrame(stockData) 
    subTimeSeries = convertArrayToTimeSeries(stockDatadf['Close'], order)
    return normalizeTimeSeries(subTimeSeries)


# %%
stockNameTrainedOn = 'NAB'



subTimeSeriesNorm = loadTimeSeriesNormalized(stockNameTrainedOn)[:-365]
print(len(subTimeSeriesNorm))


# %%
from sklearn.cluster import KMeans
import numpy as np
import numpy as np
import matplotlib.pyplot as plt




kmeans = KMeans(n_clusters=states, random_state=0).fit(subTimeSeriesNorm)
cluserCenters = kmeans.cluster_centers_


for cluster in cluserCenters:
   
    plt.plot(cluster)
    plt.show()




# %% [markdown]
# fig, axs = plt.subplots(states, states)
# fig.set_size_inches(18.5, 10.5)
# for currentState in range(states):
#     for nextState in range(states):
#         last = cluserCenters[currentState][-1]
#         dif = last - cluserCenters[nextState][0]
# 
#         clusterNextStateCopy = cluserCenters[nextState].copy()
#         clusterNextStateCopy += dif
# 
#         pltData = np.concatenate((cluserCenters[currentState], clusterNextStateCopy[1:]))
# 
#         axs[currentState, nextState].axvline(x=order)
#         
#      
# 
#         axs[currentState, nextState].plot(pltData)
# 
#         axs[currentState, nextState].set_title('Axis ' + str(currentState) + ", " + str(nextState))
# 
# plt.plot(cluster)
# 
# 

# %%

buysOrSells = np.zeros(states) #VAS

#buysOrSells = np.array([0.0,1,-1,-1,0,1,-0.7,0.5]) #ANZ

for i in range(states):
    buysOrSells[i] = cluserCenters[i][-1] - cluserCenters[i][0]


buysOrSells = (buysOrSells - np.mean(buysOrSells))/np.var(buysOrSells)

print(buysOrSells)
print(np.dot(transitionProbabilities, buysOrSells))


# %%
def predictStateVector(timeSerieNorm):
    distances = kmeans.transform([timeSerieNorm])[0]
    minDis = np.min(distances)
    maxDis = np.max(distances)
    distances -= minDis
    distances /= maxDis
    distances = 1 - distances
    distances = distances * distances * distances
    return distances


def predictStatesVector(timeSerieNorms):
    result = np.zeros((timeSerieNorms.shape[0], states))
    for i in range(len(timeSerieNorms)):
        result[i] = predictStateVector(timeSerieNorms[i])
    return result

    distances = kmeans.transform(timeSerieNorms)
    minDis = np.min(distances, axis=1)
    maxDis = np.max(distances, axis=1)
    for i in range(len(distances)):
        distances[i] -= minDis[i]
        distances[i] /= maxDis[i]
        distances[i] = 1 - distances[i]
        distances[i] = distances[i] * distances[i] * distances[i]
    return distances
    


# %%

def trainMarkokChain(timeSeriesNormalized):
    subSeriesClassifified = kmeans.predict(subTimeSeriesNorm)
    for i in range(len(subSeriesClassifified)-order):
        currentState = subSeriesClassifified[i]
        nextState = subSeriesClassifified[i+order]
        
        transitionCounts[currentState, nextState] += 1
        transitionStateCounts[currentState] += 1

def trainMarkokChainContinous(timeSeriesNormalized):
    global transitionCounts, transitionStateCounts
    timeSeriesKMeanDistances = predictStatesVector(timeSeriesNormalized)

    for i in range(len(timeSeriesKMeanDistances)-order):
        
        currentState = timeSeriesKMeanDistances[i]
        nextState = timeSeriesKMeanDistances[i+order]
        nextStateMat = np.tile(nextState, (len(nextState), 1))
        adjust = (nextStateMat.transpose() * currentState).transpose()
        transitionCounts += adjust
        transitionStateCounts += np.sum(adjust, axis=1)



#trainMarkokChain(subTimeSeriesNorm)
trainMarkokChainContinous(subTimeSeriesNorm)


print(transitionCounts)
print(transitionStateCounts)


# %%
def updateProbabilities():
    for i in range(transitionProbabilities.shape[0]):
        if(transitionStateCounts[i] != 0):
            for j in range(transitionProbabilities.shape[1]):
                transitionProbabilities[i, j] = transitionCounts[i, j]/transitionStateCounts[i]

updateProbabilities()
print(transitionProbabilities)


# %%
def loadTimeSeriesClassified(name):
    normalized = loadTimeSeriesNormalized(name)
    if(normalized.shape[0] > 0):
        return (kmeans.predict(normalized), True)
    else:
        return (None, False)


# %%
def testMarkovChain(subSeriesClassifified):
    numCorrect = 0
    length = len(subSeriesClassifified)-order
    for i in range(length):
        currentState = subSeriesClassifified[i]
        nextStateCorrect = subSeriesClassifified[i+order]
        nextStatePredict = np.argmax(transitionStateCounts[currentState])
        if(nextStatePredict == nextStateCorrect):
            numCorrect += 1
    if(length > 0):
        return numCorrect/length
    else:
        return None


# %%
from filesInPath import filesInPath

filenames = filesInPath('data')
import random

# random.shuffle(filenames)
# filenames = filenames[:40]
filenames = ['VAS', 'IOO', 'XRO', 'ANZ']

for (fileName,i) in zip(filenames, range(len(filenames))):
    #fileName = fileName[:-5]
 
    (timeSeriesClassified, success) = loadTimeSeriesClassified(fileName)  
    if(success):  
        percetage = testMarkovChain(timeSeriesClassified)  
        if(percetage != None):
            print(fileName + ",  " + str(percetage * 100 ) + "%")


# %%

def shouldBuyorSells(timeSeries):
    timeSerieNorms = normalizeTimeSeries(timeSeries)
    rankings = predictStatesVector(timeSerieNorms)
    return np.dot(rankings, np.dot(transitionProbabilities, buysOrSells))



def shoudlBuyorSell(timeSerie):
    #currentState = kmeans.predict(normalizeTimeSeries(np.array([timeSerie])))[0]
    timeSerieNorm = normalizeTimeSeries(np.array([timeSerie]))[0]
    rankings = predictStateVector(timeSerieNorm)
    return np.dot(rankings, np.dot(transitionProbabilities, buysOrSells) )


# %%
def runModelAgainst(subTimeSeries, threshold, callback=None):
    principle = 1000
    buyingPower = principle
    stocks = 0
    investedAmount = 0
    onlySellWhenHigher = False
    daysShouldWaitToSell = None

    for i in range(len(subTimeSeries)):
        stockPrice = subTimeSeries[i][-1]
        stockPosition = stocks*stockPrice
        stockAmount = int(buyingPower/stockPrice)
        cost = stockAmount * stockPrice 

        signal = shoudlBuyorSell(subTimeSeries[i])
        didBuy = False
        didSell = False
        
        if(daysShouldWaitToSell == None):
            if(signal > threshold):
                #should buy
                if(stockAmount > 0.05):
                    #buy
                    investedAmount += cost
                    buyingPower -= cost
                    stocks += stockAmount
                    daysShouldWaitToSell = order
                    didBuy = True


        elif(daysShouldWaitToSell == 0):
            #should sell
            if(stocks > 0): # and (investedAmount < stockPosition or not onlySellWhenHigher)):
                investedAmount -= stockPosition
                buyingPower += stockPosition
                stocks = 0
                daysShouldWaitToSell = None
                didSell = True
        else:
            daysShouldWaitToSell -= 1
        
        if(callback != None):
            callback((i, didBuy, didSell, buyingPower, stocks, investedAmount, daysShouldWaitToSell, stockPrice, stockPosition, stockAmount, cost, signal))

    buyingPower += stocks*stockPrice
    stocks = 0

    return buyingPower


# %%
def runModelAgainst(subTimeSeries, threshold, callback=None):
    principle = 1000
    buyingPower = principle
    stocks = 0
    investedAmount = 0
    onlySellWhenHigher = False
    daysShouldWaitToSell = None
    boughtFor = None
    stopGain = 1
    stopLoss = -0.03

    for i in range(len(subTimeSeries)):
        stockPrice = subTimeSeries[i][-1]
        stockPosition = stocks*stockPrice
        stockAmount = int(buyingPower/stockPrice)
        cost = stockAmount * stockPrice 

        signal = shoudlBuyorSell(subTimeSeries[i])
        didBuy = False
        didSell = False

        profitLoss =  1-stockPosition/boughtFor if boughtFor!=None else None
        
        if(boughtFor == None):
            if(signal > threshold):
                #should buy
                if(stockAmount > 0):
                    #buy
                    investedAmount += cost
                    buyingPower -= cost
                    stocks += stockAmount
                    didBuy = True
                    boughtFor = cost


        elif(profitLoss > stopGain or profitLoss < stopLoss):
            #should sell
            if(stocks > 0): # and (investedAmount < stockPosition or not onlySellWhenHigher)):
                investedAmount -= stockPosition
                buyingPower += stockPosition
                stocks = 0
                boughtFor = None
                didSell = True
        
        if(callback != None):
            callback((i, didBuy, didSell, buyingPower, stocks, investedAmount, daysShouldWaitToSell, stockPrice, stockPosition, stockAmount, cost, signal))

    buyingPower += stocks*stockPrice
    stocks = 0

    return buyingPower

# %% [markdown]
# def runModelAgainst(subTimeSeries, threshold, callback=None):
#     principle = 1000
#     buyingPower = principle
#     stocks = 0
#     investedAmount = 0
#     onlySellWhenHigher = False
#     daysShouldWaitToSell = None
#     boughtFor = None
#     stopGain = 0.2
#     stopLoss = -0.03
#     sellThreshold = -0.12
# 
#     for i in range(len(subTimeSeries)):
#         stockPrice = subTimeSeries[i][-1]
#         stockPosition = stocks*stockPrice
#         stockAmount = int(buyingPower/stockPrice)
#         cost = stockAmount * stockPrice 
# 
#         signal = shoudlBuyorSell(subTimeSeries[i])
#         didBuy = False
#         didSell = False
# 
#         profitLoss =  1-stockPosition/boughtFor if boughtFor!=None else None
#         
#         if(boughtFor == None):
#             if(signal > threshold):
#                 #should buy
#                 if(stockAmount > 0):
#                     #buy
#                     investedAmount += cost
#                     buyingPower -= cost
#                     stocks += stockAmount
#                     didBuy = True
#                     boughtFor = cost
# 
# 
#         elif(profitLoss > stopGain or profitLoss < stopLoss or sellThreshold > signal):
#             #should sell
#             if(stocks > 0): # and (investedAmount < stockPosition or not onlySellWhenHigher)):
#                 investedAmount -= stockPosition
#                 buyingPower += stockPosition
#                 stocks = 0
#                 boughtFor = None
#                 didSell = True
#         
#         if(callback != None):
#             callback((i, didBuy, didSell, buyingPower, stocks, investedAmount, daysShouldWaitToSell, stockPrice, stockPosition, stockAmount, cost, signal))
# 
#     buyingPower += stocks*stockPrice
#     stocks = 0
# 
#     return buyingPower

# %%
def thresholdRange(subTimeSeries, steps = 20):
    signals = np.ndarray((len(subTimeSeries)))
   
    for i in range(len(subTimeSeries)):
        signals[i] = shoudlBuyorSell(subTimeSeries[i])
   
    minSig = np.min(signals)
    maxSig = np.max(signals)
    return np.arange(minSig, maxSig, (maxSig-minSig)/steps)


# %%
print(len(subTimeSeriesNorm))


# %%

stockPriceHistory = jsonToDataFrame(getJsonDataFromFile(stockNameTrainedOn))['Close']

subTimeSeries = convertArrayToTimeSeries(stockPriceHistory, order)[len(subTimeSeriesNorm):]


# %%


bestthreshold = -np.inf
bestPrice = -np.inf
for threshold in thresholdRange(subTimeSeries):
    price = runModelAgainst(subTimeSeries, threshold)
    if(bestPrice < price):
        bestPrice = price
        bestthreshold = threshold
    print((price, threshold))


# %%

def trackModel(threshold):
    equity = np.zeros(len(subTimeSeries))
    monies = np.zeros(len(subTimeSeries))
    stockOverTime = np.zeros(len(subTimeSeries))
    buys = []
    sells = []
    stockPrices = []
    signals = []
    def record(state):
        (i, didBuy, didSell, buyingPower, stocks, investedAmount, daysShouldWaitToSell, stockPrice, stockPosition, stockAmount, cost, signal) = state
        buys.append(stockPrice if didBuy else None)
        sells.append(stockPrice if didSell else None)
        stockPrices.append(stockPrice)
        signals.append(signal)
        equity[i] = buyingPower + stocks*stockPrice
        monies[i] = buyingPower
        stockOverTime[i] = stocks*stockPrice

    runModelAgainst(subTimeSeries, threshold, record)
    
    plt.plot(equity)
    plt.xlabel('Time (days)')
    plt.ylabel('Equity ($)')
    plt.show()

    plt.plot(monies, c='blue')
    plt.plot(stockOverTime, c='red')
    plt.show()

    fig,ax = plt.subplots()
    ax.plot(stockPrices)
    ax.set_ylabel("Stock Price ($)")
    ax.set_xlabel("Time (days)")
    ax.scatter(np.arange(len(buys)), buys, c='green')
    ax.scatter(np.arange(len(sells)), sells, c='red')
    ax2=ax.twinx()
    ax2.plot(signals,c='black', alpha=0.2)
    ax2.plot([0, len(signals)-1], [threshold, threshold], c='red', alpha=0.2)
    ax2.set_ylabel('Signal, Future Prediction')
    fig.show()
    plt.show()

    plt.plot(signals)
    plt.show()


# %%
for threshold in thresholdRange(subTimeSeries, 6)[::-1]:
    print(threshold)
    trackModel(threshold)


# %%
trackModel(0.08)


# %%
buys = []
sells = []
stockPrices = []
signals = []
profits = []

daysShouldWaitToSell = None


for i in range(len(subTimeSeries) - order):
    stockPrice = subTimeSeries[i][-1]
    stockPriceFuture = subTimeSeries[i+order][-1]
    signal = shoudlBuyorSell(subTimeSeries[i])

    buys.append(None)
    sells.append(None)
    stockPrices.append(stockPrice)
    signals.append(signal)
    profits.append(0)
    if(signal > 1.8):
        #buy
        buys[i] = stockPrice  
        profit = stockPriceFuture - stockPrice
        profits[i] = profit


print(np.sum(profits))
plt.plot(profits)
plt.show()


# %%



