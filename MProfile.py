# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:08:17 2019

@author: ds
"""

import pandas as pd
import numpy as np

class MarketProfile:
    def __init__(self,pricedata):
        self.__pricedata=pricedata
         
    def getvaluearea(self,vapercent):
        totalcount=self.__pricedata['price'].count()
        #vapercent=68.5
        vacount=totalcount*vapercent/100
        counter=0

        pricecount = self.__pricedata.groupby(['price']).count()[['time']].sort_values('price')

        #print (pricecount)
        maxlocation=pricecount['time'].values.argmax()
        vafrom=maxlocation
        vato=maxlocation

        print(maxlocation)
        currlocation = maxlocation
        while counter < vacount:

            counter = counter + pricecount['time'].iloc[currlocation]
            print(counter)
            if maxlocation == 0 :
                currlocation = currlocation + 1
                vato=currlocation
            elif maxlocation == pricecount['time'].count()-1:
                currlocation = currlocation - 1
                vafrom=currlocation
            else:
                if vafrom == 0 :
                    currlocation = vato+1
                    vato = currlocation
                elif vato == pricecount['time'].count()-1:
                    currlocation = vafrom - 1
                    vafrom = currlocation
                else:
                    if pricecount['time'].iloc[vafrom-1] > pricecount['time'].iloc[vato+1] :
                        currlocation = vafrom-1
                        vafrom = currlocation
                    elif pricecount['time'].iloc[vafrom-1] < pricecount['time'].iloc[vato+1] : 
                        currlocation = vato +1
                        vato = currlocation
                    else:
                        if currlocation == vato:
                            currlocation = vafrom-1
                            vafrom = currlocation
                        else:
                            currlocation = vato+1
                            vato = currlocation


        self.vastart = pricecount.index[vafrom]
        print ('vafrom', pricecount.index[vafrom])
        self.vaend = pricecount.index[vato]
        print ('vato', pricecount.index[vato])

        
        
idData=pd.read_csv("2019 FEB NIFTY.txt",header=None,usecols=[1,2,3,4,5,6],parse_dates=[[1,2]],dtype={3:np.float64,4:np.float64,5:np.float64,6:np.float64})

idData.columns=['tradetime','open','high','low','close']

idData['price'] =(idData['open']+idData['close'])/2 

import datetime
d1=datetime.datetime(2019,2,1)#,0,0,0,0)
d2=datetime.datetime(2019,2,28)#,23,59,59,999)
  
# this will give you a list containing all of the dates
daterange = [d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)]

dailyPrice = pd.DataFrame(columns=['date','open','close','vafrom','vato','nextopen'] )

for d in daterange:
    onedaydata=idData.loc[(idData['tradetime'] >= d) & (idData['tradetime'] < d+datetime.timedelta(days=1))] 
    if onedaydata.size > 0 :
        onedaydata.index = onedaydata['tradetime']
        fivemindata = onedaydata.resample('5T').mean().copy() 
        
        fivemindata.reset_index()
        fivemindata['time']= [   pd.Timestamp(x).to_pydatetime() for x in fivemindata.index.values]
        fivemindata['price']=fivemindata['price'].map(lambda x: np.ceil(x)) 
        
        m = MarketProfile(fivemindata.copy())
        m.getvaluearea(68.5)
        
        dailyPrice=dailyPrice.append({'date':d ,'open':onedaydata.head(1)['open'][0],'close':onedaydata.tail(1)['close'][0],
                           'vafrom':m.vastart,'vato':m.vaend},ignore_index=True)
         
dailyPrice['nextopen']=dailyPrice['open'].copy().shift(-1)
 
##ML algo starts
 
X = dailyPrice.iloc[:, 1:-1].values
y = dailyPrice['nextopen'].values


 
 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)