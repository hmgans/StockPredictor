# Import the yfinance. If you get module not found error the run !pip install yfinance from your Jupyter notebook
import yfinance as yf
import pytrends as pt
from pytrends.request import TrendReq
import datetime as dt
import pandas as pd
import pandas_market_calendars as mcal
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.model_selection import train_test_split
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
from sklearn.tree import DecisionTreeRegressor


def timeFrame(start_year, start_month, start_day, end_year, end_month, end_day):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    pytrends = TrendReq(hl='en-US', tz=0)


    # Create a calendar
    nyse = mcal.get_calendar('NYSE')

    today = dt.date.today()

    start = str(start_year)+"-"+str(start_month)+"-"+str(start_day)
    end = str(end_year)+"-"+str(end_month)+"-"+str(end_day)

    early = nyse.schedule(start_date=start, end_date=end)




    #Get all dates 
    datelist = pd.date_range(start, end)

    modifiedDateList = datelist
    # Remove all dates the market was open 
    volume = len(early['market_open'])
    for i in range(volume):
        modifiedDateList = modifiedDateList.drop(early['market_open'].index[i].date())

    #At this point date list contains dates the market is not open



    # Get the data for the stock AAPL
    tesla = yf.Ticker('TSLA')

    hist = tesla.history(interval='60m', start=start, end=end)

    hourlyStockData = (hist['Open'] - hist['Close'])/ hist['Open'] # Get hourly percentage change

    kw_list = ["TSLA"]
  
    hourlyGoogleTrendData = pd.DataFrame(pytrends.get_historical_interest(kw_list, year_start=start_year, month_start=start_month, day_start=start_day, hour_start=0, year_end=end_year, month_end=end_month, day_end=end_day, hour_end=0, cat=0, geo='', gprop='', sleep=0))
    for i in modifiedDateList:
        for j in range(24):
            if j == 0:
                time = "00"
            elif j == 1:
                time = "01"
            elif j == 2:
                time = "02"
            elif j == 3:
                time = "03"
            elif j ==4:
                time = "04"
            elif j == 5:
                time = "05"
            elif j == 6:
                time = "06"
            elif j == 7:
                time = "07"
            elif j == 8:
                time = "08"
            elif j == 9:
                time = "09"
            elif j == 10:
                time = "10"
            elif j == 11:
                time = "11"
            elif j == 12:
                time = "12"
            elif j == 13:
                time = "13"
            elif j == 14:
                time = "14"
            elif j == 15:
                time = "15"
            elif j == 16:
                time = "16"
            elif j == 17:
                time = "17"
            elif j == 18:
                time = "18"
            elif j == 19:
                time = "19"
            elif j == 20:
                time = "20"
            elif j == 21:
                time = "21"
            elif j == 22:
                time = "22"
            elif j == 23:
                time = "23"


            fullDate = str(i.date()) + " " + str(time) + ":00:00" 
            Datetime = dt.datetime.strptime(fullDate, '%Y-%m-%d %H:%M:%S')
            try:
                hourlyGoogleTrendData = hourlyGoogleTrendData.drop(Datetime)
            except:
                print("Coulld not remove" + str(Datetime))

    businesshours = hourlyGoogleTrendData.between_time('8:30','15:30')

    stock_data = []
    for i in hourlyStockData:
        stock_data.append(i)


    trends = businesshours.RSG

    finalArrayOfTrendData = []

    for trendHour in trends:
    
        finalArrayOfTrendData.append(trendHour)

    

    print(finalArrayOfTrendData[0])
    print(stock_data[0])

    finalArrayOfTrendData = np.vstack(finalArrayOfTrendData)
    stock_data = np.vstack(stock_data)

    ### Linear prediction from temperature
    #-------------------------------------------------------------------------------
    # Add a column of ones to account for the bias in the data
    bias = np.ones(shape = (len(finalArrayOfTrendData),1))
    X = np.concatenate((bias, finalArrayOfTrendData), 1)

    # Matrix form of linear regression
    coeffs = la.inv(X.T @ X) @ X.T @ stock_data
    bias = coeffs[0][0]
    slope = coeffs[1][0]
    print("Linearly Predicted Hourly Stock Change From Google Trends")
    print("Bias: " + str(bias))
    print("Slope: " + str(slope))



    residuals = np.zeros(len(stock_data))
    predictions = np.zeros(len(stock_data))
    for i in range(len(stock_data)):
        predictions[i] = finalArrayOfTrendData[i]*slope + bias
        residuals[i] = stock_data[i] - predictions[i]

    MSE = np.mean(residuals**2)
    print("Mean Squared Error = " + str(MSE) +"\n\n")

    plt.plot(predictions)
    plt.plot(stock_data)
    plt.xlabel("Hour")
    plt.ylabel("Percent Change")
    plt.title("Linearly Predicted Hourly Stock Change From Google Trends")
    plt.show()

    # Corn version
    trend_train, trend_test, stock_train, stock_test = train_test_split(finalArrayOfTrendData, stock_data, test_size = 0.2)

    trend_reg = DecisionTreeRegressor()
    trend_reg.fit(trend_train, stock_train)

    #y_test corn price_test
    plt.scatter(trend_test,stock_test, color = 'red')
    idx = np.argsort(trend_test.flatten())
    plt.plot(trend_test[idx], trend_reg.predict(trend_test)[idx], color = 'blue')
    plt.title('Trends vs Stock point change - Decision Tree Regression')
    plt.xlabel('Trend')
    plt.ylabel('stock point change')
    plt.show()

    trainPercent = .8
    numberForTrain = int(trainPercent * len(finalArrayOfTrendData))
    numberForTest = len(finalArrayOfTrendData) - numberForTrain

    trainData = finalArrayOfTrendData[:numberForTrain]
    testData = finalArrayOfTrendData[numberForTrain:]

    trainResults = stock_data[:numberForTrain]
    testResults = stock_data[numberForTrain:]

    bias = np.ones(shape = (len(trainData),1))
    X = np.concatenate((bias, trainData), 1)


    # Matrix form of linear regression
    coeffs = la.inv((X.T @ X)) @ X.T @ trainResults
    bias = coeffs[0][0]
    slope = coeffs[1][0]
    

    residuals = np.zeros(numberForTest)
    predictions = np.zeros(numberForTest)
    for j in range(numberForTest):
        predictions[j] = testData[j]*slope + bias
        residuals[j] = testResults[j] - predictions[j]

    MSE = np.mean(residuals**2)
    print("R Squared:"+ str(MSE))

    plt.plot(predictions, label="Prediction")
    plt.plot(testResults, label="Results")
    plt.xlabel("Hour")
    plt.ylabel("Percent Change")
    plt.title("Linearly Predicted Change From Trends Train:" + str(trainPercent*100) +"% Test:" + str(100-trainPercent*100)  + "% ")
    plt.legend()
    plt.show()

timeFrame(2021, 1, 1, 2021, 4, 1)
