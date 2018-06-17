# builds a predictive model (overfitting, no testing) of Rocket Internet stock prices based on historical data (of a Google Finance csv)

#install dependencies csv, numpy, scikit-learn, matplotlib:
# pip3 install numpy
# sudo pip3 install scikit-learn
# pip3 install matplotlib

import csv
import time
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#try several option for different backends if it doesn't plot, TkAgg, WX, QTAgg, QT4Agg
plt.switch_backend('macosx')
start_time = time.time()
# create 2 lists with dates & prices
dates = []
date_name = []
prices = []

def get_data(filename):
#extracts prices & dates from csv

    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        # ignore first row as this is header
        next(csvFileReader)
        #iterate through every row in csv & return a string for each line using next method
        data_points_count = sum(1 for row in csvFileReader)
        count_values = data_points_count
        last_date = []

        for row in csvFileReader:
            # add date values to list, get first column in a row and remove dashes with split function
            last_date = row[0]
            # to take dates without market data into consideration (e.g. weekends)
            prices.append(float(row[1]))
            date_name.append(row[0])
            count_values -= 1
            current_date = data_points_count - count_values
            print(current_date)
            dates.append(current_date)
        return count_values, last_date

def predict_prices(dates, prices, x, count_values, last_date):
    #use numpy to format lists in nx1 matrix --> 1 dimensional array
    dates = np.reshape(dates, (len(dates),1))
    # create 3 different support vector machines (regression), c = penalty of error

    #linear
    svr_lin = SVR(kernel= 'linear', C=1e3)

    #polynomial
    svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)

    #radio-basis function
    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1)
    #train models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    #plot graph
    plt.scatter(date_name, prices, color='black', label = 'Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.plot(dates, svr_rbf.predict(dates), color='yellow', label='Linear model')
    # Add title & Legend
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("Support Vector Regression")
    plt.legend()
    # does not work: plt.show()
    plt.savefig('Rocket.png')
    elapsed_time = time.time() - start_time
    print('The graph was saved in Rocket.png.\n It took {} sec to process all {} input data points.'.format(elapsed_time, count_values))
    print('The latest date corresponds to {}'.format(last_date))
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

count_values, last_date = get_data('rocket_internet_historical_prices.csv')
predicted_price = predict_prices(dates, prices, 29, count_values, last_date)

#print(predicted_price)
#print(dates)
#print(prices)
