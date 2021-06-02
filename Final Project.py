"""Week 10 Portfolio Programming Assignment"""
"""Nico Garcia-Vicente 5/31/21"""
"""For this assignment, I gathered daily stock data (over the past 5 years) from finance.yahoo.com for the 8 stocks"""
"""that we have been working with in class. I saved this data in a csv file. I then carried out the steps from"""
"""the week 8 assignment and created a SQLite table for storing the data. To make this project my own - I then"""
"""created a function that does the following: given a stock, it runs an ordinary least squares (OLS) regression"""
"""and finds the best-fit line equation. It prints out important statistics (r and p-value, etc.) which help"""
"""determine the statistical significance of the time-stock price relationship in my model. It plots the given"""
"""stock's data around the best-fit line and uses the best-fit line equation to predict the stock's price in 2025."""

"""importing all the tools/libraries necessary to complete the tasks"""
from datetime import datetime
import csv
import os
import sqlite3
import pygal
import numpy as np
import pandas as pd
import scipy.stats as sp
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from collections import defaultdict

def ols_regression(close_price_list, date_list, symbol):
    """defining a function that takes two lists (date & close price) and the respective stock symbol as arguments"""
    """first I define my y variable (close price) and x variable (date) - the dates are passed through pandas  """
    """to_datetime function to convert to a float in order to run the regression"""
    y = np.array(close_price_list, dtype=float)
    x = np.array(pd.to_datetime(date_list), dtype=float)
    """the following lines run the regression and save the results in variables, which are finally printed out"""
    slope, intercept, r_value, p_value, std_err = sp.linregress(x,y)
    xf = np.linspace(min(x), max(x), 100)
    xf1 = xf.copy()
    xf1 = pd.to_datetime(xf1)
    yf = (slope*xf) + intercept
    print(f"{symbol} OLS regression statistics: ")
    print('r-val = ', r_value, '\n', 'p-val = ', p_value, '\n', 'std_err = ', std_err)

    """this part of the function creates an array of future dates, and uses the regression model to predict the"""
    """closing price on January 1st, 2025."""
    predict_dates = ['12/30/24', '12/31/24', '1/1/25']
    x2 = np.array(pd.to_datetime(predict_dates), dtype=float)
    xf2 = np.linspace(min(x2), max(x2), 100)
    close_2025 = round((slope * xf2[2]) + intercept, 2)
    print(f"Based on an OLS linear regression model, we expect {symbol} to have a closing price of ${close_2025} on January 1, 2025.")

    """the final piece of the function plots the stock data alongside the best-fit line"""
    f, ax = plt.subplots(1, 1)
    ax.plot(xf1, yf, label='Linear fit', lw=3)
    plt.plot_date(date_list, close_price_list, ls='')
    plt.title(f"{symbol} Stock Price Over Time")
    plt.ylabel('Close Price')
    ax.legend()
    plt.show()

if os.path.exists("Final Project Stock Data.csv"):
    with open("Final Project Stock Data.csv", "r") as stock_data:
        """reading in the csv file"""
        reader = csv.reader(stock_data, delimiter=",")
        header = next(reader)
        dates, opens, highs, lows, closes, volumes, symbols, shares, values = [], [], [], [], [], [], [], [], []
        """creating empty lists for storing each column of data"""
        for row in reader:
            """parsing the data"""
            date_str = str(row[0])
            date_format = datetime.strptime(date_str, '%m/%d/%y')
            date = datetime.date(date_format)
            dates.append(date)
            open = round(float(row[1]), 2)
            opens.append(open)
            high = round(float(row[2]), 2)
            highs.append(high)
            low = round(float(row[3]), 2)
            lows.append(low)
            close = round(float(row[4]), 2)
            closes.append(close)
            volume = float(row[5])
            volumes.append(volume)
            symbol = str(row[6])
            symbols.append(symbol)
            """we know # shares, so I used an if, elif chain to enter those here"""
            if symbol == 'GOOG':
                number_shares = 125
            elif symbol == 'MSFT':
                number_shares = 85
            elif symbol == 'RDS-A':
                number_shares = 400
            elif symbol == 'AIG':
                number_shares = 235
            elif symbol == 'FB':
                number_shares = 150
            elif symbol == 'M':
                number_shares = 425
            elif symbol == 'F':
                number_shares = 85
            elif symbol == 'IBM':
                number_shares = 80
            shares.append(number_shares)
            stock_value = round(number_shares * close, 2)
            """calculating stock value (# shares * close price)"""
            values.append(stock_value)

            """creating a list of tuples, which will allow me to use the executemany command"""
            tuples = []

            for i in range(len(dates)):
                tuples.append((dates[i], opens[i], highs[i], lows[i], closes[i], volumes[i], symbols[i], shares[i], values[i]))


    """the following code will store the data in a SQLite table"""
    """connect to the SQLite database"""
    try:
        conn = sqlite3.connect('final_project.db')

    except Error as e:
        print(e)

    finally:
        cursor = conn.cursor()
        """creating the table"""
        conn.execute("CREATE TABLE stock_portfolio_tbl (date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, symbol TEXT, shares REAL, stock_value REAL)")
        """inserting the data"""
        cursor.executemany("INSERT INTO stock_portfolio_tbl VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tuples)
        conn.commit()

        """printing the data from the table to make sure it was correctly inserted"""
        rows = cursor.execute("SELECT * FROM stock_portfolio_tbl").fetchall()
        for row in rows:
            print(row)

        conn.close()

    """The following code will get the data in the format needed to run linear regressions and plot using the function I created"""
    """creating a dictionary that stores stock symbol as the key and dates as the values"""
    dates_dict = defaultdict(list)
    [dates_dict[key].append(val) for key, val in zip(symbols, dates)]

    """creating a dictionary that stores stock symbol as the key and daily stock close price as the values"""
    close_price_dict = defaultdict(list)
    [close_price_dict[key].append(val) for key, val in zip(symbols, closes)]

    """here I'm creating lists for each stock - one for dates (which will be the x-axis) and one for close price (y-axis)"""
    goog_dates = dates_dict['GOOG']
    goog_closes = close_price_dict['GOOG']

    msft_dates = dates_dict['MSFT']
    msft_closes = close_price_dict['MSFT']

    rdsa_dates = dates_dict['RDS-A']
    rdsa_closes = close_price_dict['RDS-A']

    aig_dates = dates_dict['AIG']
    aig_closes = close_price_dict['AIG']

    fb_dates = dates_dict['FB']
    fb_closes = close_price_dict['FB']

    m_dates = dates_dict['M']
    m_closes = close_price_dict['M']

    f_dates = dates_dict['F']
    f_closes = close_price_dict['F']

    ibm_dates = dates_dict['IBM']
    ibm_closes = close_price_dict['IBM']

    """finally I am calling the ols_regression function that I created above...the output is for one stock at a"""
    """time, so once you close out of the first graph/image the output and graph for the 2nd stock will show up"""
    ols_regression(goog_closes, goog_dates, 'GOOG')
    ols_regression(msft_closes, msft_dates, 'MSFT')
    ols_regression(rdsa_closes, rdsa_dates, 'RDS-A')
    ols_regression(aig_closes, aig_dates, 'AIG')
    ols_regression(fb_closes, fb_dates, 'FB')
    ols_regression(m_closes, m_dates, 'M')
    ols_regression(f_closes, f_dates, 'F')
    ols_regression(ibm_closes, ibm_dates, 'IBM')

else:
    print("This file does not exist.")