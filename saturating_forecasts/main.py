# encoding: utf-8
"""
@author: lee
@time: 2019/5/13 15:26
@file: main.py
@desc: 
"""
import pandas as pd
from fbprophet import Prophet
from pandas.plotting import register_matplotlib_converters


def main():
    df = pd.read_csv('./data/example_wp_log_R.csv')

    df['cap'] = 8.5

    m = Prophet(growth='logistic')
    m.fit(df)

    register_matplotlib_converters()
    future = m.make_future_dataframe(periods=1826)
    future['cap'] = 8.5
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()

    df['y'] = 10 - df['y']
    df['cap'] = 6
    df['floor'] = 1.5
    future['cap'] = 6
    future['floor'] = 1.5
    m = Prophet(growth='logistic')
    m.fit(df)
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()


if __name__ == "__main__":
    main()
