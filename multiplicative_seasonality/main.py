# encoding: utf-8
"""
@author: lee
@time: 2019/8/6 8:55
@file: main.py
@desc: 
"""
from fbprophet import Prophet
import pandas as pd


def main():
    df = pd.read_csv('./data/example_air_passengers.csv')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(50, freq='MS')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    fig.show()

    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    fig.show()

    fig = m.plot_components(forecast)
    fig.show()

    m = Prophet(seasonality_mode='multiplicative')
    m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
    m.add_regressor('regressor', mode='additive')


if __name__ == "__main__":
    main()
