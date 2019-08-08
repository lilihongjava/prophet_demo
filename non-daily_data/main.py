# encoding: utf-8
"""
@author: lee
@time: 2019/8/8 10:08
@file: main.py
@desc: 
"""
from fbprophet import Prophet
import pandas as pd


def main():
    # Sub-daily data 短周期数据
    df = pd.read_csv('./data/example_yosemite_temps.csv')
    m = Prophet(changepoint_prior_scale=0.01).fit(df)
    future = m.make_future_dataframe(periods=300, freq='H')
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()

    fig = m.plot_components(fcst)
    fig.show()

    # 有规律差距的数据
    df2 = df.copy()
    df2['ds'] = pd.to_datetime(df2['ds'])
    df2 = df2[df2['ds'].dt.hour < 6]
    m = Prophet().fit(df2)
    future = m.make_future_dataframe(periods=300, freq='H')
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()

    future2 = future.copy()
    future2 = future2[future2['ds'].dt.hour < 6]
    fcst = m.predict(future2)
    fig = m.plot(fcst)
    fig.show()

    # 月数据
    df = pd.read_csv('./data/example_retail_sales.csv')
    m = Prophet(seasonality_mode='multiplicative').fit(df)
    future = m.make_future_dataframe(periods=3652)
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()

    m = Prophet(seasonality_mode='multiplicative', mcmc_samples=300).fit(df)
    fcst = m.predict(future)
    fig = m.plot_components(fcst)
    fig.show()

    future = m.make_future_dataframe(periods=120, freq='M')
    fcst = m.predict(future)
    fig = m.plot(fcst)
    fig.show()


if __name__ == "__main__":
    main()

