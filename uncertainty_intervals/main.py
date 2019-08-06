# encoding: utf-8
"""
@author: lee
@time: 2019/8/6 9:22
@file: main.py
@desc: 
"""
from fbprophet import Prophet
import pandas as pd


def main():
    df = pd.read_csv('./data/example_wp_log_peyton_manning.csv')
    df = df.loc[:180, ]  # Limit to first six months
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=60)
    # 趋势的不确定性
    forecast = Prophet(interval_width=0.95).fit(df).predict(future)
    # 季节性的不确定性
    m = Prophet(mcmc_samples=300)
    forecast = m.fit(df).predict(future)
    fig = m.plot_components(forecast)
    fig.show()


if __name__ == "__main__":
    main()
