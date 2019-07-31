# encoding: utf-8
"""
@author: lee
@time: 2019/5/14 9:04
@file: main.py
@desc: 
"""
import pandas as pd
from fbprophet import Prophet
from pandas.plotting import register_matplotlib_converters
from fbprophet.plot import add_changepoints_to_plot
from matplotlib import pyplot as plt



def main():
    df = pd.read_csv('./data/example_wp_log_peyton_manning.csv')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=366)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    for cp in m.changepoints:
        print(cp)
        plt.axvline(cp, c='gray', ls='--', lw=2)
    plt.show()


    deltas = m.params['delta'].mean(0)
    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(range(len(deltas)), deltas, facecolor='#0072B2', edgecolor='#0072B2')
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_ylabel('Rate change')
    ax.set_xlabel('Potential changepoint')
    fig.tight_layout()
    fig.show()


    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    fig.show()

    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(df).predict(future)
    fig = m.plot(forecast)
    fig.show()

    m = Prophet(changepoint_prior_scale=0.001)
    forecast = m.fit(df).predict(future)
    fig = m.plot(forecast)
    fig.show()

    m = Prophet(changepoints=['2014-01-01'])
    forecast = m.fit(df).predict(future)
    fig = m.plot(forecast)
    fig.show()


if __name__ == "__main__":
    main()
