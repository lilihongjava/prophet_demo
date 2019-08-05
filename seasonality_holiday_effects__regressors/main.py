# encoding: utf-8
"""
@author: lee
@time: 2019/5/24 14:26
@file: holiday.py
@desc: 
"""
from fbprophet import Prophet
import pandas as pd
import logging
import warnings
from fbprophet.plot import plot_yearly
import matplotlib.pyplot as plt


def main():
    logging.getLogger('fbprophet').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    df = pd.read_csv('./data/example_wp_log_peyton_manning.csv')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=366)
    # 假期和特殊事件建模
    playoffs = pd.DataFrame({
        'holiday': 'playoff',
        'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                              '2010-01-24', '2010-02-07', '2011-01-08',
                              '2013-01-12', '2014-01-12', '2014-01-19',
                              '2014-02-02', '2015-01-11', '2016-01-17',
                              '2016-01-24', '2016-02-07']),
        'lower_window': 0,
        'upper_window': 1,
    })
    superbowls = pd.DataFrame({
        'holiday': 'superbowl',
        'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
        'lower_window': 0,
        'upper_window': 1,
    })
    holidays = pd.concat((playoffs, superbowls))
    m = Prophet(holidays=holidays)
    forecast = m.fit(df).predict(future)

    print(forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
              ['ds', 'playoff', 'superbowl']][-10:])

    fig = m.plot_components(forecast)
    fig.show()

    # 内置国家假期
    m = Prophet(holidays=holidays)
    m.add_country_holidays(country_name='US')
    m.fit(df)
    print(m.train_holiday_names)

    forecast = m.predict(future)
    fig = m.plot_components(forecast)
    fig.show()

    # 季节性的傅立叶级数
    m = Prophet().fit(df)
    a = plot_yearly(m)
    plt.show()
    m = Prophet(yearly_seasonality=20).fit(df)
    a = plot_yearly(m)
    plt.show()

    # 指定自定义季节性
    m = Prophet(weekly_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    forecast = m.fit(df).predict(future)
    fig = m.plot_components(forecast)
    fig.show()

    # 季节性取决于其他因素
    def is_nfl_season(ds):
        date = pd.to_datetime(ds)
        return (date.month > 8 or date.month < 2)

    df['on_season'] = df['ds'].apply(is_nfl_season)
    df['off_season'] = ~df['ds'].apply(is_nfl_season)

    m = Prophet(weekly_seasonality=False)
    m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
    m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

    future['on_season'] = future['ds'].apply(is_nfl_season)
    future['off_season'] = ~future['ds'].apply(is_nfl_season)
    forecast = m.fit(df).predict(future)
    fig = m.plot_components(forecast)
    fig.show()

    # 节假日和季节性的先验scale
    m = Prophet(holidays=holidays, holidays_prior_scale=0.05).fit(df)
    forecast = m.predict(future)
    print(forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
              ['ds', 'playoff', 'superbowl']][-10:])
    m = Prophet()
    m.add_seasonality(
        name='weekly', period=7, fourier_order=3, prior_scale=0.1)

    # 额外回归量
    def nfl_sunday(ds):
        date = pd.to_datetime(ds)
        if date.weekday() == 6 and (date.month > 8 or date.month < 2):
            return 1
        else:
            return 0

    df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

    m = Prophet()
    m.add_regressor('nfl_sunday')
    m.fit(df)

    future['nfl_sunday'] = future['ds'].apply(nfl_sunday)

    forecast = m.predict(future)
    fig = m.plot_components(forecast)
    fig.show()


if __name__ == "__main__":
    main()
