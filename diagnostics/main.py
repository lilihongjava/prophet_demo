# encoding: utf-8
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from matplotlib import pyplot as plt
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric


def main():
    df = pd.read_csv('./data/example_wp_log_peyton_manning.csv')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=366)
    df_cv = cross_validation(
        m, '365 days', initial='1825 days', period='365 days')
    cutoff = df_cv['cutoff'].unique()[0]
    df_cv = df_cv[df_cv['cutoff'].values == cutoff]

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(m.history['ds'].values, m.history['y'], 'k.')
    ax.plot(df_cv['ds'].values, df_cv['yhat'], ls='-', c='#0072B2')
    ax.fill_between(df_cv['ds'].values, df_cv['yhat_lower'],
                    df_cv['yhat_upper'], color='#0072B2',
                    alpha=0.2)
    ax.axvline(x=pd.to_datetime(cutoff), c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('y')
    ax.set_xlabel('ds')
    ax.text(x=pd.to_datetime('2010-01-01'), y=12, s='Initial', color='black',
            fontsize=16, fontweight='bold', alpha=0.8)
    ax.text(x=pd.to_datetime('2012-08-01'), y=12, s='Cutoff', color='black',
            fontsize=16, fontweight='bold', alpha=0.8)
    ax.axvline(x=pd.to_datetime(cutoff) + pd.Timedelta('365 days'), c='gray', lw=4,
               alpha=0.5, ls='--')
    ax.text(x=pd.to_datetime('2013-01-01'), y=6, s='Horizon', color='black',
            fontsize=16, fontweight='bold', alpha=0.8)
    fig.show()

    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
    print(df_cv.head())

    df_p = performance_metrics(df_cv)
    print(df_p.head())

    fig = plot_cross_validation_metric(df_cv, metric='mape')
    fig.show()


if __name__ == "__main__":
    main()
