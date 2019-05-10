# encoding: utf-8
"""
@author: lee
@time: 2019/5/10 14:55
@file: main.py.py
@desc: 
"""
import pandas as pd


def main():
    df = pd.read_csv('./data/example_wp_log_peyton_manning.csv')
    print(df.head())


if __name__ == "__main__":
    main()
