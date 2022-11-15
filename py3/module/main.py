from model import DataBase
from model import StockWeekly

import os
import pandas as pd
from datetime import timedelta
import datetime
from pandas_datareader import data
# https://pycaret.org/regression/
from pycaret.regression import *

def main():
    # データ 取得期間の設定
    # 終了日 (仮に今日とする)
    end = datetime.datetime.today()
    # 開始日 (365日前とする)
    start = (pd.Period(end, 'D') - 365*20).start_time
    # 対象期間
    start_year = 2000
    start_month = 1
    start_day = 1
    start = datetime.date(start_year,start_month,start_day)

    # 今日の日付
    today = datetime.date.today()

    # 学習用の最終日
    end = today - timedelta(days=today.weekday()+3)
    # end = today - timedelta(days=today.weekday()+10)
    print(end)
    # 予測用の最終日
    today = today - timedelta(days=today.weekday()-4)
    # today = today - timedelta(days=today.weekday()+3)
    print(today)

    # 対象銘柄
    target_list = ['DIA', 'SPY', 'QQQ']
    table_list = [os.environ['DIA_TABLE'],os.environ['SPY_TABLE'],os.environ['QQQ_TABLE']]
    spreadsheet_id_list = [os.environ['DIA_SPREADSHEET_ID'],os.environ['SPY_SPREADSHEET_ID'],os.environ['QQQ_SPREADSHEET_ID']]
    spreadsheet_id_num_list = [int(os.environ['DIA_SPREADSHEET_ID_NUM']),int(os.environ['SPY_SPREADSHEET_ID_NUM']),int(os.environ['QQQ_SPREADSHEET_ID_NUM'])]

    # 予測する指標
    # index = 'High'
    # index = 'Low'
    index_list = ['High', 'Low']

    # インスタンス化
    stock_weekly = StockWeekly()
    db = DataBase()

    for i in range(len(target_list)):
        # モデル作成用
        df = data.get_data_yahoo(target_list[i], end=end, start=start, interval='w').drop('Adj Close', axis=1).reset_index()
        df_dgs = stock_weekly.dgs(start,end).reset_index()

        # 予測用
        df_pre = data.get_data_yahoo(target_list[i], end=today, start=start, interval='w').drop('Adj Close', axis=1).reset_index()
        df_dgs_pre = stock_weekly.dgs(start,today).reset_index()

        for index in index_list:
            # 拡張
            df_ = stock_weekly.feature(df, df_dgs, index)
            df_pre_ = stock_weekly.feature(df_pre, df_dgs_pre, index)

            # 環境セットアップ
            regression = stock_weekly.regression(df_, index)

            # モデル作成と予測
            df_pred, voting_regressor, df_fi = stock_weekly.train(df_, index)

            # 予測
            df_pre_pred = predict_model(
                voting_regressor,
                data=df_pre_
            )
            # df_pre_pred

            # 結果
            df_pre_pred.rename(columns={'Label':'predicted_Next_{}'.format(index)},inplace=True)
            df_pre_pred = df_pre_pred.reset_index()
            df_pre_pred['datetime'] = pd.to_datetime(df_pre_pred['datetime'])
            if index == 'High':
                df_High = df_pre_pred.drop('index', axis=1)
            elif index == 'Low':
                df_Low = df_pre_pred.drop('index', axis=1)

        # result_df = pd.read_csv('pre_{0}_{1}-{2}-{3}_{4}.csv'.format(target_list[i],start_year,start_month,start_day,today))
        result_df = pd.merge(df_High[['datetime','High','Low','Open','Close','Volume','Next_High', 'predicted_Next_High']], df_Low[['datetime','Next_Low', 'predicted_Next_Low']], on='datetime')
        result_df = result_df[result_df['datetime']==pd.Timestamp(today - timedelta(days=today.weekday()))]
        result_df.to_csv('pre_{0}_{1}-{2}-{3}_{4}.csv'.format(target_list[i],start_year,start_month,start_day,today))


        # 結果を SQLに格納
        db.to_sql(result_df, table_list[i])
        # SQL から CSV を作成
        csv_body = db.sql_to_csv(table_list[i])
        # CSV をスプレッドシートに書き出す
        db.upload_csv_to_spreadsheet(spreadsheet_id_list[i], spreadsheet_id_num_list[i], csv_body)

if __name__ == "__main__":
    main()