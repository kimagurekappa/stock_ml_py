from model import DataBase
from model import pycaretWeekly
from model import prophetWeekly

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import datetime
from pandas_datareader import data
import yfinance as yf

# https://pycaret.org/regression/
from pycaret.regression import *

class Main:
    def __init__(self):
        self.target_list = ['DIA', 'SPY', 'QQQ']
        self.target_list_pro = ['DIA_PRO', 'SPY_PRO', 'QQQ_PRO']
        self.target_list_etf = ['SPXL', 'SPXS', 'TQQQ', 'SQQQ']
        self.today = datetime.date.today() - datetime.timedelta(days=datetime.date.today().weekday() - 4)
        self.data_connection = DataConnection()
        self.data_update = DataUpdate()
        self.data_load = DataLoad(self.today)
        
    def main(self):
        # # レバレッジETFのデータを保存
        # self.save_stock_data(self.target_list_etf)
        
        # 三大指数ETFの週足予測
        self.prediction(self.target_list, self.target_list_pro)            
        
        
    # レバレッジETFのデータを保存  
    def save_stock_data(self,target_list):
        table_list, spreadsheet_id_list, spreadsheet_id_num_list = self.data_connection.get_spreadshee_config(target_list)
        for j in range(len(target_list)):
            # 22年前 ~ 今日までの週足を取得
            df_stock_wk = self.data_load.get_stock_data(target_list[j])
            # 結果を SQLに格納
            self.data_update.update_db(df_stock_wk, table_list[j], spreadsheet_id_list[j], spreadsheet_id_num_list[j])
    # 予測
    def prediction(self, target_list, target_list_pro):
        for j in range(len(target_list)):
            # 中期の予測
            df_stock_wk, df_dgs_wk = self.data_load.get_df(target_list[j])
            df = self.data_load.get_df_prophet(df_stock_wk, df_dgs_wk, target_list[j])
            # モデル作成
            model = prophetWeekly(df, self.today)
            model.modeling()
            # 予測結果
            model.predict()
            predict_df_prophet_4w = model.get_predict_df()
            # 結果を SQLに格納
            table_list_pro, spreadsheet_id_list_pro, spreadsheet_id_num_list_pro = self.data_connection.get_spreadshee_config(target_list_pro)
            self.data_update.update_db(predict_df_prophet_4w, table_list_pro[j], spreadsheet_id_list_pro[j], spreadsheet_id_num_list_pro[j])           
            mediate_df = model.get_predict_df_for_marge()
            
            #短期の予測
            pycaret_weekly = pycaretWeekly(self.today)
            short_df = pycaret_weekly.pycaret_main(df_stock_wk, df_dgs_wk)
            short_df = short_df[short_df['datetime']==pd.Timestamp(self.today - timedelta(days=self.today.weekday()))]
            result_df = pd.merge(short_df, mediate_df[['ds','predicted_Next_Close_Lower','predicted_Next_Close_Upper','predicted_Next_Close','Close']], left_on = 'datetime', right_on = 'ds').drop('ds', axis=1)
            # 結果を SQLに格納
            table_list, spreadsheet_id_list, spreadsheet_id_num_list = self.data_connection.get_spreadshee_config(target_list)
            self.data_update.update_db(result_df, table_list[j], spreadsheet_id_list[j], spreadsheet_id_num_list[j])           

if __name__ == "__main__":
    main = Main()
    df = main.main()