import pandas as pd
import numpy as np
from datetime import timedelta
import datetime
import pandas_datareader.data as web
# MySQLへ接続
import pymysql
import pymysql.cursors
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
# CSV
import io as StringIO
import csv
# Google Spreadsheet
import os
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
import yfinance as yf

class DataConnection:
    
    # コンストラクタ
    def __init__(self):
        self.sql_hostname = os.environ['SQL_HOSTNAME']
        self.sql_username = os.environ['SQL_USERNAME']
        self.sql_password = os.environ['SQL_PASSWORD']
        self.sql_main_database = os.environ['SQL_MAIN_DATABASE']
        self.sql_port = int(os.environ['SQL_PORT'])
        self.ssh_Host = os.environ['SSH_HOST']
        self.ssh_user = os.environ['SSH_USER']
        self.ssh_pass = os.environ['SSH_PASS']
        self.ssh_port = int(os.environ['SSH_PORT'])
        self.jsonfilename = os.environ['JSONFILENAME']

    # MYSQLへ接続して DB を上書きする
    def to_sql(self, df, table_name):
        server = SSHTunnelForwarder(
                (self.ssh_Host, self.ssh_port),
                ssh_username=self.ssh_user,
                ssh_password=self.ssh_pass,
                remote_bind_address=(self.sql_hostname, self.sql_port))
        server.start()
        local_port = server.local_bind_port
        engine = create_engine(
            f'mysql+pymysql://{self.sql_username}:{self.sql_password}@127.0.0.1:{local_port}/{self.sql_main_database}'
        )
        # データを出力
        df.to_sql(table_name, con = engine, if_exists = 'append',index = False, chunksize = 1000)
        server.close()

    # MySQLへのクエリー結果をもとにCSVデータを作成 
    def sql_to_csv(self, table_name):
        # クエリー結果の出力先ファイルオブジェクト
        csvfile = StringIO.StringIO()
        writer = csv.writer(csvfile, dialect='excel')
        # MySQL に接続
        with SSHTunnelForwarder(
            (self.ssh_Host, self.ssh_port),
            ssh_host_key=None,
            ssh_username=self.ssh_user,
            ssh_password=self.ssh_pass,
            ssh_pkey=None, 
            remote_bind_address=(self.sql_hostname, self.sql_port),
        ) as server:
            conn = pymysql.connect(
                host = '127.0.0.1',
                port = server.local_bind_port,
                user = self.sql_username,
                password = self.sql_password,
                database = self.sql_main_database,
                charset = 'utf8',
            )
            cur = conn.cursor()
            # クエリを記述
            cur.execute("""  
              SELECT *
              FROM {}
            """.format(table_name))

            for row in cur:
                writer.writerow(row)

            cur.close()
            conn.close()
            csv_body = csvfile.getvalue()
            csvfile.close()

        return csv_body

    # スプレッドシート認証
    def get_spreadshee_config(self, target_list):
        table_list = [os.environ[f'{target}_TABLE'] for target in target_list]
        spreadsheet_id_list = [os.environ[f'{target}_SPREADSHEET_ID'] for target in target_list]
        spreadsheet_id_num_list = [int(os.environ[f'{target}_SPREADSHEET_ID_NUM']) for target in target_list]
        return table_list, spreadsheet_id_list, spreadsheet_id_num_list
    
    def create_service(self):
        jsonFileName = self.jsonfilename
        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            jsonFileName, scope)
        return discovery.build('sheets', 'v4', credentials=credentials)

class DataUpdate:
    
    # コンストラクタ
    def __init__(self):
        self.data_connection = DataConnection()
    
    # CSVデータをもとにシートを更新
    def upload_csv_to_spreadsheet(self, spreadsheet_id, spreadsheet_id_num,csv_body):
        service = self.data_connection.create_service()
        SPREADSHEET_ID = spreadsheet_id
        range_ = 'A2:AJ'
        body = {
        }
        requests = service.spreadsheets().values().clear(
            spreadsheetId=SPREADSHEET_ID, range=range_, body=body
        ).execute()
        requests = [
        {
            'pasteData': {
                'coordinate':{
                  'sheetId': spreadsheet_id_num, # スプレッドシートURLの末尾、gid= に続く数字
                  'rowIndex': 1,
                  'columnIndex': 0
                },
                'data':csv_body,
                'type':'PASTE_VALUES',
                'delimiter': ',',
            }
        }
        ]
        body = {
            'requests': requests
        }
        response = service.spreadsheets().batchUpdate(
          spreadsheetId=SPREADSHEET_ID,
          body=body).execute()

    
    def update_db(self, df, table_name, spreadsheet_id, spreadsheet_id_num):
        self.data_connection.to_sql(df, table_name)
        csv_body = self.data_connection.sql_to_csv(table_name)
        self.upload_csv_to_spreadsheet(spreadsheet_id, spreadsheet_id_num, csv_body)
    
        
class DataLoad:
    def __init__(self, today):
        self.data_connection = DataConnection()
        self.today = today
        self.end = self.today - datetime.timedelta(days=self.today.weekday() + 3)
        self.today_pred = self.today - datetime.timedelta(days=self.today.weekday() - 4)

    def get_date_range(self, years):
        return (pd.Period(self.today, 'D') - 365 * years).start_time
    
    def get_stock_data(self, target):
        df_stock_wk = yf.download(target, end=self.today, start= self.today - timedelta(days=self.today.weekday()), interval = "1wk").drop('Adj Close', axis=1).reset_index()
        df_stock_wk = df_stock_wk[df_stock_wk['Date']<=np.datetime64(self.today - timedelta(days=self.today.weekday()))]
        return df_stock_wk

    # 2年,5年,10年金利とイールドカーブ
    def dgs(self, start, end):
        df_DGS2 = web.DataReader("DGS2", "fred", start, end)
        df_DGS5 = web.DataReader("DGS5", "fred", start, end)
        df_DGS10 = web.DataReader("DGS10", "fred", start, end)

        df_DGS = pd.concat([df_DGS2, df_DGS5, df_DGS10], axis=1)

        yield_curves = []
        for row in df_DGS.iterrows():
            if row[1].isnull().any():
                yield_curves.append(np.nan)
            else:
                x = [2, 5, 10]
                y = row[1].values
                yield_curves.append(np.polyfit(x, y, 1)[0])

        df_DGS["YC"] = yield_curves
        
        #VIXのデータを取得
        vix = web.DataReader('VIXCLS','fred', start, end)
        df_DGS = pd.concat([df_DGS, vix], axis=1)
        
        return df_DGS.resample('W-MON').mean()
    
    # 株と金利のデータを取得
    def get_df(self, target):
        df_stock_wk = yf.download(target, end=self.today, start=self.get_date_range(22), interval="1wk").drop('Adj Close', axis=1).reset_index()
        df_dgs_wk = self.dgs(self.get_date_range(22),self.today).reset_index()
        
        return df_stock_wk, df_dgs_wk
    
    def get_df_prophet(self, df_stock_wk, df_dgs_wk, target):

        if target == 'QQQ':
            df_prophet = df_stock_wk[df_stock_wk['Date']>=np.datetime64(self.get_date_range(6))]
        elif target == 'SPY':
            df_prophet = df_stock_wk[df_stock_wk['Date']>=np.datetime64(self.get_date_range(7))]
        else:
            df_prophet = df_stock_wk[df_stock_wk['Date']>=np.datetime64(self.get_date_range(8))]
        df_prophet['ds'] = df_prophet['Date']
        df_prophet = df_prophet.rename({'Close':'y'}, axis=1)
        df_prophet = df_prophet[['ds', 'y']]
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet = df_prophet[df_prophet['ds']<=np.datetime64(self.today - timedelta(+4))]

        return df_prophet