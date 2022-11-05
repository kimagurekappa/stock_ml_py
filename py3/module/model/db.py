import pandas as pd
from datetime import timedelta
import datetime
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

class DataBase:

    sql_hostname = os.environ['SQL_HOSTNAME']
    sql_username = os.environ['SQL_USERNAME']
    sql_password = os.environ['SQL_PASSWORD']
    sql_main_database = os.environ['SQL_MAIN_DATABASE']
    sql_port = int(os.environ['SQL_PORT'])
    ssh_Host = os.environ['SSH_HOST']
    ssh_user = os.environ['SSH_USER']
    ssh_pass = os.environ['SSH_PASS']
    ssh_port = int(os.environ['SSH_PORT'])
    jsonfilename = os.environ['JSONFILENAME']
    
    # コンストラクタ
    def __init__(self):
        pass

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
            ssh_host_key=None, # SSHホストキーがある場合は指定 (2018/10/24)
            ssh_username=self.ssh_user,
            ssh_password=self.ssh_pass, # もしくは鍵ファイルのパス、なければNone (2018/10/24)
            ssh_pkey=None, # 秘密鍵のフルパス (2018/10/24)
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
    def create_service(self):
        jsonFileName = self.jsonfilename
        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            jsonFileName, scope)
        return discovery.build('sheets', 'v4', credentials=credentials)

    # CSVデータをもとにシートを更新
    def upload_csv_to_spreadsheet(self, spreadsheet_id, spreadsheet_id_num,csv_body):
        service = self.create_service()
        SPREADSHEET_ID = spreadsheet_id

        # クリア The A1 notation of the values to clear.
        range_ = 'A2:AJ'  # TODO: Update placeholder value.
        body = {
        }
        requests = service.spreadsheets().values().clear(
            spreadsheetId=SPREADSHEET_ID, range=range_, body=body
        ).execute()

        # CSVデータでシートを更新:PasteDataRequest
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