from model import DataBase
from model import pycaretWeekly
from model import prophetWeekly

import os
import pandas as pd
from datetime import timedelta
import datetime
from pandas_datareader import data
# https://pycaret.org/regression/
from pycaret.regression import *

def main():
    # データ 取得期間の設定
    # 今日の日付
    today = datetime.date.today() - timedelta(days=datetime.date.today().weekday()-4)
    # today = datetime.date.today() - timedelta(days=datetime.date.today().weekday()+3)
    # 開始日 (365日前とする)
    start_8y = (pd.Period(today, 'D') - 365*8).start_time
    print('8y',start_8y)
    start_7y = (pd.Period(today, 'D') - 365*7).start_time
    print('7y',start_7y)
    start_6y = (pd.Period(today, 'D') - 365*6).start_time
    print('6y',start_6y)
    # 対象期間
    start_22y = (pd.Period(today, 'D') - 365*22).start_time
    print('22y',start_22y)

    # 学習用の最終日
    end = today - timedelta(days=today.weekday()+3)
    print('end',end)
    # 予測用の最終日
    today = today - timedelta(days=today.weekday()-4)
    print('today',today)

    # 対象銘柄
    target_list = ['DIA', 'SPY', 'QQQ']
    # target_list = ['DIA']
    table_list = [os.environ['DIA_TABLE'],os.environ['SPY_TABLE'],os.environ['QQQ_TABLE']]
    spreadsheet_id_list = [os.environ['DIA_SPREADSHEET_ID'],os.environ['SPY_SPREADSHEET_ID'],os.environ['QQQ_SPREADSHEET_ID']]
    spreadsheet_id_num_list = [int(os.environ['DIA_SPREADSHEET_ID_NUM']),int(os.environ['SPY_SPREADSHEET_ID_NUM']),int(os.environ['QQQ_SPREADSHEET_ID_NUM'])]

    # 予測する指標
    index_list = ['High', 'Low']
    # index_list = ['High']

    # インスタンス化
    pycaret_weekly = pycaretWeekly()
    db = DataBase()

    # 対象銘柄
    target_list_ = ['SPXL', 'SPXS', 'TQQQ', 'SQQQ']
    # target_list_ = ['SPXL']
    table_list_ = [os.environ['SPXL_TABLE'],os.environ['SPXS_TABLE'],os.environ['TQQQ_TABLE'],os.environ['SQQQ_TABLE']]
    spreadsheet_id_list_ = [os.environ['SPXL_SPREADSHEET_ID'],os.environ['SPXS_SPREADSHEET_ID'],os.environ['TQQQ_SPREADSHEET_ID'],os.environ['SQQQ_SPREADSHEET_ID']]
    spreadsheet_id_num_list_ = [int(os.environ['SPXL_SPREADSHEET_ID_NUM']),int(os.environ['SPXS_SPREADSHEET_ID_NUM']),int(os.environ['TQQQ_SPREADSHEET_ID_NUM']),int(os.environ['SQQQ_SPREADSHEET_ID_NUM'])]

    # レバレッジETFの週足
    for j in range(len(target_list_)):
        # 22年前 ~ 今日までの週足を取得
        df_stock_wk = data.get_data_yahoo(target_list_[j], end=today, start= today - timedelta(days=today.weekday()), interval='w').drop('Adj Close', axis=1).reset_index()
        df_stock_wk = df_stock_wk[df_stock_wk['Date']<=np.datetime64(today - timedelta(days=today.weekday()))]

        # 結果を SQLに格納
        db.to_sql(df_stock_wk, table_list_[j])
        # SQL から CSV を作成
        csv_body = db.sql_to_csv(table_list_[j])
        # CSV をスプレッドシートに書き出す
        db.upload_csv_to_spreadsheet(spreadsheet_id_list_[j], spreadsheet_id_num_list_[j], csv_body)

    # 三大指数ETFの週足予測
    for i in range(len(target_list)):
        # checkpoint
        print('checkpoint:',i,':',target_list[i])
        # 22年前 ~ 今日までの週足を取得
        df_stock_wk = data.get_data_yahoo(target_list[i], end=today, start=start_22y, interval='w').drop('Adj Close', axis=1).reset_index()
        df_dgs_wk = pycaret_weekly.dgs(start_22y,today).reset_index()

        # Prophetモデル作成
        # データを直近4年間にする
        if target_list[i] == 'QQQ':
            df_prophet = df_stock_wk[df_stock_wk['Date']>=np.datetime64(start_6y)]
        elif target_list[i] == 'SPY':
            df_prophet = df_stock_wk[df_stock_wk['Date']>=np.datetime64(start_7y)]
        else:
            df_prophet = df_stock_wk[df_stock_wk['Date']>=np.datetime64(start_8y)]
        df_prophet['ds'] = df_prophet['Date']
        df_prophet = df_prophet.rename({'Close':'y'}, axis=1)
        # 不要カラムの削除と並べ替え
        df_prophet = df_prophet[['ds', 'y']]
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        # パラメータ
        predict_period_weeks = 1
        model_datas = {}
        model_datas['price'] = {}
        model_datas['price']['model_type'] = 'price'
        model_datas['price']['model_data'] = df_prophet
        model_datas['price']['event'] = None
        model_datas['price']['params'] = {'growth': ['linear'],
                'changepoints': [None],
                'n_changepoints': [25],
                'changepoint_range': [0.8],
                'yearly_seasonality': [True],
                'weekly_seasonality': [False],
                'daily_seasonality': [False],
                'holidays': [None],
                'seasonality_mode': ['multiplicative','additive'],
                'seasonality_prior_scale': [10.0],
                'holidays_prior_scale': [10.0],
                'changepoint_prior_scale': [0.05],
                'mcmc_samples': [100],
                'interval_width': [0.80],
                'uncertainty_samples': [1000],
                'stan_backend': [None]
                }
        model_datas['price']['base_date'] = np.datetime64(today)
        # モデル作成
        model = prophetWeekly(model_datas=model_datas)
        model.modeling()
        model.predict(periods=predict_period_weeks)
        # 予測結果
        predict_df_prophet = model.get_predict_df()
        predict_df_prophet['last_y'] = model.get_predict_df()['y'].shift(1)
        predict_df_prophet = predict_df_prophet[predict_df_prophet['ds']>np.datetime64(today)][['ds','yhat_lower','yhat_upper','yhat','last_y']].reset_index()
        predict_df_prophet = predict_df_prophet.rename({'yhat_lower':'predicted_Next_Close_Lower','yhat_upper':'predicted_Next_Close_Upper','yhat':'predicted_Next_Close','last_y':'Close'}, axis=1)
        predict_df_prophet['ds'][0] = predict_df_prophet['ds'][0] - timedelta(days=predict_df_prophet['ds'][0].weekday())

        # Pycaretモデル作成    
        for index in index_list:
            # checkpoint
            print('checkpoint:',index)
            # 拡張
            df_pycaret_pre = pycaret_weekly.feature(df_stock_wk, df_dgs_wk, index)
            df_pycaret = df_pycaret_pre[df_pycaret_pre['datetime']<=np.datetime64(end)]
            
            if index == 'High':
                drop_list = ['Open', 'Close', 'DGS2', 'DGS5', 'Low','move_mean_{}_w52'.format(index)]
            elif index == 'Low':
                drop_list = ['Open', 'Close', 'DGS2', 'DGS5', 'High','move_mean_{}_w52'.format(index)]
            df_pycaret_pre = df_pycaret_pre.drop(drop_list,axis=1)
            df_pycaret = df_pycaret.drop(drop_list,axis=1)
            

            # 環境セットアップ
            regression = pycaret_weekly.regression(df_pycaret, index)

            # モデル作成
            df_pred, voting_regressor, df_fi = pycaret_weekly.train(df_pycaret, index)

            # 予測
            predict_df_pycaret = predict_model(
                voting_regressor,
                data=df_pycaret_pre
            )

            # 予測結果
            predict_df_pycaret.rename(columns={'Label':'predicted_Next_{}'.format(index)},inplace=True)
            predict_df_pycaret = predict_df_pycaret.reset_index()
            predict_df_pycaret['datetime'] = pd.to_datetime(predict_df_pycaret['datetime'])
            if index == 'High':
                df_High = predict_df_pycaret.drop('index', axis=1)
            elif index == 'Low':
                df_Low = predict_df_pycaret.drop('index', axis=1)

        result_df = pd.merge(df_High[['datetime','High','Next_High', 'predicted_Next_High']], df_Low[['datetime','Low','Next_Low', 'predicted_Next_Low']], on='datetime')
        result_df = result_df[result_df['datetime']==pd.Timestamp(today - timedelta(days=today.weekday()))]
        result_df = pd.merge(result_df, predict_df_prophet[['ds','predicted_Next_Close_Lower','predicted_Next_Close_Upper','predicted_Next_Close','Close']], left_on = 'datetime', right_on = 'ds').drop('ds', axis=1)
        # result_df.to_csv('result_{}_{}.csv'.format(target_list[i],today))

        # checkpoint
        print('checkpoint:',i,':',table_list[i])
        # 結果を SQLに格納
        db.to_sql(result_df, table_list[i])
        # SQL から CSV を作成
        csv_body = db.sql_to_csv(table_list[i])
        # CSV をスプレッドシートに書き出す
        db.upload_csv_to_spreadsheet(spreadsheet_id_list[i], spreadsheet_id_num_list[i], csv_body)

if __name__ == "__main__":
    main()