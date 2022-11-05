import datetime
import pandas as pd
import numpy as np
from pandas_datareader import data
from datetime import timedelta
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import seaborn as sns
# 移動平均
import bottleneck as bn
# 時系列解析
import statsmodels
import statsmodels.api as sm

# https://pycaret.org/regression/
from pycaret.regression import *

class StockWeekly:

    # コンストラクタ
    def __init__(self):

        pass
    
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

        return df_DGS.resample('W-MON').mean()
    
    #環境セットアップ
    def regression(self, df_train, index):
        regression = setup(
            data=df_train,
            target=index,
            session_id=0,
            normalize=True,
            normalize_method='zscore',
        #     feature_selection=True,
        #     transformation=True,
        #     transformation_method='yeo-johnson',
        #     pca=Ture,
        #     pca_method='kernel',
        #     create_clusters=True,
        #     profile=True,
            silent=True
        )
        return regression

    # 学習と予測
    def train(self, df_train, index):
        # Light Gradient Boostimg Machine
        tuned_lightgbm = tune_model(create_model('lightgbm',cross_validation=True),optimize='MAPE')
        # Gradient Boosting Regressor
        tuned_gbr = tune_model(create_model('gbr',cross_validation=True),optimize='MAPE')
        # Random Forest Regressor
        tuned_rf = tune_model(create_model('rf',cross_validation=True),optimize='MAPE')
        # Extra Trees Regressor
        tuned_et = tune_model(create_model('et',cross_validation=True),optimize='MAPE')
        # Decision Tree Regressor
        tuned_dt = tune_model(create_model('dt',cross_validation=True),optimize='MAPE')

        voting_regressor = blend_models(
            estimator_list=[
                tuned_lightgbm,
                tuned_gbr,
                tuned_rf,
                tuned_et,
                tuned_dt
            ],
            optimize='MAPE'
        )

        df_pred = predict_model(
            voting_regressor,
            data=df_train
        )
        df_pred.rename(columns={'Label':'predicted_Next_{}'.format(index)},inplace=True)

        # 特徴量重要度
        df_fi = pd.DataFrame(
            {
                'explanatory_varibles':get_config('X').columns.values.tolist(),
                'lightgbm_feature_importances':tuned_lightgbm.feature_importances_,
                'gbr_feature_importances':tuned_gbr.feature_importances_,
                'rf_feature_importances':tuned_rf.feature_importances_,
                'et_feature_importances':tuned_et.feature_importances_,
                'dt_feature_importances':tuned_dt.feature_importances_
            }
        )

        return df_pred,voting_regressor, df_fi

    # 学習用データの拡張
    def feature(self, df, df_dgs, index):
        # 株価
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'datetime'})
        # 金利
        df_dgs['DATE'] = pd.to_datetime(df_dgs['DATE'])
        df_dgs = df_dgs.rename(columns={'DATE': 'datetime'})
        # 週別株価+金利データ
        df = df.merge(df_dgs)
        
        # ローソク足を指標化
        df = df.assign(Open_Close = df['Open'] - df['Close'])
        df = df.assign(High_Low = df['High'] - df['Low'])
        df = df.assign(Open_High = df['Open'] - df['High'])
        df = df.assign(Open_Low = df['Open'] - df['Low'])
        df = df.assign(Close_High = df['Close'] - df['High'])
        df = df.assign(Close_Low = df['Close'] - df['Low'])
        # df

        # 移動平均
        move_mean_w13 = bn.move_mean(df[index], window=13)
        df['move_mean_{}_w13'.format(index)] = move_mean_w13
        move_mean_w26 = bn.move_mean(df[index], window=26)
        df['move_mean_{}_w26'.format(index)] = move_mean_w26
        move_mean_w52 = bn.move_mean(df[index], window=52)
        df['move_mean_{}_w52'.format(index)] = move_mean_w52
        # df

        # 時系列成分
        res = sm.tsa.seasonal_decompose(df[index],period=52)
        df_sm = pd.concat([
            df.set_index('datetime').index.to_series().reset_index(drop=True),
            pd.DataFrame({'{}_trend'.format(index):res.trend}).reset_index(drop=True).shift(26),
            pd.DataFrame({'{}_seasonal'.format(index):res.seasonal}).reset_index(drop=True),
            pd.DataFrame({'{}_resid'.format(index):res.resid}).reset_index(drop=True).shift(26),
        ],axis=1).fillna(0)
        df_sm = df_sm.rename(columns={0: 'datetime'})

        df = df.merge(df_sm)

        # 目的変数用を作成
        df['Next_{}'.format(index)] = df[index].shift(-1)
        
        return df