import datetime
import pandas as pd
import numpy as np
from pandas_datareader import data
from datetime import timedelta

# import matplotlib.pyplot as plt
# import seaborn as sns
# 移動平均
import bottleneck as bn
# 時系列解析
import statsmodels
import statsmodels.api as sm

# https://pycaret.org/regression/
from pycaret.regression import *

class pycaretWeekly:

    # コンストラクタ
    def __init__(self, today):
        # 予測する指標
        self.index_list = ['High', 'Low']
        self.today = today
        self.end = self.today - datetime.timedelta(days=self.today.weekday() + 3)
        self.today_pred = self.today - datetime.timedelta(days=self.today.weekday() - 4)
        pass
    
    #メイン処理
    def pycaret_main(self, df_stock_wk, df_dgs_wk):
        # Pycaretモデル作成    
        for index in self.index_list:
            # 拡張
            df_pycaret_pre = self.feature(df_stock_wk, df_dgs_wk, index)
            df_pycaret = df_pycaret_pre[df_pycaret_pre['datetime']<=np.datetime64(self.end)]
            
            if index == 'High':
                drop_list = ['Open', 'Close', 'DGS2', 'DGS5', 'Low','move_mean_{}_w25'.format(index)]
            elif index == 'Low':
                drop_list = ['Open', 'Close', 'DGS2', 'DGS5', 'High','move_mean_{}_w25'.format(index)]
            df_pycaret_pre = df_pycaret_pre.drop(drop_list,axis=1)
            df_pycaret = df_pycaret.drop(drop_list,axis=1)
            
            # 環境セットアップ
            regression = self.regression(df_pycaret, index)

            # モデル作成
            df_pred, voting_regressor, df_fi = self.train(df_pycaret, index)

            # 予測
            predict_df_pycaret_ = predict_model(
                voting_regressor,
                data=df_pycaret_pre.drop('Next_{}'.format(index),axis=1)
            )
            predict_df_pycaret = pd.merge(predict_df_pycaret_,df_pycaret_pre[['datetime','Next_{}'.format(index)]],on = 'datetime')

            # 予測結果
            predict_df_pycaret.rename(columns={'prediction_label':'predicted_Next_{}'.format(index)},inplace=True)
            predict_df_pycaret = predict_df_pycaret.reset_index()
            predict_df_pycaret['datetime'] = pd.to_datetime(predict_df_pycaret['datetime'])
            if index == 'High':
                df_High = predict_df_pycaret.drop('index', axis=1)
            elif index == 'Low':
                df_Low = predict_df_pycaret.drop('index', axis=1)

        result_df = pd.merge(df_High[['datetime','High','Next_High', 'predicted_Next_High']], df_Low[['datetime','Low','Next_Low', 'predicted_Next_Low']], on='datetime')
        
        return result_df
        
    
    #環境セットアップ
    def regression(self, df_train, index):
        regression = setup(
            data = df_train, 
            target = 'Next_{}'.format(index), 
            session_id = 0,
            # normalize=True,
            # normalize_method='zscore',
            # feature_selection=True,
            # transformation=True,
            # transformation_method='yeo-johnson',
            # pca=True,
            # pca_method='kernel',
            remove_multicollinearity=True,
            # profile=True,
        )

        # import RegressionExperiment and init the class
        exp = RegressionExperiment()

        # init setup on exp
        exp.setup(
            data = df_train, 
            target = 'Next_{}'.format(index), 
            session_id = 0,
            # normalize=True,
            # normalize_method='zscore',
            # feature_selection=True,
            # transformation=True,
            # transformation_method='yeo-johnson',
            # pca=True,
            # pca_method='kernel',
            remove_multicollinearity=True,
            # profile=True,
        )
        return regression

    # 学習と予測
    def train(self, df_train, index):
        # Lasso Regression
        tuned_lasso = tune_model(create_model('lasso',cross_validation=True),choose_better = True,optimize='MAPE')
        # Elastic Net
        tuned_en = tune_model(create_model('en',cross_validation=True),choose_better = True,optimize='MAPE')
        # Ridge Regression
        tuned_ridge = tune_model(create_model('ridge',cross_validation=True),choose_better = True,optimize='MAPE')
        # Bayesian Ridge
        tuned_br = tune_model(create_model('br',cross_validation=True),choose_better = True,optimize='MAPE')
        # Linear Regression
        tuned_lr = tune_model(create_model('lr',cross_validation=True),choose_better = True,optimize='MAPE')
        # Extra Trees Regressor
        tuned_et = tune_model(create_model('et',cross_validation=True),choose_better = True,optimize='MAPE')


        voting_regressor = blend_models(
            estimator_list=[
                tuned_lasso,
                tuned_en,
                tuned_ridge,
                tuned_br,
                tuned_lr,
                tuned_et
            ],
            choose_better = True,
            optimize='MAPE'
        )

        df_pred = predict_model(
            voting_regressor,
            data=df_train
        )

        # 特徴量重要度
        df_fi = pd.DataFrame(
            {
                'explanatory_varibles':get_config('X_train_transformed').columns.values.tolist(),
                # 'lasso_feature_importances':tuned_lasso.feature_importances_,
                # 'en_feature_importances':tuned_en.feature_importances_,
                # 'ridge_feature_importances':tuned_ridge.feature_importances_,
                # 'br_feature_importances':tuned_br.feature_importances_,
                # 'lr_feature_importances':tuned_lr.feature_importances_,
                'et_feature_importances':tuned_et.feature_importances_
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
        if index == 'High':
            df = df.assign(Open_Close = df['Open'] - df['Close'])
            df = df.assign(High_Open = df['High'] - df['Open'])
            df = df.assign(High_Close = df['High'] - df['Close'])
        elif index == 'Low':
            df = df.assign(Open_Close = df['Open'] - df['Close'])
            df = df.assign(Open_Low = df['Open'] - df['Low'])
            df = df.assign(Close_Low = df['Close'] - df['Low'])

        # 移動平均
        move_mean_w5 = bn.move_mean(df[index], window=5)
        df['move_mean_{}_w5'.format(index)] = move_mean_w5
        move_mean_w25 = bn.move_mean(df[index], window=25)
        df['move_mean_{}_w25'.format(index)] = move_mean_w25
        # move_mean_w50 = bn.move_mean(df[index], window=50)
        # df['move_mean_{}_w50'.format(index)] = move_mean_w50
        # 差の特徴量
        df = df.assign(w25_w5 = df['move_mean_{}_w25'.format(index)] - df['move_mean_{}_w5'.format(index)])

        # 時系列成分
        res = sm.tsa.seasonal_decompose(df[index],period=52)
        df_sm = pd.concat([
            df.set_index('datetime').index.to_series().reset_index(drop=True),
            pd.DataFrame({'{}_trend'.format(index):res.trend}).reset_index(drop=True).shift(26),
            pd.DataFrame({'{}_seasonal'.format(index):res.seasonal}).reset_index(drop=True),
            pd.DataFrame({'{}_resid'.format(index):res.resid}).reset_index(drop=True).shift(26),
        ],axis=1).fillna(0)
        df_sm = df_sm.rename(columns={0: 'datetime'})
        
        # 支持線・抵抗線
        line_turms = [50]
        lower_num = 0.001
        upper_num = 1 - lower_num

        if index == 'High':
            for line_turm in line_turms:
                # Resistance lines
                df['r' + str(line_turm)] = df['High'].rolling(line_turm).quantile(upper_num)    
                df['r' + str(line_turm)] = df['r' + str(line_turm)].shift(1)
                df = df.assign(r50_High = df['r50'] - df['High'])
        elif index == 'Low':
            for line_turm in line_turms:
                # Support lines
                df['s' + str(line_turm)] = df['Low'].rolling(line_turm).quantile(lower_num)
                df['s' + str(line_turm)] = df['s' + str(line_turm)].shift(1)
                df = df.assign(Low_s50 = df['Low'] - df['s50'])

        df = df.merge(df_sm)
        df = df[120:]

        # 目的変数用を作成
        df['Next_{}'.format(index)] = df[index].shift(-1)
        
        return df