# 参考
# https://qiita.com/Blaster36/items/ac6fdc8b14f1049dd390
# ライブラリのインポート
import pandas as pd
import numpy as np
from pandas_datareader import data
# import matplotlib.pyplot as plt

# warningを消す
import warnings
warnings.simplefilter('ignore')

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
# from fbprophet.plot import plot_yearly
import pandas_datareader.data as web

from datetime import *
import pytz
import uuid
import time
import itertools

class prophetWeekly:
    predict_datas = {}
    experiments = {}

    def __init__(self, df_prophet, today):
        self.today = today
        self.predict_period_weeks = 4
        self.model_datas = {}
        self.model_datas['price'] = {}
        self.model_datas['price']['model_type'] = 'price'
        self.model_datas['price']['model_data'] = df_prophet
        self.model_datas['price']['event'] = None
        self.model_datas['price']['params'] = {'growth': ['linear'],
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
        self.model_datas['price']['base_date'] = np.datetime64(today)
        

    # データを出力
    def get_predict_df(self):
        df = pd.concat([v for v in self.predict_datas.values()])
        df['last_y'] = df['y'].shift(1)
        df = df[df['ds']>np.datetime64(self.today - timedelta(+55))][['ds','yhat_lower','yhat_upper','yhat','y','base_date']].reset_index().drop('index',axis=1)
        df = df.rename({'yhat_lower':'predicted_Close_Lower','yhat_upper':'predicted_Close_Upper','yhat':'predicted_Close','y':'Close'}, axis=1)
        for t in range(4):
            df['ds'][(8+t)] = df['ds'][(8+t)] + timedelta(1)

        return df
    
    # データを出力
    def get_predict_df_for_marge(self):
        df = pd.concat([v for v in self.predict_datas.values()])
        df['last_y'] = df['y'].shift(1)
        df = df[df['ds']>np.datetime64(self.today)][['ds','yhat_lower','yhat_upper','yhat','last_y']].reset_index()
        df = df.rename({'yhat_lower':'predicted_Next_Close_Lower','yhat_upper':'predicted_Next_Close_Upper','yhat':'predicted_Next_Close','last_y':'Close'}, axis=1)
        df['ds'][0] = df['ds'][0] - timedelta(days=df['ds'][0].weekday())
        df = df[df['ds']==np.datetime64(self.today - timedelta(+4))][['ds','predicted_Next_Close_Lower','predicted_Next_Close_Upper','predicted_Next_Close','Close']]

        return df
    
    # モデル作成
    def modeling(self, is_best_params=True):
        # 自動でパラメータをチューニング
        def best_params(model_data, param_grid, horizon=4, cutoffs_num=12):
            time_sta = time.time()
            # Generate all combinations of parameters
            all_params = [dict(zip(param_grid.keys(), v))
                          for v in itertools.product(*param_grid.values())]
            print("Find best parameter")
            mapes = []
            try:
                # Use cross validation to evaluate all parameters
                for params in all_params:
                    m = Prophet(
                        **params).fit(model_data) # Fit model with given params
                    cuooffs_sta = - cutoffs_num - horizon
                    cuooffs_end = -horizon
                    ds_sort = model_data[cuooffs_sta:cuooffs_end]["ds"].sort_values()
                    cutoffs = ds_sort.to_list()
                    df_cv = cross_validation(
                        m, cutoffs=cutoffs, horizon='{} W'.format(horizon), parallel='processes')
                    df_p = performance_metrics(df_cv, rolling_window=1)
                    mapes.append(df_p['mape'].values[0])
                best_params = all_params[np.argmin(mapes)]
            except Exception as e:
                print ("Error")
                print(e)
                best_params = dict([(k, v[0]) for k, v in param_grid.items()])
            finally:
                time_end = time.time()
                print("Time taken for {}: {} seconds".format(
                    "best_param", time_end - time_sta)) 
                return best_params

        #対象ごとにモデルを作成
        for model, value in self.model_datas.items():
            model_data = value["model_data"]
            event = value["event"]
            params = value["params"]

            try:
                # paramsの最適化
                if is_best_params:
                    params = best_params(
                        model_data=model_data, param_grid=params)

                else:
                    params = dict([(k, v[0]) for k, v in params.items()])
                print ("params: ", params)
                m = Prophet(
                    **params, 
                )
                # モデル作成
                print(model, "Modeling...")
                m.fit(model_data)
                status = "Success"

            except Exception as e:
                print(model, "Error")
                print(e)
                status = "Error"
                pass 
            else:
                self.model_datas[model].update({"model": m})
            finally:
                # experimentを作成
                _l = ["model_type", "modeling_period_weeks",
                      "model_period", "input_data_rule", "base_date"]
                add_info = dict([(k, v) for k, v in value.items() if k in _l])
                jst = pytz.timezone('Asia/Tokyo') 
                now = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
                ex_id_now = now.replace(":", "").replace(" ","").replace("-","")
                ex_id_basedate = pd.to_datetime(
                    value["base_date"]).strftime("%Y%m%d")
                # eventがNoneのとき
                if event is None:
                    event = pd.DataFrame()
                self.experiments.update({
                    model: {
                        "experiments_id": "{}-{}-{}-{}".format(model, ex_id_basedate, ex_id_now, str(uuid.uuid4())[:8]),
                        "model_name": model,
                        "timestamp": now,
                        "status": status,
                        "params": params,
                        "event": event.to_dict (orient='index'),
                        **add_info
                    }
                })

    # 予測
    def predict(self):
        for model, value in self.model_datas.items():
            m = value["model"]
            future = m.make_future_dataframe(periods=self.predict_period_weeks, freq='w')
            print(model, "Predicting...")
            try:
                predict = m.predict(future)
                m.plot(predict)
            except Exception as e:
                print("Error")
                print(e)
                pass 
            else:
                model_data = value["model_data"]
                predict = pd.merge(
                    model_data[["y", "ds"]], predict, on="ds", how='outer')
                predict["model_name"] = model
                predict["base_date"] = value["base_date"]
                predict["experiments_id"] = self.experiments.get(
                    model).get("experiments_id")
                self.predict_datas.update({model: predict})
        return self.predict_datas