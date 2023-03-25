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

    def __init__(self, model_datas):
        self.model_datas = model_datas

    # データを出力
    def get_predict_df(self):
        df = pd.concat([v for v in self.predict_datas.values()])
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

    def predict(self, periods=53):
        for model, value in self.model_datas.items():
            m = value["model"]
            future = m.make_future_dataframe(periods=periods, freq='w')
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