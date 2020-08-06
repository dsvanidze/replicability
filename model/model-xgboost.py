import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from utils import get_data

X_train, Y_train, X_validation, Y_validation, X_test, Y_test = get_data()

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, n_estimators=100, max_depth=3, subsample=1.0)

xg_reg.fit(X_train, Y_train, eval_metric="rmse", eval_set=[(X_validation, Y_validation)], verbose=True)

print(xg_reg.evals_result())

preds = xg_reg.predict(X_test)

mse = mean_squared_error(Y_test, preds)
rmse = np.sqrt(mse)
print("MSE: %f" % (mse))
print("RMSE: %f" % (rmse))
