import mysql.connector
import pandas as pd
import time
from preprocessing_ARASAT import fit_traindata,trans_data
import numpy as np


connection = mysql.connector.connect(host='113.198.211.95',
                                    database='ENS_PV',
                                    user='adminc',
                                    password='adminc')

sql_select_Query = "SELECT * FROM `pcs` ORDER BY `id` LIMIT 120"
scalers,_ = fit_traindata()

def inputmodel(records):
    acAR = [float(x[2]) for x in records]
    acAS = [float(x[3]) for x in records]
    acAT = [float(x[4]) for x in records]
    acPowerFactor = [float(x[6]) for x in records]
    actKw = [float(x[10]) for x in records]
    df = pd.DataFrame(list(zip(acAR, acAS,acAT,acPowerFactor,actKw)),columns =['acAR', 'acAS','acAT','acPowerFactor','actKw'])
    x_pred = trans_data(df,scalers)
    return x_pred,df


def split_series(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

def getdata(test=None,n_past=120, n_future=0,n_features=5):
	X_test, y_test = split_series(test.values,n_past, n_future)
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
	y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
	return X_test

def get_result(y_test,records):
	_,train_df = inputmodel(records)
	#scalers,_ = fit_traindata()
	for index,i in enumerate(train_df.columns):
	    scaler = scalers['scaler_'+i]
	    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])
	return y_test

def train(model,records):
    x_pred,_ = inputmodel(records)
    X_test = getdata(x_pred)
    y_pred = model.predict(X_test)
    y_result = get_result(y_pred,records)

    print (y_result)
    return y_result

if __name__ == '__main__':
	#while(1):
	cursor = connection.cursor()
	cursor.execute(sql_select_Query)
	# get all records
	records = cursor.fetchall()
	x_pred = inputmodel(records)
	#time.sleep(1)