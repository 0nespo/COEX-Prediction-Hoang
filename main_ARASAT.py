import h5py
from tensorflow import keras
import time
import pickle
import numpy as np
import warnings
import datetime as dt
import mysql.connector
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import connectSQL

#file = open("mmLSTM_ARASAT.h5",'rb')
model  = keras.models.load_model('mmLSTM_ARASAT.h5')
connection = mysql.connector.connect(host='113.198.211.95',
                                    database='ENS_PV',
                                    user='adminc',
                                    password='adminc')
sql_select_Query = "SELECT * FROM `pcs` ORDER BY `id` LIMIT 120"

if __name__ == '__main__':
    while(1):
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        # get all recordsx_pred,_,scalers = inputmodel(#records)
        records = cursor.fetchall()

        #x_pred,_ = connectSQL.inputmodel(records)

        result = connectSQL.train(model,records)
        time.sleep(1)


