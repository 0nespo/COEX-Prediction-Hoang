from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

df=pd.read_csv(r'pcs_ARASAT.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')
daily_df = df.copy()
settrain_df,settest_df = daily_df[:100000], daily_df[100000:120000] 


def fit_traindata(train_df=settrain_df):
    train = train_df
    scalers={}
    for i in train_df.columns:
        scaler = MinMaxScaler()
        s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+ i] = scaler
        train[i]=s_s
    return scalers,train
def trans_data(test_df,scalers):
    test = test_df
    for i in test_df.columns:
        scaler = scalers['scaler_'+i]
        s_s = scaler.transform(test[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+i] = scaler
        test[i]=s_s
    return test
