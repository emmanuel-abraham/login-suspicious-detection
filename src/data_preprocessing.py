import pandas as pd

def preprocess_data(path):
    data = pd.read_csv(path)
    data['login_hour'] = pd.to_datetime(data['login_time']).dt.hour
    data['login_day'] = pd.to_datetime(data['login_date']).dt.dayofweek
    X = data[['login_hour', 'login_day']]
    y = data['login_success']
    return data, X, y
