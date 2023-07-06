#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def predict(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def run():
    year = int(sys.argv[1]) # 2022
    month = int(sys.argv[2]) # 3
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'yellow_pred_duration_{year:04d}-{month:02d}.parquet'
    
    print ('reading file: ', input_file)
    df = read_data(input_file)
    print ('predicting...')
    y_pred = predict(df)

    print(f'Mean predicted duration for {year:04d}-{month:02d} = {y_pred.mean()}')


if __name__ == '__main__':
    run()