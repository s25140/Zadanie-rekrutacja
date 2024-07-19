
# The file is for future use, based on new test_data
# The model results are in training_models.ipynb

import os
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *

def prepare_data(dates_path, Posting_Volumes_path, weather_path):
    keep_cols = '''dateId
dateWeekOfMonth
dateQuarter
dateIsWeekend
dateIsHolidayInd
dateWeekDayStartsMonday'''.split('\n')
    dates_df = pd.read_csv(dates_path, sep=';', usecols=keep_cols)
    # create timestamp
    dates_df['dateId'] = pd.to_datetime(dates_df['dateId'], format='%Y-%m-%d')
    dates_df.rename(columns={'dateId': 'DateId'}, inplace=True)
    dates_df['timestamp'] = dates_df['DateId'].astype('int64') / 10**9

    posting_volumes = pd.read_parquet(Posting_Volumes_path, engine='fastparquet')
    # replace negative Volume with 0
    posting_volumes['Volume'] = posting_volumes['Volume'].clip(lower=0)
    posting_volumes['timestamp'] = pd.to_datetime(posting_volumes['postingDateFk'], format='%Y%m%d').astype('int64') / 10**9
    posting_volumes['postingDateFk'] = pd.to_datetime(posting_volumes['postingDateFk'], format='%Y%m%d')
    posting_volumes.rename(columns={'postingDateFk': 'DateId'}, inplace=True)

    def get_customer_avg_volume(row, df, months_before=3, months_exclude=1):
        """
        Filters the DataFrame to include data from `months_before` months before
        the date in the row, excluding `months_exclude` months before that date.
        """
        date = row['DateId']
        start_date = date - pd.DateOffset(months=months_before)
        end_date = date - pd.DateOffset(months=months_exclude)
        avg_volume = df[(df['DateId'] > start_date) & (df['DateId'] <= end_date) & (df['Customer'] == row['Customer'])]['Volume'].mean() #[f'last{months_before}months']
        return avg_volume if not pd.isnull(avg_volume) else 0


    # add avg_volume for each row and concatenate results
    for _, row in posting_volumes.iterrows():
        posting_volumes.loc[_,'VolumeAvgLast3months'] = get_customer_avg_volume(row, posting_volumes, months_before=3)
        posting_volumes.loc[_,'VolumeAvgLast6months'] = get_customer_avg_volume(row, posting_volumes, months_before=6)
        posting_volumes.loc[_,'VolumeAvgLast9months'] = get_customer_avg_volume(row, posting_volumes, months_before=9)
        posting_volumes.loc[_,'VolumeAvgLast12months'] = get_customer_avg_volume(row, posting_volumes, months_before=12)

    weather_df = pd.DataFrame()
    for file in os.listdir(weather_path):
        temp_df = pd.read_csv(f'{weather_path}/{file}',sep=',', header=0)
        # merge to dateId
        temp_df['year'] = temp_df['Rok'].astype('int64')
        temp_df['month'] = temp_df['Miesiac'].astype('int64')
        temp_df['day'] = temp_df['Dzien'].astype('int64')
        temp_df['DateId'] = pd.to_datetime(temp_df[["year","month","day"]])
        temp_df.drop(columns=['Rok','Miesiac','Dzien'], inplace=True)
        temp_df.drop(columns=['year','month','day'], inplace=True)
        temp_df.drop('Nazwa stacji', axis=1, inplace=True)
        
        weather_df = pd.concat([weather_df, temp_df], axis=0)

    dates_df['DateId'] = pd.to_datetime(dates_df['DateId'], format='%Y%m%d')
    posting_volumes['DateId'] = pd.to_datetime(posting_volumes['DateId'], format='%Y%m%d')
    weather_df['DateId'] = pd.to_datetime(weather_df['DateId'], format='%Y-%m-%d')
    dataset = dates_df.merge(posting_volumes, on='DateId', how='right')
    dataset = dataset.merge(weather_df, on='DateId', how='left')

    # sort by date
    dataset.sort_values('DateId', inplace=True)
    dataset.to_csv('dataset.csv', index=False, sep=';')
    # get random sample
    return dataset


if __name__ == '__main__':
    '''
    Todo: test on new data...
    '''
    dates_path = 'test_data/dates.csv'
    Posting_Volumes_path = 'test_data/Posting_Volumes.parquet'
    weather_path = 'test_data/weather.csv'
    test_dataset = prepare_data(dates_path, Posting_Volumes_path, weather_path)
    model = load_model('models/LGBM_20kRMSE.pkl')
    predictions = predict_model(model, data=test_dataset)
    
    print(predictions)
    