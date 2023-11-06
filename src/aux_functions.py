import pandas as pd
import numpy as np

def compute_perc_harvest_post_march (df):
    # compute the average of the percentage harvest after march
    df1 = df.copy()
    filtered_df = df[(df['Month'] >= 3) & (df['Month'] <= 5)]
    filtered_df
    aggregated_df = filtered_df.groupby('YEAR')['PRODUCTION'].sum().reset_index()
    aggregated_df.index = pd.to_datetime(aggregated_df.YEAR, format='%Y')
    aggregated_df.index.freq = pd.tseries.offsets.YearBegin()
    aggregated_df = aggregated_df.resample('MS').ffill()
    max_index = aggregated_df.index.max()
    next_month = max_index + pd.DateOffset(months=1)
    first_day_of_next_month = pd.Timestamp(year=next_month.year, month=next_month.month, day=1)
    date_string = first_day_of_next_month.strftime('%Y-%m-%d')
    extended_date_index = pd.date_range(start= date_string, end='2023-12-01', freq='MS')
    extended_df = pd.DataFrame(index=extended_date_index)
    aggregated_df_fin = pd.concat([aggregated_df, extended_df], axis=0)
    aggregated_df_fin.fillna(method='ffill',inplace = True)
    aggregated_df_fin
    aggregated_df_fin.rename(columns={'PRODUCTION': 'PRODUCTION_POST_MARCH'}, inplace=True)
    aggregated_df_fin.drop(columns = ['YEAR'],inplace=True)
    aggregated_df_fin = pd.merge(df1, aggregated_df_fin, right_index=True,left_index=True, how= 'left')
    aggregated_df_fin['PRODUCTION_POST_MARCH'] = aggregated_df_fin['PRODUCTION_POST_MARCH'].shift(2)
    aggregated_df_fin['PERCENTAGE_HARVEST_POST_MARCH'] = aggregated_df_fin['PRODUCTION_POST_MARCH'] / aggregated_df_fin['PRODUCTION_HARVEST'] *100
    average = aggregated_df_fin.PERCENTAGE_HARVEST_POST_MARCH.mean()

    return(average,aggregated_df_fin['PERCENTAGE_HARVEST_POST_MARCH'] )
