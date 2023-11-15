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
    aggregated_df_fin.rename(columns={'PRODUCTION': 'PRODUCTION_POST_MARCH'}, inplace=True)
    aggregated_df_fin.drop(columns = ['YEAR'],inplace=True)
    aggregated_df_fin = pd.merge(df1, aggregated_df_fin, right_index=True,left_index=True, how= 'left')
    aggregated_df_fin['PRODUCTION_POST_MARCH'] = aggregated_df_fin['PRODUCTION_POST_MARCH'].shift(2)
    aggregated_df_fin['PERCENTAGE_HARVEST_POST_MARCH'] = aggregated_df_fin['PRODUCTION_POST_MARCH'] / aggregated_df_fin['PRODUCTION_HARVEST'] *100
    average = aggregated_df_fin.PERCENTAGE_HARVEST_POST_MARCH.mean()

    return(average,aggregated_df_fin['PERCENTAGE_HARVEST_POST_MARCH'] )

def shuffle_columns(df):
    columns = list(df.columns)
    np.random.shuffle(columns)
    shuffled_df = df[columns]
    return shuffled_df

def add_average_row(df):
    # used for pdf data to calculate the mean of the rows, it's useful when u have all of the value of the df to be aggregated.
    means = []
    for column_name in df.columns:
        if pd.api.types.is_numeric_dtype(df[column_name]):
            avg = df[column_name].mean()
        else:
            avg = None
        means.append(avg)

    # Create a new DataFrame with a single row containing the calculated means
    avg_df = pd.DataFrame([means], columns=df.columns)

    # Concatenate the original DataFrame and the new DataFrame
    df = pd.concat([df, avg_df], ignore_index=True)

    return df

def group_into_montly_data(df, index:bool, date_col_name ='DATE'):
    if index == False:
        df['DATE'] = pd.to_datetime(df[date_col_name], format='%d/%m/%y')
    else:
        df['DATE'] = df.index

    # Extract the month and year into separate columns
    df['Month_Year'] = df['DATE'].dt.strftime('%m-%Y')

    # Drop the original 'Date' column
    df.drop('DATE', axis=1, inplace=True)

    # Group the DataFrame by the 'Month_Year' column and calculate the mean for each group
    df_monthly_grouped = df.groupby('Month_Year').mean().reset_index()

    # Sort the DataFrame chronologically by the 'Month_Year' column
    df_monthly_grouped['Month_Year'] = pd.to_datetime(df_monthly_grouped['Month_Year'], format='%m-%Y')
    df_monthly_grouped = df_monthly_grouped.sort_values('Month_Year')

    df_monthly_grouped = df_monthly_grouped.rename(columns={'Month_Year': 'DATE'})

    return df_monthly_grouped

def drop_columns_with_zeros(df):
    # function used for pdf data
    # Check if a column has only 0 values
    columns_to_drop = [col for col in df.columns if (df[col] == 0).all()]

    # Drop the columns with only 0 values
    df = df.drop(columns=columns_to_drop)

    return df

def convert_columns_to_float(df):
    # Iterate through each column and convert its values to float
    for col in df.columns:
        if col != "DATE" and col != "YEAR":
            # Replace commas with periods, and convert values to float
            df[col] = df[col].str.replace('.', '')
            df[col] = df[col].str.replace(',', '.').astype(float)
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

    return df
