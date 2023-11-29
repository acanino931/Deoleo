



# dummy more than 4 days of 33 in a row
import pandas as pd

# Assuming your DataFrame is named 'df' and contains a 'temperature' column
# Make sure your DataFrame is sorted by date before running this code

# Example DataFrame creation (replace this with your actual DataFrame)
data = {'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
        'temperature': [30, 35, 34, 36, 34, 34, 35, 37, 33, 34]}
df = pd.DataFrame(data)

# Sort DataFrame by date
df = df.sort_values(by='date')

# Create a new column to store the result (initialize with 0)
df['four_days_above_33'] = 0

# Use rolling to efficiently check for 4 consecutive days with temperature > 33
df['four_days_above_33'] = df['temperature'].rolling(window=4, min_periods=1).apply(lambda x: 1 if all(x > 33) else 0)

# Display the resulting DataFrame
print(df)

import pandas as pd

def calculate_dummy_hot_days_ina_row(df, temperature_variable, num_days, target_temperature):
    # Sort DataFrame by date
    df = df.sort_values(by='date')

    # Create a new column to store the result (initialize with 0)
    dummy_column_name = f'{num_days}_days_above_{target_temperature}'
    df[dummy_column_name] = 0

    # Use rolling to efficiently check for consecutive days with temperature > target_temperature
    df[dummy_column_name] = df[temperature_variable].rolling(window=num_days, min_periods=1).apply(lambda x: 1 if all(x > target_temperature) else 0)

    return df

# Example usage:
data = {'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
        'temperature': [30, 35, 34, 36, 34, 34, 35, 37, 33, 34]}
df = pd.DataFrame(data)

result_df = calculate_dummy_hot_days_ina_row(df, temperature_variable='temperature', num_days=4, target_temperature=33)

# Display the resulting DataFrame
print(result_df)







# days above 30  calc new col
import pandas as pd

# Assuming your DataFrame is named 'df' and contains a 'temperature' column
# Make sure your DataFrame is sorted by date before running this code

# Example DataFrame creation (replace this with your actual DataFrame)
temperature_data = [30, 22, 34, 36, 32, 34, 35, 37, 33, 34, 30, 31, 34, 36, 32, 34, 35, 37, 33, 34]
date_range = pd.date_range(start='2023-01-01', periods=len(temperature_data))

data = {'date': date_range, 'temperature': temperature_data}
df = pd.DataFrame(data)

# Sort DataFrame by date
df = df.sort_values(by='date')

# Create a new column to store the result (initialize with 0)
df['days_above_30'] = 0

# Count the number of days above 30 for each month
df['days_above_30'] = df['temperature'].gt(36).groupby(df['date'].dt.to_period("M")).transform('sum')

# Display the resulting DataFrame
print(df)



# keeping the granularity daily adding columns
import pandas as pd

# Assuming your DataFrame is named 'df' and contains a 'precipitation' column
# Make sure your DataFrame is sorted by date before running this code

# Example DataFrame creation (replace this with your actual DataFrame)
date_range_1 = pd.date_range(start='2023-01-01', periods=20)
precipitation_data_1 = [0.0] * 20

date_range_2 = pd.date_range(start='2023-02-01', periods=20)
precipitation_data_2 = [5.2, 7.8, 3.5, 4.2, 6.1, 2.8, 5.5, 3.2, 4.8, 6.4, 1.5, 4.0, 2.3, 3.7, 5.9, 4.1, 6.0, 3.4, 4.5, 5.2]

# Concatenate the data for both months
date_range = date_range_1.append(date_range_2)
precipitation_data = precipitation_data_1 + precipitation_data_2

data = {'date': date_range, 'precipitation': precipitation_data}
df = pd.DataFrame(data)

# Sort DataFrame by date
df = df.sort_values(by='date')

# Group by month and calculate statistics
monthly_aggregated_df = df.groupby(df['date'].dt.to_period("M")).agg({
    'date': 'first',
    'precipitation': ['min', 'max', 'mean', 'std', 'var']
}).reset_index(drop=True)

# Rename columns for clarity
monthly_aggregated_df.columns = ['month', 'min_precipitation', 'max_precipitation', 'mean_precipitation', 'std_precipitation', 'var_precipitation']

# Display the resulting DataFrame
print(monthly_aggregated_df)




# making it a mountly df
import pandas as pd

# Assuming your DataFrame is named 'df' and contains a 'precipitation' column
# Make sure your DataFrame is sorted by date before running this code

# Example DataFrame creation (replace this with your actual DataFrame)
date_range_1 = pd.date_range(start='2023-01-01', periods=20)
precipitation_data_1 = [0.0] * 20

date_range_2 = pd.date_range(start='2023-02-01', periods=20)
precipitation_data_2 = [5.2, 7.8, 3.5, 4.2, 6.1, 2.8, 5.5, 3.2, 4.8, 6.4, 1.5, 4.0, 2.3, 3.7, 5.9, 4.1, 6.0, 3.4, 4.5, 5.2]

# Concatenate the data for both months
date_range = date_range_1.append(date_range_2)
precipitation_data = precipitation_data_1 + precipitation_data_2

data = {'date': date_range, 'precipitation': precipitation_data}
df = pd.DataFrame(data)

# Sort DataFrame by date
df = df.sort_values(by='date')

# Group by month and calculate statistics
monthly_aggregated_df = df.groupby(df['date'].dt.to_period("M")).agg({
    'date': 'first',
    'precipitation': ['min', 'max', 'mean', 'std', 'var']
}).reset_index(drop=True)

# Rename columns for clarity
monthly_aggregated_df.columns = ['month', 'min_precipitation', 'max_precipitation', 'mean_precipitation', 'std_precipitation', 'var_precipitation']

# Display the resulting DataFrame
print(monthly_aggregated_df)


import pandas as pd
from src import importing_data as imd
import os
from src import Aux_functions as aux
import numpy as np


start_date = '2023-01-01'
end_date = '2023-12-31'


# cambiare i prefix per la lista fare un tryparse per le 3 cifre se fallisce prova per 2
def try_parse(string):
    try:
        # Try parsing the substring [2:5]
        result = int(string[2:5])
    except ValueError:
        try:
            # If the first parse fails, try parsing the substring [2:4]
            result = int(string[2:4])
        except ValueError:
            # If both parse attempts fail, return a default value or handle the case as needed
            result = None  # You can replace this with any default value or handle the case accordingly

    return result




def include_meteo_variables(start_date='2001-01-01', end_date='2023-09-01'): # add param df
    ls_province = ["Cordoba","Jaen"]
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')
    df_meteo = pd.DataFrame(index=monthly_index)
    for province in ls_province :

        df_province = import_meteo_single_province(f"Datos/Datos_{province}", province) # "Datos/Datos_Cordoba", "Cordoba")
        #df_jaen = import_meteo_single_province("Datos/Datos_Jaen", "Jaen")
        df_meteo = df_meteo.merge(df_province, left_index=True, right_index=True, how='left')

    print("tutto ok")
    unique_vars = set(col.rsplit('_', 1)[0] for col in df_meteo.columns)

    # Create a third DataFrame with mean values for each group of similar variables
    df_andalucia = pd.DataFrame()
    for var in unique_vars:
        matching_columns = [col for col in df_meteo.columns if col.startswith(var)]
        df_andalucia[var + '_Andalucia'] = df_meteo[matching_columns].mean(axis=1)

    #df = df.merge(df_meteo, left_index=True, right_index=True, how='left')
    return df_andalucia#df

def import_meteo_single_province(path_folder, province_name, start_date='2001-01-01', end_date='2023-09-30'):
        # read all the excel files of a province and put it in a df.
        # linares and Marmolejo ( Jaen) are out because they are fragmented in 2 files from the web
        # supposibly it should be done a concat of the 2 original file and merging them separately at the end of 1st for
        # path = "C:/Users/acanino/PycharmProjects/Deoleo/Datos/Datos_Cordoba"
        # creating all the timestamps of the final df otherwise change the prefix custom to the duplicate files
        all_dates = pd.DataFrame({'Timestamp': pd.date_range(start=start_date, end=end_date)})
        all_dates['Timestamp'] = pd.to_datetime(all_dates['Timestamp'])
        all_dates.set_index('Timestamp', inplace=True)
        for file in os.listdir(path_folder):
                print(file)
                sheet = pd.read_csv(path_folder + "/" + file, sep=';', decimal=',', encoding='utf-8')
                sheet['FECHA'] = pd.to_datetime(sheet['FECHA'], format='%d/%m/%y')
                sheet.drop(columns=['DIA'], inplace=True)
                sheet.rename(columns={'FECHA': 'DATE'}, inplace=True)

                # adding a column count of days with temperature max above than 33 degrees
                tmax_column = [col for col in sheet.columns if 'TMax' in col][0]
                rain_column = [col for col in sheet.columns if 'Precip' in col][0]
                prefix_num = try_parse(sheet.columns[1])
                prefix= (sheet.columns[1])[0:2] + str(prefix_num)
                if file == 'Torreperogil_5years.csv':
                    prefix = 'JaXX'
                if file == 'Villacarrillo.csv':
                    prefix = 'JaYY'
                print(prefix)
                # Convert 'TMax' column to float, handling errors by returning None
                sheet[tmax_column] = sheet[tmax_column].apply(lambda x: pd.to_numeric(x, errors='coerce'))
                sheet[rain_column] = sheet[rain_column].apply(lambda x: pd.to_numeric(x, errors='coerce'))

                sheet[f"{prefix}days_above_33"] = sheet.groupby('DATE')[tmax_column].transform(lambda x: x.gt(33).sum())

                sheet.set_index('DATE', inplace=True)
                all_dates = all_dates.merge(sheet, left_index=True, right_index=True, how='left')
                print(f"sheet{sheet.columns}")
                print(f"all_dates{all_dates.columns}")

                # Count the number of days above 30 for each month

        # listing the variables suffix
        suffix_list = ['TMax', 'TMin', 'TMed', 'Precip','days_above_33']
        for suffix in suffix_list:
                average_lambda = lambda row: pd.to_numeric(row.filter(like=suffix), errors='coerce').mean()
                all_dates[suffix + '_Average_' + province_name] = all_dates.apply(average_lambda, axis=1)

        for col in all_dates.columns:
                if "Average" in col:
                        null_average_rows = all_dates[all_dates[col].isnull()]
                        print(null_average_rows)
                        all_dates[col].fillna(method="ffill").fillna(method="bfill")

        avg = 'Average'
        dummy_hot  = [col for col in all_dates.columns if 'days_above_33_Average' in col][0]
        dummy_precip = [col for col in all_dates.columns if 'Precip_Average' in col][0]
        ndays_rain = "ndays_" +dummy_precip
        all_dates[dummy_hot] = np.where(all_dates[dummy_hot] > 0.125, 1, 0)
        all_dates[ndays_rain] = np.where(all_dates[dummy_precip] > 0.125, 1, 0)
        cumulated_rain = "cumulated_month_" +dummy_precip
        all_dates[cumulated_rain] = all_dates[dummy_precip]


        df_selected = all_dates[all_dates.columns[all_dates.columns.str.contains(avg)]]
        df_out = aux.group_into_monthly_data_new(df_selected, index=True, sum_fields= [dummy_hot,ndays_rain,cumulated_rain])


        df_out.set_index('DATE', inplace=True)
        cumulated_rain_year = cumulated_rain.replace("month","year")
        df_out[cumulated_rain_year] = df_out.groupby(df_out.index.year)[cumulated_rain].cumsum()
        #df_out = aux.group_into_montly_data(df_selected, index=True)
        all_dates.to_excel("Output/Excel/all_date_cordoba.xlsx")
        df_out.to_excel("Output/Excel/df_cordoba.xlsx")


        return df_out


df_andaluz = include_meteo_variables()

df_andaluz.to_excel("Output/Excel/df_andaluz.xlsx")

df_andaluz.columns

df_andaluz
all_dates,df_cordoba = import_meteo_single_province("Datos/Datos_Cordoba", "Cordoba")

df_jaen = import_meteo_single_province("Datos/Datos_Jaen", "Jaen")

df_cordoba.index.year


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


df_cordoba
