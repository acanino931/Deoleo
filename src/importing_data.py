import pandas as pd
import numpy as np
import os
import warnings
from src import agridata as ag
import time
import sys
import datetime
warnings.filterwarnings("ignore")
import itertools
from itertools import product
from unidecode import unidecode
import tabula
from src import Aux_functions as aux
from src import yearly_functions as yf


def import_montly_andalucia():
    montly_sheets = ['Observatorio Mens EXTRA VIRGEN', 'Observatorio Mens VIRGEN', 'Observatorio Mens LAMPANTE']

    start_date = datetime.datetime(2007, 9, 1)
    end_date = datetime.datetime(2023, 9, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Create a DataFrame with the "DATE" column
    df_month_andalucia = pd.DataFrame({'DATE': date_range})

    for sheet in montly_sheets:

        month_andalucia_df = pd.read_excel('Datos/BASE DATOS v act 28_09.xlsx', sheet_name=sheet)
        # this data have been preprocessed in excel  changing the column name, taking care of the value of the date column
        # because in some cases appeard to be a formula and to have uncorrect values.

        month_andalucia_df = month_andalucia_df.iloc[14:, [0, -1]].reset_index(drop=True)
        month_andalucia_df.iloc[0, 0] = 'Date'

        month_andalucia_df.columns = month_andalucia_df.iloc[0]
        month_andalucia_df = month_andalucia_df.iloc[1:].reset_index(drop=True)

        month_andalucia_df['Month'] = month_andalucia_df['Date'].str.split().str[0]
        month_andalucia_df['Year'] = month_andalucia_df['Date'].str.split().str[1]

        month_mapping = {
            'Enero': '01',
            'Febrero': '02',
            'Marzo': '03',
            'Abril': '04',
            'Mayo': '05',
            'Junio': '06',
            'Julio': '07',
            'Agosto': '08',
            'Septiembre': '09',
            'Octubre': '10',
            'Noviembre': '11',
            'Diciembre': '12'
        }
        month_andalucia_df['Month'] = month_andalucia_df['Month'].map(month_mapping)
        month_andalucia_df['DATE'] = pd.to_datetime(
            month_andalucia_df['Year'] + '-' + month_andalucia_df['Month'] + '-01')
        month_andalucia_df = month_andalucia_df[['DATE', 'Precio medio']].copy()

        sheet_name = sheet.split()[-2:]
        if 'EXTRA' not in sheet_name:
            sheet_name = str(sheet_name[-1])
        else:
            sheet_name = '_'.join(sheet_name)

        month_andalucia_df.rename(columns={'Precio medio': 'Avg_Price_' + sheet_name}, inplace=True)
        df_month_andalucia = pd.merge(df_month_andalucia, month_andalucia_df, left_on='DATE', right_on='DATE', how='left')
    return df_month_andalucia


def load_data():
        # use this db to find the most updated data
        weighted_df = pd.read_excel('Datos/BASE DATOS v act 28_09.xlsx', sheet_name = 'PoolRed BD Mensual con volumen')
        # this data have been preprocessed in excel  changing the column name, taking care of the value of the date column
        # because in some cases appeard to be a formula and to have uncorrect values.

        weighted_df= weighted_df.iloc[3:, 1:8 ].reset_index(drop= True)
        weighted_df.columns = weighted_df.iloc[0]
        weighted_df=  weighted_df.iloc[1:]
        weighted_df.iloc[1:].reset_index(drop= True)
        weighted_df['DATE'] = pd.to_datetime(weighted_df['DATE'], format='%Y-%m-%d')
        weighted_df.set_index('DATE',inplace = True)
        target_columns = [col for col in weighted_df.columns if 'EUR_kg' in col]
        weighted_df = weighted_df[target_columns]
        weighted_df[target_columns] = weighted_df[target_columns].astype(float)



        file_name = 'Datos/Weekly_DB.csv'
        datos = pd.read_csv(file_name,sep = ';',decimal=',',encoding = 'utf-8')
        df = datos[['EVOO_ES', 'REFINED', 'LAMPANTE_B1','VOO_ES','Date']].copy()

        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

        # Extract the month and year into separate columns
        df['Month_Year'] = df['Date'].dt.strftime('%m-%Y')

        # Drop the original 'Date' column
        df.drop('Date', axis=1, inplace=True)

        # Group the DataFrame by the 'Month_Year' column and calculate the mean for each group
        monthly_grouped = df.groupby('Month_Year').mean().reset_index()

        # Sort the DataFrame chronologically by the 'Month_Year' column
        monthly_grouped['Month_Year'] = pd.to_datetime(monthly_grouped['Month_Year'], format='%m-%Y')
        monthly_grouped = monthly_grouped.sort_values('Month_Year')

        monthly_grouped = monthly_grouped.rename(columns = {'Month_Year': 'DATE'})

        monthly_grouped = pd.merge(weighted_df,monthly_grouped, left_on='DATE', right_on='DATE', how='left')

        # selecting all the variables since 2002

        specific_date = pd.to_datetime('2001-12-01')
        monthly_grouped = monthly_grouped.loc[monthly_grouped['DATE'] > specific_date]


        # old import file

        #file_name = 'Datos/exogenas_v2.csv'
        #additional_exog = pd.read_csv(file_name,sep = ';',decimal=',',encoding = 'utf-8')
        #additional_exog= additional_exog.iloc[:-2]
        #additional_exog['DATE'] = pd.to_datetime(additional_exog['DATE'], format='%d/%m/%Y')
       # additional_exog.to_excel('Datos/exogenas_v2.xlsx', index = False)

        file_name = 'Datos/exogenas_v2.xlsx'
        additional_exog = pd.read_excel(file_name)
        additional_exog['DATE'] = pd.to_datetime(additional_exog['DATE'], format='%d/%m/%Y')




        # CREATING A DUMMY VARIABLE FOR PRODUCTION
        additional_exog['PRODUCTION'] =additional_exog.PRODUCTION.fillna(0)
        additional_exog['DUMMY_PRODUCTION'] = additional_exog['PRODUCTION'].apply(lambda x: 1 if x != 0 else 0)
        #additional_exog.drop('PRODUCTION', axis =1,inplace = True)
      #  additional_exog['STOCK_OIL'] = additional_exog['EXIS_INIC'] + additional_exog['PRODUCTION']
        # availability of olive oil = initial existence and production of the previous period
   #     additional_exog['STOCK_OIL'] = additional_exog['EXIS_INIC'].shift(-1)
  #     additional_exog['STOCK_OIL'] = additional_exog['STOCK_OIL'].fillna(method='ffill' ,limit=1)

        # Creating the Spanish demand variable
        additional_exog['INTERNAL_DEMAND'] = additional_exog['INNER_CONS'] + additional_exog['IMPORTS']
        # creating the spanish prouction DEMAND = TOTAL DEMAND
        additional_exog['TOTAL_CONS'] = additional_exog['INNER_CONS'] + additional_exog['EXPORTS']

        additional_exog['TOTAL_DEMAND'] = additional_exog['INNER_CONS'] + additional_exog['EXPORTS'] + additional_exog['IMPORTS']



        # Merge the two DataFrames on the 'DATE' column using an outer join
        merged_df = pd.merge(monthly_grouped, additional_exog, left_on='DATE', right_on='DATE', how='left')

        # Drop the 'DATE' column
        merged_df.set_index("DATE", inplace=True)


        attempts = 4
        retry_delay = 2  # 2 seconds

        for _ in range(attempts):
            try:
                agri = ag.agridata()
                monthly_europe_seeds = agri.oil_seeds('production', granularity='monthly')
                if not monthly_europe_seeds.empty:
                    break  # If the call succeeds and the DataFrame is not empty, exit the loop
            except Exception as e:
                print(f"Failed to make the call: {e}")
                time.sleep(retry_delay)  # Wait for the specified delay before retrying

        if monthly_europe_seeds.empty:
            print(f"All attempts failed. Could not retrieve data. Specific exception occurred: {e}")
            sys.exit(1)


        monthly_europe_seeds = agri.oil_seeds('production',granularity = 'monthly')
        monthly_europe_seeds = monthly_europe_seeds.pivot_table(values = ['grossProduction','area','yield'], index = ['year','crop'], columns = 'memberStateName' , aggfunc = 'sum')
        monthly_europe_seeds.reset_index(level='crop', inplace=True)
        monthly_europe_seeds.index = pd.to_datetime(monthly_europe_seeds.index,format = '%Y')

        names_list = ['_'.join(map(str, col)) for col in monthly_europe_seeds.columns]
        list(map(lambda x: str(x) +'_crop',names_list))

        oils_seeds_df = []
        for crop in monthly_europe_seeds['crop'].unique():
            oils_seeds = monthly_europe_seeds[monthly_europe_seeds['crop'] == crop]
            names_list = ['_'.join(map(str, col)) for col in oils_seeds.columns]
            oils_seeds.columns = list(map(lambda x: str(x) + f'_{crop}', names_list))
            oils_seeds_df.append(oils_seeds)

        oils_seeds_final = pd.concat(oils_seeds_df, axis=1)
        columns_to_drop = [col for col in oils_seeds_final.columns if col.startswith('crop')]
        df_oil_seeds = oils_seeds_final.drop(columns=columns_to_drop)
        # changing granularity form year to month distributing the values of he year for each month
        df_oil_seeds = df_oil_seeds.resample('MS').ffill() / 12
        # loop to restore the previous value of the area of crops because the distribution of the quantity for every month does not apply.
        for col in df_oil_seeds.columns:
            if 'area' in col:
                df_oil_seeds[col] = df_oil_seeds[col] * 12

        # Indicators this part is not used in the analysis
        # so it might be eliminated when optimizing

        df_indicators= []
        country_codes = ['ES','IT','FR']
        for code in country_codes:
            df_indicators.append(agri.indicators('values',code))

        df = []
        for i in range(3):
            spain_ind = df_indicators[i].pivot_table(values = 'value', index = 'year', columns = 'subindicator',aggfunc = 'sum')
            spain_ind=spain_ind[spain_ind.index >=2002]
            missing_percentage = (spain_ind.isnull().sum() / len(spain_ind)) * 100
            threshold = 75
            filtered_columns = missing_percentage[missing_percentage <= (100 -threshold)].index
            spain_ind = spain_ind[filtered_columns]
            df.append(spain_ind)

        indicators_europe = pd.concat(df,axis=1)
        indicators_europe.index = pd.to_datetime(indicators_europe.index,format = '%Y')
        indicators_europe = indicators_europe.resample('MS').interpolate(method='linear')
        word_to_find = 'Spain'
        additional_data = pd.concat([indicators_europe,df_oil_seeds],axis =1)
        filtered_columns = additional_data.filter(like=word_to_find)
        #filtered_columns= filtered_columns.iloc[:,1:] #commented not sure why they decided to eliminate the column at the beginning

        # if you want to eliminate indicators whatch out to keep the oilseed db
        df1 = pd.merge(monthly_grouped, additional_exog, left_on='DATE' , right_on='DATE', how='left')
        df_month = pd.merge(df1,filtered_columns,how = 'left', left_on = 'DATE',right_index = True)



        #df_month =pd.concat([final_df['DATE'], df_month], axis=1)
        df_month.set_index('DATE',inplace = True)
        df_month['YEAR'] = df_month.index.year

        # importing the Energy Data:

        all_timestamps = pd.DataFrame({'Timestamp': pd.date_range(start='2010-01-01', end='2023-06-22')})
        # Replace 'OLD_NEXUS/Reikon_commodities.xlsx' with the path to your Excel file
        all_timestamps['Timestamp'] = pd.to_datetime(all_timestamps['Timestamp'])
        excel_file = 'Datos/Reikon_commodities.xlsx'

        # Define the list of sheet names you want to read
        sheet_names_to_read = ['EUA', 'BRENT', 'API2', 'TTF', 'MIBGAS PVB']

        # Loop over each sheet and read specific columns
        for sheet_name in sheet_names_to_read:
            # Read the data from the sheet into a DataFrame
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # in case of the co2 emission we take the december price because it is an annual contract with a few volatility.
            if sheet_name == 'EUA':
                df = df.iloc[4:, [1, 6]]
            else:
                df = df.iloc[4:, [1, 2]]
            df.columns = df.iloc[0]
            # Rename the columns with the sheet name
            df.rename(columns={df.columns[1]: df.columns[1] + "_" + sheet_name, \
                               df.columns[0]: df.columns[0] + "_" + sheet_name}, inplace=True)
            df = df.iloc[1:]
            df = df.reset_index(drop=True)

            column_name = df.columns[0]
            date_format ="%Y-%m-%d %H:%M:%S"

            df[column_name] = pd.to_datetime(df[column_name] ,format =date_format)


            all_timestamps = all_timestamps.merge(df, left_on='Timestamp', right_on=column_name, how='left')
            all_timestamps = all_timestamps.reset_index(drop=True)


        # adding the energy price located in another file
        excel_file, sheet_name = 'Datos/OMEL.xlsx', 'OMEL'
        # electricity_timestamps = pd.DataFrame({'Timestamp': pd.date_range(start='2010-01-01', end='2023-06-22')})
        df_electricity = pd.read_excel(excel_file, sheet_name=sheet_name)
        # let us take the average of the day as price spot variable
        df_electricity = df_electricity.iloc[1:, [0, -1]]
        df_electricity.columns = df_electricity.iloc[0]
        df_electricity = df_electricity.iloc[1:]
        first_column_name = df_electricity.columns[0] + "_" + sheet_name
        second_column_name =df_electricity.columns[1] + "_" + sheet_name

        df_electricity.reset_index(drop=True,inplace = True)


        # Rename the columns with the sheet name
        df_electricity.rename(columns={df_electricity.columns[1] : second_column_name , df_electricity.columns[0]: first_column_name }, inplace=True)


        #column_name = df_electricity.columns[0]
        date_format = "%Y-%m-%d %H:%M:%S"

        df_electricity[first_column_name] = pd.to_datetime(df_electricity[first_column_name], format=date_format) # pd.date_range(start='2010-01-01', end='2023-06-17') # error with the format of the date #

        all_timestamps = all_timestamps.merge(df_electricity, left_on='Timestamp', right_on=first_column_name, how='left')
        all_timestamps.reset_index(drop=False, inplace=True)
        all_timestamps.drop(columns=['index'], inplace=True)


        null_counts = all_timestamps.isnull().sum()

        # eliminating the timestapms variables ,
        # possibly we can delete and the iberian gas price MIBGAS PVB cuz we still have the european benchmark gas because
        # we have Dat just from 2015, and for electricity Price OMIP from June 20211.
        col_to_del = []
        for col in all_timestamps.columns:
            if col.startswith('Timestamp_'):  # in col  #or 'MIBGAS PVB' in col  :
                col_to_del.append(col)



        all_timestamps.drop(col_to_del, inplace=True, axis=1)
        all_timestamps.reset_index(drop=True)


        #AGGREGATING for mounth
        all_timestamps.set_index('Timestamp', inplace=True)


        #for col in all_timestamps:
        #    if col != 'TimeStamp'
        #    df['Column1'] = df['Column1'].apply(lambda x: float(x))
        #all_timestamps_bis = all_timestamps.applymap(lambda x: float(x) if pd.notna(x) else x)


        energy_month = all_timestamps.resample('MS').mean()


        #cleaning the column names
        df_month.columns = df_month.columns.str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_').str.replace(',', '_').str.replace('___', '_').str.replace('__', '_')
        df_month.columns = df_month.columns.str.replace('_seed', '')
        df_month = pd.merge(df_month, energy_month, left_index=True ,right_index=True,  how='left')



        # Including IMC data

        df_imc = pd.read_excel("Datos/IMC ENE 18_NOV 23.xlsx")
        df_imc.set_index('Date',inplace = True)
        df_month  =  pd.merge(df_month, df_imc, left_index=True ,right_index=True,  how='left')


        # INCLUDING YEARLY PRODUCTION

        # Aggregating production data using the previous year as a reference.
        df_month['Month'] = df_month.index.month
        df_month['DUMMY_PRODUCTION'] = np.where((df_month['Month'] > 9) | (df_month['Month'] < 6), 1, 0)
        # COGEMOS fin Enero como mes de referencia por la cosecha, año de referencia de la cosecha
        df_month['HARVEST_YEAR'] = np.where(df_month['Month'] > 9, df_month.index.year + 1, df_month.index.year)

        # ultima cosecha disponible, se supone que hasta noviembre se usan las existencias iniciales pero no la nuevas
        # from october until march we dont consider any shifting and for the we start to count from the beggining
        # df_month['PRODUCTION_shift'] =  np.where((df_month['Month'] > 9) | (df_month['Month'] < 5), 0, 0)

        # adding the variable PRODUCTION_BEFORE_MARCH  we are cutting the value of the harvest to what is available in March not considering the rest.
        df_month['PRODUCTION_BEFORE_MARCH'] =  np.where((df_month['Month'] > 2) & (df_month['Month'] < 9),0,df_month['PRODUCTION'])

        # aggregating the production data per year for the harvest reference year
        #decomment when test over
        df_production = df_month[['PRODUCTION_BEFORE_MARCH', 'HARVEST_YEAR']].copy()

        df_month.drop(columns=['PRODUCTION_BEFORE_MARCH'],inplace = True)

        df_production['PRODUCTION_BEFORE_MARCH'] = df_production[['PRODUCTION_BEFORE_MARCH']].fillna(0)
        df_production_agg = pd.DataFrame()

        df_production_agg['PRODUCTION_HARVEST'] = df_production.groupby('HARVEST_YEAR')['PRODUCTION_BEFORE_MARCH'].sum()
       # Start Decoment _ if you want to consider all the production comment preious lines as well
        """

        df_production = df_month[['PRODUCTION', 'HARVEST_YEAR']].copy()



        df_production['PRODUCTION'] = df_production[['PRODUCTION']].fillna(0)
        df_production_agg = pd.DataFrame()

        df_production_agg['PRODUCTION_HARVEST'] = df_production.groupby('HARVEST_YEAR')['PRODUCTION'].sum()
        """
       # End Decoment if you want to consider all the production

        # Including the aggregated production in the main df
        df_production_agg.index = pd.to_datetime(df_production_agg.index, format='%Y')
        df_production_agg.index.freq = pd.tseries.offsets.YearBegin()
        df_production_agg= df_production_agg.resample('MS').ffill()




        # we start considering the effect from the quantity of the harvest from March since the new harvest it's nearly over

        df_month = pd.merge(df_month, df_production_agg, left_index=True, right_index=True, how='left')
        # this copy is made for feed the Daniel Peña transformation where the shift will be applied to the target variable direclty
        df_month['PRODUCTION_HARVEST_ORIGINAL'] = df_month['PRODUCTION_HARVEST'].copy()
        df_month['PRODUCTION_HARVEST_ORIGINAL'] = df_month['PRODUCTION_HARVEST_ORIGINAL'].fillna(method='ffill', limit =11)
        df_month['PRODUCTION_HARVEST'] = df_month['PRODUCTION_HARVEST'].shift(2)
        average_prod_post_march, prec = aux.compute_perc_harvest_post_march(df_month)
        average_prod_post_march = 1 - (average_prod_post_march / 100)
        previsiones = load_and_transform_previsiones()
        # we are applying a penalty on the forecast of the production because the aggregated filed is not considering the production after march param:average_prod_post_march
        df_month = transform_dataframe(df_month, previsiones, average_prod_post_march)
        df_month['PRODUCTION_HARVEST_LAST_YEAR'] = df_month['PRODUCTION_HARVEST'].shift(12)
        df_month['PRODUCTION_HARVEST_2_YEARS'] = df_month['PRODUCTION_HARVEST'].shift(24)

        # importing production forecast data
        prevision_df = yf.preprocess_prevision_data(df_month)
        col_index = prevision_df.index.name
        prevision_df1 = aux.from_yearly_to_monthly(prevision_df, column_index=col_index, transpose=False)

        df_month = pd.merge(df_month, prevision_df1, right_index=True, left_index=True, how='left')

        #df_month.drop (columns =['HARVEST_YEAR'],inplace = True)


        # until we update the data let us keep the values until June

       # specific_date = pd.to_datetime('2023-07-01')
       # df_month = df_month[df_month.index < specific_date]
       # df_month = df_month.fillna(method='ffill')
        df_month.to_excel("Output/Excel/df_month.xlsx")

        return df_month

def load_and_transform_previsiones(filename = "Datos/previsiones.xlsx"):
    # Read the 'previsiones' DataFrame from the Excel file
    previsiones = pd.read_excel(filename, decimal=',')

    # Separate the 'Cosecha' column into two columns and perform necessary data transformations
    previsiones[['Año Inicio', 'Año Fin']] = previsiones['Cosecha'].str.split('/', expand=True)
    previsiones['Año Fin'] = '20' + previsiones['Año Fin']
    previsiones.drop('Cosecha', axis=1, inplace=True)
    previsiones = previsiones[['Año Inicio', 'Año Fin', 'Cosecha final Kton', 'Estimación Junta Andalucía', 'Estimación España (Junta Andalucia)', 'Estimación España COI ']]
    previsiones['Año Inicio'] = pd.to_numeric(previsiones['Año Inicio'], errors='coerce')
    previsiones['Año Fin'] = pd.to_numeric(previsiones['Año Fin'], errors='coerce')
    previsiones['Cosecha final Kton'] = pd.to_numeric(previsiones['Cosecha final Kton'], errors='coerce')
    previsiones['Estimación Junta Andalucía'] = pd.to_numeric(previsiones['Estimación Junta Andalucía'], errors='coerce')
    previsiones['Estimación España (Junta Andalucia)'] = pd.to_numeric(previsiones['Estimación España (Junta Andalucia)'], errors='coerce')
    previsiones['Estimación España COI'] = pd.to_numeric(previsiones['Estimación España COI '], errors='coerce')

    return previsiones

def transform_dataframe(dataframe, previsiones, proporcion):
    # Convert 'DATE' column to datetime
    if 'DATE' not in dataframe.columns:
        dataframe['DATE'] = dataframe.index
        print(dataframe.index)
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])

    # Define a function to transform the 'PRODUCTION_HARVEST' column
    def transform_production(row):
        month = row['DATE'].month
        year = row['DATE'].year

        if 3 <= month <= 6:
            return row['PRODUCTION_HARVEST']
        elif 7 <= month <= 12:
            condition = (previsiones['Año Inicio'] == year)
        else:
            condition = (previsiones['Año Fin'] == year)

        matching_row = previsiones[condition]

        if not matching_row.empty:
            return matching_row['Estimación España (Junta Andalucia)'].values[0]*proporcion

        return row['PRODUCTION_HARVEST']

    # Create a new column 'PRODUCTION_HARVEST_REAL_EST'
    dataframe['PRODUCTION_HARVEST_REAL_EST'] = dataframe.apply(transform_production, axis=1)

    # Fill NaN values in 'PRODUCTION_HARVEST_REAL_EST' with the previous value
    dataframe['PRODUCTION_HARVEST_REAL_EST'].fillna(method='ffill', inplace=True)

    return dataframe


def import_single_pdf(path_file ,filename) :
    print(filename)
    all_countries = pd.read_excel("Datos/all_countries_list.xlsx")

    all_countries['COUNTRY_NAME'] = all_countries['COUNTRY_NAME'].apply(lambda x: unidecode(x))
    filename
    path= path_file + filename
    #path_file = path = "Datos/PDF/Juan_Vilar/Importacion Total.pdf"
    df = tabula.read_pdf(path, pages='all')[0]

    # code to append the 1st line to the column if the info is provisional or not (knowing all the last 2 years are provisional it is not needed)
    """first_row = df.iloc[0]
    for column in df.columns:
        if not pd.isnull(first_row[column]):
            df.rename(columns={column: column + first_row[column]}, inplace=True)
    """
    # Reset the index if needed

    # getting the county col
    df.columns.values[0] = 'COUNTRY'
    # updating the coulum name so the index might see it
    for column in df.columns:
        df.rename(columns={column: column}, inplace=True)
    df = df.iloc[2:]

    df = df.reset_index(drop=True)
    # eliminating all the others with null values
    # col_to_drop = [col for col in df.columns if 'Unnamed' in col]
    col_to_drop = [col for col in df.columns if 'Unnamed' in col and col != 'COUNTRY']

    df = df.drop(columns=col_to_drop)
    df['COUNTRY'] = df['COUNTRY'].apply(lambda x: unidecode(x))

    # code to get the proper country name
    elements_to_check = all_countries['COUNTRY_NAME']
    # Loop through the DataFrame and replace row values if an element is found
    for index, row in df.iterrows():
        for element in elements_to_check:
            if element in row['COUNTRY']:
                df.at[index, 'COUNTRY'] = element

    sheet_name = filename.split(".")[0]
    df['COUNTRY'] = sheet_name + "_" + df['COUNTRY']

    newcols = []
    for col in df.columns[1:]:
        last_part = col.split("/")[-1]
        if last_part == "0":
            newyear = "2000"
        elif len(last_part) == 1:
            newyear = col[0:3] + last_part
        else:
            newyear = col[0:2] + col[-2:]
        newcols.append(int(newyear))

    for i, col in enumerate(df.columns[1:]):
        df = df.rename(columns={col: newcols[i]})

    df = df.rename(columns={'COUNTRY': 'YEAR'})

    # from year start substituting with func : aux.from_yearly_to_monthly()
    # GETTING THE CORRECT TYPE  OF THE INDEX TO RESAMPLE THE DATA INTO MOUNTHLY
    df.set_index('YEAR', inplace=True)
    df_transposed = df.transpose()
    df_transposed
    df_transposed.index = df_transposed.index.astype(int)
    df_transposed
    df_transposed.index.dtype
    df_transposed.index[0]
    date_index = pd.date_range(start=str(df_transposed.index[0]), periods=len(df_transposed.index), freq='YS')

    df_transposed.index = date_index

    # changing granularity
    df_transposed = df_transposed.resample('MS').ffill()

    max_index = df_transposed.index.max()
    next_month = max_index + pd.DateOffset(months=1)
    first_day_of_next_month = pd.Timestamp(year=next_month.year, month=next_month.month, day=1)
    first_day_of_next_month
    date_string = first_day_of_next_month.strftime('%Y-%m-%d')


    extended_date_index = pd.date_range(start= date_string, end='2023-12-01', freq='MS')
    extended_df = pd.DataFrame(index=extended_date_index)
    df_transposed_fin = pd.concat([df_transposed, extended_df], axis=0)
    # what out with the ffil we can imput automatically values to missing year make a control over the value we inputing
    df_transposed_fin = df_transposed_fin.fillna(method="ffill", limit=11)
    # to here stop substituting with func : aux.from_yearly_to_monthly() and test it

    # shift the 1st 3 month to have cross data like the starting ones
    df_transposed_fin_shifted = df_transposed_fin.shift(-3)
    # if all the values of a coloumn are null we directly eliminate the columns:
    eliminated_columns = [column for column in df_transposed_fin_shifted.columns if
                          df_transposed_fin_shifted[column].isnull().all()]

    # Drop the columns with all null values
    df_transposed_fin_shifted.drop(eliminated_columns, axis=1, inplace=True)
      #  df_transposed_fin_shifted.to_excel("Output/Excel/df_tabula.xlsx")
    #added to test
    df_transposed_fin_shifted = aux.convert_columns_to_float(df_transposed_fin_shifted)
    return df_transposed_fin_shifted

def initialize_df_month(start_date,end_date):


    # Generate a date range with monthly granularity
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    df = pd.DataFrame(index=date_range)
    return df



def import_pdf_data (path_folder_pdf =  "Datos/PDF/Juan_Vilar/" , start_date = '2001-01-01', end_date = '2023-09-01',just_total = True):

    path_folder_pdf = "Datos/PDF/Juan_Vilar/"
    df_pdf_tot = initialize_df_month(start_date,end_date)

    ls_file_pdf = [file for file in os.listdir(path_folder_pdf) if "Informe" not in file]
    for filename in ls_file_pdf:
        df_pdf = import_single_pdf("Datos/PDF/Juan_Vilar/" ,filename)
        df_pdf_tot= pd.merge (df_pdf_tot,df_pdf , right_index= True, left_index= True)
    df_pdf_tot = aux.drop_columns_with_zeros(df_pdf_tot)
    if just_total == True:
        df_pdf_tot = filter_pdf_totals(df_pdf_tot)
    return df_pdf_tot


def filter_pdf_totals (df ):
    # for the analysis we consider just the aggregate, comment that function call if you want to include all the variables
    ls_tot=[col for col in df.columns if 'TOTAL' in col]
    df = df[ls_tot]
    return df


def include_pdf_data(df):
    df_pdf = import_pdf_data()
    df = df.merge(df_pdf, left_index=True, right_index=True, how='left')
    return df

def import_meteo_single_province(path_folder,province_name, start_date ='2001-01-01',end_date = '2023-09-30'):
    # read all the excel files of a province and put it in a df.
    # linares and Marmolejo ( Jaen) are out because they are fragmented in 2 files from the web
    # supposibly it should be done a concat of the 2 original file and merging them separately at the end of 1st for
    #path = "C:/Users/acanino/PycharmProjects/Deoleo/Datos/Datos_Cordoba"
    # creating all the timestamps of the final df
    all_dates = pd.DataFrame({'Timestamp': pd.date_range(start=start_date, end=end_date)})
    all_dates['Timestamp'] = pd.to_datetime(all_dates['Timestamp'])
    all_dates.set_index('Timestamp', inplace=True)
    for file in os.listdir(path_folder):
        print(file)
        sheet = pd.read_csv(path_folder + "/" + file, sep=';', decimal=',', encoding='utf-8')
        sheet['FECHA'] = pd.to_datetime(sheet['FECHA'], format='%d/%m/%y')
        sheet.drop(columns=['DIA'],inplace=True)
        sheet.rename(columns={'FECHA': 'DATE'}, inplace=True)
        sheet.set_index('DATE',inplace = True)
        all_dates = all_dates.merge(sheet, left_index=True, right_index=True, how='left')

    # listing the variables suffix
    suffix_list = ['TMax', 'TMin', 'TMed', 'Precip']
    for suffix in suffix_list:
        average_lambda = lambda row: pd.to_numeric(row.filter(like=suffix), errors='coerce').mean()
        all_dates[suffix+'_Average_' + province_name] = all_dates.apply(average_lambda, axis=1)

    for col in all_dates.columns:
        if "Average" in col:
            null_average_rows = all_dates[all_dates[col].isnull()]
            print(null_average_rows)
            all_dates[col].fillna(method="ffill").fillna(method="bfill")

    avg = 'Average'
    df_selected = all_dates[all_dates.columns[all_dates.columns.str.contains(avg)]]

    df_out= aux.group_into_montly_data(df_selected, index =True)
    df_out.set_index('DATE',inplace=True)

    return df_out


def merge_meteo_data(list_df):
    df_temp = list_df[0]
    df_out = pd.DataFrame(index=df_temp.index)

    for df in list_df:
        df_out = df_out.merge(df, left_index=True, right_index=True, how='left')

    return df_out



