import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from src import importing_data as imd



def yearly_production_transform(path_previsiones = "Datos/previsiones.xlsx", path_df_month = "Output/Excel/df_month.xlsx"):
    dataframe = pd.read_excel(path_df_month)
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'], format='%d.%m.%Y')
    previsiones = imd.load_and_transform_previsiones(path_previsiones)
    production_data = dp_production_data(dataframe)
    previsiones_mes = dp_previsiones_data(previsiones)
    dp_production_harvest = dp_final_preproc(production_data, previsiones_mes)

    dataframe['DP_PRODUCTION_HARVEST'] = dp_production_harvest

    return(dataframe)



def dp_production_data(dataframe):
    # Create a copy of the original DataFrame to avoid modifying it
    production_data = dataframe[['DATE', 'PRODUCTION_HARVEST']].copy()
    production_data['PRODUCTION_HARVEST'].fillna(method='ffill', inplace=True)

    # Yearly deltas
    deltas = {2001: 0.9, 2002: 0.9, 2003: 0.9, 2004: 0.9, 2005: 0.9, 2006: 0.9, 2007: 0.9, 2008: 0.9, 2009: 0.9,
              2010: 0.9,
              2011: 0.9, 2012: 0.9, 2013: 0.9, 2014: 0.9, 2015: 0.9, 2016: 0.9, 2017: 0.9, 2018: 0.9, 2019: 0.9,
              2020: 0.9, 2021: 0.9, 2022: 0.9, 2023: 0.9}

    for i, row in production_data.iterrows():
        # Get year and month
        year = int(row['DATE'].year)
        month = int(row['DATE'].month)
        delta = deltas[year]
        # March - set new delta
        if month == 3:
            exponent = 0
        elif month == 1 or month == 2:
            exponent = month + 10
            delta = deltas[year - 1]
        # Calculate exponent
        else:
            exponent = month - 3

        try:
            # Exponential smoothing
            production_data.loc[i, 'PRODUCTION_HARVEST_TRANSFORMED'] = row['PRODUCTION_HARVEST'] * (delta ** exponent)

        except:
            production_data.loc[i, 'PRODUCTION_HARVEST_TRANSFORMED'] = np.nan

    production_data['PRODUCTION_HARVEST_TRANSFORMED'] = pd.to_numeric(production_data['PRODUCTION_HARVEST_TRANSFORMED'],
                                                                      errors='coerce')

    return production_data


def dp_previsiones_data(previsiones):
    previsiones_relevant = previsiones[['Año Inicio', 'Año Fin', 'Estimación España (Junta Andalucia)']]
    # Create a new DataFrame for monthly data
    previsiones_mes = pd.DataFrame(columns=["Year", "Month", "Estimación España (Junta Andalucia)"])

    # Iterate through each row in the original DataFrame
    for index, row in previsiones_relevant.iterrows():
        start_year = row["Año Inicio"]
        end_year = row["Año Fin"]
        value = row["Estimación España (Junta Andalucia)"]

        # Generate a date range from July to June for each year
        date_range = pd.date_range(start=f"{start_year}-07-01", end=f"{end_year}-06-30", freq='M')
        previsiones_mes = pd.concat([previsiones_mes, pd.DataFrame(
            {'Year': date_range.year, 'Month': date_range.month, 'Estimación España (Junta Andalucia)': value})])

    # Set the value for June of each year to the updated value
    previsiones_mes.loc[previsiones_mes['Month'] == 6, 'Estimación España (Junta Andalucia)'] = previsiones_relevant[
        'Estimación España (Junta Andalucia)'].values

    # Reset the index for the new DataFrame
    previsiones_mes = previsiones_mes.reset_index(drop=True)
    # Create a datetime column with the format "YYYY-MM-01"
    previsiones_mes['DATE'] = pd.to_datetime(
        previsiones_mes['Year'].astype(str) + '-' + previsiones_mes['Month'].astype(str) + '-01', format='%Y-%m-%d')

    # Drop the Year and Month columns if you no longer need them
    previsiones_mes = previsiones_mes.drop(['Year', 'Month'], axis=1)

    # change name Estimacion España (Junta Andalucia)
    previsiones_mes['previsiones'] = previsiones_mes['Estimación España (Junta Andalucia)']
    previsiones_mes = previsiones_mes.drop(['Estimación España (Junta Andalucia)'], axis=1)

    # Yearly deltas
    deltas = {2010: 0.9,
              2011: 0.9,
              2012: 0.9,
              2013: 0.9,
              2014: 0.9,
              2015: 0.9,
              2016: 0.9,
              2017: 0.9,
              2018: 0.9,
              2019: 0.9,
              2020: 0.9,
              2021: 0.9,
              2022: 0.9,
              2023: 0.9,
              2024: 0.9}

    for i, row in previsiones_mes.iterrows():

        # Get year and month
        year = int(row['DATE'].year)
        month = int(row['DATE'].month)
        delta = deltas[year]
        # March - set new delta
        if month == 7:

            exponent = 0

        elif month >= 1 and month <= 6:
            exponent = month + 5
            delta = deltas[year - 1]
        # Calculate exponent
        else:
            exponent = month - 7

        try:
            # Exponential smoothing
            previsiones_mes.loc[i, 'previsiones_transformed'] = row['previsiones'] * (delta ** exponent)

        except:
            previsiones_mes.loc[i, 'previsiones_transformed'] = np.nan()

    previsiones_mes['previsiones_transformed'] = pd.to_numeric(previsiones_mes['previsiones_transformed'],
                                                               errors='coerce')

    return previsiones_mes


def dp_final_preproc(production_data, previsiones_mes):
    # Merge the two DataFrames on the "DATE" column with a left join
    merged_data = production_data.merge(previsiones_mes, on="DATE", how="left")

    # Create the "DP_PRODUCTION_HARVEST" column based on your criteria
    merged_data['DP_PRODUCTION_HARVEST'] = merged_data.apply(
        lambda row: row["previsiones_transformed"] if pd.notna(row["previsiones_transformed"]) and row[
            "DATE"].month not in [3, 4, 5, 6] else row["PRODUCTION_HARVEST_TRANSFORMED"],
        axis=1
    )

    # Drop unnecessary columns
    merged_data = merged_data.drop(["previsiones", "previsiones_transformed"], axis=1)

    # Store the "DP_PRODUCTION_HARVEST" column back in the production_data DataFrame
    production_data['DP_PRODUCTION_HARVEST'] = merged_data['DP_PRODUCTION_HARVEST']

    return production_data['DP_PRODUCTION_HARVEST']
