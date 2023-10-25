import pandas as pd
import numpy as np
import datetime
from src import config as conf

## SSCC historical files

def convert_date(date_str):
    # Map month names to their corresponding numbers
    month_mapping = {
        'ene': '01',
        'feb': '02',
        'mar': '03',
        'abr': '04',
        'may': '05',
        'jun': '06',
        'jul': '07',
        'ago': '08',
        'sep': '09',
        'oct': '10',
        'nov': '11',
        'dic': '12'
    }
    
    # Split the date string into month and year parts
    month_str, year_str = date_str.lower().split('/')
    
    # Get the corresponding month number from the mapping
    month_num = month_mapping[month_str]
    
    # Convert the year part to a 4-digit format
    year = '20' + year_str if len(year_str) == 2 else year_str
    
    # Format the date as 'MM-YYYY'
    formatted_date = f'{month_num}/{year}'
    
    return formatted_date

def select_columns(dataframe, columns):
    selected_data = pd.DataFrame()
    for column in columns:
        if column in dataframe.columns:
            selected_data[column] = dataframe[column]
        else:
            selected_data[column] = pd.Series(dtype = float)
    return selected_data

# READ AND PREPROCESS FUNCTIONS

def read_sscc(path): # Read old SSCC files
    df = pd.read_csv(path, skiprows = 4,  encoding='latin-1').T
    df.columns = df.iloc[0]
   
    return df[1:]

def read_sscc_old(path, dict_transf_cols): # Read old SSCC files
    
    df = pd.read_excel(path, sheet_name='datos')
    df = df.rename(columns = dict_transf_cols).set_index('Date')
   
    return df

def preprocess_sscc(df: pd.DataFrame(), cols: list): # Read old SSCC files
    df = df[df.index.notna()]
    df = select_columns(df, cols)

        # Preprocessing: Remove NANs, column casting

    df = df.replace('-',np.NAN)
    df = df.fillna(str(0))
    try:
        df = df.astype(str).apply(lambda x: x.str.replace(',','.'))
        df = df.astype(float)

        # Change the date format

        df = df.reset_index()
        df['index']= df['index'].apply(lambda x: convert_date(x))
        df['index'] = pd.to_datetime(df['index'])
    except:
        pass

    return df


def complete_sscc( files : dict, dict_transf_cols, cols : list): # Read old SSCC files
    df_sscc = pd.DataFrame()
    for k,v in files.items():
        if k == 'old':
            df = read_sscc_old(v,dict_transf_cols)
            df_old = preprocess_sscc(df,cols)
            df_old = df_old.set_index('Date').loc['2010-01-01':'2013-12-01']
            
        else:
            df = read_sscc(v)
            df = preprocess_sscc(df,cols)
            df_sscc = pd.concat([df_sscc, df], axis=0)
    
    df_sscc = df_sscc.set_index('index')
    df_sscc_full = pd.concat([df_old,df_sscc], axis=0)
    return df_sscc_full, df_sscc_full.index[-1]


def read_new_sscc(path: str):
    '''
    Excel file that contains all data regarding the SSCC: Liquicomun.xlsx
    Input:
        path: String. Path to the file
        temp: String. Temporality desired on the data. Defaults to Daily.
    Output:
        Dataframe with the correct columns and concepts of the SSCC
        Minimum date of the dataframe
        Maximum date of the dataframe
    '''
    df_l = pd.read_excel(path).set_index('fecha')
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df_l = df_l.select_dtypes(include=numerics)
    df_l = df_l.resample("MS").mean()

    df_l['Restricciones técnicas PDBF'] = df_l['RT3'] + df_l['CT3']
    df_l['Banda de regulación secundaria'] = df_l['BS3']
    df_l['Reserva de potencia adicional a subir'] = df_l['PS3']
    df_l['Restricciones técnicas en tiempo real'] = df_l['RT6']
    df_l['Incumplimiento de energía de balance'] = df_l['BALX']
    df_l['Saldo desvíos'] = df_l['EXD']
    df_l['Control del factor de potencia'] =  df_l['CFP']
    df_l['Saldo PO 14.6'] = df_l['IN7']
    
    return df_l[['Restricciones técnicas PDBF', 'Banda de regulación secundaria',
       'Reserva de potencia adicional a subir',
       'Restricciones técnicas en tiempo real','Saldo desvíos',
       'Incumplimiento de energía de balance','Control del factor de potencia', 'Saldo PO 14.6']],df_l.index[0],df_l.index[-1]

## SSCC commodities files

def read_commodities(commodities_dict: dict, start_date:str, max_date:str, temp: str = 'Weekly'):
    '''
    Function that takes a dict containing the name and path to each commodity and creates a joined dataframe with all data
    Input:
    Commodity dictionary containing as keys the name of each concept
        commodities_dict: Dictionary. Dictionary with the name and paths of the commodities
        start_date: String. Minimum date of the dataframe
        max_date: String. Maximum date of the dataframe
        temp: String. Temporality desired on the data. Defaults to Daily.
    Output: Dataframe.
        Dataframe with the all commodities data, resampled by the temp parameter
    '''
        
    # Set the date range

    d = pd.Period(max_date,freq='M').end_time.date()
    df_range= pd.DataFrame(pd.date_range(start_date, str(d), freq='D'))
    df_range.columns = ['Date']
    df_range = df_range.set_index('Date')
    for k,v in commodities_dict.items():

        
        df = pd.read_excel(v, sheet_name=k)

        if k == 'OMEL':
            df.columns = list(df.iloc[4]) 
            df = df.iloc[5:].reset_index(drop = True)
            df = df.rename(columns = {'Fecha correcta': 'Date','Media POOL':'POOL AVG'})
            df = df[['POOL AVG', 'Date']].set_index('Date')
            df = df.astype(float)

        if k == 'OMIP': 
            df.columns = df.iloc[3]
            df = df.iloc[5:,1:]
            df.columns = ['Date', *df.columns[1:]]
            df = df.set_index(df.columns[0])
            df = df.bfill() # Impute all NAN values with the next non-NAN 
            df = df.astype(float)

        if k == 'EURUSD' or k == 'BRENT' or k == 'EUA' or k == 'API2' or  k == 'TTF'  or k == 'MIBGAS PVB':
            df.columns = list(df.iloc[2])
            df = df.iloc[5:,1:].reset_index(drop = True)
            df.columns = ['Date', *df.columns[1:]]
            df = df.set_index('Date')
            df = df.bfill() # Impute all NAN values with the next non-NAN 
            df = df.astype(float)

        df_range = pd.concat([df_range, df], axis=0)
        
    if temp == 'Monthly':
        return df_range.resample("MS").mean().loc[:max_date]
    if temp == 'Weekly':
        return df_range.resample("W").mean().loc[:max_date]
    else:
        return  df_range.resample("MS").mean().loc[:max_date]


def read_ht(path:str, df:pd.DataFrame()):
    '''
    This funciton reads the HT file. It imports it and sum all columns considered to be HT: Carbon, Fuelgas, CC and Cogen.
    Also, for out-of-sample regressions it takes the mean of the previous 24 values.
    It creates 2 new columns in the passed Dataframe.
    Input:
        path: String. File path
        df: Dataframe. Dataframe to which the HT columns are attached
    Output:
        The passed Dataframe has 2 additional columns:
        HT: Real HT value for each month
        HT_f: 'Forward' values, they are the mean of the previous 24 values. Used in out_of_sample regression like hedging
    '''
    ht = pd.read_excel(path)
    ht1 = ht.set_index('Fecha').resample('MS').mean().fillna(0)
    ht1['HT'] =  ht1[ 'CARBON']+ ht1[ 'FUELGAS'] + ht1['CC'] + ht1['COGEN']
    ht1['HT_f'] = ht1['HT'].rolling(24).mean()
    ht1 = ht1[['HT','HT_f']]
    return df.join(ht1, how = 'left')



def df_preparation():
    '''
    Function that creates the objective dataframe
    Output:
        Dataframe with the all combined data
    '''
    df_sscc,start_date, max_date = read_new_sscc(conf.sscc_path) # Read SSCC
    df_sscc.index = pd.to_datetime(df_sscc.index)
    df_sscc['sscc_4'] = df_sscc[conf.sscc_4].sum(axis=1) # Define SSCC

    df_comm = read_commodities(conf.commodities_files,start_date = start_date,  max_date = max_date,temp = 'Monthly') # Read commodities
    df_comm.index = pd.to_datetime(df_comm.index)
    df1 = df_sscc[['sscc_4']].join(df_comm, how = 'left').fillna(0)
    try:
        df1 = read_ht(conf.path_ht,df1)
    except:
        pass
    return df1

# def procesar_excel_ht(file_path):
#     # Leer el archivo de Excel
#     xls = pd.ExcelFile(file_path)
    
#     # Crear un diccionario para almacenar los datos procesados
#     data = {}
    
#     # Procesar cada hoja del archivo
#     for sheet_name in xls.sheet_names:
#         # Leer la hoja actual en un DataFrame
#         df = pd.read_excel(file_path, sheet_name=sheet_name)
        
#         # Establecer la primera columna como índice
#         df.set_index(df.columns[0], inplace=True)
        
#         # Obtener las primeras 24 columnas como encabezados
#         encabezados = df.columns[:24]
        
#         # Calcular la media de las 24 columnas anteriores y asignarla a la columna 25
#         df['Media'] = df[encabezados].mean(axis=1)
        
#         # Almacenar los datos procesados en el diccionario
#         data[sheet_name] = df['Media']
    
#     # Crear un DataFrame final a partir del diccionario de datos procesados
#     df_final = pd.DataFrame(data)
    
#     # Devolver el DataFrame final
#     return df_final