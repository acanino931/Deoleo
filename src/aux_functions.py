import pandas as pd
import numpy as np
import datetime
from sklearn.utils import resample
from src import config as conf

from dateutil.relativedelta import relativedelta




##Funciones del analisis

def rolling_vars(df: pd.DataFrame(), columns: list, window: int, calc_mean = True, calc_std=True, calc_cv = True):
    df1 = df.copy()
    for c in columns:
        if calc_mean :
            if 'MA_'  not in c:
                df1[f'MA_{c}'] = df1[c].rolling(window).mean() 
        if calc_std:
            if 'std_'  not in c:
                df1[f'std_{c}'] = df1[c].rolling(window).std()
        if calc_cv:
            if 'cv_'  not in c:
                df1[f'cv_{c}'] = df1[c].rolling(window).std()/df1[c].rolling(window).mean()
        else:
            pass

    return df1.dropna()


def rolling_corr(variable1, variable2, ventana):
    '''
    It computes the rolling correlation between the last 24 samples of the 2 variables. The window is computed in the argument 'window'
    
    '''
    roll_corr = variable1.rolling(window=ventana).corr(variable2)
    # print(roll_corr)
    return roll_corr

def rolling_corr_post(variable1, variable2, ventana):
    '''
    It computes the rolling correlation between the next 24 samples of the 2 variables. The window is computed in the argument 'window'
    
    '''
    roll_corr = variable1.rolling(window=ventana).corr(variable2)
    roll_corr_posterior = roll_corr.shift(-ventana)
    return roll_corr_posterior


def add_new_vars(df: pd.DataFrame(),spot_cols:list):
    '''
    It takes a df of monthly data
    It computes the synthetic variables of the original vars.
    Brent601: It is the average of the past 6 months
    Brent603:It is the average of the last 2 quarters before the current one

    '''
    tf_dark = 8.7  # FIXED TERMS
    tf_clean = 6.7 # FIXED TERMS

    df['Coste_carbon'] = df['TRAPI2Mc1']/df['EUR=']*(860/6000)*(1/0.35)+1*df['CFI2c5'] + tf_clean
    df['Coste_gas_brent'] = (((0.35*df['BRT-'])+5)/0.5)/df['EUR=']+0.365*df['CFI2c5'] + tf_dark
    df['Spread_carbon'] = df['POOL AVG'] - df['Coste_carbon']
    df['Spread_brent'] = df['POOL AVG'] - df['Coste_gas_brent']
    df['Brent-EURUSD'] = df['BRT-']/df['EUR=']
    for var in spot_cols:
        
        df1 = df.rename_axis('index').reset_index() 
        df1 = df1.groupby(df1['index'].dt.to_period('Q')).mean()
        df1[f'{var}603'] = df1[var].rolling(window=2, min_periods=1).mean().shift(1)
        df1 = df1.resample('M', label='right').ffill()
        df1.index = df1.index.to_timestamp(how = 'S')

        df2 = df.rename_axis('index').reset_index() 
        df2 = df2.groupby(df2['index'].dt.to_period('M')).mean()
        df2[f'{var}601'] = df2[var].rolling(window=6, min_periods=1).mean().shift(1)
        df2.index = df2.index.to_timestamp(how = 'S')


        df = df.join(df2[[f'{var}601']], how ='left')
        df = df.join(df1[[f'{var}603']], how ='left')

    # df[f'{var_brent}601-EURUSD'] = df[f'{var_brent}601']/df['EUR=']
    # df[f'{var_brent}603-EURUSD'] = df[f'{var_brent}603']/df['EUR=']

    return df.ffill()


def var_retardos(df:pd.DataFrame(), cols:list, num_desfases:int,temp:str = 'Monthly'):
    df1 = df.reset_index() 
    for c in cols:
        for desfase in range(num_desfases):
            if '_t_' not in c and '_log' not in c:
                d = desfase+1
                c_t = f'{c}_t_{d}'

                if temp == 'Monthly':
                    df2 = df1.copy()
                    df2 = df2.set_index(df1.iloc[:, 0])
                    df2[c_t] = df1.groupby(df1.iloc[:, 0])[c].mean().shift(d)
                    df2 = df2.set_index(df1.iloc[:, 0])

                    df1 = df1.set_index(df1.iloc[:, 0], drop = True).join(df2[c_t], how ='left')
                else:

                    df2 = df1.groupby(df1.iloc[:, 0].dt.to_period('M')).mean()
                    df2[c_t] = df2[c].rolling(window=1, min_periods=1).mean().shift(d)
                    df2 = df2.resample('D', label='right').ffill()
                    df2.index = df2.index.to_timestamp(how = 'S')
                    df1 = df1.set_index(df1.iloc[:, 0], drop = True).join(df2[c_t], how ='left')
                
    return df1.iloc[:,1:]

def create_log_vars(df:pd.DataFrame()):
    for c in df.columns:
        try:
            df[f'{c}_log'] = np.log(df[f'{c}'])
        except:
            pass
    return df.fillna(0)

def create_df_full(df:pd.DataFrame(), num_desfases:int = 1,new_vars:bool = True, rolling:bool = False, temp:str = 'Monthly', start_year:int = 2018):
    df1 = df.copy()
    if new_vars:
        df1 = add_new_vars(df1,conf.spot_vars)
    df1 = create_log_vars(df1)
    if num_desfases != 0:
        df1 = var_retardos(df1, cols = df1.columns, num_desfases = num_desfases, temp = temp)
    if rolling and temp == 'Monthly':
        df1 = rolling_vars(df1, df1.columns, 24,calc_mean = True, calc_std = False, calc_cv = False)
    elif rolling and temp == 'Daily':
        df1 = rolling_vars(df1, df1.columns, 365,calc_mean = True, calc_std = False, calc_cv = False)
    else:
        pass
    return df1.loc[f'{start_year}-01-01':]



def df_oversampled_weekly(df_weekly: pd.DataFrame()):
    fecha = []
    m = []
    df_os = pd.DataFrame()
    for mes in df_weekly.resample('MS'):
        # Obtiene el mes y el año correspondientes
        mes_actual = mes[0].month
        año_actual = mes[0].year
        
        # Itera sobre las semanas del mes actual
        for i in range(len(mes[1])):
            # Crea una copia del mes actual
            mes_copy = mes[1].copy()
            

            # Elimina la semana i del mes
            mes_copy = mes_copy.drop(mes_copy.index[i])
            
            # Calcula la media del mes sin la semana i
            media = mes_copy.mean().dropna()
            fecha.append(mes[0])
            # Agrega la media al DataFrame de valores oversampled
            m.append(media)

            df_os = pd.DataFrame(m, index = fecha).dropna()
    return df_os


def df_bootstrap(df_daily: pd.DataFrame, seed:int=0):
    '''
    It recieves a daily df and created various monthly df resampled
    
    '''
    df_bt = pd.DataFrame()

    for mes, datos_diarios in df_daily.resample('MS'):
        # Aplicar resample con sklearn para obtener los valores oversampled
        datos_bt = pd.DataFrame(resample(datos_diarios, replace = True, random_state = seed))

        datos_bt.index = datos_diarios.index

        # Calcular la media de los datos oversampled
        media_bt = datos_bt.mean().dropna()

        # Agregar la media al DataFrame de valores oversampled
        df_bt = pd.concat([df_bt, pd.DataFrame(media_bt).T], axis=0)

    df_bt.index.name = 'fecha'
    return df_bt



##Funciones de las coberturas

def create_forward_df(df, cols:list, date_ini,date_end):
    '''
    It extracts the forwards based on dicts where the cols are mapped for each month.
    Date_ini is the the date in which you are taking the forwards.
    Date_end: It is the last forward date (Maximum 12 months)
    '''
    
    dates = pd.date_range(start=date_ini, end=date_end, freq='MS')

    dict_keys = {key: conf.forward_1m[key] for key in cols}
    df2 = pd.DataFrame(columns = dict_keys.keys(),index=dates)
    df3 = pd.DataFrame()

    for idx,i in zip(range(len(dates) - 1), dates):
                  
        if idx < 12:
            cols_dict = f'conf.forward_{idx+1}m'

            df1 = df.loc[date_ini]
            df1 = df1[eval(cols_dict).values()]
            
            df1 = pd.DataFrame(df1).T
            df1.index = df1.index + pd.DateOffset(months=1+idx) # It takes the first forward month and it is indexed correctly
            for key, value in eval(cols_dict).items():
                df2[key] = df1[value]
        
        else: # If the date range is more than a year, for each month it takes the 1Y forward
            cols_dict = f'conf.forward_{12}m'
            df1 = df.loc[date_ini]
            df1 = df1[eval(cols_dict).values()]
        
            df1 = pd.DataFrame(df1).T
            df1.index = df1.index + pd.DateOffset(months=1+idx) # It takes the first forward month and it is indexed correctly
            for key, value in eval(cols_dict).items():
                df2[key] = df1[value]
                
        df3 = pd.concat([df3,df2]).dropna()
    return df3

def diferencia_trimestres(df, fecha_fija):
    df1 = df.copy()
    # Calcular el trimestre de la fecha fija
    trimestre_fijo = (fecha_fija.month - 1) // 3 + 1
    año_fijo = fecha_fija.year

    # Calcular la diferencia de trimestres para cada valor del índice
    df1['q'] = ((df1.index.year - año_fijo) * 4) + (df1.index.quarter - trimestre_fijo)

    return df1

def create_forward_df_60(df,df_forward, vars_60:list, date_ini,date_end):

    d = df.loc[date_ini:date_ini].index[0]  - relativedelta(months = 9)
    d1 = df.loc[d:date_ini].append(df_forward.loc[date_ini:date_end])

    for var_brent in vars_60:
        if '603' in var_brent:
            var_brent1 = var_brent[:-3]

            df1 = d1[var_brent1].rename_axis('index').reset_index() 
            df1 = df1.groupby(df1['index'].dt.to_period('Q')).mean()
            df1[f'{var_brent1}603'] = df1[var_brent1].rolling(window=2, min_periods=1).mean().shift(1)
            df1 = df1.resample('M', label='right').ffill()
            df1.index = df1.index.to_timestamp(how = 'S')
            df_forward = df_forward.join(df1[[f'{var_brent1}603']], how ='left')


        if '601' in var_brent:
            # print(var_brent)
            var_brent1 = var_brent[:-3]
            # print(var_brent1)
            df2 = d1[var_brent1].rename_axis('index').reset_index() 
            df2 = df2.groupby(df2['index'].dt.to_period('M')).mean()
            df2[f'{var_brent1}601'] = df2[var_brent1].rolling(window=6, min_periods=1).mean().shift(1)
            df2.index = df2.index.to_timestamp(how = 'S')
            
            df_forward = df_forward.join(df2[[f'{var_brent1}601']], how ='left')

    return df_forward