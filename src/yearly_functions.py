import pandas as pd
from src import graphic_functions as gf
from src import Aux_functions as aux
from src import importing_data as imd

# this module is for the analysis of the correlations between yearly variables
# and the mountly target variable

def transpose_target_variable(df1,target_variable='VIRGEN_EXTRA_EUR_kg'):
    # trasposing the olive oil prices using years as rows and months as columns
    df = pd.DataFrame()
    df[target_variable] = df1[target_variable].copy()

    # Convert the 'DATE' column to datetime
    df['DATE'] = pd.to_datetime(df1.index)

    # Extract the year from the 'DATE' column
    df['Year'] = df['DATE'].dt.year

    # Create a new DataFrame for the yearly data
    target_transposed = df.groupby('Year')[target_variable].apply(list).reset_index()

    # Rename the 'Value' column to match the corresponding year
    target_transposed.rename(columns={target_variable: 'YearlyValue'}, inplace=True)

    # Create separate columns for each month's value
    months = range(1, 13)
    if target_variable == 'VIRGEN_EXTRA_EUR_kg':
        namevar= 'VIRGEN_EXTRA'
    else:
        namevar = target_variable
    for month in months:

        target_transposed[str(namevar)+"_"+str(month)] = target_transposed['YearlyValue'].apply(
            lambda x: x[month - 1] if month <= len(x) else None)

    # Drop the 'YearlyValue' column as it's no longer needed
    target_transposed.drop(columns='YearlyValue', inplace=True)

    # Set the 'Year' column as the index
    target_transposed.set_index('Year', inplace=True)

    # Print the resulting DataFrame
    return target_transposed

def select_annual_variable_from_dic (df_origin, path_dic= 'Datos/Diccionario_Variables_02_10_2023.xlsx'):
    df_dictionary = pd.read_excel('Datos/Diccionario_Variables_02_10_2023.xlsx')
    df_dictionary = df_dictionary[df_dictionary['Granularidad'] == 'anual']
    #df_origin.columns = df_origin.columns.str.replace('_seed', '') # this line has been put in the import of data delete after testing
    # making rules for the shifting
    col_selected = [value.strip() for value in df_dictionary['Nombre']]
    df_out = df_origin[col_selected].copy()
    return df_out


# aggregating once again data per year
def aggregate_mountly_data(df1):
    df = df1.copy()
    df['Year'] = df.index.year
    df_fin = pd.DataFrame()
    for col in df.columns:
            if 'area' in col or 'HARVEST' or 'Estimación España' in col:
                df_fin[col] = df.groupby('Year')[col].mean()
            elif col == 'Year':
                df_fin[col] = df.groupby('Year')[col].mean().astype(int)
            else:
                df_fin[col] = df.groupby('Year')[col].sum()

    df_fin.reset_index(drop=True, inplace=True)
    df_fin.set_index('Year',inplace = True)
    return df_fin


def preprocess_prevision_data(df):
    average_prod_post_march, prec = aux.compute_perc_harvest_post_march(df)
    average_prod_post_march = 1 - (average_prod_post_march / 100)
    df_prev = imd.load_and_transform_previsiones()
    # putting the penalty over the prevision for confronting the data of the production since the harvest post march is not considered
    df_prev['HARVEST_FORECAST_JUNTA_ANDALUCIA'] = df_prev['Estimación España (Junta Andalucia)'].copy()
    df_prev = df_prev.apply(lambda x: x * average_prod_post_march if 'Año' not in x.name  and 'HARVEST_FORECAST_JUNTA_ANDALUCIA' not in x.name else x)
    df_prev = df_prev[['Año Inicio', 'Estimación España (Junta Andalucia)','HARVEST_FORECAST_JUNTA_ANDALUCIA']]
    df_prev.set_index('Año Inicio', inplace=True)
    return df_prev

def revert_shifting_yearly (df, month_start = 6):
    #  this function makes create a new index starting from the month shifted
    # and ordinate the columns. THE shift_used parameter is the value of the 1st mounth we are starting counting
    # from as it was january, it's one value more than the shifting used in the shifting function

    range_start = abs(month_start)
    range_stop = range_start + 12
    df['NewIndex'] = [i if i < len(df) + 1 else i % len(df) for i in
                             range(range_start, range_stop)]
    df.index = [f'{str(i).rsplit("_", 1)[0]}_{new_index}' for i, new_index in
                       zip(df.index, df['NewIndex'])]
    df = df.sort_values(by='NewIndex')
    return df

#def compute_montly_weight(df_month,'VIRGEN_EXTRA_EUR_kg'):

def calc_correlations_yearly (df_in, target_transposed):
    #
    df_out = pd.DataFrame()

    for col in df_in:
        df_temp = pd.DataFrame()
        df_temp = df_in[col]
        df_precorr = pd.merge(df_temp, target_transposed, left_index=True, right_index=True, how='left')
        target_transposed = target_transposed.fillna(method='ffill').fillna(method='bfill')
        # loop to show the graph of cross correlation function
        #for var in df_precorr :
        #    if var != df_precorr.iloc[:, 0].name:
        #        gf.cross_correlation_variable(df_precorr,df_precorr.iloc[:, 0].name,var, 24)

        df_corr = df_precorr.corr()
        # eliminating the 1st row that give us the information the correlation of the variable with itself
        df_corr = df_corr.iloc[1:, :]
        # selecting just the 1st column
        df_corr = df_corr.filter(like=col)
        df_out[col + '_weight'] = df_corr[col]

    return df_out


def revert_shifting_yearly (df, month_start = 7):
    #  this function makes create a new index starting from the month shifted
    # and ordinate the columns. THE shift_used parameter is the value of the 1st mounth we are starting counting
    # from as it was january, it's one value more than the shifting used in the shifting function

    range_start = abs(month_start)
    range_stop = range_start + 12
    df['NewIndex'] = [i if i < len(df) + 1 else i % len(df) for i in
                             range(range_start, range_stop)]
    df.index = [f'{str(i).rsplit("_", 1)[0]}_{new_index}' for i, new_index in
                       zip(df.index, df['NewIndex'])]
    df = df.sort_values(by='NewIndex')
    df.drop(columns =['NewIndex'],inplace=True)
    return df


def calc_correlations_yearly_shifted(df_original, x_var: str, shift, target_variable):
    # the shift is supposed to be the initial shift applied to the var
    # for instance if we are considering the production from March the shift applied to the yearly production is 2
    df_target = df_original.copy()
    # remove sequent line after test replace with :  #
    #df_target[f"{target_variable}_LAG_{str(abs(shift))}"] = df_target[target_variable].shift(-shift)
    df_target[target_variable] = df_target[target_variable].shift(-shift)
    target_transposed_shift = transpose_target_variable(df_target, target_variable)
    df_yearly_x = df_original[[x_var]].copy()
    # selecting all the years from 2005
    df_yearly_x = aggregate_mountly_data(df_yearly_x)
    df_yearly_x = df_yearly_x[df_yearly_x.index > 2004]
    first_non_zero_index = (df_yearly_x[x_var] != 0).idxmax()
    df_yearly_x = df_yearly_x.loc[first_non_zero_index:]
    df_corr_shifted = calc_correlations_yearly(df_yearly_x, target_transposed_shift)
    df_corr_reverted = revert_shifting_yearly(df_corr_shifted, month_start=abs(shift) + 1)
    if 'PRODUCTION_HARVEST_ORIGINAL_weight' in df_corr_reverted.columns:
        df_corr_reverted.rename(columns = {'PRODUCTION_HARVEST_ORIGINAL_weight':'PRODUCTION_HARVEST_weight'},inplace = True)

    return df_corr_reverted

