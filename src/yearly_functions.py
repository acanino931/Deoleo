import pandas as pd

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

        target_transposed[str(namevar)+str(month)] = target_transposed['YearlyValue'].apply(
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
            if 'area' in col:
                df_fin[col] = df.groupby('Year')[col].mean()
            elif col == 'Year':
                df_fin[col] = df.groupby('Year')[col].mean().astype(int)
            else:
                df_fin[col] = df.groupby('Year')[col].sum()

    df_fin.reset_index(drop=True, inplace=True)
    df_fin.set_index('Year',inplace = True)
    return df_fin


def calc_correlations_yearly (df_in, target_transposed):
    #
    df_out = pd.DataFrame()

    for col in df_in:
        df_temp = pd.DataFrame()
        df_temp = df_in[col]
        df_precorr = pd.merge(df_temp, target_transposed, left_index=True, right_index=True, how='left')
        target_transposed = target_transposed.fillna(method='ffill').fillna(method='bfill')
        # loop to show the graph of cross correlation function
        # for var in df_precorr :
        #    if var != df_precorr.iloc[:, 0].name:
        #        cross_correlation_variable(df_precorr,df_precorr.iloc[:, 0].name,var, 24)
        df_corr = df_precorr.corr()
        # eliminating the 1st row that give us the information the correlation of the variable with itself
        df_corr = df_corr.iloc[1:, :]
        # selecting just the 1st column
        df_corr = df_corr.filter(like=col)
        df_out[col + '_weight'] = df_corr[col]

    return df_out