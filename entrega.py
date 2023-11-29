# This is a sample Python script.
from src import yearly_functions as yf
import subprocess
import pandas as pd
from src import graphic_functions as gf
from docx import Document
from docx.shared import Inches
from PIL import Image
from io import BytesIO
from src import importing_data as imd
from src import Regression_Functions as rf
from src import Aux_functions as aux
import numpy as np
import statsmodels.api as sm
import os
import os
from src import Yearly_weight_transformation as year_trans
import tabula
import importlib # code to reload  lib
from unidecode import unidecode
#from src import importing_data as imd  # code to reload  lib
importlib.reload(rf)  # Reload the module # code to reload  lib

# test regression :
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from scipy.stats import kurtosis, skew
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import main


def back_testing_regression_rolling_OLD(df: pd.DataFrame(), x_cols, y_var,  initial_date: str = '2021-11-01',
                             final_date: str = '2023-09-01', signif: bool = False,
                             regr_type='Linear', num_variables: int = 4, window: int = 48, step_ahead: int = 12):
    """
    THIS FUNCTION IS USING REAL DATA FOR THE EVALUATION, NO PREDICTIONS OF REGRESSORS ARE MADE
    Rolling window hedging. It evaluates the hedging for the selected parameters, it outputs the cash flow for the selected period
    Input:
        df: Dataframe. It takes a df with the objective function, and all spot and forward columns. The hedging is done for all dates in the index.
        x_cols: List of Spot columns. These are the columns that will be selected in the regression
        y_var: String of the name of the objettve fnction (y)
        volumen: List. Volume of SSCC for each month
        initial_date: String. It is the first month in which the hedging is done. It must have the format: 'YYYY-MM-01'
        final_date: String. It is the last month in which the hedging is done. It must have the format: 'YYYY-MM-01'
        signif: Boolean. True to apply the Step Wise method in the regression. False to select all variales passed in x_cols. Only used in the regression.
        prima: Float. Selected adder (Prima). The prima is added in the final calulations as an addition to the base cash flow
        regr_type: String. Linear, for linear regression, or Huber, for robust regression. Only used in the regression.
        num_variables: Integer. Maximum number of variables to select while doing Step Wise regression, it is applied when signif is True. Only used in the regression.
        window: Integer. Training window in which the regression is calibrated
        step_ahead: Integer. Number of months ahead selected to perform hedging in. If step_ahead = 1 it means that the hedging is caculated for 1 month ahead
    Outputs: Dataframe.
        Dataframe with the final hedging calculations for each month, it has the following columns:
            vars: List of the variables used
            coefs: List of the coefficients used in the regression
            real_date: Date (Month) in which the hedging is done (m)
            forward_date: Date (Month) for which the hedging is done
            sscc_estimado: Estimated value of the SSCC of the months in the test dataframe
            sscc_spot_m1: Real value of the SSCC of the months in the test dataframe
            total_liquid: Sum of all liquidations done by the variables
            r2: R2 of the regression. It is the R2 in-sample.
            cash_flow_EUR: It is the Cash Flow resulting of the entire hedging process
            cash_flow_prima_EUR: It is the Cash Flow resulting of the entire hedging process plus an adder
            cash_flow_inicial: It is the Cash Flow resulting of the initial part of the process. It does not take into account the swap liquidations
            cash_flow_EUR_MWh: cash_flow_EUR divided by the volume (volumen)
            cash_flow_prima_EUR_MWh: cash_flow_prima_EUR_MWh divided by the volume (volumen)
            cash_flow_inicial_EUR_MWh: cash_flow_inicial divided by the volume (volumen)
            Cuadrados_Sin_C: It is used for the %Mejora metric, it is cash_flow_inicial_EUR_MWh ^2
            Cuadrados_Con_C: It is used for the %Mejora metric, it is cash_flow_EUR ^2

    """

    df_total = pd.DataFrame()

    # initial_date fecha inicial de prevision
    # final_date fecha final de prevision
    d = df.loc[initial_date:].index[0] - relativedelta(months=window)  # Date defiition
    df = df.loc[d:final_date]

    unique_dates = df.index.unique()  # List of dates of the DataFrame
    unique_dates1 = unique_dates[:-(window) - (step_ahead) + 1]  # row for the rolling windows

    for idx, i in enumerate(unique_dates1):

        ###### Date range interval for the train dataset, delimited by the window size

        date = i.date()  # Starting date of the training window
        date_max = unique_dates[
            idx + window - 1].date()  # End date of the training window. Month in which I perform the hedging


        df_out2 = df.loc[date: date_max]  # Dataframe with training window


        for step in range(1,
                          step_ahead + 1):  # Iteration on each month of the test window delimited by the step_ahead parameter
            # vol_index = step -1
            df_res = pd.DataFrame()

            date_max_step = unique_dates[idx + window + step - 1].date()  # Date of each step

            df_test = df.loc[date_max_step: date_max_step]

            ###### Regression with the forward values

            df_reg = rf.regression_OLD(df_out2, x_cols, y_var, df_test=df_test, reg_type=regr_type, significativas=signif,
                                n_vars=num_variables)

            ###### Calculate liquidations: LIQUIDATIONS

            # liquid = []
            vars = df_reg['vars'][0]
            coefs = df_reg['coef'][0]
            mape = df_reg['MAPE'][0]
            # new




            res_pred = np.concatenate([np.ravel(rr) for rr in df_reg['pred']])

            ###### CALCULATIONS

            df_res['vars'] = [vars]
            df_res['coefs'] = [coefs]

            df_res['real_date'] = date_max
            df_res['forward_date'] = date_max_step
            df_res['Prevision'] = res_pred[0]  # valor predicho
            df_res['Real'] = float(df_test[y_var][0])  # valor real
            # df_res['total_liquid'] = sum(liquid)
            df_res['r2'] = df_reg['r2'][0]
            df_res['mape'] = [mape]
            df_res['mes_vista'] = [step]

            df_total = pd.concat([df_total, df_res], axis=0)
    df_total['Mape_final'] = df_total.groupby('mes_vista')['mape'].transform('mean')
    return df_total.reset_index(drop=True)


# function that gets you the intercept for thesting
def rolling_regression_coefficient(df, target_variable, window_size):
    # calculate the regression for a rolling windows, TO be tested:
    #todo consider significativity and stepwise inside the rolling
    results = []
    n = len(df)
    df['DATE'] = df.index

    for end_date in range(window_size, n + 1):
        start_date = end_date - window_size

        # Select the data for the current window
        window_data = df.iloc[start_date:end_date]

        # Define the target variable and explanatory variables
        y = window_data[target_variable]
        X = window_data.drop(columns=[target_variable, 'DATE'])

        # Add a constant (intercept) to the model
        X = sm.add_constant(X)

        # Fit the linear regression model
        model = sm.OLS(y, X).fit()

        # Store the regression results
        r2 = model.rsquared
        p_values = model.pvalues
        coef = model.params
        result_dict = ({
            'Start Date': window_data['DATE'].iloc[0],
            'End Date': window_data['DATE'].iloc[-1],
            'R-squared': r2

        })
        # Add p-values to the dictionary
        for col, p_value, c in zip(X.columns, p_values, coef):
            result_dict[f'p-value, {col}'] = p_value
            result_dict[f'coefficient, {col}'] = c

        results.append(result_dict)
    # usage example :
    #target_variable = 'VIRGEN_EXTRA_EUR_kg'
    #window_size = 30

    # Run the rolling regression
    #rolling_results = rolling_regression(datos, target_variable, window_size)

    #rolling_results.to_excel('results_basic_model.xlsx')
    df.drop(columns=['DATE'],inplace = True)
    return pd.DataFrame(results)



mock = True
if mock == False:
    try:
        # Code that might raise an exception
        df_month = imd.load_data()

        # Other code that follows if no exception is raised
        print("No exception occurred")
    except ValueError  as e:
        # Handle the exception
        print(f"An exception occurred: {e}")
else:
    df_month = pd.read_excel("Output/Excel/df_month.xlsx")
    # in the import the field grossProduction_Spain_Soybean creates some problems with the scatterplot printing: investigate

if 'DATE' in df_month.columns:
    df_month  = df_month.set_index('DATE')
df_month_trans = year_trans.yearly_production_transform(path_df_month = "Output/Excel/df_month.xlsx")
if 'DATE' in df_month_trans.columns:
    df_month_trans = df_month_trans.set_index('DATE')

df_month = df_month_trans.copy()
df_month.columns


# declare data:
# basic model 3rd meeting :
# Modelo Basico

# pretreat variables :
#'PRODUCTION_HARVEST_REAL_EST'
#PRODUCTION_HARVEST_REAL_EST
df_month = df_month.fillna(method='ffill')

# original
df_month = df_month[['VIRGEN_EXTRA_EUR_kg', 'EXIS_INIC', 'IMPORTS', 'EXPORTS', 'INNER_CONS', 'PRODUCTION' ,'PRODUCTION_HARVEST', 'PRODUCTION_HARVEST_LAST_YEAR', 'PRODUCTION_HARVEST_2_YEARS', 'INTERNAL_DEMAND', 'TOTAL_CONS']].copy()


# production + pevision
#df_month = df_month[['VIRGEN_EXTRA_EUR_kg', 'EXIS_INIC', 'IMPORTS', 'EXPORTS', 'INNER_CONS', 'PRODUCTION' ,'PRODUCTION_HARVEST_REAL_EST', 'PRODUCTION_HARVEST_LAST_YEAR', 'PRODUCTION_HARVEST_2_YEARS', 'INTERNAL_DEMAND', 'TOTAL_CONS']].copy()
# peña model#
#df_month = df_month[['VIRGEN_EXTRA_EUR_kg', 'EXIS_INIC', 'IMPORTS', 'EXPORTS', 'INNER_CONS', 'PRODUCTION' ,'PRODUCTION_HARVEST_REAL_EST', 'PRODUCTION_HARVEST_LAST_YEAR', 'PRODUCTION_HARVEST_2_YEARS', 'INTERNAL_DEMAND', 'TOTAL_CONS','DP_PRODUCTION_HARVEST']].copy()

#df_pdf = imd.import_pdf_data()
#df_month_pdf = imd.include_pdf_data(df_month)
#df_month_pdf.columns
#df_month_pdf = rf.eliminate_rows_from_date(df_month_pdf, '2005-10-01')


#df_month.to_excel("prueba_Peña.xlsx")

basic_model_df = df_month.copy()
basic_model_df['EXPORTS_LAG15'] = basic_model_df['EXPORTS'].shift(15)
basic_model_df['INTERNAL_DEMAND_LAG_13'] = basic_model_df['INTERNAL_DEMAND'].shift(13)
basic_model_df['TOTAL_CONS_LAG_12'] = basic_model_df['TOTAL_CONS'].shift(12)
basic_model_df['TOTAL_CONS_LAG_13'] = basic_model_df['TOTAL_CONS'].shift(12)
basic_model_df['PRODUCTION_LAG_21'] = basic_model_df['PRODUCTION'].shift(21)
basic_model_df['TOTAL_CONS_LAG_13'] = basic_model_df['TOTAL_CONS'].shift(13)
basic_model_df['EXIS_INIC_18'] = basic_model_df['EXIS_INIC'].shift(18)
basic_model_df['PRODUCTION_HARVEST_LAG_8'] = basic_model_df['PRODUCTION_HARVEST'].shift(8)
#basic_model_df['PRODUCTION_LAG_21'] = basic_model_df['PRODUCTION'].shift(21)
#basic_model_df['HARVEST_REAL_EST_LAG_12'] = basic_model_df['PRODUCTION_HARVEST_REAL_EST'].shift(12)
#basic_model_df.drop(columns=['EXTERNAL_DEMAND'], inplace=True)
# danel peña variables
#basic_model_df['DP_PRODUCTION_HARVEST_LAG_15'] = basic_model_df['DP_PRODUCTION_HARVEST'].shift(15)

basic_model_df_man = basic_model_df[
    ['VIRGEN_EXTRA_EUR_kg', 'IMPORTS', 'INNER_CONS', 'TOTAL_CONS_LAG_12', 'EXPORTS_LAG15', 'TOTAL_CONS',
     'INTERNAL_DEMAND_LAG_13', 'EXIS_INIC', 'INTERNAL_DEMAND', 'PRODUCTION_HARVEST_REAL_EST','HARVEST_REAL_EST_LAG_12']]


# original 3a reu:
basic_model_df_man = basic_model_df[
    ['VIRGEN_EXTRA_EUR_kg', 'IMPORTS', 'INNER_CONS', 'TOTAL_CONS_LAG_12', 'EXPORTS_LAG15', 'TOTAL_CONS',
     'INTERNAL_DEMAND_LAG_13', 'EXIS_INIC', 'INTERNAL_DEMAND','PRODUCTION_HARVEST_LAG_8','PRODUCTION']]


basic_model_df_man.columns
#peña:
#basic_model_df_man = basic_model_df[
#    ['VIRGEN_EXTRA_EUR_kg', 'IMPORTS', 'INNER_CONS', 'TOTAL_CONS_LAG_12', 'EXPORTS_LAG15', 'TOTAL_CONS',
#     'INTERNAL_DEMAND_LAG_13', 'EXIS_INIC', 'INTERNAL_DEMAND', 'DP_PRODUCTION_HARVEST','DP_PRODUCTION_HARVEST_LAG_15']]



#df_pdf = imd.import_pdf_data()
basic_model_df_man = imd.include_pdf_data(basic_model_df_man)

basic_model_df_man = main.include_meteo_variables(basic_model_df_man)
for col in basic_model_df_man.columns:
    if 'Jaen' in col and 'Precip' in col:
        basic_model_df_man[col+'_Lag_12'] = basic_model_df_man[col].shift(12)
basic_model_df_man.drop(columns=list(basic_model_df_man.filter(like='Cordoba').columns), inplace=True)

basic_model_df_man




basic_model_df_man = rf.eliminate_rows_from_date(basic_model_df_man, '2005-10-01')

#basic_model_df_man.to_excel("DF_test_Backtesting.xlsx")

basic_model_df_man[basic_model_df_man.isnull().any(axis=1)]
basic_model_df_man.columns

var_pdf = ['Consumo Total_TOTAL  A',
       'Consumo Total_TOTAL B', 'Consumo Total_TOTAL MONDIAL WORLD',
       'Consumo UE_TOTAL  A)', 'Consumo UE_TOTAL B)', 'Consumo UE_TOTAL A + B',
       'Exportacion Total_TOTAL A','Exportacion Total_TOTAL B',
       'Exportacion Total_TOTAL MONDIAL WORLD', 'Exportacion UE_TOTAL  A)',
       'Exportacion UE_TOTAL B)', 'Exportacion UE_TOTAL A + B',
       'Importacion Total_TOTAL  A', 'Importacion Total_TOTAL  B',
       'Importacion Total_TOTAL MONDIAL WORLD', 'Importacion UE_TOTAL  A)',
       'Importacion UE_TOTAL B)', 'Importacion UE_TOTAL A + B',
       'Produccion Total_TOTAL  A', 'Produccion Total_TOTAL B','Produccion UE_TOTAL A + B',
       'Produccion Total_TOTAL MONDIAL WORLD', 'Produccion UE_TOTAL  A)']


# model to drop 11 nov
var_pdf = [ 'Consumo Total_TOTAL MONDIAL WORLD',
       'Consumo UE_TOTAL  A)', 'Consumo UE_TOTAL B)', 'Consumo UE_TOTAL A + B',
       'Exportacion Total_TOTAL A','Exportacion Total_TOTAL B',
       'Exportacion Total_TOTAL MONDIAL WORLD', 'Exportacion UE_TOTAL  A)',
       'Exportacion UE_TOTAL B)', 'Exportacion UE_TOTAL A + B',
       'Importacion Total_TOTAL  A', 'Importacion Total_TOTAL  B',
       'Importacion Total_TOTAL MONDIAL WORLD', 'Importacion UE_TOTAL  A)',
       'Importacion UE_TOTAL B)', 'Importacion UE_TOTAL A + B',
       'Produccion Total_TOTAL  A', 'Produccion Total_TOTAL B', 'Produccion UE_TOTAL A + B',
       'Produccion Total_TOTAL MONDIAL WORLD', 'Produccion UE_TOTAL  A)']

# dropping peña lag

#basic_model_df_man = basic_model_df_man.drop(columns= var_pdf)
basic_model_df_man.columns
#basic regression for control :
target_variable = 'VIRGEN_EXTRA_EUR_kg'
y = basic_model_df_man[[target_variable]].copy()
X = basic_model_df_man.drop(columns=[target_variable ]).copy()
#X = basic_model_df_man[['Consumo Total_TOTAL  A','Consumo Total_TOTAL B']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
#basic_model_df_man= basic_model_df_man.drop(columns=['Exportacion Total_TOTAL B'])
basic_model_df_man_BCK =   basic_model_df_man.copy()
basic_model_df_man = basic_model_df_man_BCK
basic_model_df_man.columns
#basic_model_df_man.to_excel("pdf_all.xlsx")
basic_model_df_man.columns
#run all
df_result1= rf.back_testing_regression_rolling_OLD(basic_model_df_man, basic_model_df_man.drop(columns=['VIRGEN_EXTRA_EUR_kg']), 'VIRGEN_EXTRA_EUR_kg',  signif= False, window = 30,initial_date= '2019-10-01',final_date = '2023-09-01')
df_result1.Mape_final.mean()
df_result1.to_excel("Prova.xlsx")


basic_model_df_man
ls_ventanas = [24, 30, 36, 40, 50]
for window in ls_ventanas:
    result_rolling = rolling_regression_coefficient(basic_model_df_man, 'VIRGEN_EXTRA_EUR_kg', window)
    result_rolling = aux.add_average_row(result_rolling)
    result_rolling.to_excel(f"Output/Rolling_Regression/Rolling{window}.xlsx")


#main.print_doc_scatter_ouliers(basic_model_df_man)
basic_model_df_man.columns
col_to_drop = ['Consumo Total_TOTAL  A',
       'Consumo Total_TOTAL B', 'Consumo Total_TOTAL MONDIAL WORLD',
       'Consumo UE_TOTAL  A)', 'Consumo UE_TOTAL B)', 'Consumo UE_TOTAL A + B', # linea canceling
       'Exportacion Total_TOTAL A',
       'Exportacion Total_TOTAL MONDIAL WORLD', 'Exportacion UE_TOTAL  A)',      # eliminar todas export
       'Exportacion UE_TOTAL B)', 'Exportacion UE_TOTAL A + B',
       'Importacion Total_TOTAL  A', 'Importacion Total_TOTAL  B',
       'Importacion Total_TOTAL MONDIAL WORLD', 'Importacion UE_TOTAL  A)',      #usar total mundial import
       'Importacion UE_TOTAL B)', 'Importacion UE_TOTAL A + B',
       'Produccion Total_TOTAL  A', 'Produccion Total_TOTAL B',
       'Produccion Total_TOTAL MONDIAL WORLD', 'Produccion UE_TOTAL  A)',
       'Produccion UE_TOTAL A + B']

col_to_drop_few = [  'Consumo UE_TOTAL B)', 'Consumo UE_TOTAL A + B', 'Consumo Total_TOTAL B', # linea canceling
       'Exportacion Total_TOTAL A', #'Exportacion Total_TOTAL B',
        # eliminar todas export,
       'Exportacion UE_TOTAL B)', 'Exportacion UE_TOTAL A + B',
        'Importacion Total_TOTAL  B',
            #usar total mundial import
       'Importacion UE_TOTAL B)', 'Importacion UE_TOTAL A + B',
        'Produccion Total_TOTAL B',
       'Produccion Total_TOTAL MONDIAL WORLD', 'Produccion UE_TOTAL A + B',

 ]

if 'Exportacion Total_TOTAL B' in basic_model_df_man.columns:
    print("azz")
else:
    print("ok")
var_pdf_a = var_pdf
for col in var_pdf:
    if 'B' in col:
        var_pdf_a.remove(col)
basic_model_df_man.columns

basic_model_df_man_nopdf = basic_model_df_man.drop(columns=var_pdf)
# for to test pdf_var
i = 0
for col in var_pdf:
    i = i +1
    basic_model_df_man_try = basic_model_df_man_nopdf.copy()
    basic_model_df_man_try[col] = basic_model_df_man[col].copy()
    if len(var_pdf) > i+1:
        basic_model_df_man_try[var_pdf[i+1]] = basic_model_df_man[var_pdf[i+1]].copy()
       # basic_model_df_man_try[var_pdf[i + 2]] = basic_model_df_man[var_pdf[i + 2]].copy()


    df_result1 = rf.back_testing_regression_rolling_OLD(basic_model_df_man_try,
                                                        basic_model_df_man_try.drop(columns=['VIRGEN_EXTRA_EUR_kg']),
                                                        'VIRGEN_EXTRA_EUR_kg', signif=False, window=30)

    if df_result1.Mape_final.mean() > 100:
        print(col,i, df_result1.Mape_final.mean())



basic_model_df_man_try = basic_model_df_man.drop(columns= col_to_drop_few)
basic_model_df_man.columns
df_result1= rf.back_testing_regression_rolling_OLD(basic_model_df_man_try, basic_model_df_man_try.drop(columns=['VIRGEN_EXTRA_EUR_kg']), 'VIRGEN_EXTRA_EUR_kg',  signif= False, window = 30)
df_result1.Mape_final.mean()
df_result1.to_excel("Output/Backtest_Regression/Rolling_Model30.xlsx")
df_result1.Mape_final.mean()

df_result1= rf.back_testing_regression_expanding_OLD(basic_model_df_man, basic_model_df_man.drop(columns=['VIRGEN_EXTRA_EUR_kg']), 'VIRGEN_EXTRA_EUR_kg',  signif= False ) # reg_type = 'Huber
df_result1.to_excel("Output/Backtest_Regression/Expanding_Basic_Model.xlsx")



#rolling correlation

ls_ventanas= [24,30,36,40,50]
for window in ls_ventanas:
    result_rolling = rf.rolling_regression(basic_model_df_man, 'VIRGEN_EXTRA_EUR_kg',window)
    result_rolling= aux.add_average_row(result_rolling)
    result_rolling.to_excel(f"Output/Rolling_Regression/Rolling{window}.xlsx")

#Regresión Robusta:
target_variable = 'VIRGEN_EXTRA_EUR_kg'
y = basic_model_df_man[[target_variable]].copy()
X = basic_model_df_man.drop(columns=[target_variable]).copy()
X = sm.add_constant(X)
model = HuberRegressor()
result = model.fit(X, y)
coeficientes = model.coef_
intercept = model.intercept_

data = {'Column': X.columns, 'Coefficient': coeficientes}
df_robust = pd.DataFrame(data)

# Save DataFrame to a file (e.g., CSV)
df_robust.to_excel('Output/coefficients_robust_all_13_11.xlsx', index=False)
coeficientes
# prediction expanding robust:
df_result1= rf.back_testing_regression_expanding_OLD(basic_model_df_man, basic_model_df_man.drop(columns=['VIRGEN_EXTRA_EUR_kg']), 'VIRGEN_EXTRA_EUR_kg',  signif= False, regr_type = 'Huber')
df_result1.to_excel("Output/Backtest_Regression/Expanding_Basic_Model_robust.xlsx")

len(coeficientes)
r2 = r2_score(y, y_pred_in_sample)
for i in range(0,len(coeficientes)):
    print(X.columns[i], str(coeficientes[i]))
