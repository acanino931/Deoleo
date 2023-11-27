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
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import numpy as np


mock = False
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
df_month['IMPORTS_LAG_9'] = df_month['IMPORTS'].shift(9)
df_month['EXPORTS_LAG_12'] = df_month['EXPORTS'].shift(12)
df_month.columns


basic_model_df = df_month[['VIRGEN_EXTRA_EUR_kg','PRODUCTION_HARVEST']]
#basic_model_df = df_month[['VIRGEN_EXTRA_EUR_kg']]

#df_month_pdf.columns
df_andaluz = imd.include_meteo_variables()

df_month = imd.include_pdf_data(df_month)
df_month
df_month['Produccion Total_TOTAL MONDIAL WORLD_no_UE'] = df_month['Produccion Total_TOTAL MONDIAL WORLD'] - df_month['Produccion UE_TOTAL A + B']

for col in df_month.columns:
    if 'Produccion' in col:
        df_month[col] = df_month[col].shift(5)

df_month['Produccion UE_TOTAL A + B'] = df_month['Produccion UE_TOTAL A + B'] * 0.93
df_month['Produccion UE_TOTAL A + B'] = df_month['Produccion UE_TOTAL A + B'] - df_month['PRODUCTION_HARVEST']
df_month['Produccion Total_TOTAL  A'] = df_month['Produccion Total_TOTAL  A'] * 0.93
df_month['Produccion Total_TOTAL  A'] = df_month['Produccion Total_TOTAL  A'] - df_month['PRODUCTION_HARVEST']
df_month['Produccion UE_TOTAL  A)'] = df_month['Produccion UE_TOTAL  A)'] * 0.93
df_month['Produccion UE_TOTAL  A)'] = df_month['Produccion UE_TOTAL  A)'] - df_month['PRODUCTION_HARVEST']
df_month['Produccion Total_TOTAL MONDIAL WORLD'] = df_month['Produccion Total_TOTAL MONDIAL WORLD'] * 0.93
df_month['Produccion Total_TOTAL MONDIAL WORLD'] = df_month['Produccion Total_TOTAL MONDIAL WORLD'] - df_month['PRODUCTION_HARVEST']
df_month['Produccion Total_TOTAL B'] = df_month['Produccion Total_TOTAL B'] * 0.93
df_month['Consumo Total_TOTAL A_no_EU'] = df_month['Consumo Total_TOTAL  A'] -df_month['Consumo UE_TOTAL A + B']


basic_model_df.columns


basic_model_df.columns
basic_model_df = df_month
basic_model_df.drop('PRODUCTION_HARVEST',axis =1,inplace = True)
basic_model_df.columns

var_pdf = ['Consumo Total_TOTAL  A',
       'Consumo Total_TOTAL B', 'Consumo Total_TOTAL MONDIAL WORLD',
       'Consumo UE_TOTAL  A)', 'Consumo UE_TOTAL B)', 'Consumo UE_TOTAL A + B',
       'Exportacion Total_TOTAL A','Exportacion Total_TOTAL B',
       'Exportacion Total_TOTAL MONDIAL WORLD', 'Exportacion UE_TOTAL  A)',
       'Exportacion UE_TOTAL B)', 'Exportacion UE_TOTAL A + B',
       'Importacion Total_TOTAL  A', 'Importacion Total_TOTAL  B',
       'Importacion Total_TOTAL MONDIAL WORLD', 'Importacion UE_TOTAL  A)',
       'Importacion UE_TOTAL B)', 'Importacion UE_TOTAL A + B',
       'Produccion Total_TOTAL  A', 'Produccion Total_TOTAL B',
       'Produccion Total_TOTAL MONDIAL WORLD', 'Produccion UE_TOTAL  A)' ,'Produccion UE_TOTAL A + B',
        'Produccion Total_TOTAL MONDIAL WORLD_no_UE','Consumo Total_TOTAL A_no_EU']



ls_target =['VIRGEN_EXTRA_EUR_kg']
var_pdf_good = ['Consumo Total_TOTAL B', 'Consumo Total_TOTAL MONDIAL WORLD',
       'Consumo UE_TOTAL  A)', 'Consumo UE_TOTAL B)', 'Consumo UE_TOTAL A + B'
       'Exportacion Total_TOTAL MONDIAL WORLD', 'Exportacion UE_TOTAL  A)',
       'Exportacion UE_TOTAL B)', 'Exportacion UE_TOTAL A + B','Importacion Total_TOTAL  B',
       'Importacion Total_TOTAL MONDIAL WORLD', 'Importacion UE_TOTAL  A)',
       'Importacion UE_TOTAL B)', 'Importacion UE_TOTAL A + B',
       'Produccion Total_TOTAL  A', 'Produccion Total_TOTAL B',
       'Produccion Total_TOTAL MONDIAL WORLD','Produccion UE_TOTAL A + B']


var_pdf_very_good = [ 'VIRGEN_EXTRA_EUR_kg','Consumo UE_TOTAL A + B', 'Exportacion UE_TOTAL A + B', 'Importacion UE_TOTAL A + B',
       'Produccion Total_TOTAL  A', 'Produccion Total_TOTAL B']


var_consumo =['Consumo Total_TOTAL  A',
 'Consumo Total_TOTAL B',
 'Consumo Total_TOTAL MONDIAL WORLD',
 'Consumo UE_TOTAL  A)',
 'Consumo UE_TOTAL B)',
 'Consumo UE_TOTAL A + B',
'Consumo Total_TOTAL A_no_EU']

var_export = [  'Exportacion Total_TOTAL A',
 'Exportacion Total_TOTAL B',
 'Exportacion Total_TOTAL MONDIAL WORLD',
 'Exportacion UE_TOTAL  A)',
 'Exportacion UE_TOTAL B)',
 'Exportacion UE_TOTAL A + B' ]

var_import = [ 'Importacion Total_TOTAL  A',
 'Importacion Total_TOTAL  B',
 'Importacion Total_TOTAL MONDIAL WORLD',
 'Importacion UE_TOTAL  A)',
 'Importacion UE_TOTAL B)',
 'Importacion UE_TOTAL A + B']

var_produc = [ 'Produccion Total_TOTAL  A',
 'Produccion Total_TOTAL B',
 'Produccion Total_TOTAL MONDIAL WORLD',
 'Produccion UE_TOTAL  A)',
 'Produccion UE_TOTAL A + B',
 'Produccion Total_TOTAL MONDIAL WORLD_no_UE'
               ]


"""
var_import = ['Importacion Total_TOTAL  A', 'Importacion Total_TOTAL  B',
       'Importacion Total_TOTAL MONDIAL WORLD', 'Importacion UE_TOTAL  A)',
 'Importacion UE_TOTAL A + B']

var_import = [
    'Importacion UE_TOTAL  A)',
    'Importacion UE_TOTAL A + B'] #mejores casi 3 R2
"""




var_pdf_rolling_good =['Consumo Total_TOTAL  A', 'Exportacion Total_TOTAL B', 'Importacion Total_TOTAL MONDIAL WORLD']
var_pdf_rolling_good =['Consumo Total_TOTAL  A', 'Exportacion Total_TOTAL B'] # bone

var_pdf_expanding_good =['Consumo UE_TOTAL A + B', 'Produccion UE_TOTAL A + B', 'Importacion UE_TOTAL B)',
                         'Consumo UE_TOTAL  A)', 'Importacion Total_TOTAL MONDIAL WORLD'] # ottime R2 = 0.727


col_energy =['VIRGEN_EXTRA_EUR_kg','Trade Close_EUA','Mid Price Close_BRENT', 'Trade Close_API2', 'Trade Close_TTF', 'Media POOL_OMEL', 'MONTHLY_INFLATION_PERC']



#basic_model_df = basic_model_df[basic_model_df.index > 2009]
#basic_model_df= basic_model_df.fillna(method='ffill')


ls_selected = ['Consumo UE_TOTAL A + B', 'Consumo UE_TOTAL  A)',
'Produccion UE_TOTAL A + B',
'Importacion UE_TOTAL B)',
'Importacion UE_TOTAL A + B']


# Best with no repetitions just UE single variable R2 above 0.87 # delete the world variable Total if needed
ls_selected = ['Consumo UE_TOTAL B)', 'Consumo UE_TOTAL  A)','Consumo Total_TOTAL B',
'Produccion UE_TOTAL A + B','Produccion Total_TOTAL MONDIAL WORLD',
'Importacion UE_TOTAL B)',
'Importacion UE_TOTAL  A)']


# Best with changing the var prod tot world
ls_selected = ['Consumo UE_TOTAL B)', 'Consumo UE_TOTAL  A)','Consumo Total_TOTAL B',
'Produccion UE_TOTAL A + B','Produccion Total_TOTAL MONDIAL WORLD_no_UE',
'Importacion UE_TOTAL B)',
'Importacion UE_TOTAL  A)']



#best after working production
ls_selected = ['Consumo UE_TOTAL B)', 'Consumo UE_TOTAL  A)']
ls_necessary = ['Consumo UE_TOTAL B)', 'Consumo UE_TOTAL  A)']





# consumi congiunti
ls_selected = ['Consumo UE_TOTAL A + B',
'Produccion UE_TOTAL A + B',
'Importacion UE_TOTAL B)',
'Importacion UE_TOTAL A + B']

results_df.index

# import congiunte
ls_selected = ['Consumo UE_TOTAL B)', 'Consumo UE_TOTAL  A)',
'Produccion UE_TOTAL A + B',
'Importacion UE_TOTAL A + B']


ls_selected = ['Consumo UE_TOTAL A + B',
'Produccion UE_TOTAL A + B',
'Importacion UE_TOTAL A + B',
               'Importacion UE_TOTAL B)',]


results_df.index

#basic_model_df = basic_model_df_bck
#basic_model_df_bck= basic_model_df.copy()
basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2010-07-01')
ls_selected=['VIRGEN_EXTRA_EUR_kg','Consumo UE_TOTAL B)',
 'Consumo UE_TOTAL  A)',
 'Exportacion UE_TOTAL B)',
 'Importacion UE_TOTAL B)']


basic_model_df.drop(columns=['Produccion Total_TOTAL MONDIAL WORLD'],inplace = True)
basic_model_df.drop(columns=['Consumo Total_TOTAL  A'],inplace = True)
basic_model_df.columns

#X = basic_model_df[ls_selected]
target_variable = 'VIRGEN_EXTRA_EUR_kg'
y = basic_model_df[[target_variable]].copy()
X = basic_model_df.drop(columns=[target_variable ]).copy()
X.columns
#X = X.drop(columns=['INNER_CONS' ]).copy()
#X = basic_model_df_man[['Consumo Total_TOTAL  A','Consumo Total_TOTAL B']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


ls_selected
type(model.pvalues)

model.pvalues

'Consumo Total_TOTAL  A'
var_pdf.remove('Consumo Total_TOTAL B')
var_pdf

var_pdf
prova = df_month[var_pdf]
df_month.columns
basic_model_df_step = basic_model_df.copy()
basic_model_2010
df_month.columns
for col in col_andaluz:
    if col not in basic_model_df_step.columns:
        basic_model_df_step[col] = df_month[col]

basic_model_df_step= basic_model_df_step.fillna(method="ffill")

col_necessary


col_andaluz = [ 'Precip_Average_Andalucia',
       'cumulated_year_Precip_Average_Andalucia', 'TMed_Average_Andalucia',
       'ndays_Precip_Average_Andalucia',
       'cumulated_month_Precip_Average_Andalucia',
       'days_above_33_Average_Andalucia', 'TMin_Average_Andalucia',
       'TMax_Average_Andalucia']
#automatic_step 3

target_variable = 'VIRGEN_EXTRA_EUR_kg'
y = basic_model_df_step[[target_variable]].copy()
for var in col_andaluz:
    ls_necessary = col_necessary
    ls_selected = ls_necessary
    if var not in ls_necessary:
        ls_selected.append(var)
    X = basic_model_df_step[ls_selected]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if (model.pvalues < 0.05).all() and len(ls_necessary) < len(ls_selected):
        print(model.summary())






basic_model_df

# gpt code



basic_model_df_copy = basic_model_df.copy()


# Create an empty list to store the results
results_list = []

var_pdf_rolling_good = ['Consumo Total_TOTAL  A', 'Exportacion Total_TOTAL B']  # Replace with your variable names
X_list = var_pdf

var_pdf
# Loop through the data
target_var = 'VIRGEN_EXTRA_EUR_kg'
y = basic_model_df_copy[[target_var]]



for i in range(len(X_list)):
    X_var = X_list[i]


    if X_var in var_produc:
        group ='Production'
    elif X_var  in var_export:
        group ='Export'
    elif X_var  in var_import:
        group = 'Import'
    else:
        group = 'Consume'

    # Add a constant term to the independent variables
    X_var_running = basic_model_df_copy[[X_var]]
    X_var_running = sm.add_constant(X_var_running)

    # Fit the OLS regression model
    model = sm.OLS(y, X_var_running).fit()

    # Get R-squared and p-values of coefficients
    r_squared = model.rsquared
    p_values = model.pvalues.drop('const')  # Drop the constant term
    p_value = p_values.to_dict().values()
    p_value= list(p_value)[0]

    # Store the results in the list
    results_list.append({
        'X_var': X_var,
        'R_squared': r_squared,
        'P_Value': p_value,
        'Group' : group
    })

# Create the DataFrame from the list of dictionaries
results_df = pd.DataFrame(results_list)
results_df.set_index('X_var',inplace = True)
results_df = results_df.sort_values(by='R_squared', ascending=False)

# Print or further analyze the results dataframe
print(results_df)


p_value
ls_selected
basic_model_df.columns
#basic_model_df_roll = basic_model_df[ls_selected]
basic_model_df_roll = basic_model_df

basic_model_df_roll.columns


basic_model_df_roll

ls_ventanas = [24, 30, 36, 40, 50]
for window in ls_ventanas:
    result_rolling = rf.rolling_regression(basic_model_df_roll, 'VIRGEN_EXTRA_EUR_kg', window)
    result_rolling = aux.add_average_row(result_rolling)
    result_rolling.to_excel(f"Output/Rolling_Regression/Rolling{window}.xlsx")


