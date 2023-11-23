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

df_inflaction = pd.read_excel("Datos/inflacion_mensual.xls",sheet_name="Sheet1")
df_inflaction.set_index('DATE',inplace = True)
df_month = df_month.merge(df_inflaction, left_index=True, right_index=True, how='left')
df_month
df_inflaction.columns

df_month.columns
col_energy =['VIRGEN_EXTRA_EUR_kg','Trade Close_EUA','Mid Price Close_BRENT', 'Trade Close_API2', 'Trade Close_TTF', 'Media POOL_OMEL', 'MONTHLY_INFLATION_PERC']


basic_model_df = df_month[['VIRGEN_EXTRA_EUR_kg']]

pdf = imd.include_pdf_data(basic_model_df)

df_andaluz = imd.include_meteo_variables()

col_andaluz_2010 = [ 'ndays_Precip_Average_Andalucia','cumulated_year_Precip_Average_Andalucia']
energy_good = ['Trade Close_EUA', 'Mid Price Close_BRENT']

basic_model_2010 = ['VIRGEN_EXTRA_EUR_kg', 'IMPORTS', 'PRODUCTION_HARVEST', 'TOTAL_CONS','PRODUCTION_HARVEST_LAST_YEAR', 'HARVEST_FORECAST_JUNTA_ANDALUCIA','IMPORTS_LAG_9', 'EXPORTS_LAG_12']


var_added = col_andaluz_2010 + energy_good

var_selected = basic_model_2010 + var_added
var_selected

df_month = df_month.merge(df_andaluz, left_index=True, right_index=True, how='left')

basic_model_df = basic_model_df.merge(pdf, left_index=True, right_index=True, how='left')



basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2010-07-01')

target_variable = 'VIRGEN_EXTRA_EUR_kg'
y = basic_model_df[[target_variable]].copy()
X = basic_model_df.drop(columns=[target_variable ]).copy()
#X = X.drop(columns=['INNER_CONS' ]).copy()
#X = basic_model_df_man[['Consumo Total_TOTAL  A','Consumo Total_TOTAL B']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
