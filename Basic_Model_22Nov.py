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


def print_doc_descriptive_vars(df1,target_var ='VIRGEN_EXTRA_EUR_kg',lag_cross_corr =24,yearly_production = True):


    # THE CCF USED HERE STARTS FROM LAG 1
    df = df1.copy()

    # calculate the column year in case is not explicited
    if 'YEAR' not in df.columns:
        if 'DATE' not in df.columns:
            df['YEAR'] = df.index.year
        else:
            df['YEAR'] = df['DATE'].year
    if yearly_production == True:
        if 'HARVEST_YEAR'not in df.columns:
            df['HARVEST_YEAR'] = df['YEAR'].shift(-2)

    correlation_matrix = df.corr()


    max_correlation_with_target = correlation_matrix['VIRGEN_EXTRA_EUR_kg'].drop(['YEAR', 'VIRGEN_EXTRA_EUR_kg']).abs().sort_values(ascending=False)
    ordered_columns = max_correlation_with_target.index.tolist()
    df = df[['YEAR', 'VIRGEN_EXTRA_EUR_kg'] + ordered_columns]
    doc = Document()
    doc.add_heading('Graficas de Todas las Variables 20 10 2023', 0)
    for col in df:
        if col != target_var and col != 'YEAR' and col != 'HARVEST_YEAR':
            doc.add_paragraph('Graficas de Variable ' + col)
            # to print the yearly var change the sequent function into : scatterplot_for_years_yearly_var() scatterplot_for_years
            image_buffer = gf.scatterplot_for_years(df, col, target_var)
            doc.add_picture(image_buffer, width=Inches(3), height=Inches(2))
            doc.add_paragraph('')
            buffer_ret = gf.custom_ccf(df, col, target_var, lag_cross_corr)
            doc.add_picture(buffer_ret, width=Inches(3), height=Inches(2))
            doc.add_paragraph('')

            buff = gf.plot_and_save_variables(df, col, target_var, temp='Monthly')
            doc.add_picture(buff, width=Inches(5), height=Inches(3))
            doc.add_paragraph('')

            buffer = gf.print_correlation_over_time(df, col, target_var)
            doc.add_picture(buffer, width=Inches(5), height=Inches(3))
            doc.add_paragraph('')

    doc.save('Output/Document/sample_with_pycharm_basic.docx')

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

#df_month_pdf = imd.include_pdf_data(df_month)
#df_month_pdf.columns
#df_andaluz = imd.include_meteo_variables()
#df_month = df_month.merge(df_andaluz, left_index=True, right_index=True, how='left')
df_month.columns

basic_model_df = df_month[
    ['VIRGEN_EXTRA_EUR_kg', 'EXIS_INIC', 'IMPORTS', 'EXPORTS', 'INNER_CONS', 'PRODUCTION', 'PRODUCTION_HARVEST',
      'TOTAL_CONS','PRODUCTION_HARVEST_REAL_EST', 'PRODUCTION_HARVEST_LAST_YEAR', 'PRODUCTION_HARVEST_2_YEARS',
     'Estimación España (Junta Andalucia)']] #'INTERNAL_DEMAND'


print_doc_descriptive_vars(basic_model_df, target_var='VIRGEN_EXTRA_EUR_kg', lag_cross_corr=24)
# END  # graficas Modelo basico Review

# SECOND VERSION Modelo Basico

basic_model_df['IMPORTS_LAG_21'] = basic_model_df['IMPORTS'].shift(21)
basic_model_df['TOTAL_CONS_LAG_12'] = basic_model_df['TOTAL_CONS'].shift(12)
basic_model_df['PRODUCTION_HARVEST_LAG_8'] = basic_model_df['PRODUCTION_HARVEST'].shift(8)
basic_model_df['EXPORTS_LAG_12'] = basic_model_df['EXPORTS'].shift(12)
basic_model_df['PRODUCTION_18'] = basic_model_df['PRODUCTION'].shift(18)
# this is probably a bad var basic_model_df['INTERNAL_DEMAND_12'] = basic_model_df['INTERNAL_DEMAND'].shift(12)
basic_model_df['PRODUCTION_HARVEST_REAL_EST_LAG_14'] = basic_model_df['PRODUCTION_HARVEST_REAL_EST'].shift(14)
#basic_model_df['DP_PRODUCTION_HARVEST_LAG_14'] = basic_model_df['DP_PRODUCTION_HARVEST'].shift(14)
#basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2005-10-01')

basic_model_df



print_doc_descriptive_vars(basic_model_df)
#df_month_pdf = rf.eliminate_rows_from_date(df_month_pdf, '2005-10-01')