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
import numpy as np
import statsmodels.api as sm
import os

import tabula
import importlib # code to reload  lib
from unidecode import unidecode
#from src import importing_data as imd  # code to reload  lib
importlib.reload(gf)  # Reload the module # code to reload  lib



# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




def print_doc_descriptive_vars(df1,target_var ='VIRGEN_EXTRA_EUR_kg',lag_cross_corr =24):
    df = df1.copy()

    # calculate the column year in case is not explicited
    if 'YEAR' not in df.columns:
        if 'DATE' not in df.columns:
            df['YEAR'] = df.index.year
        else:
            df['YEAR'] = df['DATE'].year
    correlation_matrix = df.corr()


    max_correlation_with_target = correlation_matrix['VIRGEN_EXTRA_EUR_kg'].drop(['YEAR', 'VIRGEN_EXTRA_EUR_kg']).abs().sort_values(ascending=False)
    ordered_columns = max_correlation_with_target.index.tolist()
    df = df[['YEAR', 'VIRGEN_EXTRA_EUR_kg'] + ordered_columns]
    doc = Document()
    doc.add_heading('Graficas de Todas las Variables 20 10 2023', 0)
    for col in df:
        if col != target_var and col != 'YEAR':
            doc.add_paragraph('Graficas de Variable ' + col)
            image_buffer = gf.scatterplot_for_years(df, col, target_var)
            doc.add_picture(image_buffer, width=Inches(3), height=Inches(2))
            doc.add_paragraph('')

            buffer_ret = gf.cross_correlation_variable(df, col, target_var, lag_cross_corr)
            doc.add_picture(buffer_ret, width=Inches(3), height=Inches(2))
            doc.add_paragraph('')

            buff = gf.plot_and_save_variables(df, col, target_var, temp='Monthly')
            doc.add_picture(buff, width=Inches(5), height=Inches(3))
            doc.add_paragraph('')

            buffer = gf.print_correlation_over_time(df, col, target_var)
            doc.add_picture(buffer, width=Inches(5), height=Inches(3))
            doc.add_paragraph('')

    doc.save('Output/Document/sample_with_pycharm.docx')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':



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

    df_month.columns
    # model imc start
    basic_model_df = df_month.copy()

    basic_model_df.columns
    basic_model_df = gf.compute_lags_for_custom_ccf_IMC(basic_model_df, 6 )
    basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2018-01-01')
    corr_crossimc = gf.custom_ccf_IMC(basic_model_df, 'IMC_EXTRA_VIRGEN_EUR_kg')
    corr_crossimc

    basic_model_df['VIRGEN_EXTRA_EUR_kg_LAG1'] = basic_model_df['VIRGEN_EXTRA_EUR_kg'].shift(1)
    basic_model_df['VIRGEN_EXTRA_EUR_kg_LAG2'] = basic_model_df['VIRGEN_EXTRA_EUR_kg'].shift(2)
    basic_model_df['VIRGEN_EXTRA_EUR_kg_LAG3'] = basic_model_df['VIRGEN_EXTRA_EUR_kg'].shift(3)
    basic_model_df['VIRGEN_EXTRA_EUR_kg_LAG4'] = basic_model_df['VIRGEN_EXTRA_EUR_kg'].shift(4)
    basic_model_df['VIRGEN_EXTRA_EUR_kg_LAG5'] = basic_model_df['VIRGEN_EXTRA_EUR_kg'].shift(5)
    basic_model_df['VIRGEN_EXTRA_EUR_kg_LAG6'] = basic_model_df['VIRGEN_EXTRA_EUR_kg'].shift(6)
    basic_model_df = basic_model_df[['VIRGEN_EXTRA_EUR_kg','VIRGEN_EXTRA_EUR_kg_LAG1','VIRGEN_EXTRA_EUR_kg_LAG2','VIRGEN_EXTRA_EUR_kg_LAG3','VIRGEN_EXTRA_EUR_kg_LAG4','VIRGEN_EXTRA_EUR_kg_LAG5','VIRGEN_EXTRA_EUR_kg_LAG6','IMC_EXTRA_VIRGEN_EUR_kg']]

    basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2018-01-01')
    basic_model_df.to_excel("Output/Excel/df_IMC.xlsx")
    basic_model_df.columns

    print_doc_descriptive_vars(basic_model_df, target_var='IMC_EXTRA_VIRGEN_EUR_kg', lag_cross_corr=24)

    out = gf.cross_correlation_variable_out_df(basic_model_df, 'VIRGEN_EXTRA_EUR_kg', 'IMC_EXTRA_VIRGEN_EUR_kg', 5)

    out_custom,df = gf.ccustom_ccf(basic_model_df, 'VIRGEN_EXTRA_EUR_kg','IMC_EXTRA_VIRGEN_EUR_kg' ,24)

    out_custom
    correlation = basic_model_df['VIRGEN_EXTRA_EUR_kg'].corr(basic_model_df['IMC_EXTRA_VIRGEN_EUR_kg'])
    correlation

    out_custom
    df
    out_custom
    df.columns
    df.iloc[:,-2:]

    # imc model end


    # Modelo Basico

    # pretreat variables :
    df_month = df_month.fillna(method='ffill')
    basic_model_df = df_month[
        ['VIRGEN_EXTRA_EUR_kg', 'EXIS_INIC', 'IMPORTS', 'EXPORTS', 'INNER_CONS', 'PRODUCTION', 'PRODUCTION_HARVEST',
         'PRODUCTION_HARVEST_LAST_YEAR', 'PRODUCTION_HARVEST_2_YEARS', 'INTERNAL_DEMAND', 'EXTERNAL_DEMAND']].copy()
    basic_model_df = df_month.copy()
    basic_model_df['TOTAL_CONS'] = basic_model_df['INNER_CONS'] + basic_model_df['EXPORTS']
    basic_model_df['EXPORTS_LAG15'] = basic_model_df['EXPORTS'].shift(15)
    basic_model_df['INTERNAL_DEMAND_LAG_13'] = basic_model_df['INTERNAL_DEMAND'].shift(13)
    basic_model_df['TOTAL_CONS_LAG_12'] = basic_model_df['EXTERNAL_DEMAND'].shift(12)
    basic_model_df['PRODUCTION_LAG_21'] = basic_model_df['PRODUCTION'].shift(21)
    basic_model_df['TOTAL_CONS_LAG_13'] = basic_model_df['TOTAL_CONS'].shift(13)
    basic_model_df['EXIS_INIC_18'] = basic_model_df['EXIS_INIC'].shift(18)
    basic_model_df['PRODUCTION_HARVEST_LAG_8'] = basic_model_df['PRODUCTION_HARVEST'].shift(8)
    basic_model_df.drop(columns=['EXTERNAL_DEMAND'], inplace=True)
    basic_model_df.to_excel("Output/Excel/basic_model_df.xlsx")

    gf.plot_correlation_target_variable(basic_model_df, 'IMC_EXTRA_VIRGEN_EUR_kg')

    gf.plot_correlation_matrix(basic_model_df)

    # manual, stepwise omitted
    basic_model_df_man = basic_model_df[
        ['VIRGEN_EXTRA_EUR_kg', 'IMPORTS', 'INNER_CONS', 'TOTAL_CONS_LAG_12', 'EXPORTS_LAG15', 'TOTAL_CONS',
         'INTERNAL_DEMAND_LAG_13', 'EXIS_INIC', 'PRODUCTION_HARVEST_LAG_8', 'PRODUCTION', 'INTERNAL_DEMAND']]

    x_variable = 'VIRGEN_EXTRA_EUR_kg'
    target_variable= 'IMC_EXTRA_VIRGEN_EUR_kg'
    #basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2005-10-01')

    basic_model_df.columns

    basic_model_df
    y = basic_model_df[[target_variable]].copy()
    X = basic_model_df[[x_variable]].copy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())




    # end modelo basico manual






    #selected_datetime = '2021-12-01'
    #df_month_copy= df_month.loc[:selected_datetime].copy()
    df_month_copy


    #def compute_montly_weight(df_month,'VIRGEN_EXTRA_EUR_kg')
    df_selected = yf.select_annual_variable_from_dic (df_month_copy)
    df_aggr = yf.aggregate_mountly_data(df_selected)
    df_selected.tail(14)
    df_selected.columns
    #df_month_copy = df_month.copy()
    df_aggr.columns
    target_transposed = yf.transpose_target_variable(df_month_copy, 'VIRGEN_EXTRA_EUR_kg')
    df_aggr.index

    df_weight= yf.calc_correlations_yearly(df_aggr, target_transposed)
    df_weight.to_excel("Output/Excel/df_weight.xlsx")



    #df_month.columns = df_month.columns.str.replace('_seed', '')



# 1st basic model
    # consider to add stock oil
    basic_model_df =  df_month[['VIRGEN_EXTRA_EUR_kg','EXIS_INIC','IMPORTS','EXPORTS', 'PRODUCTION','PRODUCTION_HARVEST','INTERNAL_DEMAND', 'EXTERNAL_DEMAND']].copy()




    gf.plot_correlation_target_variable(basic_model_df,'VIRGEN_EXTRA_EUR_kg')

#    basic_model_df =  df_month[['VIRGEN_EXTRA_EUR_kg','EXIS_INIC','IMPORTS','EXPORTS', 'PRODUCTION','PRODUCTION_HARVEST']].copy()
 #   basic_model_df =  df_month[['VIRGEN_EXTRA_EUR_kg','IMPORTS','EXPORTS']].copy()


    basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2005-10-01')
    df_step = rf.stepwise_eliminating(basic_model_df_man,'VIRGEN_EXTRA_EUR_kg',2)
    print(df_step.iloc[:, 0:4])
    iteration_selected = 1
    list (df_step.iloc[iteration_selected,:5])
    rf.save_model_summary_to_file(basic_model_df_man, iteration_selected, f"Output/Document/regression_summary_basic_model_stepwise_{iteration_selected}_original_2005_data.txt")
    col_selected = df_step.loc[df_step.index[iteration_selected-1],'Actual_cols']

    basic_model_df_bck = basic_model_df[col_selected].copy()
    len(basic_model_df_bck['VIRGEN_EXTRA_EUR_kg'])
  #  df_pred, MSFE,MAPE = rf.back_testing_actual_time(basic_model_df_bck,50, 24, 'VIRGEN_EXTRA_EUR_kg') # montly model 50 obs out # 24 horizons previewd
    #df_pred.columns
    df_pred, MSFE, MAPE = rf.back_testing_regression(basic_model_df_bck, 50, 24 ,'VIRGEN_EXTRA_EUR_kg')  # montly model 50 obs out # 24 horizons previewd


    df_pred
    MAPE
    basic_model_df_bck.columns

    basic_model_df.columns

    selected_rows
    selected_rows1
    df_month_copy.Month
    print_doc_descriptive_vars(selected_rows,'VIRGEN_EXTRA_EUR_kg',24)
    print_doc_descriptive_vars(selected_rows1, 'VIRGEN_EXTRA_EUR_kg', 24)
    gf.plot_correlation_matrix(basic_model_df_man)

    df_month.columns
    basic_model_df_man
#   anticipo -2 production harvest -1 exports (intorno ai 40 y 35)

    basic_model_df_man = df_month[['VIRGEN_EXTRA_EUR_kg', 'EXIS_INIC', 'IMPORTS', 'EXPORTS', 'PRODUCTION','INNER_CONS' ,'PRODUCTION_HARVEST','INTERNAL_DEMAND', 'EXTERNAL_DEMAND', ]].copy()
    basic_model_df_man['PRODUCTION_HARVEST_OLD'] = basic_model_df_man['PRODUCTION_HARVEST'].shift(12)
    basic_model_df_man['PRODUCTION_HARVEST_OLD'] = basic_model_df_man['PRODUCTION_HARVEST_OLD'].fillna(method='ffill' ,limit=12)
    basic_model_df_man['PRODUCTION_HARVEST'] = basic_model_df_man['PRODUCTION_HARVEST'].shift(-3)
    basic_model_df_man['PRODUCTION_HARVEST'] = basic_model_df_man['PRODUCTION_HARVEST'].fillna(method='ffill' ,limit=3)
    basic_model_df_man['EXPORTS'] = basic_model_df_man['EXPORTS'].shift(12)
    basic_model_df_man['EXPORTS'] = basic_model_df_man['EXPORTS'].fillna(method='ffill' ,limit=12)
    basic_model_df_man['INTERNAL_DEMAND'] = basic_model_df_man['INTERNAL_DEMAND'].shift(13)
    basic_model_df_man['INTERNAL_DEMAND'] = basic_model_df_man['INTERNAL_DEMAND'].fillna(method='ffill' ,limit=13)
    basic_model_df_man['EXTERNAL_DEMAND'] = basic_model_df_man['EXTERNAL_DEMAND'].shift(13)
    basic_model_df_man['EXTERNAL_DEMAND'] = basic_model_df_man['EXTERNAL_DEMAND'].fillna(method='ffill' ,limit=13)
    basic_model_df_man['PRODUCTION'] = basic_model_df_man['PRODUCTION'].shift(6)
    basic_model_df_man['PRODUCTION'] = basic_model_df_man['PRODUCTION'].fillna(method='ffill' ,limit=6)


   #    basic_model_df_man = df_month[['VIRGEN_EXTRA_EUR_kg', 'EXIS_INIC', 'IMPORTS', 'EXPORTS', 'PRODUCTION','INNER_CONS' ,'PRODUCTION_HARVEST','INTERNAL_DEMAND', 'EXTERNAL_DEMAND',PRODUCTION_HARVEST_OLD ]].copy()
    basic_model_df_man = basic_model_df_man[['VIRGEN_EXTRA_EUR_kg', 'IMPORTS', 'INNER_CONS' , 'EXPORTS', 'PRODUCTION_HARVEST','EXTERNAL_DEMAND','PRODUCTION_HARVEST_OLD','INTERNAL_DEMAND']].copy()
    basic_model_df_man = basic_model_df_man.drop(columns = ['EXTERNAL_DEMAND','PRODUCTION'])
    basic_model_df_man = sm.add_constant(basic_model_df_man)
    target_variable = 'VIRGEN_EXTRA_EUR_kg'
    basic_model_df_man = rf.eliminate_rows_from_date(basic_model_df_man, '2005-10-01')
    y = basic_model_df_man[target_variable].copy()
    X = basic_model_df_man.drop(columns = [target_variable])
    model = sm.OLS(y, X).fit()
    print (model.summary())
    df_pred, MSFE, MAPE = rf.back_testing_regression(basic_model_df_man, 50, 24 ,'VIRGEN_EXTRA_EUR_kg')  # montly model 50 obs out # 24 horizons previewd
    MAPE
    model_sar = SARIMAX(y, exog=X, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0))
    model_fit = model_sar.fit()
    print(model_fit.summary())
    MAPE_df = pd.DataFrame(MAPE)
    MSFE_df = pd.DataFrame(MSFE)
    df_mapes = pd.merge(MAPE_df,MSFE_df , left_index = True, right_index= True  )
    df_mapes.to_excel("Output/Excel/MAPE.xlsx")
    type(MAPE)
    print(X[X.isnull()])

    X.info()

    # Generate some example data
    np.random.seed(0)
    X = np.random.rand(100, 1)  # Independent variable
    y = 2 * X + 1 + np.random.randn(100, 1)  # Dependent variable with noise

    # Add a constant term to the independent variable (intercept)
    X = sm.add_constant(X)

    # Fit a simple OLS linear regression model
    model = sm.OLS(y, X).fit()

    # Predict new values
    new_X = np.array([[1, 0.5], [1, 0.75]])  # Example data for prediction
    predictions = model.predict(new_X)

    # Display the model summary and predictions
    print(X)
    print(model.summary())
    print("Predictions:", predictions)

    predictions




    #def back_testing_actual_time(X, y, horizontes):



    """
    X, y, max_data_no_missing, column_data_max = rf.remove_null_rows(basic_model_df,target_variable = 'VIRGEN_EXTRA_EUR_kg')

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # View the model summary
    summary_text = model.summary().as_text()
    file_path = 'Output/Document/regression_summary_basic_model.txt'
    with open(file_path, 'w') as file:
        file.write(summary_text)
        file.write(summary_with_parameter)

"""







    """df_month_andalucia = imd.import_montly_andalucia()
    df_month_andalucia.columns

    df_month_andalucia = df_month_andalucia.replace(0, np.nan)
    df_month_andalucia = df_month_andalucia.set_index('DATE')
    df_monthcol = df_month.copy()

    gf.plot_correlation_matrix(df_month)


    df_month2 = pd.merge(df_monthcol, df_month_andalucia, left_on='DATE', right_on='DATE', how='left')


    print_doc_descriptive_vars(df_month2,'VIRGEN_EXTRA_EUR_kg')
    """

























