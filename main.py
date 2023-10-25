# This is a sample Python script.
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


import importlib # code to reload  lib
#from src import importing_data as imd  # code to reload  lib
importlib.reload(rf)  # Reload the module # code to reload  lib



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

    df_month.columns

    df_month_copy = df_month.copy()

# 1st basic model

    basic_model_df =  df_month[['VIRGEN_EXTRA_EUR_kg','EXIS_INIC','IMPORTS','EXPORTS', 'PRODUCTION','PRODUCTION_HARVEST']].copy()

    basic_model_df = rf.eliminate_rows_from_date(basic_model_df, '2005-10-01')
    df_step = rf.stepwise_eliminating(basic_model_df,'VIRGEN_EXTRA_EUR_kg',3)
    print(df_step.iloc[:, 0:4])
    iteration_selected = 1
    rf.save_model_summary_to_file(df_step, iteration_selected, f"Output/Document/regression_summary_basic_model_stepwise_{iteration_selected}_original_2005_data.txt")
    col_selected = df_step.loc[df_step.index[iteration_selected-1],'Actual_cols']

    basic_model_df_bck = basic_model_df[col_selected].copy()
    len(basic_model_df_bck['VIRGEN_EXTRA_EUR_kg'])
    df_pred, MSFE,MAPE = rf.back_testing_actual_time(basic_model_df_bck,50, 24, 'VIRGEN_EXTRA_EUR_kg') # montly model 50 obs out # 24 horizons previewd
    df_pred.columns
    MAPE[0]

    print_doc_descriptive_vars(basic_model_df,'VIRGEN_EXTRA_EUR_kg',24)
    gf.plot_correlation_matrix(basic_model_df)














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

























