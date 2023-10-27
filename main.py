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

import tabula
import importlib # code to reload  lib
from unidecode import unidecode
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


    all_countries = pd.read_excel("Datos/all_countries_list.xlsx")

    all_countries['COUNTRY_NAME'] = all_countries['COUNTRY_NAME'].apply(lambda x: unidecode(x))


    path = "Datos/PDF/Juan_Vilar/Importacion Total.pdf"
    df = tabula.read_pdf(path, pages='all')[0]

    # code to append the 1st line to the column if the info is provisional or not (knowing all the last 2 years are provisional it is not needed)
    """first_row = df.iloc[0]
    for column in df.columns:
        if not pd.isnull(first_row[column]):
            df.rename(columns={column: column + first_row[column]}, inplace=True)
    """
    # Reset the index if needed

    # getting the county col
    df.columns.values[0] = 'COUNTRY'
    #updating the coulum name so the index might see it
    df.rename(columns={column: column }, inplace=True)
    df = df.iloc[2:]
    df.columns
    df = df.reset_index(drop=True)
    # eliminating all the others with null values
   # col_to_drop = [col for col in df.columns if 'Unnamed' in col]
    col_to_drop = [col for col in df.columns if 'Unnamed' in col and col != 'COUNTRY']

    #col_to_drop= df.columns[1]
    col_to_drop
    df = df.drop(columns=col_to_drop)
    df
    df['COUNTRY'] = df['COUNTRY'].apply(lambda x: unidecode(x))

    # code to get the proper country name
    elements_to_check = all_countries['COUNTRY_NAME']
    # Loop through the DataFrame and replace row values if an element is found
    for index, row in df.iterrows():
        for element in elements_to_check:
            if element in row['COUNTRY']:
                df.at[index, 'COUNTRY'] = element

    sheet_name = 'IMPORT_GLOBAL'
    df['COUNTRY'] =  sheet_name +"_"+ df['COUNTRY']

    newcols = []
    for col in df.columns[1:]:
        last_part = col.split("/")[-1]
        if last_part == "0":
            newyear = "2000"
        elif len (last_part ) ==1:
            newyear = col[0:3] + last_part
        else:
            newyear= col[0:2] + col[-2:]
        newcols.append(int(newyear))

    for i, col in enumerate(df.columns[1:]):
        df = df.rename(columns={col: newcols[i]})

    df = df.rename(columns={'COUNTRY': 'YEAR'})

    # GETTING THE CORRECT TYPE  OF THE INDEX TO RESAMPLE THE DATA INTO MOUNTHLY
    df.set_index('YEAR',inplace = True)
    df_transposed = df.transpose()
    df_transposed
    df_transposed.index = df_transposed.index.astype(int)
    df_transposed
    df_transposed.index.dtype
    df_transposed.index[0]
    date_index = pd.date_range(start=str(df_transposed.index[0]), periods=len(df_transposed.index), freq='YS')


    df_transposed.index = date_index

    # changing granularity
    df_transposed = df_transposed.resample('MS').ffill()

    extended_date_index = pd.date_range(start='2023-02-01', end='2023-12-01', freq='MS')
    extended_date_index
    extended_df = pd.DataFrame(index=extended_date_index)
    df_transposed_fin = pd.concat([df_transposed, extended_df], axis=0)
    # what out with the ffil we can imput automatically values to missing year make a control over the value we inputing
    df_transposed_fin = df_transposed_fin.fillna(method="ffill",limit=11)

    # shift the 1st 3 month to have cross data like the starting ones
    df_transposed_fin_shifted  = df_transposed_fin.shift(-3)


    df_transposed_fin_shifted.to_excel("Output/Excel/df_tabula.xlsx")




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

    print_doc_descriptive_vars(basic_model_df_man,'VIRGEN_EXTRA_EUR_kg',24)
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

























