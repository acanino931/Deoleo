import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA


def remove_null_rows(df,target_variable = 'VIRGEN_EXTRA_EUR_kg'):
    # this function eliminate all the rows with null values in the x and gives you back the X the y and the min date that the model is considering


    ls_min = [df.index[df[col].notnull()].min() for col in df.columns ]
    max_data_no_missing = max(list(set(ls_min)))
    column_data_max = {}
    for col in df.columns:
        if df.index[df[col].notnull()].min() == max_data_no_missing:
            column_data_max[col] = max_data_no_missing

    x_data = df.loc[max_data_no_missing:, :].drop(columns= [target_variable])
    y_data = df.loc[max_data_no_missing:, target_variable]
    x_data = sm.add_constant(x_data)
    return (x_data,y_data,max_data_no_missing, column_data_max)

def shuffle_columns(df):
    columns = list(df.columns)
    np.random.shuffle(columns)
    shuffled_df = df[columns]
    return shuffled_df

def stepwise_eliminating(df, target_variable, iterations):
    ls_to_del = []
    data = []  # Initialize an empty list to collect dictionaries

    for i in range(iterations):
        df1 = shuffle_columns(df)
        df1 = df1.drop(columns=ls_to_del)
        X, y, max_data_no_missing, column_data_max = remove_null_rows(df1, target_variable=target_variable)
        # Fit the regression model
        model = sm.OLS(y, X).fit()
        p_values = model.pvalues

        # Identify the greatest p value to eliminate the variable
        p_value_max = model.pvalues.max()
        selected_vars = list(X.columns)
        newvar = [var for var, p_value in zip(selected_vars, p_values) if p_value == p_value_max]

        actual_cols = [col for col in df.columns if col not in ls_to_del]
        summary_text = model.summary().as_text()
        info_rows = f"The 1st Date considered is: {max_data_no_missing} and the columns with more null values are: {column_data_max}"
        summary_text = summary_text + "\n" + info_rows

        new_row = {
            "Iteration": i,
            "Rsquared": model.rsquared,
            "Variable_out": newvar[0],
            "P-value": p_value_max,
            "Actual_cols": actual_cols,
            "Model_summary": summary_text
        }
        ls_to_del.append(newvar[0])
        data.append(new_row)  # Append the new row to the list of dictionaries

    # Create a DataFrame from the list of dictionaries
    df_stepwise_eliminating = pd.DataFrame(data)

    return df_stepwise_eliminating


def save_model_summary_to_file(df, index, file_path):
    try:
        # Get the Model_summary from the specified row based on the index
        model_summary = df.loc[index-1, 'Model_summary']

        # Write the Model_summary to the specified file
        with open(file_path, 'w') as file:
            file.write(model_summary)

        print(f"Model summary for index {index} has been saved to {file_path}")
    except KeyError:
        print(f"Index {index} not found in the DataFrame.")


def eliminate_rows_from_date(df,specific_date ):
    # this function make you select a the df from a specific date
    # filling all the null values
    selected_datetime = pd.to_datetime(specific_date)
    df1 = df.loc[selected_datetime:].copy()
    df1 = df1.fillna(method='ffill').fillna(method='bfill')
    return df1

def back_testing_SARIMAX( df,test_sample,horizontes, target_variable = 'VIRGEN_EXTRA_EUR_kg' ):
    # the X are not forecasted, use with actual values
    y = df[target_variable].copy()
    X = df.drop(columns = [target_variable])
    X = sm.add_constant(X)


    long = len(y)
    n_estimation = len(y) - test_sample
    real = y[(n_estimation):len(y)]

    n_forecasting = long - n_estimation

    predicc = np.zeros((horizontes, n_forecasting))
    MSFE = np.zeros((horizontes, 1))
    MAPE = np.zeros((horizontes, 1))

    for Periods_ahead in range(horizontes):
        for i in range(0, n_forecasting):
            aux_y = y[0:(n_estimation - Periods_ahead + i)]
            aux_x = X[0:(n_estimation - Periods_ahead + i)]
            model = sm.OLS(aux_y, aux_x).fit()
            X_forecast = X[n_estimation + i - Periods_ahead:n_estimation + i + 1]

            y_pred = model.predict( exog=X_forecast)
            predicc[Periods_ahead][i] = y_pred.iloc[0]
           # print(f"periods_ahead{Periods_ahead} ")
           # print(y_pred.iloc[0])

        error = np.array(real) - predicc[Periods_ahead]
        MSFE[Periods_ahead] = np.mean(error ** 2)
        MAPE[Periods_ahead] = np.mean(np.abs(error / np.array(real))) * 100

    data_pred = {}
    for i in range (n_forecasting):
        column_name = f"Prediction_{i}"
        data_pred[column_name] = predicc[:, i]
    df_pred= pd.DataFrame(data_pred)

 #   data_error = { 'MSFE': MSFE, 'MAPE': MAPE}
 #   df_error = pd.DataFrame(data_error)

    return df_pred ,MSFE, MAPE

  #  df_pred = pd.DataFrame({"V1": predicc[0]})


def back_testing_regression(df, test_sample, horizontes, target_variable='VIRGEN_EXTRA_EUR_kg'):
    # the X are not forecasted, use with actual values
    y = df[target_variable].copy()
    X = df.drop(columns=[target_variable])
    X = sm.add_constant(X)

    long = len(y)
    n_estimation = len(y) - test_sample
    real = y[n_estimation:]

    n_forecasting = long - n_estimation

    predicc = np.zeros((horizontes, n_forecasting))
    MSFE = np.zeros(horizontes)
    MAPE = np.zeros(horizontes)

    for Periods_ahead in range(horizontes):
        for i in range(0, n_forecasting):
            aux_y = y[:n_estimation - Periods_ahead + i]
            aux_x = X[:n_estimation - Periods_ahead + i]
            model = sm.OLS(aux_y, aux_x).fit()
            X_forecast = X[n_estimation + i - Periods_ahead:n_estimation + i + 1]

            y_pred = model.predict(exog=X_forecast)
            predicc[Periods_ahead][i] = y_pred.iloc[Periods_ahead] #unsure wheter to select the 1st value or the pred

        # Convert 'real' to a NumPy array before calculating MAPE and MSFE
        real_np = np.array(real)
        # Calculate MAPE for all periods up to the current one
        error = real_np - predicc[:(Periods_ahead + 1)]
        MAPE[Periods_ahead] = np.mean(np.abs(error / real_np)) * 100

        # Calculate MSFE for all periods up to the current one
        MSFE[Periods_ahead] = np.mean(error ** 2)

    data_pred = {}
    for i in range(n_forecasting):
        column_name = f"Prediction_{i}"
        data_pred[column_name] = predicc[:, i]
    df_pred = pd.DataFrame(data_pred)

    return df_pred, MSFE, MAPE
