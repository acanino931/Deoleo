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
        df1 = df.copy()
        df1['New_Index'] = range(len(df))
        index_to_sel = df1.loc['New_Index']
        index_to_sel = df1.loc[df1['New_Index'] == index].index
        # Get the Model_summary from the specified row based on the index
        model_summary = df.loc[index_to_sel, 'Model_summary']

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

def eliminate_rows_after_date(df,specific_date ):
    # this function make you select a the df from a specific date
    # filling all the null values
    selected_datetime = pd.to_datetime(specific_date)
    df1 = df.loc[:selected_datetime].copy()
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


import pandas as pd
import numpy as np
import statsmodels.api as sm

def eliminate_multicollinearity(df, target_var, num_iterations=10, correlation_threshold=0.7):
   # function to be tested to eliminate multicollinearity
    results = []

    for iteration in range(num_iterations):
        y = df[target_var]
        X = df.drop([target_var], axis=1)

        # Add a constant to the predictor matrix (for intercept in the regression)
        X = sm.add_constant(X)

        # Fit a linear regression model
        model = sm.OLS(y, X).fit()

        # Calculate the correlation matrix for predictor variables
        correlation_matrix = X.corr()

        # Find variables with high covariance
        high_covariance_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    high_covariance_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

        # Remove the variable less correlated with the target variable in each high covariance pair
        for var1, var2 in high_covariance_pairs:
            if abs(model.pvalues[var1]) > abs(model.pvalues[var2]):
                X = X.drop(var1, axis=1)
            else:
                X = X.drop(var2, axis=1)

        # Re-fit the model with the selected variables
        new_model = sm.OLS(y, X).fit()
        r_squared = new_model.rsquared
        multicollinearity_metric = np.mean(np.abs(correlation_matrix.values) - np.eye(len(X.columns)))

        results.append({
            'Iteration': iteration + 1,
            'R-squared': r_squared,
            'Selected Columns': list(X.columns),
            'Multicollinearity Metric': multicollinearity_metric
        })

    results_df = pd.DataFrame(results)
    return results_df

# Example usage prevuois funct :
# Replace 'your_data.csv' and 'YourTargetVariable' with your data and target variable
#df = pd.read_csv('your_data.csv')
#target_var = 'YourTargetVariable'
#results_df = eliminate_multicollinearity(df, target_var, num_iterations=10)
#print(results_df)




def rolling_regression(df, target_variable, window_size):
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
        result_dict = ({
            'Start Date': window_data['DATE'].iloc[0],
            'End Date': window_data['DATE'].iloc[-1],
            'R-squared': r2,

        })
        # Add p-values to the dictionary
        for col, p_value in zip(X.columns, p_values):
            result_dict[f'p-value, {col}'] = p_value

        results.append(result_dict)
    # usage example :
    #target_variable = 'VIRGEN_EXTRA_EUR_kg'
    #window_size = 30

    # Run the rolling regression
    #rolling_results = rolling_regression(datos, target_variable, window_size)

    #rolling_results.to_excel('results_basic_model.xlsx')
    df.drop(columns=['DATE'],inplace = True)
    return pd.DataFrame(results)




