import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src import Aux_functions as aux
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from scipy.stats import kurtosis, skew
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


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



def stepwise_eliminating(df, target_variable, iterations):
    ls_to_del = []
    data = []  # Initialize an empty list to collect dictionaries

    for i in range(iterations):
        df1 = aux.shuffle_columns(df)
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


def save_model_summary_to_file(df, iteration, file_path =f"Output/Document/Out_reg_stepwise/regression_summary_basic_model_stepwise_original_2005_data.txt" ):
    try:
        df1 = df.copy()
        file_path = f"Output/Document/Out_reg_stepwise/regression_summary_basic_model_stepwise_{iteration}_original_2005_data.txt"
        #df1['New_Index'] = range(len(df))
        #index_to_sel = df1.loc[df1['New_Index'] == iteration].index
        # Get the Model_summary from the specified row based on the index
        model_summary = df.loc[iteration-1, 'Model_summary']

        # Write the Model_summary to the specified file
        with open(file_path, 'w') as file:
            file.write(model_summary)

        print(f"Model summary for index {iteration} has been saved to {file_path}")
    except KeyError:
        print(f"Index {iteration} not found in the DataFrame.")


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



def back_testing_regression_expanding_OLD(df: pd.DataFrame(), x_cols, y_var, initial_date: str = '2021-11-01',
                                final_date: str = '2023-09-01', signif: bool = False,
                                regr_type='OLS', num_variables: int = 4, window: int = 48, step_ahead: int = 12):
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
    if regr_type=='OLS':
        x_cols = sm.add_constant(x_cols)
    df_total = pd.DataFrame()


    # initial_date fecha inicial de prevision
    # final_date fecha final de prevision
    d = df.loc[initial_date:].index[0] - relativedelta(months=1)  # Date defiition
    df = df.loc[:final_date]

    unique_dates = df.index.unique()  # List of dates of the DataFrame
    date_object = pd.to_datetime(initial_date, format='%Y-%m-%d')
    d = unique_dates.get_loc(date_object)
  #  d = unique_dates.index(date_object)

    # unique_dates1 = unique_dates[:-(window) - (step_ahead) + 1]  # row for the rolling windows
    unique_dates1 = unique_dates[d-1:- (step_ahead) ]  # List of dates to iterate over
    for idx, i in enumerate(unique_dates1):


        ###### Date range interval for the train dataset, delimited by the window size

        date_max = i.date()  # final date of the training window , now it's datemax

        df_out2 = df.loc[: date_max]  # Dataframe with training window
        #print(df_out2.index)

        for step in range(1,
            step_ahead + 1):  # Iteration on each month of the test window delimited by the step_ahead parameter
            # vol_index = step -1
            df_res = pd.DataFrame()

            ind_test = df.loc[date_max:date_max].index[0] + relativedelta(months=step)  # Date defiition
            df_test = df.loc[ind_test: ind_test]

            #print(df_test.index)

            ###### Regression with the forward values
            if regr_type=='OLS':
                df_out2 = sm.add_constant(df_out2)
                df_test.insert(0, 'const', 1.0)
                df_reg = regression_OLD_OLS(df_out2, x_cols, y_var, df_test=df_test, reg_type=regr_type,
                                    significativas=signif,
                                    n_vars=num_variables)

            if regr_type == 'Linear':
                df_reg = regression_OLD_original(df_out2, x_cols, y_var, df_test=df_test, reg_type=regr_type,
                                                 significativas=signif,
                                                 n_vars=num_variables)

            ###### Calculate liquidations: LIQUIDATIONS

            # liquid = []
            vars = df_reg['vars'][0]
            coefs = df_reg['coef'][0]
            mape = df_reg['MAPE'][0]
            if regr_type != 'OLS':
                intercept = df_reg['intercept'][0]
            # new

            res_pred = np.concatenate([np.ravel(rr) for rr in df_reg['pred']])

            ###### CALCULATIONS

            df_res['vars'] = [vars]
            if regr_type != 'OLS':
                df_res['intercept'] = [intercept]
            df_res['coefs'] = [coefs]
            df_res['real_date'] = date_max
            df_res['forward_date'] = ind_test
            df_res['Prevision'] = res_pred[0]  # valor predicho
            df_res['Real'] = float(df_test[y_var][0])  # valor real
            # df_res['total_liquid'] = sum(liquid)
            df_res['r2'] = df_reg['r2'][0]
            df_res['mape'] = [mape]
            df_res['mes_vista'] = [step]

            df_total = pd.concat([df_total, df_res], axis=0)
    df_total['Mape_final'] = df_total.groupby('mes_vista')['mape'].transform('mean')
    return df_total.reset_index(drop=True)


def back_testing_regression_rolling_OLD(df: pd.DataFrame(), x_cols, y_var,  initial_date: str = '2021-11-01',
                             final_date: str = '2023-09-01', signif: bool = False,
                             regr_type='OLS', num_variables: int = 4, window: int = 48, step_ahead: int = 12):
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
    if regr_type=='OLS':
        x_cols = sm.add_constant(x_cols)


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
            print(F"date{date} ,date_max_step: {date_max_step}, step {step}")


            ###### Regression with the forward values
            #adding the constant cause the library statsmodel need it while skkit learn does not
            if regr_type=='OLS':
                df_out2 = sm.add_constant(df_out2)
                df_test.insert(0, 'const', 1.0)
                df_reg = regression_OLD_OLS(df_out2, x_cols, y_var, df_test=df_test, reg_type=regr_type, significativas=signif,n_vars=num_variables)

            if regr_type == 'Linear':
                df_reg = regression_OLD_original(df_out2, x_cols, y_var, df_test=df_test, reg_type=regr_type,
                                                 significativas=signif,
                                                 n_vars=num_variables)

            ###### Calculate liquidations: LIQUIDATIONS

            # liquid = []
            vars = df_reg['vars'][0]
            coefs = df_reg['coef'][0]
            mape = df_reg['MAPE'][0]
            if regr_type != 'OLS':
                intercept = df_reg['intercept'][0]

            # new




            res_pred = np.concatenate([np.ravel(rr) for rr in df_reg['pred']])

            ###### CALCULATIONS

            df_res['vars'] = [vars]
            if regr_type != 'OLS':
                df_res['intercept'] = [intercept]
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



def stepwise_regression_OLD(X, y, n_vars: int = 4):
    '''
    It performs the step wise method for the variables entered in the dataframe X. It is delimited with the parameter n_vars.
    The step-wise method in regression is a systematic approach to select the most relevant variables and build an optimal regression mode.
    If no variables are found relevant, it defaults to the Pool Avg variable
    Input:
        X: Dataframe. Dataframe with all coluns to performa the Step Wise regression
        y: Array. Array constaining the objective variable values
        n_cols: Integer. The maximum number of columns to select

    Output: List.
        It outputs a list containing the name of the relevant columns found with 2 particular rules applied:
            When no column is found significant, it defaults to the Pool Avg column
            When more than n_vars variables are found, it takes only the n_vars more relevant columns
    '''
    included = list(X.columns)
    while True:
        if 'const' in included:
            model = sm.OLS(y, (X[included])).fit()
        else:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
        #print(model.summary())

        max_pval = model.pvalues[1:].max()  # Excluye el p-valor del intercepto

        if max_pval > 0.05:
            excluded_feature = model.pvalues[1:].idxmax()
            included.remove(excluded_feature)
        else:
            pvals = model.pvalues[1:].sort_values(ascending=False)
            break
    if len(included) == 0:
        included = ['EXIS_INIC', 'IMPORTS', 'EXPORTS', 'INNER_CONS', 'PRODUCTION', 'PRODUCTION_HARVEST','INTERNAL_DEMAND','TOTAL_DEMAND','TOTAL_CONS']
    elif len(included) > (n_vars):
        vars_to_rmv = len(included) - n_vars
        included = list(pvals.index[vars_to_rmv:])
    else:
        pass

    return included


def regression_OLD_original(dataframe: pd.DataFrame(), variables: list, y_var: str, df_test=None, reg_type: str = 'Linear',
               significativas: bool = False, combs: bool = False, n_vars: int = 4):
    '''
    It performs the regression based on the dataframe passed. It outputs a dataframe with all metrics and data relevant to the regression.

    Input:
        dataframe: Dataframe. Dataframe that contains all variables, the objective function
        variables: List. List constaining the objective variable values
        y_var: String. The objective variable  of the regression
        df_test: Dataframe. It is the test dataframe in which the regression is evaluated
        reg_type: String. Linear for linear regression or Huber for robust regression
        significativas: Boolean. True to apply the Step Wise method in the regression. False to select all variales passed in x_cols.
        combs: Boolean. It calculates the regression for each combination of variables, with a maximum of n_vars
        n_vars: Integer. Maximum number of variables passed to the regression. It is used when using the combinations (The parameter combs must be True) and when using Step Wise regression (The parameter signif must be True)

    Output: Dataframe.
        It outputs a dataframe with multiple columns:
            Num variables: Number of variables used in the regression
            Variables: String with the concatenation of the variables used in the regression
            vars: List of the variables used in the regression
            formula: String with the formula used in the regression
            coef: Coefficients of the variables used in the regression
            intercept: Intercept of the regression
            r2: R2 of the regression
            adj_r2: Adjusted R2 of the regression
            MSE:  Mean Square Error of the regression
            MAE:  Mean Average Error of the regression
            MAPE: Mean Average Percentage Error of the regression
            skewness: skewness of the regression
            kurtosis: kurtosis of the regression
            perc_80: Quantile 0.8 of the residuals of the regression
            perc_95: Quantile 0.95 of the residuals of the regression
            Cointegrados: If the residuals are cointegrated. 1 if yes, 0 if no
            Estructura: If the residuals has structure. 1 if yes, 0 if no
            test: Objective variable real values of the test dataframe
            pred: Objective variable predictions
            residuos: Residuals. Objective variable real values minus its predictions
    '''
    resultados = []
    combinaciones = []
    combinaciones_filtradas = []
    res_test = []
    res_pred = []

    if combs:
        max_variables = min(len(variables), n_vars)
        for r in range(3, max_variables + 1):
            combinaciones.extend(itertools.combinations(variables, r))

        for combinacion in combinaciones:
            has_similar_name = False
            if len(set([var[:4] for var in combinacion])) == len(combinacion):
                combinaciones_filtradas.append(combinacion)
    else:
        combinaciones_filtradas = list([variables])

    for combinacion in combinaciones_filtradas:
        variables_comb = list(combinacion)
        x = dataframe[variables_comb]
        y = dataframe[y_var]
        x_test_in_sample = x.copy()
        y_test_in_sample = y.copy()

        if df_test is None:
            x_test = x.copy()
            y_test = y.copy()
        else:  # IF we are using out-of-sample bt
            x_test = df_test[variables_comb]
            try:
                y_test = df_test[y_var]
            except:
                y_test = pd.Series([0] * len(df_test))

        if significativas:  # Apply Step Wise Regression

            variables_comb = stepwise_regression_OLD(x, y, n_vars)

            x = x[variables_comb]
            x_test = x_test[variables_comb]
            x_test_in_sample = x[variables_comb].copy()

        if reg_type == 'Huber':  # Robust regression

            model = HuberRegressor()
            result = model.fit(x, y)

            coeficientes = model.coef_
            intercept = model.intercept_

            y_pred_in_sample = result.predict(x_test_in_sample)
            y_pred = result.predict(x_test)
            r2 = r2_score(y_test_in_sample, y_pred_in_sample)  # get the r2 in-sample

        if reg_type == 'Linear':  # Linear regression

            model = LinearRegression()
            result = model.fit(x, y)

            coeficientes = model.coef_
            intercept = model.intercept_

            y_pred = result.predict(x_test)
            y_pred_in_sample = result.predict(x_test_in_sample)
            r2 = r2_score(y_test_in_sample, y_pred_in_sample)  # get the r2 in-sample

        # Get the formula

        formula = f" Y = {round(intercept, 3)}"

        for i, coef in enumerate(coeficientes):
            formula += f" + {round(coef, 3)} * {variables_comb[i]}"

        n = len(y_test)  # número de observaciones
        p = len(variables_comb)  # número de predictores (p): Multivariante
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        if '_log' in y_var:  # If the columns are logs, we have to retransform the results

            y_test = np.exp(y_test)
            y_pred = np.exp(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Residuals analysis

        residuos = y_test - y_pred

        skewness = skew(residuos)
        kurt = kurtosis(residuos)
        perc_80 = residuos.quantile(0.8)
        perc_95 = residuos.quantile(0.95)

        if df_test is None:
            res_adf = adfuller(residuos)
            if res_adf[1] < 0.05:
                co_int = 1
            else:
                co_int = 0

            lb = acorr_ljungbox(residuos)
            if any(float(p) < 0.05 for p in lb['lb_pvalue']):
                estruc = 1
            else:
                estruc = 0

        else:
            co_int = 0
            estruc = 0

        #   Dataframe creation

        res_test.append(y_test)
        res_test1 = np.concatenate(res_test).tolist()
        res_test1 = np.concatenate([np.ravel(rr) for rr in res_test1])

        res_pred.append(y_pred)
        res_pred1 = np.concatenate(res_pred).tolist()
        res_pred1 = np.concatenate([np.ravel(rr) for rr in res_pred1])

        resultados.append((len(variables_comb), ', '.join(variables_comb), variables_comb, formula, coeficientes,
                           intercept, r2, adj_r2, mse, mae, mape, skewness, kurt, perc_80, perc_95, co_int, estruc,
                           res_test1, res_pred1, np.array(residuos)))

    df_resultados = pd.DataFrame(resultados,
                                 columns=['Num Variables', 'Variables', 'vars', 'formula', 'coef', 'intercept', 'r2',
                                          'adj_r2', 'MSE', 'MAE', 'MAPE', 'skewness', 'kurtosis', 'perc_80', 'perc_95',
                                          'Cointegrados', 'Estructura', 'test', 'pred', 'residuos'])

    return df_resultados


def regression_OLD_OLS(dataframe: pd.DataFrame(), variables: list, y_var: str, df_test=None, reg_type: str = 'OLS',
                       significativas: bool = False, combs: bool = False, n_vars: int = 4):
    '''
    It performs the regression based on the dataframe passed. It outputs a dataframe with all metrics and data relevant to the regression.

    Input:
        dataframe: Dataframe. Dataframe that contains all variables, the objective function
        variables: List. List constaining the objective variable values
        y_var: String. The objective variable  of the regression
        df_test: Dataframe. It is the test dataframe in which the regression is evaluated
        reg_type: String. Linear for linear regression or Huber for robust regression
        significativas: Boolean. True to apply the Step Wise method in the regression. False to select all variales passed in x_cols.
        combs: Boolean. It calculates the regression for each combination of variables, with a maximum of n_vars
        n_vars: Integer. Maximum number of variables passed to the regression. It is used when using the combinations (The parameter combs must be True) and when using Step Wise regression (The parameter signif must be True)

    Output: Dataframe.
        It outputs a dataframe with multiple columns:
            Num variables: Number of variables used in the regression
            Variables: String with the concatenation of the variables used in the regression
            vars: List of the variables used in the regression
            formula: String with the formula used in the regression
            coef: Coefficients of the variables used in the regression
            intercept: Intercept of the regression
            r2: R2 of the regression
            adj_r2: Adjusted R2 of the regression
            MSE:  Mean Square Error of the regression
            MAE:  Mean Average Error of the regression
            MAPE: Mean Average Percentage Error of the regression
            skewness: skewness of the regression
            kurtosis: kurtosis of the regression
            perc_80: Quantile 0.8 of the residuals of the regression
            perc_95: Quantile 0.95 of the residuals of the regression
            Cointegrados: If the residuals are cointegrated. 1 if yes, 0 if no
            Estructura: If the residuals has structure. 1 if yes, 0 if no
            test: Objective variable real values of the test dataframe
            pred: Objective variable predictions
            residuos: Residuals. Objective variable real values minus its predictions
    '''
    resultados = []
    combinaciones = []
    combinaciones_filtradas = []
    res_test = []
    res_pred = []

    if combs:
        max_variables = min(len(variables), n_vars)
        for r in range(3, max_variables + 1):
            combinaciones.extend(itertools.combinations(variables, r))

        for combinacion in combinaciones:
            has_similar_name = False
            if len(set([var[:4] for var in combinacion])) == len(combinacion):
                combinaciones_filtradas.append(combinacion)
    else:
        combinaciones_filtradas = list([variables])

    for combinacion in combinaciones_filtradas:
        variables_comb = list(combinacion)
        x = dataframe[variables_comb].copy()
       # x = sm.add_constant(x)
        y = dataframe[y_var]
        x_test_in_sample = x.copy()
        y_test_in_sample = y.copy()

        if df_test is None:
            x_test = x.copy()
            y_test = y.copy()
        else:  # IF we are using out-of-sample bt
            x_test = df_test[variables_comb].copy()

            try:
                y_test = df_test[y_var]
            except:
                y_test = pd.Series([0] * len(df_test))

        if significativas:  # Apply Step Wise Regression

            variables_comb = stepwise_regression_OLD(x, y, n_vars)

            x = x[variables_comb].copy()
            x_test = x_test[variables_comb]
            x_test_in_sample = x[variables_comb].copy()

        if reg_type == 'Huber':  # Robust regression

            model = HuberRegressor()
            result = model.fit(x, y)

            coeficientes = model.coef_
            intercept = model.intercept_

            y_pred_in_sample = result.predict(x_test_in_sample)
            y_pred = result.predict(x_test)
            r2 = r2_score(y_test_in_sample, y_pred_in_sample)  # get the r2 in-sample

        if reg_type == 'OLS':  # Linear regression

            model = sm.OLS(y, x)
            result = model.fit()

            #print(x_test.columns)
            coeficientes = result.params
            #            intercept = model.intercept_
            y_pred = result.predict(x_test)
            y_pred_in_sample = result.predict(x_test_in_sample)
            r2 = result.rsquared  # get the r2 in-sample

        # Get the formula

        formula = " Y = "

        for i, coef in enumerate(coeficientes):
            formula += f" + {round(coef, 3)} * {variables_comb[i]}"

        n = len(y_test)  # número de observaciones
        p = len(variables_comb)  # número de predictores (p): Multivariante
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        if '_log' in y_var:  # If the columns are logs, we have to retransform the results

            y_test = np.exp(y_test)
            y_pred = np.exp(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f"y_test{y_test}, y_pred{y_pred}")

        # Residuals analysis

        residuos = y_test - y_pred

        skewness = skew(residuos)
        kurt = kurtosis(residuos)
        perc_80 = residuos.quantile(0.8)
        perc_95 = residuos.quantile(0.95)

        if df_test is None:
            res_adf = adfuller(residuos)
            if res_adf[1] < 0.05:
                co_int = 1
            else:
                co_int = 0

            lb = acorr_ljungbox(residuos)
            if any(float(p) < 0.05 for p in lb['lb_pvalue']):
                estruc = 1
            else:
                estruc = 0

        else:
            co_int = 0
            estruc = 0

        #   Dataframe creation

        res_test.append(y_test)
        res_test1 = np.concatenate(res_test).tolist()
        res_test1 = np.concatenate([np.ravel(rr) for rr in res_test1])

        res_pred.append(y_pred)
        res_pred1 = np.concatenate(res_pred).tolist()
        res_pred1 = np.concatenate([np.ravel(rr) for rr in res_pred1])

        resultados.append((len(variables_comb), ', '.join(variables_comb), variables_comb, formula, coeficientes,
                         r2, adj_r2, mse, mae, mape, skewness, kurt, perc_80, perc_95, co_int, estruc,
                           res_test1, res_pred1, np.array(residuos)))

    df_resultados = pd.DataFrame(resultados,
                                 columns=['Num Variables', 'Variables', 'vars', 'formula', 'coef', 'r2',
                                          'adj_r2', 'MSE', 'MAE', 'MAPE', 'skewness', 'kurtosis', 'perc_80', 'perc_95',
                                          'Cointegrados', 'Estructura', 'test', 'pred', 'residuos'])

    return df_resultados

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






