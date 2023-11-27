from src import Regression_Functions as rf
from src import Aux_functions as aux
import statsmodels.api as sm
import pandas as pd
from src import importing_data as imd

def semiautomatic_adding_feature (df, all_columns:list, model_columns:list , target_variable: str = 'VIRGEN_EXTRA_EUR_kg'):
    # this funtion prints all the valuable (significative) added variables for a specific model already built
    y = df[[target_variable]].copy()
    ls_significative = []
    for var in all_columns:
        if var not in model_columns and var != target_variable:
            print(var)
            ls_necessary = model_columns.copy()
            ls_selected = ls_necessary.copy()
            if var not in ls_necessary:
                ls_selected.append(var)
            X = df[ls_selected]
            X = sm.add_constant(X)
            if target_variable in X.columns:
                X.drop(columns =[target_variable],inplace = True)
            model = sm.OLS(y, X).fit()
            print(len(ls_necessary) , len(ls_selected))
            if (model.pvalues < 0.05).all()  and len(ls_necessary) < len(ls_selected): #
                print(model.summary(), X)
                ls_significative.append(var)
    return ls_significative




def stepwise_eliminating(df, target_variable, iterations):
    ls_to_del = []
    data = []  # Initialize an empty list to collect dictionaries

    for i in range(iterations):
        df1 = aux.shuffle_columns(df)
        df1 = df1.drop(columns=ls_to_del)
        X, y, max_data_no_missing, column_data_max = aux.remove_null_rows(df1, target_variable=target_variable)
        X = sm.add_constant(X)
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


def semimanual_single_regressions(df,  all_columns : list ,target_var: str = 'VIRGEN_EXTRA_EUR_kg'):
    # from a list of vaiables that function returns the sorted df of all the single regression parameters
    #use that function to analyze sigificativitiy for each variable in single regression and magnitude co correlations:

    results_list = []

    X_list = all_columns


    target_var = 'VIRGEN_EXTRA_EUR_kg'
    y = df[[target_var]]

    # custom loop for the pdf variables
    for i in range(len(X_list)):
        X_var = X_list[i]

        if 'Produccion' in X_var:
            group = 'Production'
        elif 'Exportacion' in  X_var:
            group = 'Export'
        elif 'Importacion' in  X_var :
            group = 'Import'
        elif 'Consumo' in  X_var :
            group = 'Consume'
        else:
            group = 'Other'

        # Add a constant term to the independent variables
        X_var_running = df[[X_var]]
        X_var_running = sm.add_constant(X_var_running)

        # Fit the OLS regression model
        model = sm.OLS(y, X_var_running).fit()

        # Get R-squared and p-values of coefficients
        r_squared = model.rsquared
        p_values = model.pvalues.drop('const')  # Drop the constant term
        p_value = p_values.to_dict().values()
        p_value = list(p_value)[0]

        # Store the results in the list
        results_list.append({
            'X_var': X_var,
            'R_squared': r_squared,
            'P_Value': p_value,
            'Group': group
        })

    # Create the DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)
    results_df.set_index('X_var', inplace=True)
    results_df = results_df.sort_values(by='R_squared', ascending=False)

    # Print or further analyze the results dataframe
    return results_df



