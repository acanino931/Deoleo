import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from scipy.stats import kurtosis, skew
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import ndiffs, nsdiffs
from sklearn.utils import resample
from statsmodels.stats.diagnostic import acorr_ljungbox
from src import aux_functions as aux
from src import config as conf

def stepwise_regression(X, y,n_vars:int = 4):
    included = list(X.columns)
    while True:
        model = sm.OLS(y, sm.add_constant(X[included])).fit()

        max_pval = model.pvalues[1:].max()  # Excluye el p-valor del intercepto


        if max_pval > 0.05:
            excluded_feature = model.pvalues[1:].idxmax()
            included.remove(excluded_feature)
        else:
            pvals = model.pvalues[1:].sort_values(ascending = False)
            break
    if len(included) == 0:
        included = ['POOL AVG']
    elif len(included) > (n_vars):
        vars_to_rmv = len(included) - n_vars
        included = list(pvals.index[vars_to_rmv:])
    else:
        pass
    return included
    
def regression(dataframe: pd.DataFrame(), variables:list, y_var : str , df_test = None,summary:bool = True, reg_type:str = 'Huber',significativas:bool = False, plot_regression:bool = False, combs:bool = False,n_vars:int = 4):
    resultados = []
    combinaciones = []
    combinaciones_filtradas = []
    best_r2 = None
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
        
    # Realizar las regresiones lineales para cada combinación de variables
    

    for combinacion in combinaciones_filtradas:
        variables_comb =  list(combinacion)
        x = dataframe[variables_comb]
        y = dataframe[y_var]
        x_test_in_sample = x.copy()
        y_test_in_sample = y.copy()

        if df_test is None:
            x_test = x.copy()
            y_test = y.copy()
        else:                               # IF we are using out-of-sample bt
            x_test = df_test[variables_comb]
            try:
                y_test = df_test[y_var]
            except:
                y_test = pd.Series( [0] * len(df_test))

        if significativas: # Aplicar Step Wise Regression

            variables_comb = stepwise_regression(x,y,n_vars)
            
            print(variables_comb)
            x = x[variables_comb]
            x_test = x_test[variables_comb]
            x_test_in_sample = x[variables_comb].copy()

        if reg_type == 'Huber':

            model = HuberRegressor()
            result = model.fit(x,y)

            coeficientes = model.coef_
            intercept = model.intercept_

            y_pred_in_sample = result.predict(x_test_in_sample)
            y_pred = result.predict(x_test)
            perc_out = 0
            r2 = r2_score(y_test_in_sample, y_pred_in_sample)# Siempre obtengo el r2 in-sample

        if reg_type == 'Ransac':   
            '''
            In most statistical applications, some distributions may have heavy tails, and therefore small sample numbers may skew statistical estimation. 
            Robust estimators solve this by weighing the data differently. RANSAC on the other hand makes no attempt to accommodate the outliers,
            it's built for cases where the data points genuinely don't belong, not just distributed non-normaly.
            '''
            total_samples = len(x)
            residual_threshold = np.percentile(np.abs(y_test - np.median(y_test)), 75) # Max un 25% de outliers

            model = RANSACRegressor(random_state=0, max_trials=1000,residual_threshold = residual_threshold,
                                min_samples = total_samples)
                                    
                                        
            result = model.fit(x,y)

            coeficientes = model.estimator_.coef_
            intercept = model.estimator_.intercept_

            y_pred = result.predict(x_test)
            y_pred_in_sample = result.predict(x_test_in_sample)
            if df_test is None:
                r2 = r2_score(y_test_in_sample[model.inlier_mask_], y_pred_in_sample[model.inlier_mask_])# Siempre obtengo el r2 in-sample
            else:
                r2 = r2_score(y_test_in_sample, y_pred_in_sample)# Siempre obtengo el r2 in-sample
            
            
            inlier_mask = model.inlier_mask_
            perc_out =  100 * (1 - np.mean(inlier_mask))
            
        if reg_type == 'Linear':
            
            model = LinearRegression()
            
            
            result = model.fit(x,y)

            coeficientes = model.coef_  
            intercept = model.intercept_
            
            y_pred = result.predict(x_test)
            y_pred_in_sample = result.predict(x_test_in_sample)
            perc_out = 0
            r2 = r2_score(y_test_in_sample, y_pred_in_sample) # Siempre obtengo el r2 in-sample
        

        variables_implicadas = variables_comb

        # Get the formula

        formula = f" Y = {round(intercept,3)}"

        for i, coef in enumerate(coeficientes):
            formula += f" + {round(coef,3)} * {variables_implicadas[i]}"

            # Save the best model to plot
    
        if best_r2 is None or r2 > best_r2:
            best_r2 = r2
            y_pred_plot = y_pred
            best_formula = formula


        n = len(y_test) #número de observaciones
        p = len(variables_comb) #número de predictores (p): Multivariante
        
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        if  '_log' in y_var: # If the columns are logs, we have to retransform the results
            
            y_test = np.exp(y_test)
            y_pred = np.exp(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) 
        
        # Estudio de los residuos

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

        #   Results

        res_test.append(y_test)
        res_test1 = np.concatenate(res_test).tolist()
        res_test1 = np.concatenate([np.ravel(rr) for rr in res_test1])

        res_pred.append(y_pred)
        res_pred1 = np.concatenate(res_pred).tolist()
        res_pred1 = np.concatenate([np.ravel(rr) for rr in res_pred1])

        resultados.append((len(variables_implicadas),', '.join(variables_implicadas), variables_implicadas, formula, coeficientes,intercept,r2, adj_r2,mse, mae, mape,skewness, kurt, perc_80,perc_95,  perc_out, co_int, estruc,res_test1,res_pred1, np.array(residuos)))
    
    df_resultados = pd.DataFrame(resultados, columns=['Num Variables', 'Variables','vars','formula','coef','intercept', 'r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','perc_outliers','Cointegrados', 'Estructura','test','pred','residuos'])

    # PLOT
    if plot_regression:

        plt.scatter(y, y_pred_plot, label='Predicted values')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='r', label='Regression')
        plt.xlabel('Real (y)')
        plt.ylabel('Predicted values')
        plt.title(best_formula)
        plt.text(np.min(y), np.max(y_pred_plot), f' R2 = {round(best_r2,3)}', color='black')
        plt.show()

    return df_resultados





def regression_backtest(df: pd.DataFrame(),years:list, x_cols:list, y_var:str, iterations:int = 10 , reg_type:str =  'Huber', window:int = 12, combinations = False):

    df_total = pd.DataFrame()

    for ye in years:
        df = df.loc[f'{ye}-01-01':]

        unique_dates = df.index.unique()
        
        n_blocks = len(unique_dates) // window

        unique_dates1 = unique_dates[:-(window+1)]

        for it in range(iterations):

            if it == 0:

                df_res = regression(df, x_cols, y_var, reg_type =  reg_type, combs = combinations)
                df_res['Origin'] = 'Original'
                df_res['year'] = ye
                df_total = pd.concat([df_total, df_res], axis=0)

            else:

            # # Bootstrap in-sample

            #     df_in = resample(df, random_state = 0)

            #     df_in_res = regression(df_in, x_cols, y_var, reg_type =  reg_type, combs = combinations)
            #     df_in_res['Origin'] = 'In-sample'
            #     df_in_res['year'] = ye
            #     df_total = pd.concat([df_total, df_in_res], axis=0)
                

            # Bootstrap out-of-sample
                
                        # Train-test bootstraping
               
                mask = np.ones(len(df), dtype=bool)
                mask[np.random.choice(len(df), size=int(len(df) * 0.2), replace=False)] = False
                df_out = df[mask]
                df_test = df[~mask]

                df_out_res = regression(df_out, x_cols, y_var, df_test = df_test,reg_type =  reg_type, combs = combinations)
                df_out_res['Origin'] = 'Out-sample-traintest'
                df_out_res['year'] = ye
                df_total = pd.concat([df_total, df_out_res], axis=0)

                # Rolling window bootstraping
                
                t = 3
                unique_dates2 = unique_dates1[:-(t+1)]

                for idx,i in enumerate(unique_dates2):
                    
                    date = i.date()
                    date_max = unique_dates[idx + window -1].date()
                    date_max_1 = unique_dates[idx + window ].date()
                    date_max_t = unique_dates[idx + window + t -1].date()
                    
                    
                    df_out2 = df.loc[date : date_max]
                    df_test2 = df.loc[date_max_1 : date_max_t]

                    df_out_res = regression(df_out2, x_cols, y_var, df_test = df_test2, reg_type =  reg_type, combs = combinations)
                    
                    df_out_res['Origin'] = 'Out-sample-Rolling'
                    df_out_res['year'] = ye
                    
                    df_total = pd.concat([df_total, df_out_res], axis=0)
                

                # Expanding window bootstraping

                t = 3
                unique_dates2 = unique_dates1[:-(t+1)]
                ii = pd.Timestamp('2019-01-01 00:00:00')
                for idx,i in enumerate(unique_dates2):
                    
                    if idx == 1:
                        ii = i

                    date = i.date()
                    date_ini = ii.date()
                    date_max = unique_dates[idx + window -1].date()
                    date_max_1 = unique_dates[idx + window ].date()
                    date_max_t = unique_dates[idx + window + t -1].date()
                    
                    df_out3 = df.loc[date_ini : date_max]
                    df_test3 = df.loc[date_max_1 : date_max_t]

                    df_out_res = regression(df_out3, x_cols, y_var, df_test = df_test3, reg_type =  reg_type, combs = combinations)

                    df_out_res['Origin'] = 'Out-sample-Expand'
                    df_out_res['year'] = ye
                    
                    df_total = pd.concat([df_total, df_out_res], axis=0)

                        # Block bootstraping

                block_date = pd.Timestamp(np.random.choice(unique_dates1))
                position = unique_dates1.get_loc(block_date)
                date_max = unique_dates[position + window].date()
                
                block = df.loc[block_date : date_max ]
                df_out4 = df.drop(block.index)


                df_out_res = regression(df_out4, x_cols, y_var, df_test = block,reg_type =  reg_type, combs = combinations)
                df_out_res['Origin'] = 'Out-sample-block'
                df_out_res['year'] = ye
                df_total = pd.concat([df_total, df_out_res], axis=0)
                
                        
                   
                    
                
    return df_total.groupby(['Num Variables', 'Variables','Origin', 'year'])['r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','perc_outliers','residuos'].mean().reset_index()



# Toma un df diario, hace resampling con sustitución en los días y luego calcula la media mensual. Esto lo hace un numero de iteraciones.

def regression_bootstrap(df:pd.DataFrame(),years:list, x_cols:list, y_var:str,  bt_iterations: int = 10,  reg_type:str =  'Huber',  combinations = False):
    df_total = pd.DataFrame()
    res = []
    for i in range(bt_iterations):
        if i == 0:
            df1 = df.resample('MS').mean()
        else:
            df1 = aux.df_bootstrap(df, seed = i+1)
        
        df2 = regression(df1,x_cols,y_var,reg_type =reg_type ,plot_regression = False, combs = combinations )
        if i == 0:
            df2['Origin'] = 'Original'
        else:
            df2['Origin'] = 'Bootstrap'
        
        df_total = pd.concat([df_total, df2], axis=0)
        # res1 = np.mean(df_total['residuos'][0])
        res.append(df_total['residuos'][0])

    res1 = np.concatenate(res).tolist()
    res1 = np.concatenate([np.ravel(rr) for rr in res1])

    return res1,df_total.groupby(['Num Variables', 'Variables','Origin'])['r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','perc_outliers'].mean().reset_index()



def traintest_bt(df: pd.DataFrame(), x_cols, y_var,reg_type =  'Huber',iterations:int = 10, window:int = 12, combs:bool = False):
    """Bootstraps the data using train test split """
    df_total = pd.DataFrame()
    res = pd.DataFrame()
    for i in range(iterations):
        mask = np.ones(len(df), dtype=bool)
        mask[np.random.choice(len(df), size=int(len(df) * 0.2), replace=False)] = False
        df_out = df[mask]
        df_test = df[~mask]

        df_out_res = regression(df_out, x_cols, y_var, df_test = df_test,reg_type =  reg_type, combs = combs)

        df_total = pd.concat([df_total, df_out_res], axis=0)

    res = np.concatenate([np.ravel(r) for r in df_total['residuos']])
    return res,df_total.groupby(['Num Variables', 'Variables'])['r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','perc_outliers','residuos'].mean().reset_index()

def rolling_window_bt(df:pd.DataFrame(), x_cols, y_var,regr_type =  'Huber',iterations:int = 10, window:int = 12, combs :bool = False, step_ahead:int = 3):
    """Bootstraps the data using rolling window
    Extracts the mresulting metrics and the y_pred and y_real
    """
    df_total = pd.DataFrame()
    preds = pd.DataFrame()
    unique_dates = df.index.unique()
    n_blocks = len(unique_dates) // window
    unique_dates1 = unique_dates[:-(window+1)]
    unique_dates2 = unique_dates1[:-(step_ahead+1)]
    
    for i in range(iterations):
        for step in range(step_ahead + 1 ):  #To extract the value for each indivicual step
        
            for idx,i in enumerate(unique_dates2):
                
                # Date range interval for the train dataset
                date = i.date()
                date_max = unique_dates[idx + window -1].date()
                name = int(step)

                if step == 0:
                    date_max_1 = unique_dates[idx + window ].date()
                    date_max_t = unique_dates[idx + window + step_ahead -1].date()
                else:
                    date_max_1 = unique_dates[idx + window + step -1].date()
                    date_max_t = unique_dates[idx + window + step -1].date()

                df_out2 = df.loc[date : date_max]
                df_test2 = df.loc[date_max_1 : date_max_t]

                df_out_res = regression(df_out2, x_cols, y_var, df_test = df_test2, reg_type =  regr_type, combs = combs)
                df_out_res['step'] = name
                # Save the data from all iterations
                df_total = pd.concat([df_total, df_out_res], axis=0)


                # Extract pred and real values for posterior analysis
                res_test1 = np.concatenate([np.ravel(rr[0]) for rr in df_out_res['test']])
                res_pred1 = np.concatenate([np.ravel(rr[0]) for rr in df_out_res['pred']])
                step1 = [name] * len(res_pred1)
                
                predictions = pd.DataFrame({'real':res_test1,'pred': res_pred1, 'step' : step1})

                preds = pd.concat([preds, predictions], axis=0)                

    return preds.reset_index(drop = True),df_total.groupby(['Num Variables', 'Variables','step'])['r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95'].mean().reset_index()

def rolling_window_forwards_bt(df:pd.DataFrame(),df_f:pd.DataFrame(), x_cols, y_var,regr_type =  'Huber',iterations:int = 10, window:int = 12, combs :bool = False, step_ahead:int = 3):

    """Bootstraps the data using train test split 
    Extracts the resulting metrics and the y_pred and y_real
    """
    df_total = pd.DataFrame()
    preds = pd.DataFrame()
    unique_dates = df.index.unique()
    unique_dates1 = unique_dates[:-(window+1)]
    unique_dates2 = unique_dates1[:-(step_ahead+1)]
    
    for i in range(iterations):
        for step in range(step_ahead + 1 ):  #To extract the value for each indivicual step
        
            for idx,i in enumerate(unique_dates2):
                
                # Date range interval for the train dataset
                date = i.date()
                date_max = unique_dates[idx + window -1].date()

                date_max_1 = unique_dates[idx + window ].date()
                date_max_t = unique_dates[idx + window + step_ahead -1].date()
                name = int(step)

                if step != 0:
                    date_max_1 = unique_dates[idx + window + step -1].date()
                    date_max_t = unique_dates[idx + window + step -1].date()

                df_out2 = df.loc[date : date_max]
                df_obj = df[y_var].loc[date_max_1 : date_max_t]

                df_forwards = aux.create_forward_df(df_f,cols = conf.forward_1m.keys(),date_ini = str(date_max),date_end = date_max_t)
                df_forwards2 = df_forwards.loc[date_max_1 : date_max_t]

                df_test = df_forwards2.join(df_obj, how = 'left')
                # print(df_test)
                try:
                    df_out_res = regression(df_out2, x_cols, y_var, df_test = df_test, reg_type =  regr_type, combs = combs)
                    df_out_res['step'] = name
                except:
                    pass
                # Save the data from all iterations
                df_total = pd.concat([df_total, df_out_res], axis=0)


                # Extract pred and real values for posterior analysis
                res_test1 = np.concatenate([np.ravel(rr[0]) for rr in df_out_res['test']])
                res_pred1 = np.concatenate([np.ravel(rr[0]) for rr in df_out_res['pred']])
                step1 = [name] * len(res_pred1)
                
                predictions = pd.DataFrame({'real':res_test1,'pred': res_pred1, 'step' : step1})

                preds = pd.concat([preds, predictions], axis=0)
                
                

    return preds.reset_index(drop = True),df_total.groupby(['Num Variables', 'Variables','step'])['r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','perc_outliers'].mean().reset_index()
    

    # Expanding window bootstraping
def expanding_window_bt(df:pd.DataFrame(), x_cols, y_var,reg_type =  'Huber',iterations:int = 10, window:int = 12, combs :bool = False, win_to_predict:int = 3):
    """Bootstraps the data using expanding window """

    df_total = pd.DataFrame()
    unique_dates = df.index.unique()
        
    n_blocks = len(unique_dates) // window

    unique_dates1 = unique_dates[:-(window+1)]
    ii = pd.Timestamp('2019-01-01 00:00:00')
    unique_dates2 = unique_dates1[:-(win_to_predict+1)]
    for i in range(iterations):
        for idx,i in enumerate(unique_dates2):
            
            if idx == 1:
                ii = i

            date = i.date()
            date_ini = ii.date()
            date_max = unique_dates[idx + window -1].date()
            date_max_1 = unique_dates[idx + window ].date()
            date_max_t = unique_dates[idx + window + win_to_predict -1].date()
            
            df_out3 = df.loc[date_ini : date_max]
            df_test3 = df.loc[date_max_1 : date_max_t]

            df_out_res = regression(df_out3, x_cols, y_var, df_test = df_test3, reg_type =  reg_type, combs = combs)

            
            df_total = pd.concat([df_total, df_out_res], axis=0)

    return df_total.groupby(['Num Variables', 'Variables'])['r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','perc_outliers'].mean().reset_index()


def random_block_bt(df:pd.DataFrame(), x_cols, y_var,reg_type =  'Huber',iterations:int = 10, window:int = 12, combs :bool = False):
    """Bootstraps the data using a random block approach"""

    df_total = pd.DataFrame()
    unique_dates = df.index.unique()
        
    n_blocks = len(unique_dates) // window

    unique_dates1 = unique_dates[:-(window+1)]
    for i in range(iterations):
        block_date = pd.Timestamp(np.random.choice(unique_dates1))
        position = unique_dates1.get_loc(block_date)
        date_max = unique_dates[position + window].date()
        
        block = df.loc[block_date : date_max ]
        df_out4 = df.drop(block.index)

        df_out_res = regression(df_out4, x_cols, y_var, df_test = block,reg_type =  reg_type, combs = combs)


        df_total = pd.concat([df_total, df_out_res], axis=0)

    return df_total.groupby(['Num Variables', 'Variables'])['r2','adj_r2', 'MSE', 'MAE','MAPE','skewness','kurtosis','perc_80','perc_95','perc_outliers'].mean().reset_index()