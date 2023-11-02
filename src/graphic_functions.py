import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import numpy as np
from scipy.stats import pearsonr
from src import plot_functions as plf
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from io import BytesIO
import seaborn as sns


def order_by_correlation_size(df,target_variable ='VIRGEN_EXTRA_EUR_kg'):

    correlation_matrix = df.corr()
    max_correlation_with_target = correlation_matrix[target_variable].drop([target_variable]).abs().sort_values(ascending=False)
    ordered_columns = max_correlation_with_target.index.tolist()
    df = df[[target_variable] + ordered_columns]
    return df

def scatterplot_for_years(df, col, target_variable):
    # Create a figure and axes for each variable, making it one-third smaller
    fig, ax = plt.subplots(figsize=(7, 4))

    # Extract unique 'YEAR' values
    unique_years = df['YEAR'].unique()
    num_unique_years = len(unique_years)

    # Create a custom colormap based on the number of unique years
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_unique_years))
    custom_cmap = mcolors.ListedColormap(colors)

    # Create a normalization object for the colormap
    norm = mcolors.Normalize(vmin=unique_years.min(), vmax=unique_years.max())

    # Extract data for the scatter plot

    # first not null value
    x_data = df.loc[df.index[df[col].notnull()].min():]
    x_data = x_data[col].fillna(method='ffill').fillna(method='bfill')
    df.loc[x_data.index[0 :],target_variable]

    y_data = df.loc[x_data.index[0 :],target_variable]

    #y_data = df[target_variable]
    year_values = df.loc[x_data.index[0 :],'YEAR']

    # Create scatter plot
    scatter = ax.scatter(x_data, y_data, c=year_values, cmap=custom_cmap, norm=norm)

    # Fit the linear regression model
    x_data_filled = x_data.fillna(method='ffill').fillna(method='bfill')
    X = sm.add_constant(x_data_filled)
    model = sm.OLS(y_data, X).fit()

    # Get the coefficients and R-squared value
    intercept, slope = model.params
    r_squared = model.rsquared

    # Add regression line to the scatter plot
    ax.plot(x_data_filled, intercept + slope * x_data_filled, color='red', label=f'Line')

    # Display R-squared value on the plot
    ax.text(0.05, 0.9, f'R-squared = {r_squared:.2f}', transform=ax.transAxes, fontsize=10, color='red')

    # Set labels and title for the plot
    ax.set_xlabel(col)
    ax.set_ylabel(target_variable)
    ax.set_title(f'Scatter Plot of {col}')

    # Create legend elements based on unique 'YEAR' values
    legend_elements = [mlines.Line2D([], [], marker='o', color='w', label=str(year),
                                      markersize=6, markerfacecolor=custom_cmap(norm(year)))
                       for year in unique_years]

    # Add legend with legend elements on the right
    ax.legend(handles=legend_elements, title='YEAR', loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

    # Save the figure as an image (PNG format)
  #  plt.savefig(f'scatter_plot_{col}.png', bbox_inches='tight')

    # Show the plot
    #plt.show()
    print(col)
    image_buffer = BytesIO()
    image_buffer.seek(0)
    plt.savefig(image_buffer, bbox_inches='tight')
    plt.close()
    return image_buffer


def plot_and_save_variables(df, y1_var_name, y2_var_name, temp='Monthly'):
    # Obtener los valores de las variables
    x = df.index
    y1 = df[y1_var_name]
    y2 = df[y2_var_name]

    # Crear el gráfico
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'b-', label=y1_var_name)
    ax2.plot(x, y2, 'r-', label=y2_var_name)

    # Ajustar las etiquetas del eje x
    if temp == 'Monthly':
        plt.xticks(x[::12], pd.to_datetime(x[::12]).year)
    if temp == 'Weekly':
        plt.xticks(x[::52], pd.to_datetime(x[::52]).year)

    # Personalizar el gráfico
    ax1.set_ylabel(y1_var_name, color='b')
    ax2.set_ylabel(y2_var_name, color='r')
    plt.title(f'{y1_var_name} vs {y2_var_name}')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    image_buffer = BytesIO()
    plt.savefig(image_buffer)
    image_buffer.seek(0)
    plt.close()
    return image_buffer

    """"doc.add_picture(image_buffer, width=Inches(4), height=Inches(3))

    # Add two line breaks after the image
    doc.add_paragraph('')
    doc.add_paragraph('')  # You can add more if needed"""


# ROLLING CORRELATION


def rolling_corr(variable1, variable2, ventana):
    roll_corr = variable1.rolling(window=ventana).corr(variable2)
    return roll_corr


def rolling_corr_post(variable1, variable2, ventana):
    roll_corr = variable1.rolling(window=ventana).corr(variable2)
    roll_corr_posterior = roll_corr.shift(-ventana)
    return roll_corr_posterior


def print_correlation_over_time(df, col, target_variable):
    # this function show how the rolling correlation between the 2 variable analyzed in the same moment has
    # changed over time
    for variable_explicativa in [col]:
        if col != target_variable and col != 'YEAR':
            variable_objetivo = target_variable
            df1 = df[[variable_explicativa, variable_objetivo]]
            df1 = df1.loc[df.index[df[col].notnull()].min():]
            name = f'corr_{variable_explicativa}_{variable_objetivo}'
            df1[name] = rolling_corr(df[variable_explicativa], df[variable_objetivo], 36)
            buffer = plot_and_save_variables(df1, variable_objetivo, name, temp='Monthly')
            return buffer

            # saving in a document:


def cross_correlation_variable(df, col, target_col, nlags):
    # using fill method starting from the 1st not null value for every column
    # this function show the correlations between t variables in different time lags, using the cross correlation function
    df = df.loc[df.index[df[col].notnull()].min():]
    #line commented, leave like this if no errors occur
   # df[col] = df[col].fillna(method='ffill')

    #modified the order of the 1st 2 params as test to see if it makes much more sense
    cross_corr = sm.tsa.stattools.ccf(df[target_col],df[col], adjusted=False)


    # Get the first 'nlags' lines of cross-correlation
    cross_corr_first_lags = cross_corr[:nlags]

    # Calculate the average correlation for the selected period
    average_corr = np.mean(cross_corr_first_lags)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot cross-correlation
    ax.plot(cross_corr_first_lags, marker='o', linestyle='-', color='b', label='Cross-Correlation')

    # Add horizontal line for the average correlation
    ax.axhline(y=average_corr, color='purple', linestyle='--', label='Average Correlation')

    # Set labels and title
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Cross-Correlation for {col} with {target_col}')
    ax.legend()

    # Show the plot
    image_buffer = BytesIO()
    plt.savefig(image_buffer, bbox_inches='tight')
    image_buffer.seek(0)
    plt.close()
    return image_buffer

def cross_correlation_variable_out_df(df, col, target_col, nlags):
    # using fill method starting from the 1st not null value for every column
    # this function show the correlations between t variables in different time lags, using the cross correlation function
    df = df.loc[df.index[df[col].notnull()].min():]
    #line commented, leave like this if no errors occur
   # df[col] = df[col].fillna(method='ffill')

    #modified the order of the 1st 2 params as test to see if it makes much more sense
    cross_corr = sm.tsa.stattools.ccf(df[col],df[target_col], adjusted=False)


    # Get the first 'nlags' lines of cross-correlation
    cross_corr_first_lags = cross_corr[:nlags]


    # Calculate the average correlation for the selected period

    return cross_corr_first_lags


def custom_ccf(df,col, target_variable, nlags):
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Initialize an array to store correlation values
    correlations = []

    # Loop to create and store lagged variables
    for lag in range(1, nlags + 1):
        # Create a new variable name for the lagged variable
        varname_lag = f"{col}_lag_{lag}"

        # Perform the lag by shifting the target variable
        df_copy[varname_lag] = df_copy[col].shift(lag)

        # Calculate the correlation between the lagged variable and the target variable
        correlation = df_copy[varname_lag].corr(df_copy[target_variable])

        # Append the correlation value to the array
        correlations.append(correlation)

    # Return the array of correlation values
    average_corr = np.mean(correlations)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot cross-correlation
    ax.plot(correlations, marker='o', linestyle='-', color='b', label='Cross-Correlation')

    # Add horizontal line for the average correlation
    ax.axhline(y=average_corr, color='purple', linestyle='--', label='Average Correlation')

    # Set labels and title
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Cross-Correlation for lagged {col} with {target_variable}')
    ax.legend()

    # Show the plot
    image_buffer = BytesIO()
    plt.savefig(image_buffer, bbox_inches='tight')
    image_buffer.seek(0)
    plt.close()
    return image_buffer


def compute_lags_for_custom_ccf_IMC(df,nlags, col_to_lag = 'VIRGEN_EXTRA_EUR_kg',target_variable= 'IMC_EXTRA_VIRGEN_EUR_kg'):
    df1 = df[[target_variable,col_to_lag]]

    for lag in range (1,nlags+1):
        varname_lag = f"{col_to_lag}_lag_{lag}"
        df1[varname_lag] = df[col_to_lag].shift(lag)
    return df1


def custom_ccf_IMC(df,col='VIRGEN_EXTRA_EUR_kg', target_variable='IMC_EXTRA_VIRGEN_EUR_kg'):
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Initialize an array to store correlation values
    correlations = []

    # Loop to create and store lagged variables
    for col_name in df.columns:
        if col_name != target_variable:

            # Calculate the correlation between the lagged variable and the target variable
            correlation = df_copy[col_name].corr(df_copy[target_variable])

            # Append the correlation value to the array
            correlations.append(correlation)
    average_corr = np.mean(correlations)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot cross-correlation
    ax.plot(correlations, marker='o', linestyle='-', color='b', label='Cross-Correlation')

    # Add horizontal line for the average correlation
    ax.axhline(y=average_corr, color='purple', linestyle='--', label='Average Correlation')

    # Set labels and title
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Cross-Correlation for lagged {col} with {target_variable}')
    ax.legend()

    # Show the plot
    plt.show()

    return correlations


def plot_correlation_matrix(df):
    df = order_by_correlation_size(df)
    correlation_matrix_month = df.corr()
    # Create custom diverging colorscale with blue for values close to 1 and red for values close to -1
    cmap = LinearSegmentedColormap.from_list("custom", ["red", "white", "blue"])

    # Create heatmap using seaborn with custom colorscale
    plt.figure() # figsize=(28, 26)
    sns.heatmap(correlation_matrix_month, annot=True, fmt=".2f", cmap=cmap, center=0,cbar=False)  # Use custom colorscale
    plt.title("Correlation Heatmap with Custom Colorscale")
    plt.show()

def plot_correlation_target_variable(df, target_variable = 'VIRGEN_EXTRA_EUR_kg'):
    # change the df to df_outliers
    # Calculate correlation between predictors (features) and the target variable (y)
    correlation_with_y = df.corr()[target_variable].drop(
        target_variable)  # Replace 'EVOO_ES' with your actual target variable name

    # Create a bar plot to visualize the correlations
    plt.figure(figsize=(16, 12))  # Adjust the figure size as needed
    sns.barplot(x=correlation_with_y.values, y=correlation_with_y.index, orient='h', palette='viridis')
    plt.xlabel(f"Correlation with {target_variable}")  # Update with your target variable's name
    plt.ylabel('Predictor Variables')
    plt.title(f"Correlation of Predictor Variables with {target_variable}")
    plt.show()



def mark_outliers(df, column_name):
    df1= df[[column_name]]
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the outlier threshold (3 times the IQR)
    outlier_threshold = 3 * IQR

    # Create a new column 'IsOutlier' in the DataFrame
    df1[column_name+'_is_outlier'] = np.where((df[column_name] > Q3 + outlier_threshold) | (df[column_name] < Q1 - outlier_threshold), 1, 0)

    return df1[column_name+'_is_outlier']


def plot_time_series_with_outliers(df, column_name, is_outlier_column):
    # Extract the time series values and 'IsOutlier' values
    time_series = df[column_name]
    is_outlier = df[is_outlier_column] == 1

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the time series
    ax.plot(df.index, time_series, label=column_name)

    # Highlight the outlier values with red markers
    ax.scatter(df.index[is_outlier], time_series[is_outlier], color='red', marker='o', label='Outliers')

    # Set plot labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel(column_name)
    ax.set_title(f'Time Series with Outliers for {column_name}')
    ax.legend()

    # Show the plot
    plt.show()