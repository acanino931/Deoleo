import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from docx import Document
from docx.shared import Inches
from PIL import Image


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

def calcular_acf_pacf(dataframe):
    for columna in dataframe.columns:
        serie_temporal = dataframe[columna]
        
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        
        plot_acf(serie_temporal, lags=25, ax=axs[0])
        axs[0].set_title(f'ACF - {columna}')
        axs[0].xaxis.set_ticks(range(0, 21, 2))  
        
        plot_pacf(serie_temporal, lags=25, ax=axs[1])
        axs[1].set_title(f'PACF - {columna}')
        axs[1].xaxis.set_ticks(range(0, 21, 2)) 
        
        plt.tight_layout()
        plt.show()


def plot_variables(df, y1_var_name, y2_var_name, temp = 'Monthly'):
    # Obtener los valores de las variables
    x = df.index
    y1 = df[y1_var_name]
    y2 = df[y2_var_name]

    # Crear el gráfico
    fig, ax1 = plt.subplots(figsize=(14,8))
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

    return plt.show()




def scatter_regression_adjr2(df, x_var_name, y_var_name, extract_r2: bool = False):

    x = df[x_var_name].to_numpy()
    y = df[y_var_name].to_numpy()

    # Crear el gráfico de dispersión
    plt.scatter(df[x_var_name], df[y_var_name])
    plt.xlabel(x_var_name)
    plt.ylabel(y_var_name)

    # Ajustar una línea de regresión
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    # Calcular el coeficiente de determinación (R-cuadrado)
    r2 = r2_score(y, y_pred)

    # Calcular el número de observaciones (n) y el número de predictores (p)
    n = len(y)
    p = 1 # Bivariante

    # Calcular el coeficiente de determinación ajustado (adjR2)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Obtener la fórmula de la línea de regresión
    slope = model.coef_[0]
    intercept = model.intercept_
    formula = f'{y_var_name} = {slope:.2f} * {x_var_name} + {intercept:.2f}'

    # Mostrar la línea de regresión en el gráfico
    plt.plot(x, y_pred, color='red' )

    # Mostrar la fórmula y el adjR2 en el gráfico
    plt.text(np.min(x), np.max(y), formula, color='black')
    plt.text(np.min(x), np.max(y) - (np.max(y) - np.min(y)) * 0.1, f'adjR2 = {adj_r2:.2f}', color='black')

    plt.title(f'Scatter with adjR2: {x_var_name} vs {y_var_name}')
    
    if extract_r2:
        plt.show()
        return adj_r2,plt.show()
    else:
        return plt.show()
    
def get_plot_analysis(df, x_var_name, y_var_name, extract_coef: bool = False, save_fig: bool = False, temp = 'Monthly'):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    ###### Scatter plot 2 vars with regression line

    plt.subplot(212)
    
    x = df[x_var_name].to_numpy()
    y = df[y_var_name].to_numpy()
    years = df.index.year.to_numpy()  # Columna con los años, ajusta el nombre según tus datos

    # Crear el gráfico de dispersión con colores diferentes según el año
    unique_years = np.unique(years)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_years)))  # Colormap para los colores
    patches = []  # Lista para almacenar los elementos de la leyenda
    for i, year in enumerate(unique_years):
        mask = years == year
        plt.scatter(x[mask], y[mask], color=colors[i], label=str(year))
        patches.append(Patch(color=colors[i], label=str(year)))

    plt.xlabel(x_var_name)
    plt.ylabel(y_var_name)

    # Ajustar una línea de regresión
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    # Obtener el coeficiente de correlación de Pearson: r
    r = df[x_var_name].corr(df[y_var_name])

    # Calcular el coeficiente de determinación (R-cuadrado)
    r2 = r2_score(y, y_pred)

    # Calcular el número de observaciones (n) y el número de predictores (p)
    n = len(y)
    p = 1  # Bivariante

    # Calcular el coeficiente de determinación ajustado (adjR2)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Obtener la fórmula de la línea de regresión
    slope = model.coef_[0]
    intercept = model.intercept_
    formula = f'{y_var_name} = {slope:.2f} * {x_var_name} + {intercept:.2f}'

    # Mostrar la línea de regresión en el gráfico
    plt.plot(x, y_pred, color='red')

    # Mostrar la fórmula y el adjR2 en el gráfico
    plt.text(np.min(x), np.max(y), formula, color='black')
    plt.text(np.min(x), np.max(y) - (np.max(y) - np.min(y)) * 0.1, f'adjR2 = {adj_r2:.2f}', color='black')
    plt.text(np.min(x), np.max(y) - (np.max(y) - np.min(y)) * 0.2, f'r = {r:.2f}', color='black')
    plt.title(f'Scatter with adjR2: {x_var_name} vs {y_var_name}')

    # Agregar leyenda de colores en la esquina superior derecha
    plt.legend(handles=patches, title='Year', loc='upper right')
    
    ####### Temporal plot 2 vars

    plt.subplot(211)

    # Obtener los valores de las variables
    x = df.index
    y1 = df[x_var_name]
    y2 = df[y_var_name]

    # Crear el gráfico
    ax3 = ax1.twinx()
    ax1.plot(x, y1, 'b-', label=x_var_name)
    ax3.plot(x, y2, 'r-', label=y_var_name)

    
    
    
    # Personalizar el gráfico
    ax1.set_ylabel(x_var_name, color='b')
    ax3.set_ylabel(y_var_name, color='r')
    plt.title(f'{x_var_name} vs {y_var_name}')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')

    # Ajustar las etiquetas del eje x
    if temp == 'Monthly':
        plt.xticks(x[::12], pd.to_datetime(x[::12]).year)
    if temp == 'Weekly':
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6)) 
    
    if extract_coef:
        plt.show()
        if save_fig: # To implement
            pass
        return r, adj_r2
    else:
        if save_fig: # To implement
            pass
        return plt.show()
    

def plot_oversampled_data(df: pd.DataFrame(),column_to_plot:str,column_to_look:str = 'value'):
    '''
    Function to plot oversampled data, it must have a column referencing if the data is real or oversampled (column_to_look)
    '''
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extrae los valores del eje x (índice del DataFrame)
    x = df.index

    # Extrae los valores del eje y (columna 'sscc_4')
    y = df[column_to_plot]

    # Extrae los valores de la variable para colorear y trazar líneas
    variable = df[column_to_look]

    # Grafica los puntos donde la variable sea diferente de 1 con scatter
    ax.scatter(x[variable != 1], y[variable != 1], color='blue')

    # Grafica la serie temporal donde la variable sea igual a 1 con una línea
    ax.plot(x[variable == 1], y[variable == 1], color='red')

    # Configura las etiquetas de los ejes
    ax.set_xlabel('Date')
    ax.set_ylabel(column_to_plot)

    ax.legend(['Oversampled values', 'Real values'], loc='upper left')  # Etiquetas de la leyenda y ubicación

    # Muestra el gráfico
    plt.title('Línea de valores reales vs Oversampled')
    plt.show()



def plot_histogram_with_tendencia(residuos, num_bloques):
    plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura según tus preferencias
    plt.hist(residuos, bins=num_bloques, color='skyblue', edgecolor='black', density=True)  # Crea el histograma

    # Estimación de la densidad de kernel
    kde = gaussian_kde(residuos)
    x_vals = np.linspace(min(residuos), max(residuos), 100)
    y_vals = kde(x_vals)
    
    # Agrega la línea de tendencia
    plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Línea de Tendencia')

    # Agrega líneas verticales en los cuantiles 0.05 y 0.2
    quantiles = np.quantile(residuos, [0.05, 0.2, 0.8])
    colors = ['green', 'blue', 'orange']
    labels = ['Cuantil 0.05', 'Cuantil 0.2','Cuantil 0.8']
    for quantile, color, label in zip(quantiles, colors, labels):
        plt.axvline(x=quantile, color=color, linestyle='--', linewidth=2, label=label)

    
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Histogram')
    plt.legend()
    plt.show()

def grafico_columnas_agrupadas(df, title= None, eje:str = 'EUR/MWh'):
    # Convertir el índice a tipo 'datetime' si no está configurado así
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_mes = df.copy()
    # Convertir el índice a una lista de fechas
    fechas = df_mes.index.tolist()

    # Crear el gráfico de columnas agrupadas
    fig, ax = plt.subplots(figsize=(22, 6))  # Ajustar el tamaño de la figura para aprovechar el espacio

    # Seleccionar una paleta de colores más vivos y con mayor contraste
    colores = ['C0', 'C1', 'C2', 'C3', 'C4']

    # Distribuir uniformemente las columnas agrupadas
    ancho = 0.8 / len(df_mes.columns)  # Ajustar el ancho de las columnas
    posiciones = np.arange(len(df_mes.columns)) * ancho

    # Generar las columnas agrupadas
    for i, columna in enumerate(df_mes.columns):
        x = np.arange(len(fechas))
        y = df_mes[columna]
        ax.bar(x + posiciones[i], y, width=ancho, label=columna, color=colores[i])

    # Configurar el título y las etiquetas de los ejes
    ax.set_title(f'Gráfico de cajas {title}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel(eje)

    # Configurar la leyenda
    ax.legend()

    # Configurar las etiquetas del eje x
    ax.set_xticks(np.arange(len(fechas)))
    ax.set_xticklabels([fecha.strftime('%Y-%m') for fecha in fechas], rotation=90)

    # Ajustar el espaciado entre las etiquetas del eje x
    plt.subplots_adjust(bottom=0.2)

    # Ajustar los límites de los ejes x
    ax.set_xlim(-ancho, len(fechas)  + ancho)

    # Mostrar el gráfico
    plt.show()