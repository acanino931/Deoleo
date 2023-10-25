# Dict for the sscc files with data extracted form the REE webpage

sscc_files = {'2014': '../1. Data/sscc/sscc_2014.csv',
              '2015': '../1. Data/sscc/sscc_2015.csv',
              '2016': '../1. Data/sscc/sscc_2016.csv',
              '2017': '../1. Data/sscc/sscc_2017.csv',
              '2018': '../1. Data/sscc/sscc_2018.csv',
              '2019': '../1. Data/sscc/sscc_2019.csv',
              '2020': '../1. Data/sscc/sscc_2020.csv',
              '2021': '../1. Data/sscc/sscc_2021.csv',
              '2022': '../1. Data/sscc/sscc_2022.csv',
              '2023': '../1. Data/sscc/sscc_2023.csv',
              'old' : '../1. Data/sscc/Spots_mensuales.xlsm'

            }

# Path of the HT file:

path_ht = '../1. Data/PBF_2014_2023.xlsx'

# SSCC file path

sscc_path = '../1. Data/sscc/liquicomun.xlsx'
# Dict for the sscc files with data extracted form the Commodies webpage

commodities_files = {'OMEL': '../1. Data/OMEL.xlsx',
                     'OMIP': '../1. Data/OMEL.xlsx',
                     'EURUSD': '../1. Data/Reikon_commodities.xlsx',
                    #  'GBPUSD': '../1. Data/Reikon_commodities.xlsx',
                     'EUA': '../1. Data/Reikon_commodities.xlsx',
                     'BRENT': '../1. Data/Reikon_commodities.xlsx',
                    #  'NBP': '../1. Data/Reikon_commodities.xlsx',
                     'API2': '../1. Data/Reikon_commodities.xlsx',
                     'TTF': '../1. Data/Reikon_commodities.xlsx',
                     'MIBGAS PVB': '../1. Data/Reikon_commodities.xlsx'
            }

# Columns to select from the SSCC 
columns = [
         'Restricciones técnicas PDBF',
         'Banda de regulación secundaria',
         'Reserva de potencia adicional a subir',
         'Restricciones técnicas en tiempo real',
         'Incumplimiento de energía de balance', 
         'Coste desvíos',
         'Saldo desvíos', 
         'Control del factor de potencia', 
         'Saldo PO 14.6',
         'Servicios de ajuste',
         'Servicio de interrumpibilidad'
         ]

# Dict for the Spot_mensuales file with historical data of the SSCC until 2010

dict_cols_old = {
         'Fecha_Matlab' : 'Date',
         'Restricciones técnicasPBF' : 'Restricciones técnicas PDBF',
         'Banda de regulación secundaria' : 'Banda de regulación secundaria',
         'Reserva de potencia adicional a subir' : 'Reserva de potencia adicional a subir',
         'Restricciones técnicas en tiempo real' : 'Restricciones técnicas en tiempo real',
         'Incumplimiento energía balance' : 'Incumplimiento de energía de balance', 
         'Desvíos (2)' : 'Coste desvíos', 
         'Excedente desvíos' : 'Saldo desvíos', 
         'Control del factor de potencia' : 'Control del factor de potencia', 
         'saldo entre sistemas' : 'Saldo PO 14.6',
         'BASE' : 'Servicios de ajuste',
         'Servicio de interrumpibilidad' : 'Servicio de interrumpibilidad'
        }


# List with all SPOT variables in the commodities file to consider

spot_vars = [
      'POOL AVG',
      'EUR=',
      'CFI2c5', # Cojo la cotizacion de los EUA de diciembre de ese año
      'BRT-',
      'TRAPI2Mc1',
      'TRNLTTFD1',
      'MIBG-DA1-ES'
      ]

spots_to_var = {
            'POOL AVG':     'POOL AVG',
            'EUR=':         'EUR/USD',
            'CFI2c5':       'EUA',
            'BRT-':         'BRENT',
            'TRAPI2Mc1':    'API2',
            'TRNLTTFD1':    'TTF',
            'MIBG-DA1-ES':  'MIBGAS - PVB',
            'HT':           'HT'
}

# Definición variable objetivo: sscc_4, sscc_8

sscc_4 = ['Restricciones técnicas PDBF', 'Banda de regulación secundaria',
       'Reserva de potencia adicional a subir',
       'Restricciones técnicas en tiempo real']

sscc_8 = ['Restricciones técnicas PDBF', 'Banda de regulación secundaria',
       'Reserva de potencia adicional a subir',
       'Restricciones técnicas en tiempo real','Saldo desvíos',
       'Incumplimiento de energía de balance','Control del factor de potencia', 'Saldo PO 14.6']


# Definicion variables forward:

# Definicion variables forward:

forward_1m = {
            'POOL AVG': 'OMIPFTBMc1',
            'EUR=': 'EUR1MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc1',
            'TRAPI2Mc1': 'TRAPI2Mc2',
            'TRNLTTFD1': 'TRNLTTFMc1',
            'MIBG-DA1-ES': 'MIBGMESMc1'
}

forward_2m = {
            'POOL AVG': 'OMIPFTBMc2',
            'EUR=': 'EUR2MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc2',
            'TRAPI2Mc1': 'TRAPI2Mc3',
            'TRNLTTFD1': 'TRNLTTFMc2',
            'MIBG-DA1-ES': 'MIBGMESMc2'
}

forward_3m = {
            'POOL AVG': 'OMIPFTBMc3',
            'EUR=': 'EUR3MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc3',
            'TRAPI2Mc1': 'TRAPI2Qc1',
            'TRNLTTFD1': 'TRNLTTFMc3',
            'MIBG-DA1-ES': 'MIBGMESQc1'
}

forward_4m = {
            'POOL AVG': 'OMIPFTBMc4',
            'EUR=': 'EUR4MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc4',
            'TRAPI2Mc1': 'TRAPI2Qc2',
            'TRNLTTFD1': 'TRNLTTFMc4',
            'MIBG-DA1-ES': 'MIBGMESQc2'
}

forward_5m = {
            'POOL AVG': 'OMIPFTBMc5',
            'EUR=': 'EUR5MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc5',
            'TRAPI2Mc1': 'TRAPI2Qc2',
            'TRNLTTFD1': 'TRNLTTFQc2',
            'MIBG-DA1-ES': 'MIBGMESQc2'
}

forward_6m = {
            'POOL AVG': 'OMIPFTBMc6',
            'EUR=': 'EUR6MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc6',
            'TRAPI2Mc1': 'TRAPI2Qc2',
            'TRNLTTFD1': 'TRNLTTFQc2',
            'MIBG-DA1-ES': 'MIBGMESQc2'
}

forward_7m = {
            'POOL AVG': 'OMIPFTBQc3',
            'EUR=': 'EUR7MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc7',
            'TRAPI2Mc1': 'TRAPI2Qc3',
            'TRNLTTFD1': 'TRNLTTFQc3',
            'MIBG-DA1-ES': 'MIBGMESQc3'
}
forward_8m = {
            'POOL AVG': 'OMIPFTBQc3',
            'EUR=': 'EUR8MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc8',
            'TRAPI2Mc1': 'TRAPI2Qc3',
            'TRNLTTFD1': 'TRNLTTFQc3',
            'MIBG-DA1-ES': 'MIBGMESQc3'
}
forward_9m = {
            'POOL AVG': 'OMIPFTBQc3',
            'EUR=': 'EUR9MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc9',
            'TRAPI2Mc1': 'TRAPI2Qc3',
            'TRNLTTFD1': 'TRNLTTFQc3',
            'MIBG-DA1-ES': 'MIBGMESQc3'
}
forward_10m = {
            'POOL AVG': 'OMIPFTBQc4',
            'EUR=': 'EUR10MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc10',
            'TRAPI2Mc1': 'TRAPI2Qc4',
            'TRNLTTFD1': 'TRNLTTFQc4',
            'MIBG-DA1-ES': 'MIBGMESYc1'
}
forward_11m = {
            'POOL AVG': 'OMIPFTBQc4',
            'EUR=': 'EUR11MV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc11',
            'TRAPI2Mc1': 'TRAPI2Qc4',
            'TRNLTTFD1': 'TRNLTTFQc4',
            'MIBG-DA1-ES': 'MIBGMESYc1'
}
forward_12m = {
            'POOL AVG': 'OMIPFTBQc4',
            'EUR=': 'EUR1YV=',
            'CFI2c5': 'CFI2c9',
            'BRT-': 'LCOc12',
            'TRAPI2Mc1': 'TRAPI2Qc4',
            'TRNLTTFD1': 'TRNLTTFQc4',
            'MIBG-DA1-ES': 'MIBGMESYc1'
}
# forward_1m = {
#             'POOL AVG': 'OMIPFTBMc1',
#             'EURUSD': 'EUR1MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc1',
#             'API2': 'TRAPI2Mc2',
#             'TTF': 'TRNLTTFMc1',
#             'MIBGAS PVB': 'MIBGMESMc1'
# }

# forward_2m = {
#             'POOL AVG': 'OMIPFTBMc2',
#             'EURUSD': 'EUR2MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc2',
#             'API2': 'TRAPI2Mc3',
#             'TTF': 'TRNLTTFMc2',
#             'MIBGAS PVB': 'MIBGMESMc2'
# }

# forward_3m = {
#             'POOL AVG': 'OMIPFTBMc3',
#             'EURUSD': 'EUR3MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc3',
#             'API2': 'TRAPI2Mc4',
#             'TTF': 'TRNLTTFMc3',
#             'MIBGAS PVB': 'MIBGMESQc1'
# }

# forward_4m = {
#             'POOL AVG': 'OMIPFTBMc4',
#             'EURUSD': 'EUR4MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc4',
#             'API2': 'TRAPI2Qc2',
#             'TTF': 'TRNLTTFMc4',
#             'MIBGAS PVB': 'MIBGMESQc2'
# }

# forward_5m = {
#             'POOL AVG': 'OMIPFTBMc5',
#             'EURUSD': 'EUR5MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc5',
#             'API2': 'TRAPI2Qc2',
#             'TTF': 'TRNLTTFQc2',
#             'MIBGAS PVB': 'MIBGMESQc2'
# }

# forward_6m = {
#             'POOL AVG': 'OMIPFTBMc6',
#             'EURUSD': 'EUR6MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc6',
#             'API2': 'TRAPI2Qc2',
#             'TTF': 'TRNLTTFQc2',
#             'MIBGAS PVB': 'MIBGMESQc2'
# }

# forward_7m = {
#             'POOL AVG': 'OMIPFTBQc3',
#             'EURUSD': 'EUR7MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc7',
#             'API2': 'TRAPI2Qc3',
#             'TTF': 'TRNLTTFQc3',
#             'MIBGAS PVB': 'MIBGMESQc3'
# }
# forward_8m = {
#             'POOL AVG': 'OMIPFTBQc3',
#             'EURUSD': 'EUR8MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc8',
#             'API2': 'TRAPI2Qc3',
#             'TTF': 'TRNLTTFQc3',
#             'MIBGAS PVB': 'MIBGMESQc3'
# }
# forward_9m = {
#             'POOL AVG': 'OMIPFTBQc3',
#             'EURUSD': 'EUR9MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc9',
#             'API2': 'TRAPI2Qc3',
#             'TTF': 'TRNLTTFQc3',
#             'MIBGAS PVB': 'MIBGMESQc3'
# }
# forward_10m = {
#             'POOL AVG': 'OMIPFTBQc4',
#             'EURUSD': 'EUR10MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc10',
#             'API2': 'TRAPI2Qc4',
#             'TTF': 'TRNLTTFQc4',
#             'MIBGAS PVB': 'MIBGMESYc1'
# }
# forward_11m = {
#             'POOL AVG': 'OMIPFTBQc4',
#             'EURUSD': 'EUR11MV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc11',
#             'API2': 'TRAPI2Qc4',
#             'TTF': 'TRNLTTFQc4',
#             'MIBGAS PVB': 'MIBGMESYc1'
# }
# forward_12m = {
#             'POOL AVG': 'OMIPFTBYc1',
#             'EURUSD': 'EUR1YV=',
#             'EUA': 'CFI2c9',
#             'BRENT': 'LCOc12',
#             'API2': 'TRAPI2Yc1',
#             'TTF': 'TRNLTTFYc1',
#             'MIBGAS PVB': 'MIBGMESYc1'
# }

forward_1q = {
            'POOL AVG': 'OMIPFTBQc1',
            'EURUSD': ['EUR1MV=','EUR2MV=','EUR3MV='],
            'EUA': 'CFI2c9',
            'BRENT': ['LCOc1','LCOc2','LCOc3'],
            'API2': 'TRAPI2Qc1',
            'TTF': 'TRNLTTFQc1',
            'MIBGAS PVB': 'MIBGMESQc1'
}

forward_2q = {
            'POOL AVG': 'OMIPFTBQc2',
            'EURUSD': ['EUR4MV=','EUR5MV=','EUR6MV='],
            'EUA': 'CFI2c9',
            'BRENT': ['LCOc4','LCOc5','LCOc6'],
            'API2': 'TRAPI2Qc2',
            'TTF': 'TRNLTTFQc2',
            'MIBGAS PVB': 'MIBGMESQc2'
}

forward_3q = {
            'POOL AVG': 'OMIPFTBQc3',
            'EURUSD': ['EUR7MV=','EUR8MV=','EUR9MV='],
            'EUA': 'CFI2c9',
            'BRENT': ['LCOc7','LCOc8','LCOc9'],
            'API2': 'TRAPI2Qc3',
            'TTF': 'TRNLTTFQc3',
            'MIBGAS PVB': 'MIBGMESQc3'
}

forward_4q = {
            'POOL AVG': 'OMIPFTBQc4',
            'EURUSD': ['EUR10MV=','EU11MV=','EUR1YV='],
            'EUA': 'CFI2c9',
            'BRENT': ['LCOc10','LCOc11','LCOc12'],
            'API2': 'TRAPI2Qc4',
            'TTF': 'TRNLTTFQc4',
            'MIBGAS PVB': 'xxxxx'
}

forward_1y = {
            'POOL AVG': 'OMIPFTBYc1',
            'EURUSD': 'EUR1YV=',
            'EUA': 'CFI2c9',
            'BRENT': 'LCOc12',
            'API2': 'TRAPI2Yc1',
            'TTF': 'TRNLTTFYc1',
            'MIBGAS PVB': 'MIBGMESYc1'
}