import requests
import numpy as np
import pandas as pd

class agridata:
    def __init__(self) -> None:
        self.host_url = 'https://www.ec.europa.eu/agrifood/api'
        pass

    def olive_oil(self,info_type, membercode = None, begindate = None, enddate = None, marketingyear = None,products =  None ,
                  markets = None , weeks = None,granularity = None, productionYears = None , months = None)-> pd.DataFrame :
        
        self.info_type = info_type
        self.membercode = membercode
        self.begindate = begindate
        self.enddate = enddate
        self.marketingyear = marketingyear
        self.products =products
        self.markets = markets
        self.weeks = weeks

        

        if info_type ==  'prices':
            
            params = {'memberStateCodes':membercode,
                      'marketingYears':marketingyear,
                      'beginDate': begindate,
                      'endDate'
                      'produts':products,
                      'markets':markets,
                      'weeks':weeks}

            url =  self.host_url +f'/oliveOil/prices?'
            for key,value in params.items():
                if value != None:
                    url = url +f'&{key}={value}'
            

        elif info_type == "production":
            params = {'memberStateCodes':membercode,
                      'marketingYears':marketingyear,
                      'granularity': granularity,
                      'productionYears': productionYears}

            url =  self.host_url +f'/oliveOil/production?'
            for key,value in params.items():
                if value != None:
                    url = url +f'&{key}={value}'

        self.response = requests.get(url)
        if self.response.status_code != 200:
            raise Exception 
        else:
            return pd.DataFrame(self.response.json())
        
    def indicators(self, info_type,membercode = None, years= None,categories= None,indicators= None,
                subindicators= None,parameters= None,units= None,codes= None)-> pd.DataFrame :
        self.info_type =  info_type


        if info_type == 'values':
            params = {'memberStateCodes':membercode,
                        'years':years,
                        'categories': categories,
                        'indicators':indicators,
                        'subindicators':subindicators,
                        'parameters':parameters,
                        'units':units,
                        'codes':codes}
            
            url =  self.host_url +f'/cmefIndicators/values?'
            for key,value in params.items():
                if value != None:
                    url = url +f'&{key}={value}'
        self.response = requests.get(url)
        if self.response.status_code != 200:
            raise Exception 
        else:
            return pd.DataFrame(self.response.json())

    def oil_seeds(self,info_type, membercode = None, begindate = None, enddate = None, marketingyear = None,products =  None ,
                  markets = None , weeks = None,granularity = None, productionYears = None , months = None) -> pd.DataFrame :
        
        self.info_type = info_type
        self.membercode = membercode
        self.begindate = begindate
        self.enddate = enddate
        self.marketingyear = marketingyear
        self.products =products
        self.markets = markets
        self.weeks = weeks

        

        if info_type ==  'prices':
            
            params = {'memberStateCodes':membercode,
                      'marketingYears':marketingyear,
                      'beginDate': begindate,
                      'endDate': enddate,
                      'produts':products,
                      'markets':markets,
                      'weeks':weeks}

            url =  self.host_url +f'/oilseeds/prices?'
            for key,value in params.items():
                if value != None:
                    url = url +f'&{key}={value}'
            

        elif info_type == "production":
            params = {'memberStateCodes':membercode,
                      'marketingYears':marketingyear,
                      'granularity': granularity,
                      'productionYears': productionYears}

            url =  self.host_url +f'/oilseeds/production?'
            for key,value in params.items():
                if value != None:
                    url = url +f'&{key}={value}'

        self.response = requests.get(url)
        if self.response.status_code != 200:
            raise Exception 
        else:
            return pd.DataFrame(self.response.json())