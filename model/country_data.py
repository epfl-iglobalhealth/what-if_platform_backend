import numpy as np
import pandas as pd
import yaml


class CountryData:

    def __init__(self):
        self.data = pd.read_csv('../model/data/final_data.csv', parse_dates=['date']).set_index('date')
        self.features = CountryData.extract_feature_names()

    def get_constant_features(self, iso_code: str):
        return self.data[self.data.iso_code == iso_code][self.features['constant']].iloc[0].to_dict()

    @staticmethod
    def extract_feature_names():
        with open('../model/config/r_estim_features.yaml', 'r', encoding='utf-8') as file:
            swissre_features = yaml.load(file, Loader=yaml.FullLoader)
        constant_columns = swissre_features['demography'] + \
                         swissre_features['sanitary'] + \
                         swissre_features['economic']
        variable_columns = swissre_features['weather'] +\
                            swissre_features['policies']

        return {'constant': constant_columns, 'variable': variable_columns}