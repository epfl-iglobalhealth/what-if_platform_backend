import numpy as np
import pandas as pd
import yaml


class CountryData:

  def __init__(self):
    self.data = pd.read_csv('./model/data/final_data.csv', parse_dates=['date']).set_index('date')
    self.shap = pd.read_csv('./model/data/final_shap.csv')
    self.features = CountryData.extract_feature_names()

  @staticmethod
  def extract_feature_names():
    with open('./model/config/r_estim_features.yaml', 'r', encoding='utf-8') as file:
      swissre_features = yaml.load(file, Loader=yaml.FullLoader)
    constant_columns = swissre_features['demography'] + \
                       swissre_features['sanitary'] + \
                       swissre_features['economic']
    variable_columns = swissre_features['weather'] + \
                       swissre_features['policies']

    return {'constant': constant_columns, 'variable': variable_columns}

  @staticmethod
  def get_sundays_between_dates(start_date: str, end_date: str):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    sundays = pd.date_range(start_date, end_date, freq='W-SUN')
    return sundays

  def get_shap_for_country(self, iso_code):
    data = self.shap[self.shap.iso3 == iso_code][['variable', 'shap_value_normalized']].\
      sort_values(by='shap_value_normalized', ascending=False)
    # approximate shap_values_normalized to 4 decimal places
    data['shap_value_normalized'] = data['shap_value_normalized'].round(4)
    return {'x': data['variable'].values.tolist(), 'y': [{'data':data['shap_value_normalized'].values.tolist()}]}

  def get_constant_features(self, iso_code: str):
    return self.data[self.data.iso_code == iso_code][self.features['constant']].iloc[0].to_dict()

  def get_policies_name(self):
    return self.features['variable'][12:]

  def get_policies_for_a_period(self, iso_code: str, start_date: str, end_date: str):
    # call the function get_sundays to get the sundays of the given period
    sundays = CountryData.get_sundays_between_dates(start_date, end_date)
    end_date = np.datetime64(end_date)
    # if the last value of sundays is different from end_date append end_date to sundays, converting it
    # to a datetime64 object
    if sundays[-1] != end_date:
      sundays = np.append(sundays, end_date)

    # get the data of the given country where the index is equal to the sundays
    data = self.data[(self.data.iso_code == iso_code) & (self.data.index.isin(sundays))][self.get_policies_name()]
    # return the dataframe
    return data

  def get_all_countries(self):
    # get iso2 and iso3 from self.shap dataframe
    data = self.shap[['iso2', 'iso3']].merge(self.data[['iso_code', 'name']], left_on = 'iso3', right_on='iso_code')
    # lowercase iso2
    data['iso2'] = data['iso2'].str.lower()
    # create a column "flag" which is a string. It is equal to "flag-icon flag-icon-{iso2}"
    data['flag'] = data['iso2'].apply(lambda x: 'flag-icon flag-icon-{}'.format(x))
    result = []
    # sort data by name
    data = data.sort_values(by='name')
    for i, iso_code in enumerate(data['iso3'].unique()):
      # add to result id: i, iso_code, country, flag
      result.append({'id': i, 'iso_code': iso_code, 'name': data[data['iso3'] == iso_code]['name'].values[0],
                     'flag': data[data['iso3'] == iso_code]['flag'].values[0]})
    return result








