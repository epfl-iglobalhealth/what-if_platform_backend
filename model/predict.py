from datetime import datetime

import numpy as np
import pandas as pd
import simplejson as json
import torch

from model.country_data import CountryData
from model.hybrid import HybridLSTM


class Predict:
  def __init__(self, iso_code: str, window_size: int, economic=False):
    self.iso_code = iso_code
    self.economic = economic
    if economic:
      self.model = HybridLSTM.load_from_checkpoint(checkpoint_path=f"./model/checkpoints/unemp_rate/{self.iso_code}/model.ckpt")
      self.final_data = pd.read_csv('./model/data/final_data_economic.csv', parse_dates=['date']).set_index('date') #
    else:
      self.model = HybridLSTM.load_from_checkpoint(checkpoint_path=f"./model/checkpoints/reproduction_rate/{self.iso_code}/model.ckpt")
      self.final_data = pd.read_csv('./model/data/final_data.csv', parse_dates=['date']).set_index('date')
    self.columns_to_use = CountryData.extract_feature_names(economic)

    self.window_size = window_size

  def predict_for_a_period(self, start_date: str, end_date: str, data=None):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    original_data = self.final_data[(self.final_data['iso_code'] == self.iso_code) &
                                    (self.final_data.index >= start_date) &
                                    (self.final_data.index <= end_date)]

    ground = original_data['shifted_r_estim'] if not self.economic else original_data['unemployment_rate_idx']

    if data is not None:
      if not self.economic:
        weather_cols = self.columns_to_use['variable'][:12]
        # prediction data is data concatenated to original_data of weather_cols
        prediction_data = pd.concat([data, original_data[weather_cols]], axis=1)
      else:
        prediction_data = pd.concat([data, original_data['shifted_r_estim']], axis=1)
    else:
      prediction_data = original_data

    constant_data = prediction_data[self.columns_to_use['constant']]
    variable_data = prediction_data[self.columns_to_use['variable']]
    training_data = self.final_data[self.final_data['iso_code'] != self.iso_code]
    constant_training_data = training_data[self.columns_to_use['constant']]
    variable_training_data = training_data[self.columns_to_use['variable']]
    constant_mean = constant_training_data.mean()
    variable_mean = variable_training_data.mean()
    constant_std = constant_training_data.std()
    variable_std = variable_training_data.std()
    if self.economic:
      variable_mean['shifted_r_estim'] = 0
      variable_std['shifted_r_estim'] = 1

    # Normalizing the new data with the same mean and std as the training data
    constant_data = (constant_data - constant_mean) / constant_std
    variable_data = (variable_data - variable_mean) / variable_std

    # Join done on the date as we only have one country so the date is unique
    final_data_for_prediction = constant_data.merge(variable_data, left_index=True, right_index=True)

    # Creating the data with a shape that can be fed into the network
    sliced = self.slice_data(final_data_for_prediction, self.columns_to_use['constant'],
                             self.columns_to_use['variable'])
    print(self.columns_to_use['constant'])
    print(self.columns_to_use['variable'])
    net_input = (torch.from_numpy(sliced[0]), torch.from_numpy(sliced[1]))
    prediction_mask = sliced[2]

    # Generate Final Prediction
    pred = self.model.eval()(*(net_input[0], net_input[1])).detach()
    pred = pred.reshape(pred.size(0)).numpy()
    pred = np.append([np.nan] * (self.window_size- 1), pred).flatten()


    # Appending NaNs for impossible predictions (missing data in features)
    pred = self.inject_nans(pred, prediction_mask)

    # We return the prediction, the ground truth and the error
    x = final_data_for_prediction.index.strftime('%Y-%m-%d').values.tolist()
    error = [round(value, 4) for value in np.abs(pred - ground).values.tolist()]
    if self.economic:
      df2 = pd.DataFrame({'index': final_data_for_prediction.index, 'ground': ground, 'pred': pred, 'error': np.abs(pred - ground)}).sort_values(
        by='index')
      df2 = df2[(df2['index'] >= '2020-05-01') & (df2['index'].dt.day == 1)]
      pred = df2['pred'].values
      x = df2.index.strftime('%Y-%m').values.tolist()
      error = [round(value, 4) for value in df2['error'].values.tolist()]
      ground = df2['ground']

    if data is not None:
      if not self.economic:
        y = [
          {'label': 'Reported viral transmission', 'data': [round(value, 4) for value in ground.values.tolist()]},
          {'label': 'Predicted viral transmission by our model', 'data': [round(value, 4) for value in pred.tolist()]},
          {'label': 'Epidemic tipping point: Viral transmission becomes exponential', 'data':[1 for _ in x]}
        ]
      else:
        y = [
          {'label': 'Reported (and interpolated) unemployment rate', 'data': [round(value, 4) for value in ground.values.tolist()]},
          {'label': 'Predicted unemployment rate by our model', 'data': [round(value, 4) for value in pred.tolist()]},
        ]
    else:
      if not self.economic:
        y = [
          {'label': 'Reported viral transmission', 'data': [round(value, 4) for value in ground.values.tolist()]},
          {'label': 'Predicted viral transmission by our model', 'data': [round(value, 4) for value in pred.tolist()]},
          {'label': 'Epidemic tipping point: Viral transmission becomes exponential', 'data':[1 for _ in x]},
          {'label': 'Error (MAE)', 'data': error}
        ]
      else:
        y = [{'label': 'Reported (and interpolated) unemployment rate', 'data': [round(value, 4) for value in ground.values.tolist()]},
        {'label': 'Predicted unemployment rate by our model', 'data': [round(value, 4) for value in pred.tolist()]},
        {'label': 'Error (MAE)', 'data': error}]


    return json.loads(json.dumps({'x': x, 'y': y}, ignore_nan=True))

  def slice_data(self, df, const_cols, var_cols):
    """ Slices a df to generate hybrid_lstm training data (stride=1), assumes the df is sorted by date and has no date
    dropped """

    # Regular slicing
    train_cols = sorted(var_cols) + sorted(const_cols)
    # window size is 7
    slices = np.array([df[train_cols].values[i:i + self.window_size] for i in
                       range(len(df) - self.window_size + 1)])

    # Mask for training and prediction
    valid_dates = np.array([not np.isnan(x).any() for x in slices])
    slices = slices[valid_dates]

    # Pop Mean of const features
    const_features = slices[:, :, -len(const_cols):].mean(axis=1)
    var_features = slices[:, :, :len(var_cols)]

    return const_features, var_features, valid_dates

  def inject_nans(self, pred, prediction_mask):
    """Fill array with Nans, assume it has appended nans already to account for past_window size"""

    idx_to_fill = [e for e, x in enumerate(prediction_mask) if x]

    # Count appended nan and remove them
    nb_appended = np.isnan(pred).sum()
    pred = pred[~np.isnan(pred)]

    new_pred = np.full(len(prediction_mask), np.nan)

    for p, idx in zip(pred, idx_to_fill):
      new_pred[idx] = p

    # Restore nan
    new_pred = np.append([np.nan] * nb_appended, new_pred)
    return new_pred

  def predict_for_a_period_personalized(self, start_date: str, end_date: str, features: dict):
    # given start_date and end_date get all the dates in between
    dates = pd.date_range(start_date, end_date)
    sundays = CountryData.get_sundays_between_dates(start_date, end_date)
    constant_features = features['constant']
    variable_features = features['variable']
    # for each constant feature, repeat it a number of times as long as dates
    for feature in constant_features:
      feature_values = [float(constant_features[feature])] * len(dates)
      constant_features[feature] = feature_values
    # get the difference in terms of days between sundays[0] and start_date
    repeat_first = (sundays[0] - pd.to_datetime(start_date)).days + 1  # +1 because we change policies on monday
    # get the difference in terms of days between end_date and sundays[-1]
    repeat_last = (pd.to_datetime(end_date) - sundays[-1]).days
    for feature in variable_features:
      list_of_values = variable_features[feature]
      new_list_of_values = []
      for i, value in enumerate(list_of_values):
        # if the value is first repeat it repeat_first times,
        # if the value is last repeat_it repeat_last times,
        # else repeat it 7 times
        if i == 0:
          new_list_of_values += [value] * repeat_first
        elif i == len(list_of_values) - 1 and repeat_last > 0:
          new_list_of_values += [value] * repeat_last
        else:
          new_list_of_values += [value] * 7
      variable_features[feature] = new_list_of_values

    # merge the two dictionaries
    features_dict = {**constant_features, **variable_features}
    df = pd.DataFrame(index=dates, data=features_dict)
    return self.predict_for_a_period(start_date, end_date, df)
