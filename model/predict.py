import torch
import pandas as pd
import numpy as np
import yaml
import simplejson as json
from datetime import datetime
from model.hybrid import HybridLSTM


class Predict:
    def __init__(self, iso_code: str):
        self.iso_code = iso_code
        self.model = HybridLSTM.load_from_checkpoint(checkpoint_path=f"../model/checkpoints/{self.iso_code}/model.ckpt")
        # TODO: one model for each country to avoid leakage
        self.columns_to_use = self.extract_feature_names()
        self.data_swissre = pd.read_csv('../model/data/final_data.csv', parse_dates=['date']).set_index('date')

    def extract_feature_names(self):
        with open('../model/config/r_estim_features.yaml', 'r', encoding='utf-8') as file:
            swissre_features = yaml.load(file, Loader=yaml.FullLoader)
        constant_columns = swissre_features['demography'] + \
                         swissre_features['sanitary'] + \
                         swissre_features['economic']
        variable_columns = swissre_features['weather'] +\
                            swissre_features['policies']

        return {'constant': constant_columns, 'variable': variable_columns}

    def predict_for_a_period(self, start_date: str, end_date: str ):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        prediction_data = self.data_swissre[(self.data_swissre['iso_code'] == self.iso_code) &
                                            (self.data_swissre.index >= start_date) &
                                            (self.data_swissre.index <= end_date)]
        ground = prediction_data['shifted_r_estim']

        constant_data = prediction_data[self.columns_to_use['constant']]
        variable_data = prediction_data[self.columns_to_use['variable']]
        training_data = self.data_swissre[self.data_swissre['iso_code'] != self.iso_code]
        constant_training_data = training_data[self.columns_to_use['constant']]
        variable_training_data = training_data[self.columns_to_use['variable']]
        constant_mean = constant_training_data.mean()
        variable_mean = variable_training_data.mean()
        constant_std = constant_training_data.std()
        variable_std = variable_training_data.std()

        # Normalizing the new data with the same mean and std as the training data
        constant_data = (constant_data - constant_mean) / constant_std
        variable_data = (variable_data - variable_mean) / variable_std

        # Join done on the date as we only have one country so the date is unique
        final_data_for_prediction = constant_data.merge(variable_data, left_index=True, right_index=True)

        # Creating the data with a shape that can be fed into the network
        sliced = self.slice_data(final_data_for_prediction, self.columns_to_use['constant'],
                                 self.columns_to_use['variable'])
        net_input = (torch.from_numpy(sliced[0]), torch.from_numpy(sliced[1]))
        prediction_mask = sliced[2]

        # Generate Final Prediction
        pred = self.model.eval()(*(net_input[0], net_input[1])).detach()
        pred = pred.reshape(pred.size(0)).numpy()
        pred = np.append([np.nan] * (7 - 1), pred).flatten()

        # Appending NaNs for impossible predictions (missing data in features)
        pred = self.inject_nans(pred, prediction_mask)

        # We return the prediction, the ground truth and the error
        y = [
            {'label': 'Ground', 'data': ground.values.tolist()},
            {'label': 'Predictions', 'data': pred.tolist()},
            {'label': 'Error', 'data':np.abs(pred-ground).values.tolist()}
        ]
        x = final_data_for_prediction.index.strftime('%Y-%m-%d').values.tolist()
        return json.loads(json.dumps({'x':x, 'y':y}, ignore_nan=True))

    def slice_data(self, df, const_cols, var_cols):
        """ Slices a df to generate hybrid_lstm training data (stride=1), assumes the df is sorted by date and has no date
        dropped """

        # Regular slicing
        train_cols = sorted(var_cols) + sorted(const_cols)
        # window size is 7
        slices = np.array([df[train_cols].values[i:i + 7] for i in
                           range(len(df) - 7 + 1)])

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
        
if __name__ == '__main__':
    pred = Predict('CHE')
    print(pred.predict_for_a_period('2020-04-01', '2020-06-01'))







