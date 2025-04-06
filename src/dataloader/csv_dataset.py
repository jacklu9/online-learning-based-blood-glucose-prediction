from src.utils import normalise
from typing import Tuple, List, Any, Dict
from torch.utils.data import Dataset
import pandas as pd
import torch
import argparse
import logging
logging = logging.getLogger('pytorch_lightning')


class CSVDataset(Dataset):
    """This class is used to create a dataset from a .csv file for blood glucose prediction.

    Args:
        site_id (int): ID of the dataset (e.g., patient ID).
        lookback_window (int): Number of time steps to look back.
        prediction_horizon (int): Prediction horizon time steps into the future.
        train_batchsize (int): Number of training samples for each timestep.
        valid_batchsize (int): Number of validation or test samples for each timestep.
        input_features (List[str]): List of input features to use.
        output_features (List[str]): List of output features to predict.
    """

    def __init__(self,
                 site_id: int,
                 lookback_window: int,
                 prediction_horizon: int,
                 train_batchsize: int,
                 valid_batchsize: int,
                 input_features: List = ['BloodGlucose', 'basal','bolus', 'CHO'],
                 output_features: List = ['BloodGlucose'],
                 ) -> None:

        self._data = pd.read_csv('data/site_' + str(site_id) + '/dataset.csv')
        self.available_features = list(self._data.columns.values)

        # Check if the input and output features are available in the `available_features` list
        for input_feature in input_features:
            if input_feature not in self.available_features:
                raise ValueError(
                    f"Input feature {input_feature} is not available.")

        for output_feature in output_features:
            if output_feature not in self.available_features:
                raise ValueError(
                    f"Output feature {output_feature} is not available.")

        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon

        self.train_batchsize = train_batchsize
        self.valid_batchsize = valid_batchsize

        self.input_features = input_features
        self.output_features = output_features
        self.bypass_features = [feature for feature in self.available_features
                                if feature not in self.output_features]

        # Get indices of input, output, and bypass features
        self.indices_input_features = [i for i in range(len(self.available_features))
                                       if self.available_features[i] in self.input_features]
        self.indices_output_features = [i for i in range(len(self.available_features))
                                        if self.available_features[i] in self.output_features]
        self.indices_bypass_features = [i for i in range(len(self.available_features))
                                        if i not in self.indices_output_features]
        self.scaling_factors_min: torch.Tensor = None
        self.scaling_factors_max: torch.Tensor = None
        self._preprocess_and_split_data()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'scaling_factors_min': self.scaling_factors_min,
            'scaling_factors_max': self.scaling_factors_max,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scaling_factors_min = state_dict['scaling_factors_min']
        self.scaling_factors_max = state_dict['scaling_factors_max']
        self._preprocess_and_split_data()

    def _preprocess_and_split_data(self) -> None:
        data = self._data[self.available_features].values
        data = torch.from_numpy(data).type(torch.float32)
  
        # Ensure normalization factors are initialized
        self.scaling_factors_min = data.min(dim=0).values if self.scaling_factors_min is None else self.scaling_factors_min
        self.scaling_factors_max = data.max(dim=0).values if self.scaling_factors_max is None else self.scaling_factors_max
  
        # Normalize the data
        data_normalised = normalise(data, self.scaling_factors_min, self.scaling_factors_max)
  
        # Initialize as lists
        self._inputs_train, self._targets_train = [], []
        self._inputs_validation, self._targets_validation = [], []
        self._inputs_test, self._targets_test = [], []
  
        self.t_i = self.train_batchsize + self.valid_batchsize + \
            self.lookback_window + self.prediction_horizon - 1
        self.t_total = len(data) - self.prediction_horizon + 1
  
        for t in range(self.t_i, self.t_total):
            # Construct extended input (lookback + prediction horizon)
            extended_input = torch.zeros(1, self.lookback_window + self.prediction_horizon, len(self.input_features))  
            for j, feature_name in enumerate(self.input_features):
                if feature_name in ['Basal', 'Bolus', 'CHO']:
                    extended_input[0, :, j] = data_normalised[t - self.lookback_window:t + self.prediction_horizon, self.indices_input_features[j]]
                elif feature_name == 'BloodGlucose':
                    bg_extended = torch.zeros(self.lookback_window + self.prediction_horizon)
                    # Fill past values
                    bg_extended[:self.lookback_window] = data_normalised[t - self.lookback_window:t, self.indices_input_features[j]]
                    # Fill future values with the last known blood glucose value
                    last_known_value = data_normalised[t - 1, self.indices_input_features[j]]
                    bg_extended[self.lookback_window:] = last_known_value
                    extended_input[0, :, j] = bg_extended
  
            # Construct target output
            target_output = torch.zeros(1, self.prediction_horizon, len(self.available_features))
            for j in range(len(self.available_features)):
                target_output[0, :, j] = data_normalised[t:t + self.prediction_horizon, j]
  
            # Add to test set
            self._inputs_test.append(extended_input)
            self._targets_test.append(target_output)
  
            # Construct validation set
            extended_input_valid = torch.zeros(self.valid_batchsize, self.lookback_window + self.prediction_horizon, len(self.input_features))
            target_output_valid = torch.zeros(self.valid_batchsize, self.prediction_horizon, len(self.available_features))
            for k in range(self.valid_batchsize):
                for j, feature_name in enumerate(self.input_features):
                    if feature_name in ['Basal', 'Bolus', 'CHO']:
                        extended_input_valid[k, :, j] = data_normalised[t - self.lookback_window - self.prediction_horizon - k:t + self.prediction_horizon - self.prediction_horizon - k, self.indices_input_features[j]]
                    elif feature_name == 'BloodGlucose':
                        bg_extended = torch.zeros(self.lookback_window + self.prediction_horizon)
                        bg_extended[:self.lookback_window] = data_normalised[t - self.lookback_window - self.prediction_horizon - k:t - self.prediction_horizon - k, self.indices_input_features[j]]
                        last_known_value = data_normalised[t - self.prediction_horizon - k - 1, self.indices_input_features[j]]
                        bg_extended[self.lookback_window:] = last_known_value
                        extended_input_valid[k, :, j] = bg_extended
  
                for j in range(len(self.available_features)):
                    target_output_valid[k, :, j] = data_normalised[t - self.prediction_horizon - k:t + self.prediction_horizon - self.prediction_horizon - k, j]
  
            self._inputs_validation.append(extended_input_valid)
            self._targets_validation.append(target_output_valid)
  
            # Construct training set
            extended_input_train = torch.zeros(self.train_batchsize, self.lookback_window + self.prediction_horizon, len(self.input_features))
            target_output_train = torch.zeros(self.train_batchsize, self.prediction_horizon, len(self.available_features))
            for k in range(self.train_batchsize):
                for j, feature_name in enumerate(self.input_features):
                    if feature_name in ['Basal', 'Bolus', 'CHO']:
                        extended_input_train[k, :, j] = data_normalised[t - self.valid_batchsize - self.lookback_window - self.prediction_horizon - k:t + self.prediction_horizon - self.prediction_horizon - self.valid_batchsize - k, self.indices_input_features[j]]
                    elif feature_name == 'BloodGlucose':
                        bg_extended = torch.zeros(self.lookback_window + self.prediction_horizon)
                        bg_extended[:self.lookback_window] = data_normalised[t - self.valid_batchsize - self.lookback_window - self.prediction_horizon - k:t - self.valid_batchsize - self.prediction_horizon - k, self.indices_input_features[j]]
                        last_known_value = data_normalised[t - self.valid_batchsize - self.prediction_horizon - k - 1, self.indices_input_features[j]]
                        bg_extended[self.lookback_window:] = last_known_value
                        extended_input_train[k, :, j] = bg_extended
  
                for j in range(len(self.available_features)):
                    target_output_train[k, :, j] = data_normalised[t - self.valid_batchsize - self.prediction_horizon  - k:t + self.prediction_horizon - self.prediction_horizon - self.valid_batchsize - k, j]
  
            self._inputs_train.append(extended_input_train)
            self._targets_train.append(target_output_train)
  
        # Convert lists to tensors
        self._inputs_train = torch.stack(self._inputs_train, dim=0)
        self._targets_train = torch.stack(self._targets_train, dim=0)
  
        self._inputs_validation = torch.stack(self._inputs_validation, dim=0)
        self._targets_validation = torch.stack(self._targets_validation, dim=0)
  
        self._inputs_test = torch.stack(self._inputs_test, dim=0)
        self._targets_test = torch.stack(self._targets_test, dim=0)

    def get_profile(self) -> torch.Tensor:
        data = self._data[self.available_features].values
        data = torch.from_numpy(data).type(torch.float32)
        profiles = data[self.t_i:self.t_total]
        return profiles

    def __len__(self) -> int:
        return len(self._inputs_train)

    def __getitem__(self, time: int) -> Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor]]:
        inputs_train = self._inputs_train[time]
        targets_train = self._targets_train[time]

        inputs_val = self._inputs_validation[time]
        targets_val = self._targets_validation[time]

        inputs_test = self._inputs_test[time]
        targets_test = self._targets_test[time]

        return (inputs_train, targets_train), (inputs_val, targets_val), (inputs_test, targets_test)

    @staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument("--site_id", type=int,
                            default=1, help="The ID of the dataset (e.g., patient ID).")
        parser.add_argument('--train_batchsize', type=int, default=1,
                            help='The number of samples used for each training step.')
        parser.add_argument('--valid_batchsize', type=int, default=1,
                            help='The number of samples used for each validation or test step.')
        parser.add_argument('--input_features',  nargs='+', default=['BloodGlucose', 'basal','bolus', 'CHO'],
                            help='Features to use: BloodGlucose, Insulin, CHO.')
        parser.add_argument('--output_features',  nargs='+', default=['BloodGlucose'],
                            help='Features to predict: BloodGlucose.')
        parser.add_argument('--lookback_window', type=int, default=60,
                            help='Number of past time steps used for training.')
        parser.add_argument('--prediction_horizon', type=int, default=8,
                            help='The number of predicted future samples.')
        return parser
