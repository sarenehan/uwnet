import os
import pickle
import numpy as np
import torch
from scipy import linalg
from stochastic_parameterization.get_transition_matrix import \
    get_transition_matrix

dataset_dt_seconds = 10800


class StochasticStateModel(object):

    def __init__(
            self,
            precip_quantiles=[0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1],
            dims=(128, 64),
            dt_seconds=10800):
        self.is_trained = False
        self.dims = dims
        self.dt_seconds = dt_seconds
        self.precip_quantiles = precip_quantiles
        self.possible_etas = list(range(len(precip_quantiles)))
        self.setup_eta()
        self.transition_matrix = get_transition_matrix(self.precip_quantiles)
        self.setup_transition_matrix()

    def setup_transition_matrix(self):
        if self.dt_seconds != dataset_dt_seconds:
            continuous_transition_matrix = linalg.logm(
                self.transition_matrix) / dataset_dt_seconds
            self.transition_matrix = linalg.expm(
                continuous_transition_matrix * self.dt_seconds)

    def setup_eta(self):
        self.eta = np.random.choice(
            self.possible_etas,
            self.dims,
            p=np.ediff1d([0] + list(self.precip_quantiles))
        )

    def train_conditional_model(
            self,
            eta,
            training_config_file,
            **kwargs):
        cmd = f'python -m uwnet.train with {training_config_file}'
        cmd += f' eta_to_train={eta}'
        cmd += f' output_dir=models/stochastic_state_model_{eta}'
        cmd += f" precip_quantiles='{self.precip_quantiles}'"
        for key, val in kwargs.items():
            cmd += f' {key}={val}'
        os.system(cmd)

    def train(
            self,
            training_config_file='assets/training_configurations/default.json',
            **kwargs):
        conditional_models = {}
        if not self.is_trained:
            for eta in self.possible_etas:
                self.train_conditional_model(
                    eta, training_config_file, **kwargs)
                conditional_models[eta] = torch.load(
                    f'models/stochastic_state_model_{eta}/1.pkl'
                )
            self.conditional_models = conditional_models
            self.predictor = np.vectorize(
                lambda eta, x:
                    self.conditional_models[eta].forward(x)
            )
            self.is_trained = True
        else:
            raise Exception('Model already trained')

    def update_eta(self):
        new_eta = np.zeros(self.eta.size)
        for eta in self.possible_etas:
            indices = np.ravel_multi_index(
                np.argwhere(self.eta == eta).T, self.dims)
            next_etas = np.random.choice(
                self.possible_etas,
                len(indices),
                p=self.transition_matrix[eta]
            )
            np.put(new_eta, indices, next_etas)
        self.eta = new_eta.reshape(self.dims)

    def predict(self, x):
        if not self.is_trained:
            raise Exception('Model is not trained.')
        self.update_eta()
        output = np.zeros(x.size)
        for eta, model in self.conditional_models.items():
            indices = np.ravel_multi_index(
                np.argwhere(self.eta == eta).T, self.dims)
            predictions = model.forward(np.take(x, indices))
            np.put(output, indices, predictions)
        return output.reshape(self.dims)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save(self, save_location):
        with open(save_location, 'wb') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    model = StochasticStateModel()
    kwargs = {'batch_size': 1000, 'epochs': 1}
    model.train(**kwargs)
    model.save('stochastic_parameterization/stochastic_model.pkl')
