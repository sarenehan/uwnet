import os
import pickle
import numpy as np
import torch
from stochastic_parameterization.get_transition_matrix import \
    get_transition_matrix


class StochasticStateModel(object):

    def __init__(
            self,
            precip_quantiles=[0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1],
            dims=(128, 64)):
        self.is_trained = False
        self.precip_quantiles = precip_quantiles
        self.possible_etas = list(range(len(precip_quantiles)))
        self.eta = np.random.choice(
            self.possible_etas,
            dims,
            p=np.ediff1d([0] + list(precip_quantiles))
        )
        self.transition_matrix = get_transition_matrix(precip_quantiles)
        self.eta_stepper = np.vectorize(
            lambda eta:
            np.random.choice(
                self.possible_etas,
                p=self.transition_matrix[eta]
            )
        )

    def train_conditional_model(
            self,
            eta,
            training_config_file,
            **kwargs):
        cmd = f'python -m uwnet.train with {training_config_file}'
        cmd += f' eta_to_train={eta}'
        for key, val in kwargs.items():
            cmd += f' {key}={val}'
        os.system(cmd)

    def train(self):
        conditional_models = {}
        if not self.is_trained:
            for eta in self.possible_etas:
                self.train_conditional_model(eta)
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

    def update_current_state(self):
        self.eta = self.eta_stepper(self.eta)

    def predict(self, data):
        if not self.is_trained:
            raise Exception('Model is not trained.')
        self.update_current_state()
        return self.predictor(data)

    def save(self, save_location):
        with open(save_location, 'wb') as f:
            pickle.dump(self, f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
