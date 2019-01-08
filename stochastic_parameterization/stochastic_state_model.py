import os
import pickle
import numpy as np
import torch
from stochastic_parameterization.get_transition_matrix import \
    get_transition_matrix


class StochasticStateModel(object):

    def __init__(
            self,
            precip_quantiles=[0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1]):
        self.is_trained = False
        self.precip_quantiles = precip_quantiles
        self.possible_etas = list(range(len(precip_quantiles)))
        self.eta = np.random.choice(
            self.possible_etas,
            p=np.ediff1d([0] + list(precip_quantiles))
        )
        self.transition_matrix = get_transition_matrix(precip_quantiles)

    def train_conditional_model(self, eta):
        os.system(
            """
                python -m uwnet.train with \
                    assets/training_configurations/default.json \
                    eta_to_train=2 \
                    batch_size=20000 \
                    epochs=1 \
                    output_dir=models/stochastic_state_model_{}
            """.format(eta)
        )

    def train(self):
        conditional_models = {}
        if not self.is_trained:
            for eta in range(len(self.precip_quantiles)):
                self.train_conditional_model(eta)
                conditional_models[eta] = torch.load(
                    f'models/stochastic_state_model_{eta}/1.pkl'
                )
            self.conditional_models = conditional_models
            self.is_trained = True
        else:
            raise Exception('Model already trained')

    def update_current_state(self):
        probabilities = self.transition_matrix[self.eta]
        self.eta = np.random.choice(self.possible_etas, p=probabilities)

    def predict(self, data):
        model = self.conditional_models[self.eta]
        self.update_current_state
        return model.predict(data)

    def save(self, save_location):
        with open(save_location, 'wb') as f:
            pickle.dump(self, f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
