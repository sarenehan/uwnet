import os
import numpy as np


class StochasticStateModel(object):

    def __init__(
            self,
            precip_quantiles=[0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1]):
        self.is_trained = False
        self.precip_quantiles = precip_quantiles
        self.eta = np.random.choice(
            list(range(len(precip_quantiles))),
            np.ediff1d([0] + list(precip_quantiles))
        )
        self.setup_transition_matrix()

    def setup_transition_matrix(self):
        pass

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
        if not self.is_trained:
            for eta in range(len(self.precip_quantiles)):
                self.train_conditional_model(eta)
            self.is_trained = True
        else:
            raise Exception('Model already trained')

    @property
    def current_state(self):
        pass

    def predict(self, data):

