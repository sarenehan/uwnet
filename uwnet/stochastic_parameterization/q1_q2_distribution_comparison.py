from matplotlib import pyplot as plt
import numpy
from stewart_intro.utils import load_model, model_name
from stewart_intro.evaluate_model import get_predicted_q1_and_q2
from stewart_intro.generate_training_data import get_layer_mass

mlp = load_model(model_name)
q1_trues, q2_trues, q1_preds, q2_preds = get_predicted_q1_and_q2(mlp)
layer_mass = get_layer_mass()


def get_mass_weighted_qs(qs):
    return ((qs * layer_mass) / layer_mass.sum()).mean(axis=1)


def plot_true_vs_predicted_distribution(
        q_true,
        q_pred,
        name_,
        units):
    q_true_mass_weighted = get_mass_weighted_qs(q_true)
    q_pred_mass_weighted = get_mass_weighted_qs(q_pred)
    left_limit = min(
        q_true_mass_weighted.min(),
        q_pred_mass_weighted.min()
    )
    right_limit = max(
        q_true_mass_weighted.max(),
        q_pred_mass_weighted.max()
    )

    bins = numpy.linspace(left_limit, right_limit, 75)

    plt.hist(q_true_mass_weighted, bins, alpha=0.5, label=f'Q{name_} True')
    plt.hist(
        q_pred_mass_weighted, bins, alpha=0.5, label=f'Q{name_} Predicted')
    plt.legend(loc='upper right')
    plt.title(f'Distribution of Q{name_} True vs Predicted')
    plt.ylabel('Count')
    plt.xlabel(f'Q{name_} ({units})')
    plt.show()
    plt.gcf().clear()


print(f'Q1 True mean: {get_mass_weighted_qs(q1_trues).mean()}')
print(f'Q1 Pred mean: {get_mass_weighted_qs(q1_preds).mean()}')
print(f'Q1 True var: {get_mass_weighted_qs(q1_trues).var()}')
print(f'Q1 Pred var: {get_mass_weighted_qs(q1_preds).var()}')

print(f'\n\n\nQ2 True mean: {get_mass_weighted_qs(q2_trues).mean()}')
print(f'Q2 Pred mean: {get_mass_weighted_qs(q2_preds).mean()}')
print(f'Q2 True var: {get_mass_weighted_qs(q2_trues).var()}')
print(f'Q2 Pred var: {get_mass_weighted_qs(q2_preds).var()}')


plot_true_vs_predicted_distribution(
    q1_trues, q1_preds, '1', 'K/s')
plot_true_vs_predicted_distribution(
    q2_trues, q2_preds, '2', 'g/kg*s')
