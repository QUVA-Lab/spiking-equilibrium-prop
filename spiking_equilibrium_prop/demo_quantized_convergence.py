from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt

from artemis.experiments import ExperimentFunction
from artemis.experiments.experiment_record import ExperimentRecord, get_experiment_to_latest_record_mapping
from artemis.experiments.experiment_record_view import separate_common_args
from artemis.general.display import pyfuncstring_to_tex
from artemis.general.numpy_helpers import get_rng
from artemis.general.progress_indicator import ProgressIndicator
from artemis.general.test_mode import is_test_mode
from artemis.plotting.db_plotting import dbplot_collection
from artemis.plotting.easy_plotting import funplot
from artemis.plotting.matplotlib_backend import MovingPointPlot
from artemis.plotting.pyplot_plus import outside_right_legend
from spiking_equilibrium_prop.eq_prop import initialize_states, initialize_params, eqprop_step, \
    SimpleLayerController
from spiking_equilibrium_prop.quantized_eqprop import EncodingDecodingNeuronLayer

"""
Ok, so this demonstrates how we can schedule the epsilons and lambdas of the predictive coder to guarantee convergence.  

Long story short: 1/t annealing is too fast - Early values stay remembered and network does not converge to fixed point.
Same with exponential scaling.  With no annealing, we hit a noise-floor quickly.  In general 1/t**k annealing seems to 
work when k>0<1 with 1/sqrt(t) working pretty well for both epsilons and lambdas.  

Annealing lambdas seems to lead to faster convergence than annealing epsilons.  
"""


def rho(x):
    return np.clip(x, 0, 1)


def drho(x):
    return ((x>=0) & (x<=1)).astype(float)


def last(iterable):
    gen = iter(iterable)
    x = next(gen)
    for x in gen:
        pass


def compare_quantized_convergence_records(records: Sequence[ExperimentRecord], add_reference_lines = True, label_reference_lines = True, ax=None, show_now = True, include_legend_now = True, legend_squeeze=0.7):

    if ax is None:
        ax = plt.gca()

    argcommon, argdiffs = separate_common_args(records, as_dicts=True, only_shared_argdiffs=True)

    for record_ix, (rec, args) in enumerate(zip(records, (dict(d) for d in argdiffs))):
        rs_online_errors, rs_end_errors, rr_end_errors, ss_end_errors = rec.get_result()
        argstr = ', '.join(f'{k}={args[k]}' for k in ['quantizer', 'epsilons', 'lambdas'])\
            .replace("epsilons", "$\epsilon$")\
            .replace(', lambdas=None', '')\
            .replace("lambdas", "$\lambda$")\
            .replace('**', '^')\
            .replace('sqrt(t)', '$\sqrt{t}$')\
            .replace('quantizer=', 'q:')\
            .replace('sigma_delta', '$\Sigma\Delta$')\
            .replace('t^2', '$t^{2}$')\
            .replace('0.832/t^0.584', '$0.83/t^{0.58}$')\
            .replace('0.843/t^0.0923', '$0.84/t^{0.092}$')\
            # .replace('t^0.88', '$t^{0.88}$')
        h, = plt.loglog(np.arange(1, len(rs_end_errors)+1), rs_end_errors.mean(axis=1), label=pyfuncstring_to_tex(argstr))

        ax.set_ylabel('$\\|error(t)|$')
        ax.grid(True)

        if record_ix==len(records)-1:
            if add_reference_lines:
                arbitrary_base = .3
                funplot(lambda t: arbitrary_base/np.sqrt(t), color='k', label='$\propto 1/\sqrt{t}$' if label_reference_lines else None, linestyle = ':', keep_ylims=True)
                funplot(lambda t: arbitrary_base/t, color='k', label='$\propto 1/t$' if label_reference_lines else None, linestyle = '--', keep_ylims=True)
            if include_legend_now:
                outside_right_legend(width_squeeze=legend_squeeze)

    if show_now:
        plt.show()


def show_error(errors):
    rs_online_errors, rs_end_errors, rr_end_errors, ss_end_errors = errors
    return f'Mean: {np.mean(rs_online_errors):6.3g},\tFinal: {np.mean(rs_online_errors[-1]):.3g}'


@ExperimentFunction(is_root=True, one_liner_function=show_error, compare=compare_quantized_convergence_records)
def demo_quantized_convergence(
        quantized_layer_constructor,
        smooth_epsilon=0.5,
        layer_sizes=(500, 500, 10),
        initialize_acts_randomly = False,
        minibatch_size = 1,
        # n_steps = 100,
        n_steps = 10000,
        initial_weight_scale = 1.,
        prop_direction = 'neutral',
        data_seed=1241,
        param_seed = 1237,
        hang = True,
        plot=False
        ):
    """
    """

    smooth_layer_constructor = SimpleLayerController.get_partial_constructor(epsilon=smooth_epsilon)

    print('Params:\n' + '\n'.join(list(f'  {k} = {v}' for k, v in locals().items())))

    data_rng = get_rng(data_seed)
    param_rng = get_rng(param_seed)

    HISTORY_LEN = n_steps
    N_NEURONS_TO_PLOT = 10

    if is_test_mode():
        n_steps = 10

    pi = ProgressIndicator(update_every='2s', expected_iterations=2*n_steps)
    n_in, n_out = layer_sizes[0], layer_sizes[-1]

    x_data = data_rng.rand(minibatch_size, n_in)

    params = initialize_params(
        layer_sizes=layer_sizes,
        initial_weight_scale=initial_weight_scale,
        rng=param_rng
        )

    def run_update(layer_constructor, mode):

        plt.gca().set_prop_cycle(None)

        states = initialize_states(
            layer_constructor=layer_constructor,
            n_samples=minibatch_size,
            params = params
            )

        for t in range(n_steps):

            states = eqprop_step(layer_states=states, x_data=x_data, beta=0, y_data=None, direction=prop_direction)
            acts = [s.potential for s in states]
            yield acts
            if plot:
                dbplot_collection([a[0, :N_NEURONS_TO_PLOT] for a in acts], f'{mode} acts', axis='acts', draw_every='5s', cornertext=f'Negative Phase: {t}', plot_type = lambda: MovingPointPlot(buffer_len=HISTORY_LEN, plot_kwargs=dict(linestyle = '-.' if mode=='Smooth' else '-'), reset_color_cycle=True))
                # dbplot_collection([a[0, :N_NEURONS_TO_PLOT] for a in acts], f'{mode} acts', axis='acts', draw_every=1, cornertext=f'Negative Phase: {t}', plot_type = lambda: MovingPointPlot(buffer_len=HISTORY_LEN, plot_kwargs=dict(linestyle = '-.' if mode=='Smooth' else '-'), reset_color_cycle=True))
            pi()

    smooth_record = list(run_update(layer_constructor=smooth_layer_constructor, mode='Smooth'))
    smooth_acts = smooth_record[-1]

    rough_record = list(run_update(layer_constructor=quantized_layer_constructor, mode='Rough'))
    rough_acts = rough_record[-1]

    rs_online_errors = np.array([[np.mean(np.abs(hr - hs)) for hr, hs in zip(hs_rough, hs_smooth)] for hs_rough, hs_smooth in zip(rough_record, smooth_record)])
    rs_end_errors =    np.array([[np.mean(np.abs(hr - hs)) for hr, hs in zip(hs_rough, hs_smooth)] for hs_smooth in [smooth_record[-1]] for hs_rough in rough_record])
    rr_end_errors =    np.array([[np.mean(np.abs(hr - hs)) for hr, hs in zip(hs_rough, hs_smooth)] for hs_smooth in [rough_record[-1]] for hs_rough in rough_record])
    ss_end_errors =    np.array([[np.mean(np.abs(hr - hs)) for hr, hs in zip(hs_rough, hs_smooth)] for hs_smooth in [smooth_record[-1]] for hs_rough in smooth_record])

    mean_abs_error = np.mean(rs_online_errors, axis=0)
    final_abs_error = rs_online_errors[-1]
    print(f'Mean Abs Layerwise Errors: {np.array_str(mean_abs_error, precision=5)}\t Final Layerwise Errors: {np.array_str(final_abs_error,  precision=5)}')

    return rs_online_errors, rs_end_errors, rr_end_errors, ss_end_errors


X = demo_quantized_convergence.add_config_root_variant('scheduled', quantized_layer_constructor = lambda epsilons, lambdas=None, quantizer='sigma_delta': EncodingDecodingNeuronLayer.get_simple_constrcutor(epsilons=epsilons, lambdas=lambdas, quantizer=quantizer))


for epsilons in ('1/sqrt(t)', '1/t'):
    for quantizer in ('sigma_delta', 'stochastic', 'threshold', 'second_order_sd'):
        X.add_variant(epsilons =epsilons, quantizer=quantizer)


XX = X.add_root_variant('epsilon_decay')
XX.add_variant(epsilons='1')
XX.add_variant(epsilons='1/2')
XX.add_variant(epsilons='1/sqrt(t)')
XX.add_variant(epsilons='1/t')
XX.add_variant(epsilons='1/t**2')
XX.add_variant(epsilons='exp(-t/20)')
XX.add_variant(epsilons='1*exp(-t/40)')
XX.add_variant(epsilons='1/t**(1-1/sqrt(t))')


XX = X.add_root_variant('predictive_coding')
for epsilons in ('0.5', '1/sqrt(t)'):
    XX.add_variant(epsilons=epsilons, lambdas = '0.125')
    XX.add_variant(epsilons=epsilons, lambdas = '0.25')
    XX.add_variant(epsilons=epsilons, lambdas = '0.5')
    XX.add_variant(epsilons=epsilons, lambdas = '0.75')
    XX.add_variant(epsilons=epsilons, lambdas = '1')



XX = X.add_root_variant('what_power')
XX.add_variant(epsilons='1/t**0.75')
XX.add_variant(epsilons='1/t**0.7')
XX.add_variant(epsilons='1/t**0.6')
XX.add_variant(epsilons='1/t**0.5')
XX.add_variant(epsilons='1/t**0.4')

# XX = X.add_root_variant('predictive_coding_with_epsilon_decay')
# XX.add_variant(epsilons='1/t', lambdas = '0.125')
# XX.add_variant(epsilons='1/t', lambdas = '0.25')
# XX.add_variant(epsilons='1/t', lambdas = '0.5')
# XX.add_variant(epsilons='1/t', lambdas = '0.75')
# XX.add_variant(epsilons='1/t', lambdas = '1')

XX = X.add_root_variant('predictive_coding_with_epsilon_decay')
XX.add_variant(epsilons='1/sqrt(t)', lambdas = '0.125')
XX.add_variant(epsilons='1/sqrt(t)', lambdas = '0.25')
XX.add_variant(epsilons='1/sqrt(t)', lambdas = '0.5')
XX.add_variant(epsilons='1/sqrt(t)', lambdas = '0.75')
XX.add_variant(epsilons='1/sqrt(t)', lambdas = '1')


XX = X.add_root_variant('predictive_coding_decay')
XX.add_variant(epsilons='0.5', lambdas = '1')
XX.add_variant(epsilons='0.5', lambdas = '1/t**0.25')
XX.add_variant(epsilons='0.5', lambdas = '1/t**0.5')
XX.add_variant(epsilons='0.5', lambdas = '1/t**0.75')
XX.add_variant(epsilons='0.5', lambdas = '1/t**0.875')
XX.add_variant(epsilons='0.5', lambdas = '1/t')
XX.add_variant(epsilons='0.5', lambdas = 'exp(-t/20)')
XX.add_variant(epsilons='0.5', lambdas = '1/t**(1-1/sqrt(t))')


XX = X.add_root_variant('anneal_both')
XX.add_variant(epsilons='0.5', lambdas = '1')
XX.add_variant(epsilons='0.5/t**0.5', lambdas = '1')
XX.add_variant(epsilons='0.5', lambdas = '1/t**0.5')
XX.add_variant(epsilons='0.5/t**0.5', lambdas = '1/t**0.5')
XX.add_variant('best_mean', epsilons = '0.9505/t**0.2905', lambdas = '0.8683/t**0.7853')
XX.add_variant('best_final', epsilons = '0.8864/t**0.4268', lambdas = '0.8997/t**0.8798')
XX.add_variant('best_final_compact_name', epsilons = '0.89/t**0.43', lambdas = '0.9/t**0.88')


XX = X.add_root_variant('quantizer', epsilons='0.5')
for lambdas in (None, '1/t**0.75'):
    XXX = XX.add_root_variant(lambdas = lambdas)
    XXX.add_variant(quantizer='sigma_delta')
    XXX.add_variant(quantizer='stochastic')


# ======================================================================================
# Random Search

rng = np.random.RandomState(1234)
XX=X.add_root_variant('search')
n_trials = 5000
for n in range(n_trials):
    eps_initial, eps_exponent, lambda_initial, lambda_exponent = rng.rand(4)
    XX.add_variant(epsilons = f'{eps_initial:.3g}/t**{eps_exponent:.3g}', lambdas = f'{lambda_initial:.3g}/t**{lambda_exponent:.3g}')



def find_best_search_results():
    print('Hello')
    search_experiments = X.get_variant('search').get_all_variants()
    experiment_to_record = get_experiment_to_latest_record_mapping(search_experiments)
    search_records = list(experiment_to_record.values())  # [s.get_latest_record() for s in X.get_variant('search').get_all_variants()][:100]
    print(f'Retrieved {len(search_records)} records')
    mean_errors, final_erros, early_errors = zip(*[(np.mean(err), np.mean(err[-1]), np.mean(err[:20])) for record in search_records for _, err, _, _ in [record.get_result()]])
    for name, err in [('Early', early_errors), ('Mean ', mean_errors), ('Final', final_erros)]:
        ix = np.argmin(err)
        print(f'Best {name} Error: Early: {early_errors[ix]:.3g}, Mean: {mean_errors[ix]:.3g}, Final: {final_erros[ix]:.3g}, : {search_records[ix].get_experiment().get_id()}')


# ======================================================================================


if __name__ == '__main__':
    # demo_quantized_convergence.browse(display_format='flat', remove_prefix=False)
    demo_quantized_convergence.browse(display_format='flat', filterexp='~has:search', remove_prefix=False)
    # find_best_search_results()
