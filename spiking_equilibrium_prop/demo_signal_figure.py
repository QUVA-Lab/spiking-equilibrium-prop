import numpy as np
from matplotlib import pyplot as plt

from artemis.general.should_be_builtins import unzip
from artemis.plotting.expanding_subplots import add_subplot, vstack_plots, set_figure_border_size
from artemis.plotting.pyplot_plus import plot_stacked_signals, event_raster_plot, stemlight
from spiking_equilibrium_prop.quantized_eqprop import IStepSizer
from spiking_equilibrium_prop.quantized_eqprop import create_step_sizer, PredictiveEncoder, \
    SigmaDeltaQuantizer, PredictiveDecoder
from spiking_equilibrium_prop.synthesized_signals import lowpass_random


def demo_create_signal_figure(
        w = [-.7, .8, .5],
        eps = 0.2,
        lambda_schedule = '1/t**.75',
        eps_schedule = '1/t**.4',
        n_samples = 200,
        seed = 1247,
        input_convergence_speed=3,
        scale = 0.3,
        ):
    rng = np.random.RandomState(seed)

    varying_sig = lowpass_random(n_samples = n_samples, cutoff=0.03, n_dim=len(w), normalize=(-scale, scale), rng=rng)
    frac = 1-np.linspace(1, 0, n_samples)[:, None]**input_convergence_speed
    x = np.clip((1-frac)*varying_sig + frac*scale*rng.rand(3), 0, 1)
    true_z = [s for s in [0] for xt in x for s in [np.clip((1-eps)*s + eps*xt.dot(w), 0, 1)]]

    # Alternative try 2
    eps_stepper = create_step_sizer(eps_schedule)  # type: IStepSizer
    lambda_stepper = create_step_sizer(lambda_schedule)  # type: IStepSizer
    encoder = PredictiveEncoder(lambda_stepper=lambda_stepper, quantizer=SigmaDeltaQuantizer())
    decoder = PredictiveDecoder(lambda_stepper=lambda_stepper)
    q = np.array([qt for enc in [encoder] for xt in x for enc, qt in [enc(xt)]])
    inputs = [qt.dot(w) for qt in q]
    sig, epsilons, lambdaas = unzip([(s, eps, dec.lambda_stepper(inp)[1]) for s, dec, eps_func in [[0, decoder, eps_stepper]] for inp in inputs for dec, decoded_input in [dec(inp)] for eps_func, eps in [eps_func(decoded_input)] for s in [np.clip((1-eps)*s + eps*decoded_input, 0, 1)]])

    fig=plt.figure(figsize=(3, 4.5))
    set_figure_border_size(0.02, bottom=0.1)

    with vstack_plots(spacing=0.1, xlabel='t', bottom_pad=0.1):

        ax = add_subplot()

        sep = np.max(x)*1.1
        plot_stacked_signals(x, sep=sep, labels=False)
        plt.gca().set_color_cycle(None)

        event_raster_plot(events = [np.nonzero(q[:, i])[0] for i in range(len(w))], sep=sep, s=100)
        ax.legend(labels=[f'$s_{i}$' for i in range(1, len(w)+1)], loc='lower left')

        ax = add_subplot()
        stemlight(inputs, ax=ax, label='$u_j$', color='k')
        # plt.plot(inputs, label='$u_j$', color='k')
        ax.axhline(0, color='k')
        ax.tick_params(axis='y', labelleft='off')
        ax.legend(loc='upper left')
        # plt.grid()

        ax = add_subplot()
        # ax.plot([eps_func(t) for t in range(n_samples)], label='$\epsilon$', color='k')
        ax.plot(epsilons, label='$\epsilon$', color='k')
        ax.plot(lambdaas, label='$\lambda$', color='b')
        ax.axhline(0, color='k')
        ax.legend(loc='upper right')
        ax.tick_params(axis='y', labelleft='off')

        ax = add_subplot()
        ax.plot(true_z, label='$s_j$ (real)', color='r')
        ax.plot(sig, label='$s_j$ (binary)', color='k')
        ax.legend(loc='lower right')
        ax.tick_params(axis='y', labelleft='off')
        ax.axhline(0, color='k')

    plt.show()


if __name__ == '__main__':
    demo_create_signal_figure()