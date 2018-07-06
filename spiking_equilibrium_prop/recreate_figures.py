from matplotlib import pyplot as plt
from artemis.plotting.expanding_subplots import set_figure_border_size, add_subplot
from spiking_equilibrium_prop.demo_mnist_eqprop import report_score_from_result
from spiking_equilibrium_prop.demo_mnist_quantized_eqprop import experiment_mnist_eqprop
from spiking_equilibrium_prop.demo_quantized_convergence import \
    compare_quantized_convergence_records, demo_quantized_convergence
from spiking_equilibrium_prop.demo_signal_figure import demo_create_signal_figure


def create_neuron_figure():

    demo_create_signal_figure()


def create_convergence_figure():

    X = demo_quantized_convergence.get_variant('scheduled')

    records = [
        X.get_variant('epsilon_decay').get_variant(epsilons='1/2').get_latest_record(if_none='run'),
        X.get_variant('epsilon_decay').get_variant(epsilons='1/t').get_latest_record(if_none='run'),
        X.get_variant('epsilon_decay').get_variant(epsilons='1/sqrt(t)').get_latest_record(if_none='run'),
        X.get_variant(epsilons ='1/sqrt(t)', quantizer ='stochastic').get_latest_record(if_none='run'),
        X.get_variant('search').get_variant('epsilons=0.843_SLASH_t**0.0923,lambdas=0.832_SLASH_t**0.584').get_latest_record(if_none='run'),  # Best Mean Error
        ]
    compare_quantized_convergence_records(
            records = records,
            ax = add_subplot(), show_now=False, include_legend_now = True, label_reference_lines=True, legend_squeeze=0.5
            )
    plt.xlabel('t')
    plt.show()


def create_mnist_figure():

    ex_continuous = experiment_mnist_eqprop.get_variant('vanilla').get_variant('one_hid_swapless')
    ex_binary = experiment_mnist_eqprop.get_variant('quantized').get_variant('one_hid_swapless').get_variant(epsilons='0.843/t**0.0923', lambdas='0.832/t**0.584', quantizer='sigma_delta')

    rec_continuous = ex_continuous.get_latest_record(if_none='run')
    rec_binary = ex_binary.get_latest_record(if_none='run')

    print(f"Continuous: {report_score_from_result(rec_continuous.get_result())}")
    print(f"Binary: {report_score_from_result(rec_binary.get_result())}")

    plt.figure(figsize=(4.5, 3))
    set_figure_border_size(bottom=0.15, left=0.15)
    ex_binary.compare([rec_continuous, rec_binary], show_now=False)

    plt.legend(['Continuous Eq-Prop: Train', 'Continuous Eq-Prop: Test', 'Binary Eq-Prop: Train', 'Binary Eq-Prop: Test'])
    plt.show()


if __name__ == '__main__':
    {1: create_neuron_figure, 2: create_convergence_figure, 3: create_mnist_figure}[int(input('Which figure would you like to create?  (1-3) >> '))]()
