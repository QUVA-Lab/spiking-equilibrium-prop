from spiking_equilibrium_prop.demo_mnist_eqprop import experiment_mnist_eqprop, settings
from spiking_equilibrium_prop.quantized_eqprop import EncodingDecodingNeuronLayer

"""
Here we add quantized experiments
"""

experiment_quantized_eqprop = experiment_mnist_eqprop.add_config_root_variant('quantized', layer_constructor = lambda epsilons, quantizer, lambdas=None: EncodingDecodingNeuronLayer.get_simple_constrcutor(epsilons=epsilons, lambdas=lambdas, quantizer=quantizer))

for network_type, network_settings in settings.items():

    X = experiment_quantized_eqprop.add_root_variant(network_type, **network_settings)
    for eps_function in (0.5, '0.5/sqrt(t)', '0.5/t'):
        X.add_variant(epsilons = eps_function, quantizer='sigma_delta')
        for lambda_function in (0.25, 0.5, 0.8):
            X.add_variant(epsilons = eps_function, lambdas = lambda_function, quantizer='sigma_delta')
    X.add_variant(epsilons='0.89/t**0.43', lambdas='0.89/t**0.43', quantizer='sigma_delta')  # Found in optimal convergence search
    X.add_variant(epsilons='0.843/t**0.0923', lambdas='0.832/t**0.584', quantizer='sigma_delta')  # Found in optimal convergence search


if __name__ == '__main__':
    experiment_mnist_eqprop.browse()
