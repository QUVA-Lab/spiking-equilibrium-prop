from abc import abstractmethod
from typing import Tuple

import numpy as np
from dataclasses import dataclass

from artemis.general.should_be_builtins import bad_value
from spiking_equilibrium_prop.eq_prop import ISimpleUpdateFunction, IDynamicLayer, LayerParams, \
    drho, rho


@dataclass
class StochasticQuantizer(ISimpleUpdateFunction):

    rng: np.random.RandomState = np.random.RandomState()  # Note... this is not quite correct because rng is stateful

    def __call__(self, x: np.ndarray) -> Tuple['StochasticQuantizer', np.ndarray]:
        q = np.array(x>self.rng.rand(*x.shape), copy=False).astype(int)
        return StochasticQuantizer(rng=self.rng), q


@dataclass
class ThresholdQuantizer(ISimpleUpdateFunction):

    thresh: float = 0.5

    def __call__(self, x: np.ndarray) -> Tuple['SigmaDeltaQuantizer', np.ndarray]:
        q = (x>self.thresh).astype(int)
        return ThresholdQuantizer(self.thresh), q


@dataclass
class SigmaDeltaQuantizer(ISimpleUpdateFunction):

    phi: np.ndarray = 0.

    def __call__(self, x: np.ndarray) -> Tuple['SigmaDeltaQuantizer', np.ndarray]:
        phi_prime = x + self.phi
        q = (phi_prime>0.5).astype(int)
        new_phi = phi_prime - q
        return SigmaDeltaQuantizer(new_phi), q


@dataclass
class SecondOrderSigmaDeltaQuantizer(ISimpleUpdateFunction):

    phi1: np.ndarray = 0.
    phi2: np.ndarray = 0.

    def __call__(self, x: np.ndarray) -> Tuple['SecondOrderSigmaDeltaQuantizer', np.ndarray]:
        phi_1_ = self.phi_1 + x
        phi_2_ = self.phi_2 + phi_1_
        q = (phi_2_ > 0.5).astype(int)
        return SecondOrderSigmaDeltaQuantizer(phi1=phi_1_-q, phi2=phi_2_-q), q


@dataclass
class IdentityFunction(ISimpleUpdateFunction):

    def __call__(self, x):
        return self, x


class IStepSizer(object):

    @abstractmethod
    def __call__(self, x: np.ndarray) -> Tuple['IStepSizer', float]:
        pass


@dataclass
class ConstantStepSizer(IStepSizer):

    step_size: float

    def __call__(self, x):
        return self, self.step_size


@dataclass
class ScheduledStepSizer(IStepSizer):

    schedule: str
    t: int = 1

    def __call__(self, x):
        step_size = eval(self.schedule, {'exp': np.exp, 'sqrt': np.sqrt}, {'t': self.t})
        return ScheduledStepSizer(schedule=self.schedule, t=self.t+1), step_size


def create_step_sizer(step_schedule):
    return ConstantStepSizer(step_schedule) if isinstance(step_schedule, (int, float)) else \
        ScheduledStepSizer(step_schedule) if isinstance(step_schedule, str) else \
        step_schedule if callable(step_schedule) else \
        bad_value(step_schedule)


@dataclass
class KestonsStepSizer(IStepSizer):

    a: float
    b: float
    k: int = 1.
    last_err: np.ndarray = None
    initial_step: float = 1
    avg: np.ndarray = 0.

    def __call__(self, x):

        if self.k == 1:
            last_err = None
            k = self.k+1
        elif self.k==2:
            last_err = x - self.avg
            k = self.k+1
        else:
            last_err = x - self.avg
            k = self.k + (1 if (self.last_err*last_err).sum() < 0 else 0)
            last_err = x - self.avg
        step_size = self.initial_step*(self.a/(self.b+k))
        avg = (1-step_size)*self.avg + step_size*x
        new_obj = KestonsStepSizer(initial_step=self.initial_step, last_err=last_err, k=k, avg=avg, a=self.a, b=self.b)
        return new_obj, step_size


@dataclass
class PredictiveEncoder(ISimpleUpdateFunction):

    lambda_stepper: IStepSizer
    quantizer: ISimpleUpdateFunction
    last_input: np.ndarray = 0.

    def __call__(self, x):
        new_lambda_state, lambdaa = self.lambda_stepper(x)
        prediction_error = x - (1-lambdaa)*self.last_input
        pre_code = prediction_error/lambdaa
        new_quantizer, q = self.quantizer(pre_code)
        new_encoder =PredictiveEncoder(lambda_stepper=new_lambda_state, quantizer=new_quantizer, last_input=x, )
        return new_encoder, q


@dataclass
class PredictiveDecoder(ISimpleUpdateFunction):

    lambda_stepper: IStepSizer
    last_reconstruction: np.ndarray = 0.

    def __call__(self, x):
        lambdaa_stepper, lambdaa = self.lambda_stepper(x)
        reconstruction = (1-lambdaa)*self.last_reconstruction + lambdaa*x
        return PredictiveDecoder(lambda_stepper=lambdaa_stepper, last_reconstruction=reconstruction), reconstruction


@dataclass
class EncodingDecodingNeuronLayer(IDynamicLayer):

    encoder: ISimpleUpdateFunction
    decoder: ISimpleUpdateFunction
    stepper: IStepSizer

    @classmethod
    def get_partial_constructor(cls, encoder, decoder, stepper):
        def partial_constructor(n_samples: int, params: LayerParams):
            return EncodingDecodingNeuronLayer(
                params=params,
                output=np.zeros((n_samples, params.n_units)),
                potential = np.zeros((n_samples, params.n_units)),
                encoder=encoder,
                decoder=decoder,
                stepper=stepper
            )
        return partial_constructor

    @classmethod
    def get_simple_constrcutor(cls, epsilons, quantizer, lambdas=None):

        stepper = create_step_sizer(epsilons)

        quantizer = \
            SigmaDeltaQuantizer() if quantizer == 'sigma_delta' else \
            StochasticQuantizer() if quantizer == 'stochastic' else \
            ThresholdQuantizer() if quantizer == 'threshold' else \
            SecondOrderSigmaDeltaQuantizer() if quantizer == 'second_order_sd' else \
            bad_value(quantizer)

        if lambdas is None:
            encoder = quantizer
            decoder = IdentityFunction()
        else:
            lambda_stepper = create_step_sizer(lambdas)
            encoder = PredictiveEncoder(lambda_stepper=lambda_stepper, quantizer=quantizer)
            decoder = PredictiveDecoder(lambda_stepper=lambda_stepper)

        return cls.get_partial_constructor(encoder=encoder, decoder=decoder, stepper=stepper)

    def __call__(self, x_aft=None, x_fore=None, pressure = None, clamp = None) -> 'EncodingDecodingNeuronLayer':

        if clamp is not None:
            potential = clamp
            decoder = self.decoder
            stepper = self.stepper
        else:
            n_samples = x_aft.shape[0] if x_aft is not None else x_fore.shape[0]
            u = np.zeros((n_samples, self.params.n_units))
            if x_aft is not None:
                u += x_aft @ self.params.w_aft
            if x_fore is not None:
                u += x_fore @ self.params.w_fore
            decoder, v = self.decoder(u)
            if pressure is not None:
                v += pressure
            if self.params.b is not None:
                v += self.params.b
            stepper, eps = self.stepper(u)
            potential = np.clip(self.potential - eps * drho(self.potential)* (self.potential - v), 0, 1)
        post_potential = rho(potential)

        encoder, output = self.encoder(post_potential)

        return EncodingDecodingNeuronLayer(
            potential=potential,
            params=self.params,
            output = output,
            encoder=encoder,
            decoder=decoder,
            stepper=stepper
        )