from abc import abstractmethod
from typing import Sequence, NamedTuple, Optional, Tuple, Callable, Iterable
import dataclasses
import numpy as np
from dataclasses import dataclass
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import izip_equal


def rho(x):
    return np.clip(x, 0, 1)


def drho(x):
    return np.array((x>=0) & (x<=1)).astype(float)


def last(iterable):
    gen = iter(iterable)
    x = next(gen)
    for x in gen:
        pass
    return x


class PropDirectionOptions:

    FORWARD = 'forward'
    BACKWARD = 'backward'
    NEUTRAL = 'neutral'
    FASTFORWARD = 'fast-forward'
    SWAP = 'swap'


class ISimpleUpdateFunction(object):

    @abstractmethod
    def __call__(self, x: np.ndarray) -> Tuple['ISimpleUpdateFunction', np.ndarray]:
        pass


class LayerParams(NamedTuple):
    w_fore: Optional[np.ndarray] = None
    w_aft: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None

    @property
    def n_units(self):
        return len(self.b) if self.b is not None else self.w_fore.shape[1] if self.w_fore is not None else self.w_aft.shape[0]


@dataclass
class IDynamicLayer(object):
    params: LayerParams  # Parameters (weight, biases)
    output: np.ndarray  # The last output produced by the layer
    potential: np.ndarray  # The Potential (i.e. "value") of the neuron.

    @abstractmethod
    def __call__(self, x_aft=None, x_fore=None, pressure = None, clamp = None) -> 'IDynamicLayer':
        pass


@dataclass
class SimpleLayerController(IDynamicLayer):

    epsilon: float

    @staticmethod
    def get_partial_constructor(epsilon):
        def partial_constructor(n_samples: int, params: LayerParams):
            return SimpleLayerController(
                params=params,
                output=np.zeros((n_samples, params.n_units)),
                potential = np.zeros((n_samples, params.n_units)),
                epsilon=epsilon
            )
        return partial_constructor

    def __call__(self, x_aft=None, x_fore=None, pressure = None, clamp = None):

        if clamp is not None:
            potential = clamp
        else:
            n_samples = x_aft.shape[0] if x_aft is not None else x_fore.shape[0]
            v = np.zeros((n_samples, self.params.n_units))
            if self.params.b is not None:
                v += self.params.b
            if x_aft is not None:
                v += x_aft @ self.params.w_aft
            if x_fore is not None:
                v += x_fore @ self.params.w_fore
            if pressure is not None:
                v += pressure
            potential = np.clip(self.potential - self.epsilon * drho(self.potential) * (self.potential - v), 0, 1)
        output = rho(potential)

        return SimpleLayerController(
            epsilon=self.epsilon,
            params = self.params,
            potential = potential,
            output = output
        )


def eqprop_step(layer_states: Sequence[IDynamicLayer], x_data, beta, y_data: Optional[np.ndarray] = None, direction ='neutral') -> Sequence[IDynamicLayer]:

    assert direction in (PropDirectionOptions.FORWARD, PropDirectionOptions.BACKWARD, PropDirectionOptions.NEUTRAL)
    layer_ix = range(len(layer_states)) if direction in (PropDirectionOptions.FORWARD, PropDirectionOptions.NEUTRAL) else range(len(layer_states))[::-1]
    layer_states = list(layer_states)
    new_layers = [None]*len(layer_states)
    for ix in layer_ix:
        new_state = layer_states[ix](
            x_aft = None if ix==0 else layer_states[ix - 1].output,
            x_fore = layer_states[ix + 1].output if ix < len(layer_states) - 1 else None,
            clamp = x_data if ix==0 else None,
            pressure = beta * 2*(y_data - layer_states[-1].potential) if (y_data is not None and ix == len(layer_states) - 1) else None
        )
        new_layers[ix] = new_state
        if direction in (PropDirectionOptions.FORWARD, PropDirectionOptions.BACKWARD):
            layer_states[ix] = new_state

    # dbplot_collection([layer_states[0].output.reshape(-1, 28, 28)]+[s.output for s in layer_states[1:]], 'outputs')
    # dbplot_collection([layer_states[0].potential.reshape(-1, 28, 28)]+[s.potential for s in layer_states[1:]], 'outputs')
    return new_layers


def eqprop_fast_forward_step(layer_states: Sequence[IDynamicLayer], x_data):
    layer_states = list(layer_states)
    new_layers = []
    for ix, layer in enumerate(layer_states):
        new_state = layer_states[ix](
            x_aft = None if ix==0 else new_layers[-1].output,
            x_fore = None,
            clamp = x_data if ix==0 else None,
            pressure = None
        )
        # new_state = layer_states[ix](
        #     x_aft = None if ix==0 else new_layers[-1].output,
        #     x_fore = None,
        #     clamp = x_data if ix==0 else rho(new_layers[-1].output @ layer_states[ix].params.w_aft + layer_states[ix].params.b),
        #     pressure = None
        # )
        new_layers.append(new_state)
    return new_layers


def eqprop_update(negative_acts, positive_acts, ws, bs, learning_rate, beta, bidirectional, l2_loss = None):

    n_samples = negative_acts[0].shape[0]
    w_grads = [-np.transpose(na_pre) @ (pa_post-na_post) / n_samples for na_pre, pa_post, na_post in izip_equal(negative_acts[:-1], positive_acts[1:], negative_acts[1:])]

    if bidirectional:
        w_grads = w_grads[:1] + [wg - np.transpose(pa_pre-na_pre) @ na_post / n_samples for wg, pa_pre, na_pre, na_post in izip_equal(w_grads[1:], positive_acts[1:-1], negative_acts[1:-1], negative_acts[2:])]

    b_grads = [-np.mean(pa_post-na_post, axis=0) for pa_post, na_post in izip_equal(positive_acts[1:], negative_acts[1:])]
    if l2_loss is not None:
        w_grads = [(1-l2_loss)*wg for wg in w_grads]
        b_grads = [(1-l2_loss)*bg for bg in b_grads]

    if not isinstance(learning_rate, (list, tuple)):
        learning_rate = [learning_rate]*len(ws)

    new_ws = [w - lr/beta * w_grad for w, w_grad, lr in izip_equal(ws, w_grads, learning_rate)]
    new_bs = [b - lr/beta * b_grad for b, b_grad, lr in izip_equal(bs, b_grads, learning_rate)]
    return new_ws, new_bs


def _params_vals_to_params(ws: Sequence[np.ndarray], bs: Sequence[np.ndarray]):
    return [LayerParams(w_aft = None if i==0 else ws[i-1], w_fore = ws[i].T if i<len(ws) else None, b=None if i==0 else bs[i-1]) for i in range(len(ws)+1)]


def initialize_params(layer_sizes: Sequence[int], initial_weight_scale=1., rng = None) -> Sequence[LayerParams]:
    rng = get_rng(rng)
    ws = [rng.uniform(low=-initial_weight_scale*np.sqrt(6./(n_pre+n_post)), high=np.sqrt(6./(n_pre+n_post)), size=(n_pre, n_post))
      for n_pre, n_post in izip_equal(layer_sizes[:-1], layer_sizes[1:])]
    bs = [np.zeros(n_post) for n_post in layer_sizes[1:]]
    return _params_vals_to_params(ws, bs)


def initialize_states(layer_constructor: Callable[[int, LayerParams], IDynamicLayer], n_samples: int, params: Sequence[LayerParams]) -> Sequence[IDynamicLayer]:
    return [layer_constructor(n_samples, p) for p in params]


def output_from_state(states: Sequence[IDynamicLayer]):
    return states[-1].potential


def run_negative_phase(x_data, layer_states: Sequence[IDynamicLayer], n_steps, prop_direction) -> Iterable[Sequence[IDynamicLayer]]:
    if prop_direction==PropDirectionOptions.SWAP:
        prop_direction = PropDirectionOptions.FORWARD
    if prop_direction==PropDirectionOptions.FASTFORWARD:
        for t in range(n_steps):
            layer_states = eqprop_fast_forward_step(layer_states=layer_states, x_data=x_data)
            yield layer_states
    else:
        for t in range(n_steps):
            layer_states = eqprop_step(layer_states=layer_states, x_data = x_data, y_data=None, beta=0, direction=prop_direction)
            yield layer_states


def run_positive_phase(x_data, y_data, beta, layer_states: Sequence[IDynamicLayer], n_steps, prop_direction) -> Iterable[Sequence[IDynamicLayer]]:
    if prop_direction==PropDirectionOptions.SWAP:
        prop_direction = PropDirectionOptions.BACKWARD
    for t in range(n_steps):
        layer_states = eqprop_step(layer_states=layer_states, x_data = x_data, y_data=y_data, beta=beta, direction=prop_direction)
        yield layer_states


def run_inference(x_data, states: Sequence[IDynamicLayer], n_steps: int, prop_direction=PropDirectionOptions.NEUTRAL):

    negative_states = last(run_negative_phase(x_data=x_data, layer_states=states, n_steps=n_steps, prop_direction=prop_direction))
    return output_from_state(negative_states)


def run_eqprop_training_update(x_data, y_data, layer_states: Sequence[IDynamicLayer], beta: float, random_flip_beta: bool,
                               learning_rate: float, n_negative_steps: int, n_positive_steps: int, layer_constructor: Optional[Callable[[int, LayerParams], IDynamicLayer]]=None,
                               bidirectional:bool=True, l2_loss:Optional[float]=None, renew_activations:bool = True, prop_direction=PropDirectionOptions.NEUTRAL, splitstream=False, rng=None) -> Sequence[IDynamicLayer]:

    if isinstance(prop_direction, (list, tuple)):
        negative_prop_direction, positive_prop_direction = prop_direction
    else:
        negative_prop_direction, positive_prop_direction = prop_direction, prop_direction

    rng = get_rng(rng)
    this_beta = beta * rng.choice([-1, 1]) if random_flip_beta else beta
    negative_states = last(run_negative_phase(x_data=x_data, layer_states=layer_states, n_steps=n_negative_steps, prop_direction=negative_prop_direction))
    positive_states = last(run_positive_phase(x_data=x_data, layer_states=negative_states, beta=this_beta, y_data=y_data, n_steps=n_positive_steps, prop_direction=positive_prop_direction))
    if splitstream:
        negative_states = last(run_negative_phase(x_data=x_data, layer_states=negative_states, n_steps=n_positive_steps, prop_direction=positive_prop_direction))

    ws, bs = zip(*((s.params.w_aft, s.params.b) for s in layer_states[1:]))
    neg_acts, pos_acts = [[ls.potential for ls in later_state] for later_state in (negative_states, positive_states)]
    new_ws, new_bs = eqprop_update(
        negative_acts=neg_acts,
        positive_acts=pos_acts,
        ws=ws,
        bs=bs,
        learning_rate=learning_rate,
        beta=this_beta,
        bidirectional=bidirectional,
        l2_loss=l2_loss
    )
    new_params = _params_vals_to_params(new_ws, new_bs)
    if renew_activations:
        assert layer_constructor is not None, 'If you choose renew_activations true, you must provide a layer constructor.'
        new_states = initialize_states(n_samples=x_data.shape[0], params=new_params, layer_constructor=layer_constructor)
    else:
        new_states = [dataclasses.replace(s, params=p) for s, p in izip_equal(positive_states, new_params)]
    return new_states
