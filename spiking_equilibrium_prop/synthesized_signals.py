from scipy.signal import butter
import numpy as np
from artemis.general.numpy_helpers import get_rng
from scipy import signal


def lowpass(sig, cutoff):
    b, a = butter(N=2, Wn=cutoff)
    new_sig = signal.lfilter(b, a, sig, axis=0)
    return new_sig


def lowpass_random(n_samples, cutoff, n_dim=None, rng = None, normalize = False, slope=0):
    """
    Return a random lowpass-filtered signal.
    :param n_samples:
    :param cutoff:
    :param rng:
    :return:
    """
    rng = get_rng(rng)
    assert 0<=cutoff<=1, "Cutoff must be in the range 0 (pure DC) to 1 (sample frequency)"
    base_signal = rng.randn(n_samples) if n_dim is None else rng.randn(n_samples, n_dim)
    lowpass_signal = lowpass(base_signal, cutoff)
    if normalize is True:
        lowpass_signal = lowpass_signal/np.std(lowpass_signal)
    elif isinstance(normalize, tuple):
        lower, upper = normalize
        minsig, maxsig= np.min(lowpass_signal, axis=0), np.max(lowpass_signal, axis=0)
        lowpass_signal = ((lowpass_signal - minsig)/(maxsig-minsig)) * (upper-lower) + lower
    if slope != 0:
        ramp = slope*np.arange(len(lowpass_signal))
        lowpass_signal = lowpass_signal+(ramp if n_dim is None else ramp[:, None])
    return lowpass_signal
