import numpy as np
import numpy.typing as npt


def gen_audio_array_baseline(
        sampling_rate,
        time,
        bit_depth,
        start=0,
        stop=None,
        step=1) -> npt.NDArray:
    if  not stop:
        stop = time * sampling_rate
    return np.arange(start, stop, step, dtype=bit_depth)


def gen_sine_wave():
    pass


# ToDo
# plot_sig
# plot_dft
# dft_diff