import numpy as np

from utilities.utils import gen_audio_array_baseline
from constants.globals import FRAMES_PER_SECOND_DVD_S


def main() -> None:
    ndarray_1 = gen_audio_array_baseline(
        FRAMES_PER_SECOND_DVD_S,
        2,
        np.int16)

    ndarray_2 = np.arange(
        0,
        2*FRAMES_PER_SECOND_DVD_S,
        1,
        dtype=np.int16)

    print(np.array_equal(ndarray_1, ndarray_2))


if __name__=="__main__":
    main()