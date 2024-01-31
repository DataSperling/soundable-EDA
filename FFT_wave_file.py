"""
script to test the methods scipy.fft, fft fftfreq, rfft and rfftfreq produce
consistent (normalisable) output when applied to calculated and measured .wav
type audio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq, rfft, rfftfreq

SAMPLING_FREQUENCY = 44100;   # frequency in Hz / s-1
SAMPLE_DURATION  = 4;         # time in s 
NO_SAMPLES = SAMPLING_FREQUENCY * SAMPLE_DURATION

# step 1 define function to generate periodic sinwave
def gen_simple_sinwave(frequency, rate, duration):
  x = np.linspace(0, duration, (rate*duration), endpoint=False)
  frequencies = x*frequency
  y = np.sin( (np.pi*2) * frequencies)
  return x,y

# step 2 use function to generate mixed sample and save as waveform file
# intensities modeled as the square of frequency
_, sig_low = gen_simple_sinwave(150, SAMPLING_FREQUENCY, SAMPLE_DURATION)
sig_low = sig_low*0.3
_, sig_med = gen_simple_sinwave(600, SAMPLING_FREQUENCY, SAMPLE_DURATION)
sig_med = sig_med*0.6
_, sig_hi  = gen_simple_sinwave(2400, SAMPLING_FREQUENCY, SAMPLE_DURATION)
sig_hi  = sig_hi*0.9
sig_comp = sig_low + sig_med + sig_hi
plt.plot(sig_comp[:1000])
plt.show()
write('sig_comp.wav', SAMPLING_FREQUENCY, sig_comp)



# step 3 calculate FFT of generated waveform
fy = rfft(sig_comp)
fx = rfftfreq(NO_SAMPLES, 1/SAMPLING_FREQUENCY)
plt.plot(fx, np.abs(fy))
plt.show()

# step 4 check by subtraction of the FT's that the algorithmically generated FFT
# and the FFT from waveform analysis are the same

print("EOF---EOF---EOF")
