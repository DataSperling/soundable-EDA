"""
Test of methods scipy.fft, fft fftfreq, rfft and rfftfreq produce consistent 
(normalisable) output when applied to simulated and measured (.wav type) audio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import wave

SAMPLE_FREQUENCY = 44100;                 # frequency (Hz)
SAMPLING_DURATION  = 4;                   # time (s)
N = SAMPLE_FREQUENCY * SAMPLING_DURATION  # number of samples


# Generate periodic signal of given frequency
def gen_simple_sinwave(frequency, rate, duration):
  x = np.linspace(0, duration, (rate*duration), endpoint=False)
  frequencies = x*frequency
  y = np.sin( (np.pi*2) * frequencies)
  return x,y


# Simulate composite signal consisting of 3 frequencies with weighting
_, sig_1 = gen_simple_sinwave(100, SAMPLE_FREQUENCY, SAMPLING_DURATION)
_, sig_2 = gen_simple_sinwave(200, SAMPLE_FREQUENCY, SAMPLING_DURATION)
_, sig_3 = gen_simple_sinwave(800, SAMPLE_FREQUENCY, SAMPLING_DURATION)
sig_comp = sig_1 + (0.4*sig_2) + (0.2*sig_3)
plt.figure(figsize=(20,10))
plt.plot(sig_comp[:3000], color='red')
plt.title('800Hz, 200Hz, 100Hz Composite Signal in 1:2:5 Intensity',
          fontsize=25)
plt.ylabel('f(x)', fontsize=20)
plt.xlabel('Sample Number', fontsize=20)
plt.show()


# Normalise composite signal for 16-bit "type"
sig_norm = np.int16( (sig_comp / sig_comp.max() ) * 32767)
plt.figure(figsize=(20,10))
plt.plot(sig_norm[:3000], color='green')
plt.title('800Hz, 200Hz, 100Hz Normalised Signal in 1:2:5 Intensity', fontsize=25)
plt.ylabel('f(x)', fontsize=20)
plt.xlabel('Sample Number', fontsize=20)
plt.show()


# Export composite signal for input processing later
write('test_sample.wav', SAMPLE_FREQUENCY, sig_norm)


# Plot FFT to check 3 frequencies present at correct intensities (5:2:1)
yj = rfft(sig_norm)
xj = rfftfreq(N, 1/SAMPLE_FREQUENCY)
plt.figure(figsize=(20,10))
plt.plot(xj, np.abs(yj), color='blue')
plt.title('FFT (SciPy.rfft) 800Hz, 200Hz, 100Hz Simulated Signal in 1:2:5 Intensity',
          fontsize=25)
plt.ylabel('Relative Intensity',
          fontsize=20)
plt.xlabel('Frequency (Hz)',
          fontsize=20)
plt.xlim(0, 1000)
plt.show()


# Check parameters of exported audio before plotting FFT
try:
  test_sample = wave.open('test_sample.wav')
  number_samples = test_sample.getnframes()
  sample_rate = test_sample.getframerate()
  print('sample rate: ', sample_rate, '\n',
        'number channels: ', test_sample.getnchannels(), '\n',
        'number samples: ', number_samples, '\n',
        'duration audio: ', number_samples / test_sample.getframerate())

# Plot FFT
  sig_wav = test_sample.readframes(number_samples)
  sig_arr = np.frombuffer(sig_wav, dtype=np.int16)      
  yk = rfft(sig_arr)
  xk = rfftfreq(number_samples, 1/sample_rate)
  plt.figure(figsize=(20,10))
  plt.plot(xk, np.abs(yk), color='black')
  plt.title('FFT (SciPy.rfft) 800Hz, 200Hz, 100Hz Real Signal',
            fontsize=25)
  plt.ylabel('Relative Intensity',
             fontsize=20)
  plt.xlabel('Frequency (Hz)',
             fontsize=20)
  plt.xlim(0, 1000)
  plt.show()
finally:
  test_sample.close()



print("EOF---EOF---EOF---EOF---EOF---EOF---EOF---EOF---EOF---EOF---EOF---EOF")
