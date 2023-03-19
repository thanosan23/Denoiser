import numpy as np
import matplotlib.pyplot as plt

# construct a function (with noise)
dt = 0.001
t = np.arange(0, 1, dt)

f1 = 1 * np.sin(2*np.pi*50*t)
f2 = 1 * np.sin(2*np.pi*120*t)
f = f1 + f2

noise = np.random.randn(len(t)) * 1.2
f += noise

# compute fft
def computeFFT(fn):
    n = len(fn)
    F = np.fft.fft(fn, n)
    F_power = np.abs(F) / n
    F_power *= 2
    F_power[0] /= 2
    return F, F_power

F, F_power = computeFFT(f)

# denoise
indices = F_power > 0.5
F *= indices

denoised = np.fft.ifft(F)

freq = np.arange(len(f))
I = np.arange(1, np.floor(len(f)/2), dtype='int')

fig, axs = plt.subplots(3, 1)

# fast fourier transform graph
plt.sca(axs[0])
plt.plot(freq[I], F_power[I])

# Original Function graph
plt.sca(axs[1])
plt.plot(freq[I], f[I])

# Denoised function graph
plt.sca(axs[2])
plt.plot(freq[I], denoised[I])
plt.show()
