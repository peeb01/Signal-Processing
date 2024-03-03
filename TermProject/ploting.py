import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 


df = pd.read_csv('Dublin IRE AQI.csv')


values = df['AQI'].values

fft_result = np.fft.fft(values)
freq = np.fft.fftfreq(len(values), df.index[1] - df.index[0])

positive_freq = freq[:len(freq)//2]
amplitude = np.abs(fft_result[:len(fft_result)//2])

# # Plot the amplitude spectrum
# plt.plot(amplitude)
# plt.title('Fourier Transform - Amplitude Spectrum')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# # plt.savefig('FTT')
# plt.show()



# print(df)
n = np.arange(0, len(df))


plt.figure(figsize=(12,6))
plt.plot(n, df['AQI'].values)
plt.xlabel('n')
plt.ylabel('AQI')
# plt.title('Plot AQI Data')
# plt.savefig('AQI')
plt.show()

