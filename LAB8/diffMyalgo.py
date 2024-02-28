import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# from numpy.linalg import inv

Normal = mpimg.imread('Image\\neutralCK.png')

img1 = mpimg.imread('Image\\happyCK.png')
img2 = mpimg.imread('Image\\angerCK.png')
img3 = mpimg.imread('Image\\disgustCK.png')
img4 = mpimg.imread('Image\\fearCK.png')
img5 = mpimg.imread('Image\\sadCK.png')
img6 = mpimg.imread('Image\\surpriseCK.png')

diff1 = img1 - Normal
diff2 = img2 - Normal
diff3 = img3 - Normal
diff4 = img4 - Normal
diff5 = img5 - Normal
diff6 = img6 - Normal

diff1T = img1.T - Normal
diff2T = img2.T - Normal
diff3T = img3.T - Normal
diff4T = img4.T - Normal
diff5T = img5.T - Normal
diff6T = img6.T - Normal

grad_1 = np.sqrt(diff1**2)
grad_2 = np.sqrt(diff2**2)
grad_3 = np.sqrt(diff3**2)
grad_4 = np.sqrt(diff4**2)
grad_5 = np.sqrt(diff5**2)
grad_6 = np.sqrt(diff6**2)


def calculate_angle(diff_T, diff):
    with np.errstate(divide='ignore', invalid='ignore'): 
        angles = np.arctan(np.divide(diff_T, diff, where=diff!=0, out=np.full_like(diff, np.pi/2)))
    return angles


theta1 = calculate_angle(diff1T, diff1)
theta2 = calculate_angle(diff2T, diff2)
theta3 = calculate_angle(diff3T, diff3)
theta4 = calculate_angle(diff4T, diff4)
theta5 = calculate_angle(diff5T, diff5)
theta6 = calculate_angle(diff6T, diff6)


Vec1 = (grad_1*np.exp(1j*theta1)).flatten()
Vec2 = (grad_2*np.exp(1j*theta2)).flatten()
Vec3 = (grad_3*np.exp(1j*theta3)).flatten()
Vec4 = (grad_4*np.exp(1j*theta4)).flatten()
Vec5 = (grad_5*np.exp(1j*theta5)).flatten()
Vec6 = (grad_6*np.exp(1j*theta6)).flatten()


data1 = np.concatenate((theta1.reshape(-1,1), np.real(Vec1).reshape(-1,1)), axis=1)
data2 = np.concatenate((theta2.reshape(-1,1), np.real(Vec2).reshape(-1,1)), axis=1)
data3 = np.concatenate((theta3.reshape(-1,1), np.real(Vec3).reshape(-1,1)), axis=1)
data4 = np.concatenate((theta4.reshape(-1,1), np.real(Vec4).reshape(-1,1)), axis=1)
data5 = np.concatenate((theta5.reshape(-1,1), np.real(Vec5).reshape(-1,1)), axis=1)
data6 = np.concatenate((theta6.reshape(-1,1), np.real(Vec6).reshape(-1,1)), axis=1)

data1 = data1[data1[:,0].argsort()]
data2 = data2[data2[:,0].argsort()]
data3 = data3[data3[:,0].argsort()]
data4 = data4[data4[:,0].argsort()]
data5 = data5[data5[:,0].argsort()]
data6 = data6[data6[:,0].argsort()]

# data1[:, 1] = np.real(np.fft.fft(data1[:, 1]))
# data2[:, 1] = np.real(np.fft.fft(data2[:, 1]))
# data3[:, 1] = np.real(np.fft.fft(data3[:, 1]))
# data4[:, 1] = np.real(np.fft.fft(data4[:, 1]))
# data5[:, 1] = np.real(np.fft.fft(data5[:, 1]))
# data6[:, 1] = np.real(np.fft.fft(data6[:, 1]))

plt.figure()
# plt.plot(theta1.flatten(), np.real(Vec1), label='Vec1')
# plt.plot(theta2.flatten(), np.real(Vec2), label='Vec2')
# plt.plot(theta3.flatten(), np.real(Vec3), label='Vec3')
# plt.plot(theta4.flatten(), np.real(Vec4), label='Vec4')
# plt.plot(theta5.flatten(), np.real(Vec5), label='Vec5')
# plt.plot(theta6.flatten(), np.real(Vec6), label='Vec6')

# plt.plot(data1[:, 0], data1[:, 1], label='Vec1')
# plt.plot(data2[:, 0], data2[:, 1], label='Vec2')
# plt.plot(data3[:, 0], data3[:, 1], label='Vec3')
# plt.plot(data4[:, 0], data4[:, 1], label='Vec4')
# plt.plot(data5[:, 0], data5[:, 1], label='Vec5')
# plt.plot(data6[:, 0], data6[:, 1], label='Vec6')

plt.plot(grad_1.flatten(), label='Vec1')
plt.plot(grad_2.flatten(), label='Vec2')
plt.plot(grad_3.flatten(), label='Vec3')
plt.plot(grad_4.flatten(), label='Vec4')
plt.plot(grad_5.flatten(), label='Vec5')
plt.plot(grad_6.flatten(), label='Vec6')

plt.xlabel('Angle (theta) radiant')
plt.ylabel('Real Part of Vector')
plt.legend()
plt.show()