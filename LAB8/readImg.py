import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pprint
import pandas as pd
# from numpy.linalg import inv

img1 = mpimg.imread('Image\\happyCK.png')
img2 = mpimg.imread('Image\\angerCK.png')
img3 = mpimg.imread('Image\\disgustCK.png')
img4 = mpimg.imread('Image\\fearCK.png')
img5 = mpimg.imread('Image\\neutralCK.png')
img6 = mpimg.imread('Image\\sadCK.png')
img7 = mpimg.imread('Image\\surpriseCK.png')

# img1, img2, img3, img4 = img1.astype(np.int32), img2.astype(np.int32), img3.astype(np.int32), img4.astype(np.int32)
# img5, img6, img7 = img5.astype(np.int32), img6.astype(np.int32), img7.astype(np.int32)

# if shape all image are equal, for this all image are size in array 48x48

zeros = np.zeros((img1.shape[0], 1), dtype=int) 

img1_ = np.concatenate((zeros, img1), axis=1)
img2_ = np.concatenate((zeros, img2), axis=1)
img3_ = np.concatenate((zeros, img3), axis=1)
img4_ = np.concatenate((zeros, img4), axis=1)
img5_ = np.concatenate((zeros, img5), axis=1)
img6_ = np.concatenate((zeros, img6), axis=1)
img7_ = np.concatenate((zeros, img7), axis=1)



diff1 = np.diff(img1_)
diff2 = np.diff(img2_)
diff3 = np.diff(img3_)
diff4 = np.diff(img4_)
diff5 = np.diff(img5_)
diff6 = np.diff(img6_)
diff7 = np.diff(img7_)

img1_t = np.concatenate((zeros, img1.T), axis=1)
img2_t = np.concatenate((zeros, img2.T), axis=1)
img3_t = np.concatenate((zeros, img3.T), axis=1)
img4_t = np.concatenate((zeros, img4.T), axis=1)
img5_t = np.concatenate((zeros, img5.T), axis=1)
img6_t = np.concatenate((zeros, img6.T), axis=1)
img7_t = np.concatenate((zeros, img7.T), axis=1)


diff1_T = np.diff(img1_t).T
diff2_T = np.diff(img2_t).T
diff3_T = np.diff(img3_t).T
diff4_T = np.diff(img4_t).T
diff5_T = np.diff(img5_t).T
diff6_T = np.diff(img6_t).T
diff7_T = np.diff(img7_t).T



"""
|Grad| = sqrt(diff_i^2 + diff_T_i)

theta = arctan(diff_T_i/diff_i)
"""
grad_1 = np.sqrt(diff1**2 + diff1_T**2)
grad_2 = np.sqrt(diff2**2 + diff1_T**2)
grad_3 = np.sqrt(diff3**2 + diff1_T**2)
grad_4 = np.sqrt(diff4**2 + diff1_T**2)
grad_5 = np.sqrt(diff5**2 + diff1_T**2)
grad_6 = np.sqrt(diff6**2 + diff1_T**2)
grad_7 = np.sqrt(diff7**2 + diff1_T**2)


def calculate_angle(diff_T, diff):
    with np.errstate(divide='ignore', invalid='ignore'): 
        angles = np.arctan(np.divide(diff_T, diff, where=diff!=0, out=np.full_like(diff, np.pi/2)))
    return angles

# Calculate angles in radian
theta1 = calculate_angle(diff1_T, diff1)*180/np.pi
theta2 = calculate_angle(diff2_T, diff2)*180/np.pi
theta3 = calculate_angle(diff3_T, diff3)*180/np.pi
theta4 = calculate_angle(diff4_T, diff4)*180/np.pi
theta5 = calculate_angle(diff5_T, diff5)*180/np.pi
theta6 = calculate_angle(diff6_T, diff6)*180/np.pi
theta7 = calculate_angle(diff7_T, diff7)*180/np.pi

# print(np.max(theta1).flatten())

# # print(theta1[0])
# max_index = np.unravel_index(np.argmax(theta1), theta1.shape)
# min_index = np.unravel_index(np.argmin(theta1), theta1.shape)
# # print(max_index)
# print(theta1[min_index[0]][min_index[1]])

"""
find grad Vector = |Grad|e^jtheta
"""
Vec1 = (grad_1*np.exp(1j*theta1*np.pi/180))
Vec2 = (grad_2*np.exp(1j*theta2*np.pi/180))
Vec3 = (grad_3*np.exp(1j*theta3*np.pi/180))
Vec4 = (grad_4*np.exp(1j*theta4*np.pi/180))
Vec5 = (grad_5*np.exp(1j*theta5*np.pi/180))
Vec6 = (grad_6*np.exp(1j*theta6*np.pi/180))
Vec7 = (grad_7*np.exp(1j*theta7*np.pi/180))


imggr1 = np.real(Vec1)
imggr2 = np.real(Vec2)
imggr3 = np.real(Vec3)
imggr4 = np.real(Vec4)
imggr5 = np.real(Vec5)
imggr6 = np.real(Vec6)
imggr7 = np.real(Vec7)

# plt.imshow(img1, cmap='gray')
# plt.show()
# plt.imshow(grad_1, cmap='gray')
# plt.show()
# plt.imshow(imggr1, cmap='gray')
# plt.show()

Vec1 = Vec1.flatten()
Vec2 = Vec2.flatten()
Vec3 = Vec3.flatten()
Vec4 = Vec4.flatten()
Vec5 = Vec5.flatten()
Vec6 = Vec6.flatten()
Vec7 = Vec7.flatten()


data1 = np.concatenate((theta1.reshape(-1,1), np.real(Vec1).reshape(-1,1)), axis=1)
data2 = np.concatenate((theta2.reshape(-1,1), np.real(Vec2).reshape(-1,1)), axis=1)
data3 = np.concatenate((theta3.reshape(-1,1), np.real(Vec3).reshape(-1,1)), axis=1)
data4 = np.concatenate((theta4.reshape(-1,1), np.real(Vec4).reshape(-1,1)), axis=1)
data5 = np.concatenate((theta5.reshape(-1,1), np.real(Vec5).reshape(-1,1)), axis=1)
data6 = np.concatenate((theta6.reshape(-1,1), np.real(Vec6).reshape(-1,1)), axis=1)
data7 = np.concatenate((theta7.reshape(-1,1), np.real(Vec7).reshape(-1,1)), axis=1)

# print(np.max(data1[:,0].flatten()))
# print(data1)

data1 = data1[data1[:,0].argsort()]
data2 = data2[data2[:,0].argsort()]
data3 = data3[data3[:,0].argsort()]
data4 = data4[data4[:,0].argsort()]
data5 = data5[data5[:,0].argsort()]
data6 = data6[data6[:,0].argsort()]
data7 = data7[data7[:,0].argsort()]

data1[:, 0] = data1[:, 0].astype(np.int64)
data2[:, 0] = data2[:, 0].astype(np.int64)
data3[:, 0] = data3[:, 0].astype(np.int64)
data4[:, 0] = data4[:, 0].astype(np.int64)
data5[:, 0] = data5[:, 0].astype(np.int64)
data6[:, 0] = data6[:, 0].astype(np.int64)
data7[:, 0] = data7[:, 0].astype(np.int64)

# print(pd.DataFrame(data1))


df1 = pd.DataFrame(data1[:,1])
df2 = pd.DataFrame(data2[:,1])
df3 = pd.DataFrame(data3[:,1])
df4 = pd.DataFrame(data4[:,1])
df5 = pd.DataFrame(data5[:,1])
df6 = pd.DataFrame(data6[:,1])
df7 = pd.DataFrame(data7[:,1])


df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1)

thes = np.array([i for i in range(-89, 90)])
# # print(thes)
M1 = []
M2 = []
M3 = []
M4 = []
M5 = []
M6 = []
M7 = []

for i in thes:
    dt1 = data1[data1[:, 0] == i]
    if dt1.size == 0:  # Check if dt1 is empty
        ave1 = 0
    else:
        ave1 = np.mean(dt1[:, 1])
    M1.append(ave1)
    
    dt2 = data2[data2[:, 0] == i]
    if dt2.size == 0:  # Check if dt2 is empty
        ave2 = 0
    else:
        ave2 = np.mean(dt2[:, 1])
    M2.append(ave2)
    
    dt3 = data3[data3[:, 0] == i]
    if dt3.size == 0:  # Check if dt3 is empty
        ave3 = 0
    else:
        ave3 = np.mean(dt3[:, 1])
    M3.append(ave3)
    
    dt4 = data4[data4[:, 0] == i]
    if dt4.size == 0:  # Check if dt4 is empty
        ave4 = 0
    else:
        ave4 = np.mean(dt4[:, 1])
    M4.append(ave4)
    
    dt5 = data5[data5[:, 0] == i]
    if dt5.size == 0:  # Check if dt5 is empty
        ave5 = 0
    else:
        ave5 = np.mean(dt5[:, 1])
    M5.append(ave5)
    
    dt6 = data6[data6[:, 0] == i]
    if dt6.size == 0:  # Check if dt6 is empty
        ave6 = 0
    else:
        ave6 = np.mean(dt6[:, 1])
    M6.append(ave6)
    
    dt7 = data7[data7[:, 0] == i]
    if dt7.size == 0:  # Check if dt7 is empty
        ave7 = 0
    else:
        ave7 = np.mean(dt7[:, 1])
    M7.append(ave7)

df = pd.DataFrame({'L1': M1, 
      'L2': M2, 
      'L3': M3,
      'L4': M4,
      'L5': M5,
      'L6': M6,
      'L7': M7
      })

lst = []
for i in range(len(df)):
    max = df.iloc[i, :].argmax()
    lst.append(max)

print(lst)

plt.figure(figsize=(12,5))

plt.plot(data1[:,0], data1[:,1], label='happy')
plt.plot(data2[:,0], data2[:,1], label='anger')
plt.plot(data3[:,0], data3[:,1], label='disgust')
plt.plot(data4[:,0], data4[:,1], label='fear')
plt.plot(data5[:,0], data5[:,1], label='neutral')
plt.plot(data6[:,0], data6[:,1], label='sad')
plt.plot(data7[:,0], data7[:,1], label='surprise')

# fig, axes = plt.subplots(7, 1, figsize=(8, 12))

# axes[0].plot(data1[:, 0], data1[:, 1])
# axes[0].set_title('happy')

# axes[1].plot(data2[:, 0], data2[:, 1])
# axes[1].set_title('anger')

# axes[2].plot(data3[:, 0], data3[:, 1])
# axes[2].set_title('disgust')

# axes[3].plot(data4[:, 0], data4[:, 1])
# axes[3].set_title('fear')

# axes[4].plot(data5[:, 0], data5[:, 1])
# axes[4].set_title('neutral')

# axes[5].plot(data6[:, 0], data6[:, 1])
# axes[5].set_title('sad')

# axes[6].plot(data7[:, 0], data7[:, 1])
# axes[6].set_title('surprise')

# plt.xlabel('Angle (theta) radiant')
# plt.ylabel('Real Part of Vector')
# plt.tight_layout()
# plt.savefig('Grad.png')
# plt.show()



# plt.plot(grad_1.flatten(), label='happy')
# plt.plot(grad_2.flatten(), label='anger')
# plt.plot(grad_3.flatten(), label='disgust')
# plt.plot(grad_4.flatten(), label='fear')
# plt.plot(grad_5.flatten(), label='neutral')
# plt.plot(grad_6.flatten(), label='sad')
# plt.plot(grad_7.flatten(), label='surprise')

plt.xlabel('Angle (theta) Degree')
plt.ylabel('Real Part of Gradiant')
plt.legend()
# plt.savefig('grad2.png')
plt.show()

