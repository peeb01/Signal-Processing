
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.linalg import inv

img1 = mpimg.imread('happyCK.png').T.flatten()
img2 = mpimg.imread('angerCK.png').T.flatten()
img3 = mpimg.imread('disgustCK.png').T.flatten()
img4 = mpimg.imread('fearCK.png').T.flatten()
img5 = mpimg.imread('neutralCK.png').T.flatten()
img6 = mpimg.imread('sadCK.png').T.flatten()
img7 = mpimg.imread('surpriseCK.png').T.flatten()


def create_matrix(img):
    N = len(img)
    n = np.arange(1,N+1)
    # print(len(n) == N) # True

    ''' Create Matrix ==> R.A = S'''
    # Matrix R
    R = np.array([[N, np.sum(n)], [np.sum(n), np.sum(n**2)]])

    # Martix S
    # n.s[n]
    ns = np.array([n*data for data in img])
    S = np.array([[np.sum(img)], [np.sum(ns)]])

    # Matrix A
    ''' Matrix A = R^-1 . S '''
    A = np.dot(inv(R), S)
    return A

n = np.arange(1, 500)

[a1, b1] = create_matrix(img1)
y1 = a1 + b1*n
[a2, b2] = create_matrix(img2)
y2 = a2 + b2*n
[a3, b3] = create_matrix(img3)
y3 = a3 + b3*n
[a4, b4] = create_matrix(img4)
y4 = a4 + b4*n
[a5, b5] = create_matrix(img5)
y5 = a5 + b5*n
[a6, b6] = create_matrix(img6)
y6 = a6 + b6*n
[a7, b7] = create_matrix(img7)
y7 = a7 + b7*n


plt.figure()
plt.plot(y1, label = 'happy')
plt.plot(y2, label = 'anger')
plt.plot(y3, label = 'disgust')
plt.plot(y4, label = 'fear')
plt.plot(y5, label = 'neutral')
plt.plot(y6, label = 'sad')
plt.plot(y7, label = 'surprise')

plt.legend()
# plt.savefig('LinearT.png')
plt.show()

