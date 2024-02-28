''' Regression '''

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm

import matplotlib.image as mpimg
img1 = mpimg.imread('happyCK.png').flatten()
img2 = mpimg.imread('angerCK.png')
img3 = mpimg.imread('disgustCK.png')
img4 = mpimg.imread('fearCK.png')
img5 = mpimg.imread('neutralCK.png')
img6 = mpimg.imread('sadCK.png')
img7 = mpimg.imread('surpriseCK.png')
img_test = mpimg.imread('testB.png')

# imgs = np.array([img1, img2, img3, img4, img5, img6, img7])
# print('\n',img_test, '\n')

stext =     ''' Create Matrix ==> R.A = S\n\n
        Matrix R : 2x2\n
        Martix S : 2x2\n
        Matrix A = R^-1 . S
    '''

def create_max(img):
    N = len(img)
    n = np.arange(1,N+1)
    # print(len(n) == N) # True
    R = np.array([[N, np.sum(n)], [np.sum(n), np.sum(n**2)]])
    ns = np.array([n[i]*img[i] for i in range(N)])
    S = np.array([[np.sum(img)], [np.sum(ns)]])
    A = np.dot(inv(R), S)
    return A    

def create_matrix(img):
    Ax = []

    for i in range(len(img)):
        N = len(img[i])
        n = np.arange(1, N+1)
        R = np.array([[N, np.sum(n)], [np.sum(n), np.sum(n**2)]])
        ns = np.array([n[i]*img[i] for i in range(N)])
        S = np.array([[np.sum(img[i])], [np.sum(ns)]])
        A = np.dot(inv(R), S)
        Ax.append(A)
    return np.array(Ax)

# xx = create_mx(img1)
# print(xx)


n = np.arange(1, len(img1)+1)



A1 = create_matrix(img1)

A2 = create_matrix(img2)

A3 = create_matrix(img3)

A4 = create_matrix(img4)

A5 = create_matrix(img5)

A6 = create_matrix(img6)

A7 = create_matrix(img7)

At = create_matrix(img_test)

print(np.max(A1 - A2, axis=0))

# y1 = a1 + b1*n
# y2 = a2 + b2*n
# y3 = a3 + b3*n
# y4 = a4 + b4*n
# y5 = a5 + b5*n
# y6 = a6 + b6*n
# y7 = a7 + b7*n
# y_test = a_test + b_test*n




# plt.figure(figsize=(12,7))
# plt.plot(y1, label = 'happy')
# plt.plot(y2, label = 'anger')
# plt.plot(y3, label = 'disgust')
# plt.plot(y4, label = 'fear')
# plt.plot(y5, label = 'neutral')
# plt.plot(y6, label = 'sad')
# plt.plot(y7, label = 'surprise')
# plt.scatter(n, y_test, label = 'y_Test')
# # plt.title('A regression between hap, ang, dis, fea, neu, sad, sur')

# plt.legend()
# # plt.savefig('LinearNH21.png')
# plt.show()

