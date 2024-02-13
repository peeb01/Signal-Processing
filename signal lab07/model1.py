''' Regression '''

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm

import matplotlib.image as mpimg
img1 = mpimg.imread('happyCK.png').flatten()
img2 = mpimg.imread('angerCK.png').flatten()
img3 = mpimg.imread('disgustCK.png').flatten()
img4 = mpimg.imread('fearCK.png').flatten()
img5 = mpimg.imread('neutralCK.png').flatten()
img6 = mpimg.imread('sadCK.png').flatten()
img7 = mpimg.imread('surpriseCK.png').flatten()


# imgs = np.array([img1, img2, img3, img4, img5, img6, img7])
print('\n',img1, '\n')

stext =     ''' Create Matrix ==> R.A = S\n\n
        Matrix R : 2x2\n
        Martix S : 2x2\n
        Matrix A = R^-1 . S
    '''

def create_matrix(img):
    N = len(img)
    n = np.arange(1,N+1)
    # print(len(n) == N) # True
    R = np.array([[N, np.sum(n)], [np.sum(n), np.sum(n**2)]])
    ns = np.array([n[i]*img[i] for i in range(N)])
    S = np.array([[np.sum(img)], [np.sum(ns)]])
    A = np.dot(inv(R), S)
    return A    


n = np.arange(1, len(img1)+1)



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


plt.figure(figsize=(12,7))
plt.plot(y1, label = 'happy')
plt.plot(y2, label = 'anger')
plt.plot(y3, label = 'disgust')
plt.plot(y4, label = 'fear')
plt.plot(y5, label = 'neutral')
plt.plot(y6, label = 'sad')
plt.plot(y7, label = 'surprise')
# plt.title('A regression between hap, ang, dis, fea, neu, sad, sur')

plt.legend()
plt.savefig('LinearNH21.png')
plt.show()

