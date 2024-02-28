
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
img_test = mpimg.imread('testB.png').flatten()

def create_matrix(img):
    N = len(img)
    n = np.arange(1,N+1)
    # print(len(n) == N) # True

    ''' Create Matrix ==> R.A = S'''
    # Matrix R
    R = np.array([[N, np.sum(n)], [np.sum(n), np.sum(n**2)]])

    # Martix S
    # n.s[n]
    print(inv(R))
    ns = np.array([n*data for data in img])
    S = np.array([[np.sum(img)], [np.sum(ns)]])

    # Matrix A
    ''' Matrix A = R^-1 . S '''
    A = np.dot(inv(R), S)
    return A

n = np.arange(1, 500)

[a1, b1] = create_matrix(img1)
y1 = a1 + b1*n