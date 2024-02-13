''' Regression '''

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm

import matplotlib.image as mpimg
img1 = mpimg.imread('happyCK.png')
img2 = mpimg.imread('angerCK.png')
img3 = mpimg.imread('disgustCK.png')
img4 = mpimg.imread('fearCK.png')
img5 = mpimg.imread('neutralCK.png')
img6 = mpimg.imread('sadCK.png')
img7 = mpimg.imread('surpriseCK.png')


# imgs = np.array([img1, img2, img3, img4, img5, img6, img7])
print('\n',img1, '\n')

