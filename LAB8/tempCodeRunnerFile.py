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

print(diff1.shape)
