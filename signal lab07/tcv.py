import cv2
import matplotlib.pyplot as plt


# Load the image
image = cv2.imread('happyCK.png')

# Apply a filter (e.g., Gaussian blur) to the image
filtered_image = cv2.GaussianBlur(image, (15, 15), 0)
print(filtered_image)
# # Display the original and filtered images
# cv2.imshow('Original Image', image)
# cv2.imshow('Filtered Image', filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(filtered_image)
plt.show()