import l0_module
import cv2
import numpy as np

# Example using the main function
# Image to Image

l0_module.main_l0("country_house.jpg",0.2,"country_house_example.png")

# l0_module.main_l0("jungle_cat.jpg",0.2,"jungle_cat_l0.png")


# Example using l0_norm function
# nparray to nparray
# Parameters:
# input_image as nparray (require)
# lambda as float (require)
# maxSize as int (optional - default 32)
# maxLoop as int (optional - default 100)

#load the image
image = cv2.imread('country_house.jpg')
cv2.imwrite('country_house_input.png', image)

#convert to np
image_np = np.asarray(image)

# Example passing all parameters

res = l0_module.l0_norm(image_np, 0.2, 32, 100)
cv2.imwrite('country_house_output.png', res)

# Example using default parameters
res = l0_module.l0_norm(image_np, 0.2)
cv2.imwrite('country_house_output_default.png', res)