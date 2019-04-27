from scipy.ndimage import measurements,morphology
from numpy import *
from PIL import Image
im = array(Image.open('qwe.jpg').convert('L'))
im = 1*(im<128)
"""
labels,nbr_objects = measurements.label(im)
print("Number of object:",nbr_objects)

pil_im = Image.fromarray(labels)
pil_im.show()
"""
#im_open = morphology.binary_opening(im,ones((9,5)),iterations = 2)
im_open = morphology.binary_opening(im,ones((1,1)),iterations = 2)
labels,nbr_objects = measurements.label(im_open)
print("Number of object:",nbr_objects)

pil_im = Image.fromarray(labels)
pil_im.show()