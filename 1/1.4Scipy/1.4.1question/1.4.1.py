from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open('synth_rof.jpg').convert('L'))
im2 = filters.gaussian_filter(im,5)
pil_im = Image.fromarray(im2)
pil_im.show()
pil_im.save('absqwe.jpg')
"""
im = array(Image.open('qwe.jpg'))
im2 = zeros(im.shape)
for i in range(3):
	im2[:,:,i] = filters.gaussian_filter(im[:,:,i],5)
im2 = uint8(im2)
im2 = array(im2,'uint8')
pil_im = Image.fromarray(im2)
pil_im.show()"""

