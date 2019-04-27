from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open('qwe.jpg').convert('L'))
imx = zeros(im.shape)
filters.sobel(im,1,imx)
"""
pil_im = Image.fromarray(imx)
pil_im.show()
"""
imy = zeros(im.shape)
filters.sobel(im,0,imy)
#pil_im = Image.fromarray(imy)
#pil_im.show()
magnitude = sqrt(imx**2+imy**2)

#pil_im = Image.fromarray(magnitude)
#pil_im.show()



sigma = 1
imxx = zeros(im.shape)
filters.gaussian_filter(im,(sigma,sigma),(0,1),imxx)



pil_im = Image.fromarray(imxx)
pil_im.show()


imyy = zeros(im.shape)
filters.gaussian_filter(im,(sigma,sigma),(1,0),imyy)


pil_im = Image.fromarray(imyy)
pil_im.show()