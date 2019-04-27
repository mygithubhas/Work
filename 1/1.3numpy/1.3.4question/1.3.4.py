from PIL import Image
from numpy import *
import imtools
from pylab import *
im = array(Image.open("asd.jpg").convert('L'),'f')
im2,cdf = imtools.histeq(im)
pil_im = Image.fromarray(im2)
pil_im.show()

"""
figure()
gray()
contour(pil_im,origin = 'image')

axis('equal')
axis('off')
figure()"""
hist(im.flatten(),cdf)
show()
hist(im2.flatten(),cdf)
show()