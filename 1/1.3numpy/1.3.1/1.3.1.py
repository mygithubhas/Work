from PIL import Image
from pylab import *

im = array(Image.open("asd.jpg"))
print(im.shape,im.dtype)

im = array(Image.open("asd.jpg").convert('L'),'f')
print(im.shape,im.dtype)#形状，数据类型