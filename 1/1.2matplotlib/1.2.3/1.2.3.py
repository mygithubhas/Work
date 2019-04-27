from PIL import Image
from pylab import *

im = array(Image.open("asd.jpg"))#转化为numpy.array结构

imshow(im)
#show()

print("please click 3 points")

x = ginput(3)

print("you clicked:",x)
show()