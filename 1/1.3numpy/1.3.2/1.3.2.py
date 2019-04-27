from PIL import Image
from numpy import *
from pylab import *
im = array(Image.open("asd.jpg").convert('L'),'f')
im2 = 255-im
imshow(im2)
show()
im3 = (100.0/255)*im +100
imshow(im3)
show()
im4 = 255.0*(im/255.0)**2
imshow(im4)
show()
print(int(im.min()),int(im.max()))

im5 = Image.fromarray(im)

im5.show()