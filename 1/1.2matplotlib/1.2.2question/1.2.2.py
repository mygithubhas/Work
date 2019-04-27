from PIL import Image
from pylab import *

im = array(Image.open('asd.jpg').convert('L'))

figure()#创建新图
gray()
contour(im,origin = 'image')

axis('equal')
axis('off')
figure()
hist(im.flatten(),128)
show()#到上一个figure之间是代码块