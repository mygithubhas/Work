from PIL import Image
from pylab import *

im = array(Image.open('asd.jpg'))

imshow(im)#绘制图像


x = [100,100,400,400]
y = [200,500,200,500]

plot(x,y,'r*')

plot(x[:2],y[:2],':')

title('Plotting:"asd.jpg"')
show()#此为代码块
