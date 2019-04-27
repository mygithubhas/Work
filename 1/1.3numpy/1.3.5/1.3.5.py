from PIL import Image
from numpy import *
import imtools
from pylab import *

path = "D:/project/1/1.3numpy/1.3.5"
#im = array(Image.open("asd.jpg").convert('L'),'f')
im2 = imtools.compute_average(imtools.get_imlist(path))

pil_im = Image.fromarray(im2)
pil_im.show()

