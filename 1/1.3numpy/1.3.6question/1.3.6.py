from PIL import Image
from numpy import *
from pylab import *
import pca
import imtools
path = "D:/project/1/1.3numpy/1.3.6question"
#path = "fws.jpg"
imlist = imtools.get_imlist(path)
im = array(Image.open(imlist[0]))
m,n = im.shape[0:2]
imnbr = len(imlist)
immatrix = array([array(Image.open(im)).flatten()for im in imlist],'f')

V,S,immean = pca.pca(immatrix)

figure()
gray()
subplot(2,1,1)
#imshow(immean.reshape(500,500))
imshow(immean.reshape(m,n))
for i in range(7):
	subplot(2,4,i+2)
	imshow(V[i].reshape(m,n))
show()