import os
from PIL import Image
from pylab import *
def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
	#目录和文件名合成一个目录，for返回文件的所有文件列表，判断是否以指定后缀结尾。

def imresize(im,sz):#传入np.array
	pil_im = Image.fromarray(uint8(im))#array转化为Image对象
	return array(pil_im.resize(sz))
	
def histeq(im,nbr_bins = 256):#传入np.array
	imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)#一维化，长度为256
	cdf = imhist.cumsum()#计算各行累加和
	cdf = 255*cdf/cdf[-1]#-1指列表中最后一个
	im2 = interp(im.flatten(),bins[:-1],cdf)#[:-1]从0到最后一个
	return im2.reshape(im.shape),cdf#还原图像原始尺寸
	
def compute_average(imlist):
	averageim = array(Image.open(imlist[0]),'f')
	for imname in imlist[1:]:
		try:
			averageim += array(Image.open(imname))
		except:
			printf(imname+'....skipped')
	averageim /= len(imlist)
	return array(averageim,'uint8')