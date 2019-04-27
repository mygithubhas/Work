# -*- coding: utf-8 -*-
import os
from numpy import *
from PIL import Image
from pylab import *
"""
def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
	if imagename[-3:]!='pgm':
		im = Image.open(imagename).convert('L')
		im.save('tmp.pgm')
		imagename = 'tmp.pgm'
		
	cmmd = str("sift"+imagename+"--output="+resultname+" "+params)
	os.system(cmmd)
	print('processed',imagename,'to',resultname)
	
def read_features_from_file(filename):
	f = loadtxt(filename)
	return f[:,:4],f[:,4:]
	
def write_features_to_file(filename,locs,desc):
	savetxt(filename,hstack((locs,desc)))
	
	
def plot_features(im,locs,circle=False):
	def draw_circle(c,r):
		t = arange(0,1.01,.01)*2*pi
		x = r*cos(t)+c[0]
		y = r*sin(t)+c[1]
		plot(x,y,'b',linewidth=2)
		imshow(im)
		
	if circle:
		for p in locs:
			draw_circle(p[:2],p[2])
	else:
		plot(locs[:,0],locs[:,1],'ob')
	axis('off')"""


def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
   """ Process an image and save the results in a file. """

   if imagename[-3:] != 'pgm':
       # create a pgm file
       im = Image.open(imagename).convert('L')  #.convert('L') 将RGB图像转为灰度模式，灰度值范围[0,255]
       im.save('tmp.pgm')                       #将灰度值图像信息保存在.pgm文件中
       imagename = 'tmp.pgm'

   cmmd = str("E:\code\Sift\sift_python\sift.exe "+imagename+" --output="+resultname+
               " "+params)
   os.system(cmmd)                              #执行sift可执行程序，生成resultname(test.sift)文件
   print('processed', imagename, 'to', resultname)
def read_features_from_file(filename):
  # Read feature properties and return in matrix form. 

    f = loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors


def plot_features(im,locs,circle=True):
   """ Show image with features. input: im (image as array), 
       locs (row, col, scale, orientation of each feature). """

   def draw_circle(c,r):
       t = arange(0,1.01,.01)*2*pi
       x = r*cos(t) + c[0]
       y = r*sin(t) + c[1]
       plot(x,y,'b',linewidth=2)

   imshow(im)
   if circle:
       for p in locs:
           draw_circle(p[:2],p[2]) 
   else:
       plot(locs[:,0],locs[:,1],'ob')
   axis('off')