import os
from numpy import *
from PIL import Image
from pylab import *
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
	axis('off')