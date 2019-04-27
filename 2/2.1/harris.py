from scipy.ndimage import filters
from numpy import *
from pylab import *
def compute_harris_response(im,sigma=3):
	imx = zeros(im.shape)
	filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
	imy = zeros(im.shape)
	filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)
	Wxx = filters.gaussian_filter(imx*imx,sigma)
	Wxy = filters.gaussian_filter(imx*imy,sigma)
	Wyy = filters.gaussian_filter(imy*imy,sigma)
	
	Wdet = Wxx*Wyy-Wxy**2
	Wtr = Wxx+Wyy
	return Wdet/Wtr
	
	
def get_harris_points(harrisim,min_dist=10,threshold=0.1):
	corner_threshold = harrisim.max() *threshold
	harrisim_t = (harrisim>corner_threshold)*1
	coords = array(harrisim_t.nonzero()).T
	
	candidate_values = [harrisim[c[0],c[1]]for c in coords]
	
	index = argsort(candidate_values)
	
	allowed_locations = zeros(harrisim.shape)
	allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
	filtered_coords = []
	for i in index:
		if allowed_locations[coords[i,0],coords[i,1]] == 1:
			filtered_coords.append(coords[i])
			allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
	return filtered_coords
	
def plot_harris_points(image,filtered_coords):
	figure()
	gray()
	imshow(image)
	plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
	axis('off')
	show()
	
	
def get_descriptors(image,filtered_coords,wid=5):
	desc = []
	for coords in filtered_coords:
		patch = image[coords[0]-wid:coords[0]+wid+1,coords[1]-wid:coords[1]+wid+1].flatten()
		desc.append(patch)
	return desc
	
def match(desc1,desc2,threshold=0.5):
	n = len(desc1[0])
	d = -ones((len(desc1),len(desc2)))
	for i in range(len(desc1)):
		for j in range(len(desc2)):
			d1 = (desc1[i]-mean(desc1[i]))/std(desc1[i])
			d2 = (desc2[j]-mean(desc2[j]))/std(desc2[j])
			ncc_value = sum(d1*d2)/(n-1)
			if ncc_value>threshold:
				d[i,j] = ncc_value
	ndx = argsort(-d)
	matchscores = ndx[:,0]
	return matchscores
	
def match_twosided(desc1,desc2,threshold = 0.5):
	matches_12 = match(desc1,desc2,threshold)
	matches_21 = match(desc2,desc1,threshold)
	
	ndx_12 = where(matches_12>=0)[0]
	for n in ndx_12:
		if matches_21[matches_12[n]]!=n:
			matches_12[n] = -1
	return matches_12
	
def appendimages(im1,im2):
	rows1 = im1.shape[0]
	rows2 = im2.shape[0]
	
	if rows1<rows2:
		im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
	elif rows1>rows2:
		im2 = concatenate((im1,im2),axis=1)
	return concatenate((im1,im2),axis = 1)
	
def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
	im3 = appendimages(im1,im2)
	if show_below:
		im3 = vstack((im3,im3))
	imshow(im3)
	
	cols1 = im1.shape[1]
	for i,m in enumerate(matchscores):
		if m>0:
			plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
	axis('off')
	
