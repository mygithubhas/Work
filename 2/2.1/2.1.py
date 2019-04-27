from scipy import *
from numpy import *
from PIL import Image
import harris

im = array(Image.open('asd.jpg').convert('L'))
harrisim = harris.compute_harris_response(im)
filtered_coords = harris.get_harris_points(harrisim,6)
harris.plot_harris_points(im,filtered_coords)