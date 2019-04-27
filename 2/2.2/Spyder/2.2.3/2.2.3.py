# -*- coding: utf-8 -*-
import sift
from numpy import *
from PIL import Image
from pylab import *
"""
imname = 'qaz.jpg'
im1 = array(Image.open(imname).convert('L'))
sift.process_image(imname,'empire.sift')
l1,d1 = sift.read_features_from_file('empire.sift')

figure()
gray()
sift.plot_features(im1,l1,circle=ture)
show()"""

if __name__ == '__main__':
    imname = "qaz.jpg"              #待处理图像路径
    im=Image.open(imname)
    sift.process_image(imname,'test.sift')
    l1,d1 = sift.read_features_from_file('test.sift')           #l1为兴趣点坐标、尺度和方位角度 l2是对应描述符的128 维向
    figure()
    gray()
    plot_features(im,l1,circle = True)
    title('sift-features')
    show()
