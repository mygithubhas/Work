from PIL import Image
import numpy
pil_im = Image.open('asd.jpg')
box = (100,100,300,300)#坐标值，并非偏移量
region = pil_im.crop(box)#剪裁指定区域
region = region.transpose(Image.ROTATE_180)#转置
pil_im.paste(region,box)#放回指定区域
pil_im.show()#显示图片
pil_im.save('qwe.jpg')