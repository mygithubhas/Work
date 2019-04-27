from PIL import Image
pil_im = Image.open('asd.jpg')
pil_im.resize((128,128)).show()#缩小为指定尺寸，不改变原始数据，创建副本。

pil_im.rotate(45).show()#旋转图像

