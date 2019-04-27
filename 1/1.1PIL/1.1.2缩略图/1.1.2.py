from PIL import Image
pil_im = Image.open('test.jpg')#打开文件
pil_im.thumbnail((128,128))#等比例缩小到指定尺度，若无法缩小到指定尺度，则选择其中一个参数处理
pil_im.save('qwe.jpg')