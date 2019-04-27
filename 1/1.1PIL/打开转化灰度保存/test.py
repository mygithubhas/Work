from PIL import Image
pil_im = Image.open('test.jpg').convert('L')#转化为灰度图L
pil_im.show()
pil_im.save('qwe.jpg')