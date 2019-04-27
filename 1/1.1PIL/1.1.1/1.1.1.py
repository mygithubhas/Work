from PIL import Image
import imtools
import os
filelist = "D:/project/PIL/1.1.1/e"#文件名，或文件绝对地址
print(filelist)
print(os.path.splitext(filelist))#返回文件名绝对地址（“绝对路径”,"后缀名"），
for infile in filelist:
	outfile = os.path.splitext(infile)[0]+".jpg"#在绝对路径名后+".jpg"
	if infile != outfile:
		try:
			Image.open(infile).save(outfile)#打开文件并加上jpg后缀后保存
		except IOError:
			print("cannot convert",infile)