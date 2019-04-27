from PIL import Image
from numpy import *

def pca(X):
	num_data,dim = X.shape#获取图片维度（元组）
	mean_X = X.mean(axis = 0)#维度均值（0行1列）
	X = X - mean_X
	
	if(dim>num_data):
		M = dot(X,X.T)
		e,EV = linalg.eigh(M)
		tmp = dot(X.T,EV).T#dot矩阵乘，T转置
		V = tmp[::-1]#平方根
		S = sqrt(e)[::-1]
		for i in range(V.shape[1]):
			V[:,i]/=S
	else:
		U,S,V = linalg.svd(X)
		V = V[:num_data]
	return V,S,mean_X