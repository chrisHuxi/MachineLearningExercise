# coding=utf-8

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt


def createSample(sample_amount = 50,mu = np.array([0, 0]),Sigma = np.array([[9, 0], [0, 1]])):
	R = cholesky(Sigma)
	x, y = np.random.multivariate_normal(mu, Sigma, sample_amount).T
	return x,y
	
def rotateData(xy,degree):
	new_xy = []
	for everypoint in xy.T:
		new_x = everypoint[0]*np.cos(degree) - everypoint[1]*np.sin(degree)
		new_y = everypoint[0]*np.sin(degree) + everypoint[1]*np.cos(degree)
		new_xy.append((new_x,new_y))
	return np.array(new_xy).T
	
def display(x,y,marker):
	#print(x)
	plt.plot(x, y,marker)
	plt.title('Distribution (two classes)')
	plt.ylabel('y')
	plt.xlabel('x')
	plt.show()	

if __name__ == '__main__':
	x,y = createSample()
	x_mean = np.mean(x)
	y_mean = np.mean(y)

	data_xy = np.vstack((x,y))
	cov = np.cov(data_xy)
	print(cov)
	rotate_data_xy = rotateData(data_xy,3.14/6)
	new_cov = np.cov(rotate_data_xy)
	print(new_cov)

	true_rotate_x,true_rotate_y = createSample(50,[0,0],np.array([[6,2*np.sqrt(3)],[2*np.sqrt(3),3.0]]))
	'''
	new data cov:
	6			8*sqrt(3)
	8*sqrt(3) 	3
	'''
	plt.plot(rotate_data_xy[0],rotate_data_xy[1],'o')
	plt.plot(true_rotate_x, true_rotate_y,'x')
	
	plt.title('Distribution (two classes)')
	plt.ylabel('y')
	plt.xlabel('x')
	plt.show()	
	
	
	