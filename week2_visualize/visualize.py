# coding=utf-8

import csv
import numpy as np
from locale import *
import matplotlib.pyplot as plt
import pandas as pd

def readinData(fileName):
	with open(fileName,"r",encoding ="utf-8") as f:
		reader = csv.reader(f,delimiter='\t')
		data = []
		for row in reader:
			temp_list = [row[0]]
			for colum in row[1:]:
				temp_list.append(float(colum.replace(',','.')))
			data.append(temp_list)
	return data
	
def visualize_subplot(data1,data2):
	real_data1 = []
	for row in data1:
		real_data1.append(row[1:])
	real_data1 = np.array(real_data1)
	plt.figure(figsize = (14,8))
	plt.subplot(211)
	plt.plot(np.arange(0,len(data1)),real_data1[:,0],'o')
	plt.plot(np.arange(0,len(data1)),real_data1[:,1],'o')
	plt.plot(np.arange(0,len(data1)),real_data1[:,2],'o')
	
	real_data2 = []
	for row in data2:
		real_data2.append(row[1:])
	real_data2 = np.array(real_data2)
	
	plt.subplot(212)
	plt.plot(np.arange(0,len(data2)),real_data2[:,0],'o')
	plt.plot(np.arange(0,len(data2)),real_data2[:,1],'o')
	plt.plot(np.arange(0,len(data2)),real_data2[:,2],'o')
	plt.show()
	
def calculateCov(data1,data2):
	df = pd.DataFrame(data1)
	
	df2 = pd.DataFrame(data2)
	df[4] = df2[1]
	df[5] = df2[2]
	df[6] = df2[3]
	del df[0]
	corr = df.corr()
	fig, ax = plt.subplots()
	ax.matshow(corr)
	plt.xticks(range(len(corr.columns)), corr.columns)
	plt.yticks(range(len(corr.columns)), corr.columns)
	plt.show()
		
if __name__ == '__main__':
	data1 = readinData('DatAccel.txt')
	data2 = readinData('DatGyr.txt')
	
	visualize_subplot(data1,data2)
	