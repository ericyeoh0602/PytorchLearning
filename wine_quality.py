import numpy as np
import torch 
import torchvision
import imageio 
import csv

wine_path="../data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy=np.loadtxt(wine_path,dtype=np.float32,delimiter=";",skiprows=1)

print(wine_numpy)

#out=img.permute(2,0,1)


print("hello world")

'''
a=tt.ones(3)
a[1]=float(a[1])
print(a)

point=tt.tensor([[4.0,4.1],[4.3,4.5],[2.1,3.3]])
print(point[0][1])
print(point.shape)

print("git refresh")
print("ddd")
'''
