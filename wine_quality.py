from tracemalloc import start
import numpy as np
import torch 
import torchvision
import imageio 
import csv

wine_path="../PytorchLearning/dlwpt-code-master/data/p1ch4/tabular-wine/winequality-white.csv"
wine_numpy=np.loadtxt(wine_path,dtype=np.float32,delimiter=";",skiprows=1)
col_list=next(csv.reader(open(wine_path),delimiter=";"))

wine_tensor=torch.from_numpy(wine_numpy)
print(wine_numpy.shape,wine_numpy)
print(wine_tensor ,end="\n")

data=wine_tensor[:,:-1]
print( data.shape,data)

wine_score=wine_tensor[:,-1]
print(wine_score)

#print(col_list)

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
