# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 10:31:17 2022

@author: lenovo
"""
import pandas as pd
#import numpy as np
from scipy.sparse import coo_matrix
import random

def biaoqian():
    # n = ['IL4','IL17A','CSF2','IL13','IL5','IL15','CXCL10','CXCL8','CCL2',
    #      'IL2','CCL3','CXCL9','TNF','IL1B','CCL4','IL1A','IL2RA','IL23A',
    #      'TNFRSF1A','ANGPT2']
    n = ['IL6','IL10','IFNG','IL4','IL17A','CSF2','IL13','IL5','IL15','CXCL10','CXCL8','CCL2',
         'IL2','CCL3','CXCL9','TNF','IL1B','CCL4','IL1A','IL2RA']
    #n1 = ['成纤维细胞','基质细胞','长寿浆细胞']
    a = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\全部的细胞因子\string_interactions.xlsx')
    aa = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\总的(不含细胞因子).xlsx',header = None)
    c = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\全部的细胞因子\string_functional_annotations.xlsx')

    b = pd.DataFrame(pd.concat([a[0],a[1]]).unique())
    bb = pd.DataFrame(pd.concat([aa[0],aa[1]]).unique())
    cc = pd.DataFrame(c[0].unique())

    ##邻接矩阵
    #将细胞因子和细胞编上序号
    dic = {}
    for i,j in zip(list(b[0]),list(b.index)):
        dic[i] = j
    le = len(dic)
    for ccc in cc[0]:
        if ccc not in dic.keys():
            dic[ccc] = le
            le = le+1
    e = pd.DataFrame()
    e[0] = list(dic.keys())
    ez1 = []
    ez1[1:1] = n
    ez1[1:1] = n1
    ez2 = [i for i in e[0] if i not in ez1]
    e[1] = [i for i in range(len(dic))]

    ls1 = []
    for y in e[0]:
        if y in n:
            ls1.append(1)
        else:
            ls1.append(0)

    xx = random.sample(list(ez2),30)
    ls2 = []
    for y1 in e[0]:
        if y1 in xx:
            ls2.append(1)
        # elif y1 in ez1:#细胞有负标签的时候加进去
        #     ls2.append(1)
        else:
            ls2.append(0)

    ls3 = []
    for y2 in e[0]:
        if y2 in n:
            ls3.append(0)
        elif y2 in xx:
            ls3.append(0)
        else:
            ls3.append(1)
    e[2] = ls1
    e[3] = ls2
    e[4] = ls3
    e.to_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\标签.xlsx',index = None)

biaoqian()