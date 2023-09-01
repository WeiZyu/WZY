import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from 邻接矩阵 import ad

def lyi():
    a = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\全部的细胞因子\string_interactions.xlsx')
    aa = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\总的(不含细胞因子).xlsx', header=None)
    c = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\全部的细胞因子\string_functional_annotations.xlsx')

    b = pd.DataFrame(pd.concat([a[0], a[1]]).unique())
    bb = pd.DataFrame(pd.concat([aa[0], aa[1]]).unique())
    cc = pd.DataFrame(c[0].unique())

    ##邻接矩阵
    # 将细胞因子和细胞编上序号
    dic = {}
    for i, j in zip(list(b[0]), list(b.index)):
        dic[i] = j
    le = len(dic)
    for ccc in cc[0]:
        if ccc not in dic.keys():
            dic[ccc] = le
            le = le + 1

    a1 = pd.DataFrame()
    a1[0] = dic.keys()
    a1[1] = dic.keys()
    a1[2] = [1 for f in range(len(dic))]
    lp = [item for item in list(dic.keys()) if item not in list(cc[0])]

    return lp,dic

def fea():
    c = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\特征矩阵.xlsx')
    c0 = pd.DataFrame(c['GO'].unique())[0].tolist()

    # #细胞因子为列，细胞为行的特征矩阵
    # M ,dic= ad()
    # M1 = pd.DataFrame(M)
    # M1.index = list(dic.keys())
    # M1.columns = list(dic.keys())

    lp,dic = lyi()
    # M2 = M1[M1.index.isin(lp)]
    # cyt = [i for i in list(dic.keys())[:1714] if i not in lp[:1674]]
    # M3 = M2[M2.index.isin(cyt)]
    # M3.to_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\细胞特征新.xlsx')

    dd = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\细胞特征新.xlsx', index_col='细胞')

    c1 = pd.DataFrame(dd.columns)[0].tolist()
    c2 = pd.DataFrame((c0 + c1)).drop_duplicates()  # 细胞特征去重
    # c2 = pd.concat([c0, c1], axis=0).set()
    c2.index = range(len(c2))
    # 将细胞因子和细胞编上序号
    lp, dic = lyi()
    # pd.DataFrame.from_dict(dic,orient='index').to_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\标签.xlsx',header = None)

    dic1 = {}
    for i1, j1 in zip(list(c2[0]), list(c2.index)):
        dic1[i1] = j1

    d = c['labels'].str.split(',')

    # 将细胞因子特征中的细胞因子序号和细胞因子特征的序号对应起来
    f = pd.DataFrame()
    lis = []
    lis1 = []
    lis3 = []
    for k, o in enumerate(c2[0]):
        for ii, jj in enumerate(d):
            if k == ii:
                for mm in jj:
                    if mm in dic.keys():
                        lis.append(dic[mm])
                        lis1.append(o)
                        lis3.append(mm)

    lis3 = pd.DataFrame(lis3).drop_duplicates()

    # 英文对应关系变成序号对应关系
    lis2 = []  # 特征的编号
    for mm in lis1:
        if mm in dic1.keys():
            lis2.append(dic1[mm])
    lis.append(len(dic) - 1)
    f[0] = lis  # 细胞因子的编号
    f3 = pd.DataFrame(f[0]).drop_duplicates()
    lis2.append(len(dic1))
    f[1] = lis2
    f[2] = [1 for iiii in range(len(lis))]

    # 将细胞的特征给转化为矩阵
    # dd = np.array(dd)
    dd[np.isnan(dd)] = 0
    ind = dd.index.tolist()  # 细胞有哪些
    jo = [i for i in dic.keys() if i not in ind]  # 细胞因子有哪些
    list1 = []
    list2 = []
    list3 = []
    for iii in dd:
        for jjj, ooo in enumerate(dd[iii]):
            if ooo == 1:
                list1.append(ind[jjj])
                list2.append(iii)
                list3.append(1)
            else:
                list1.append(ind[jjj])
                list2.append(iii)
                list3.append(0)
    f1 = pd.DataFrame()
    list1_ = []
    list2_ = []
    for fi in list1:
        if fi in dic.keys():
            list1_.append(dic[fi])
    for fj in list2:
        if fj in dic1.keys():
            list2_.append(dic1[fj])
    f1[0] = list1_
    f4 = pd.DataFrame(f1[0]).drop_duplicates()
    f1[1] = list2_
    f1[2] = list3

    fz = pd.concat([f, f1], axis=0)
    f5 = pd.DataFrame(fz[0]).drop_duplicates()
    M11 = coo_matrix((fz.iloc[:, 2], (fz.iloc[:, 0], fz.iloc[:, 1]))).toarray()

    return M11

#fea()
