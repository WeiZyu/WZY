import pandas as pd
#import numpy as np
from scipy.sparse import coo_matrix

def ad():
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

    # 每个因子连接了多少因子，用字典来表示
    dic2 = {}
    s = 0
    for t in a[0]:
        if t != s:
            ww = a[a[0] == t]
            ww.index = range(len(ww))
            s = ww[0][0]
            dic2[ww[0][0]] = list(ww[1])
        else:
            continue


    # 元路径细胞因子-细胞因子-KEGG通路-细胞因子-细胞因子所构建的邻接矩阵
    d = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\全部的细胞因子\KEGG\enrichment.KEGG.xlsx')
    # 实现遍历几几相连几几相连的方法
    def Combinations(L, k):
        """List all combinations: choose k elements from list L"""
        na = len(L)
        result = []  # To Place Combination result
        for i in range(na - k + 1):
            if k > 1:
                newL = L[i + 1:]
                Comb, _ = Combinations(newL, k - 1)
                for item in Comb:
                    item.insert(0, L[i])
                    result.append(item)
            else:
                result.append([L[i]])
        return result, len(result)

    # 每个细胞因子经过通路之后和别的细胞因子之间的连接
    x2 = pd.DataFrame()
    for yy in d[1]:
        yy = yy.split(',')
        n1 = Combinations(yy, 2)[0]
        n2 = pd.DataFrame(n1)
        n2[2] = [1 for i in range(len(n2))]
        x2 = pd.concat([x2, n2], axis=0, ignore_index=True)
    x21 = pd.DataFrame()
    x21[0] = x2[1]
    x21[1] = x2[0]
    x21[2] = x2[2]
    xz = x2.append(x21, ignore_index=True)
    x_2 = xz.drop_duplicates(subset=[0, 1])
    x22 = x_2.append(aa, ignore_index=True)
    # 细胞经过细胞因子和细胞因子的连接
    xibao = [i for i in dic.keys() if i not in list(cc[0])]
    pp1 = pd.DataFrame()
    for uu in xibao:
        if uu in list(x22[0]):
            lj = x22[x22[0] == uu]
            for uu0, uu00 in enumerate(lj[2]):  # score=uu00
                for uu1, uu11 in enumerate(lj[1]):
                    if uu0 == uu1:
                        if uu11 in list(x22[0]):
                            pp = pd.DataFrame()
                            lj1 = x22[x22[0] == uu11]
                            pp[0] = [uu for i in range(len(lj1))]
                            pp[1] = list(lj1[1])
                            pp[2] = [i * uu00 for i in lj1[2]]
                            pp1 = pp1.append(pp, ignore_index=True)
    pp2 = pd.DataFrame()
    pp2[0] = pp1[1]
    pp2[1] = pp1[0]
    pp2[2] = pp1[2]
    pp3 = pp2.append(pp1, ignore_index=True)
    pp4 = pp3.drop_duplicates(subset=[0, 1])
    pp5 = pp4.append(x22, ignore_index=True)
    pp5 = pp5.drop_duplicates(subset=[0, 1])
    x3 = pp5.append(a1, ignore_index=True)

    # 细胞因子和细胞中文对应关系变成序号对应关系
    da1 = pd.DataFrame()
    li = []
    for m in x3[0]:
        if m in dic.keys():
            li.append(dic[m])
    li1 = []
    for m in x3[1]:
        if m in dic.keys():
            li1.append(dic[m])
    da1[0] = li
    da1[1] = li1
    da1[2] = x3[2]

    M1 = coo_matrix((da1.iloc[:, 2], (da1.iloc[:, 0], da1.iloc[:, 1])), shape=(len(dic), len(dic))).toarray()

    return M1,dic
