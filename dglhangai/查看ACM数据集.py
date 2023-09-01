# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:03:01 2022

@author: lenovo
"""

import os
import pickle


data_list = []
with open(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\data\ACM3025.pkl', "rb") as f:
        while True:
            try:
                data = pickle.load(f)    
                data_list.append(data)
            except EOFError:
                break