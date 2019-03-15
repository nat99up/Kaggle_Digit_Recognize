#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:49:45 2019

@author: xiaohaoren
"""

import csv
import numpy as np
import tensorflow as tf




def CsvLoading(file_name):
    
    file_contents = []
    labels = []
    datas = []
    
    
    with open(file_name) as file :
        
        lines = csv.reader(file)
        
        for line in lines:
            
            file_contents.append(line)
    
    file_contents.remove(file_contents[0])
    
    for line in file_contents:
        
        labels.append(int(line[0]))
        datas.append(list(map(binarizeChar,line[1:])))
    
    labels = np.array(labels)
    datas = np.array(datas)
    
    return (labels,datas)
    

def binarizeChar(c):
    if c == '0':
        return 0
    else :
        return 1



def Network(x):
    
    input_layer = tf.reshape(x,[-1,28,28,1])
    
    return input_layer
    
    



    
if __name__ == '__main__':
    
    (l,d) = CsvLoading('train.csv')
    
    x = tf.placeholder(tf.int32,[None,784])
    y_ = tf.placeholder(tf.int32,[None,10])
    
    
    
    
    
    
    
    