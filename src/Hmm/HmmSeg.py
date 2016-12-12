#encoding:utf-8
#author:lvshanchun

import sys
import math
import os
import numpy as np
class TrainHmm:
    def __init__(self,TrainingFile,TestFile):
        self.train_set = TrainingFile
        self.test_set = TestFile
        self.training()
        self.testing()

    def findIndex(self,i,lens):
        if lens == 1:#单字词
            return 'S'
        if i == 0 :#词开头
            return 'B'
        if i == lens-1 :#词结尾
            return 'E'
        return 'M'#其他 词中间

    def training(self):
        print ('training ...')
        a = '  '
        with open(self.train_set,'r') as fpo:
            self.count = dict()  # key是状态序列，值是字典类型，值中每个字典存放的是每个字出现的次数
            self.cnt = dict()  # 统计B M E S出现的次数
            self.words = list()  # 字典
            strs = ""  # 存放的是字标注序列 每一行用逗号隔开

            for line in fpo:
                # line = line.decode('utf8')
                line = line.replace(' \n', '')
                # line = line.replace('\n','')
                grap = line.split(a)
                for scen in grap:
                    scen = scen.strip()
                    lens = len(scen)
                    for i in range(0, lens):
                        wd = scen[i]
                        if wd not in self.words:  # words字典
                            self.words.append(wd)  # 如果字典中没有这个字，就加入字典中
                        st = self.findIndex(i, lens)  # 查询字在词中的位置
                        # 如果键在字典中，返回这个键所对应的值。如果键不在字典中，向字典中插入这个键，
                        # 并且以default为这个键的值，并返回 default。default的默认值为None
                        self.cnt.setdefault(st, 0)
                        self.cnt[st] += 1
                        strs += st
                        self.count.setdefault(st, {})
                        self.count[st].setdefault(wd, 0)
                        self.count[st][wd] += 1
                strs += ','
        ast = dict()
        #状态转移 ast key为状态序列 值为字典 值中存放的是状态序列和次数 ast[i][j]表示 状态i向状态j转移的次数
        #B->M,E M->M，E E->B,S S->B,S
        for i in range(len(strs)-2):#因为有i+1故要减2
            st1 = strs[i]
            st2 = strs[i+1]
            if st1 == ',' or st2 == ',' :# 一行结束
                continue
            ast.setdefault(st1,{})
            ast[st1].setdefault(st2,0)
            ast[st1][st2] += 1
        #初始状态,因为第一个词不可能是 M E所以初始概率为0
        self.pi = {'B':0.5,'M':0,'E':0,'S':0.5}
        self.matrixA = dict()
        self.matrixB = dict()
        state = ['B','M','E','S']
        #初始化状态转移矩阵 a
        for st1 in state:
            self.matrixA.setdefault(st1,{})
            for st2 in state:
                self.matrixA[st1].setdefault(st2,0)

        for st1,item in ast.items():
            for st2 in item.keys():
                self.matrixA[st1][st2] = float(item[st2])/float(self.cnt[st1])
        #初始化发射矩阵，做了平滑处理 这里假定所有的词都出现了一次
        for st in state:
            self.matrixB.setdefault(st,{})
            for wd in self.words:
                self.matrixB[st].setdefault(wd,1.0/float(self.cnt[st]))
        #item[wd]=self.count[st][wd]表示 st到wd的发射次数 self.cnt[st]表示st出现的次数
        #发射概率b[st][wd]=self.count[st][wd]/self.cnt[st]
        for st,item in self.count.items():
            for wd in item.keys():
                self.matrixB[st][wd] = float(item[wd])/float(self.cnt[st])

        print ('training completed')

    def testing(self):
        print ('testing ...')
        filename,_ = self.test_set.split('.')
        filename += '_result.utf8'

        fpo = open(self.test_set,'r')
        fpw = open(filename,'w')

        fi = dict()
        state = ['B','E','M','S']

        num = 0
        for eachline in fpo: #一句一句地进行
            num += 1
            # line = eachline.decode('utf8').strip()
            line = eachline.strip()
            lens = len(line)
            if lens < 1 :
                continue
            wd = line[0]
            for st in state:#每一句t=0时刻
                #fi[t][st]表示的是第t时刻，到达st的最大概率
                fi.setdefault(1,{})
                if wd not in self.matrixB[st].keys():
                    self.matrixB[st].setdefault(wd,1.0/float(self.cnt[st]))#平滑处理，假定至少有一次
                fi[1].setdefault(st,self.pi[st]*self.matrixB[st][wd])#初始化每一句的第一步
            for i in range(1,lens):#每一句t>0时刻
                wd = line[i]
                fi.setdefault(i+1,{})
                for st1 in state:#当前状态
                    fi[i+1].setdefault(st1,0)
                    max_num = 0
                    for st2 in state:#前一状态
                        max_num = max(max_num,fi[i][st2]*self.matrixA[st2][st1])#博客上有说明，求最大时其实和发射状态无关
                    if wd not in self.matrixB[st1].keys():#平滑处理，假定至少有一次
                        self.matrixB[st1][wd] = 1.0/float(self.cnt[st1])
                    fi[i+1][st1] = max_num*self.matrixB[st1][wd]
            #处理完一句
            links = list()
            tmp = list()
            #求出最后一个时刻的最大概率对应的状态，用于回溯
            for st in state:
                tmp.append([st,fi[lens][st]])
            st1,_ = max(tmp,key=lambda x:x[1])
            links.append(st1)
            #这里的实现有点不好，本来可以在求最大概率是记录下路径的，没必要重新算一次路径
            for i in range(lens,1,-1):#从最后一个时刻开始回溯
                tmp = list()
                for st in state:
                    tmp.append([st,fi[i-1][st]*self.matrixA[st][st1]])
                st1,sc = max(tmp,key=lambda x:x[1])
                links.append(st1)

            links.reverse()
            strs = ""
            #处理一句输出一句
            for i in range(len(links)):
                st = links[i]
                if st == 'S':
                    strs += (line[i]+'  ')
                    continue
                if st == 'B' or st == 'M':
                    strs += line[i]
                    continue
                if st == 'E':
                    strs += (line[i]+'  ')
            strs += '\n'#处理完一句换行输出
            fpw.writelines(strs)

        fpo.close()
        fpw.close()
        print ('test completed')

if __name__ == "__main__":
    # args = len(sys.argv)
    # if(args < 3):
    #     print ("Usage [trainingSet] [testSet] for utf-8")
    #     sys.exit(1)
    trainPath='/home/longzc/PycharmProjects/WordSeg/icwb2-data/training/pku_training.utf8'
    testPath='/home/longzc/PycharmProjects/WordSeg/icwb2-data/testing/pku_test.utf8'
    # hmm_train = TrainHmm(sys.argv[1],sys.argv[2])
    hmm_train = TrainHmm(trainPath,testPath)
