#!/usr/bin/env python
# -*- coding:UTF-8 -*-

'''
Created on 20**-**-**

@author: fangmeng
'''

'''from numpy import *

#==================================
# 输入:
#        fileName: 数据文件名(含路径)
# 输出:
#        dataMat: 数据集
#==================================
def loadDataSet(fileName):
    '载入数据文件'

    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#==================================================
# 输入:
#        vecA: 样本a
#        vecB: 样本b
# 输出:
#        sqrt(sum(power(vecA - vecB, 2))): 样本距离
#==================================================
def distEclud(vecA, vecB):
    '计算样本距离'

    return sqrt(sum(power(vecA - vecB, 2)))

#===========================================
# 输入:
#        dataSet: 数据集
#        k: 簇个数
# 输出:
#        centroids: 簇划分集合(每个元素为簇质心)
#===========================================
def randCent(dataSet, k):
    '随机初始化质心'

    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

#===========================================
# 输入:
#        dataSet: 数据集
#        k: 簇个数
#        distMeas: 距离生成器
#        createCent: 质心生成器
# 输出:
#        centroids: 簇划分集合(每个元素为簇质心)
#        clusterAssment: 聚类结果
#===========================================
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    'K-Means基本实现'

    m = shape(dataSet)[0]
    # 簇分配结果矩阵。一列为簇分类结果，一列为误差。
    clusterAssment = mat(zeros((m,2)))
    # 创建原始质心集
    centroids = createCent(dataSet, k)
    # 簇更改标记
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False

        # 每个样本点加入其最近的簇。
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2

        # 更新簇
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)

    return centroids, clusterAssment

def main():
    'k-Means聚类操作展示'

    datMat = mat(loadDataSet('/home/fangmeng/testSet.txt'))
    myCentroids, clustAssing = kMeans(datMat, 3)

    #print myCentroids
    print(clustAssing)

if __name__ == "__main__":
   main()'''

import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)
np.set_printoptions(threshold = sys.maxsize)
d = pd.read_excel('globalterrorismdb_0221dist.xlsx', index_col = 0, na_values=[''])
#d.info(verbose = True)
#0-20
con = [13, 21, 26, 39, 41, 44, 47, 50, 52, 56, 60, 62, 66, 84, 87, 92, 94, 98, 105, 105, 108, 111]

gtd = d[['attacktype1', 'targtype1','targsubtype1','nkill','nwound']]
#data2 = gtd[gtd.QuadClass.isin([3, 4])].reset_index()

gtd1 = gtd.groupby(['targtype1'])
attack_list = np.zeros((9, 111), dtype=int)
nkill_list = np.zeros((9, 111), dtype=int)
nwound_list = np.zeros((9, 111), dtype=int)

num = 0
for index, data in gtd1:

        attack = data.groupby(['attacktype1','targsubtype1'], as_index = False).size().rename('Count').reset_index()
        nkill = data.groupby(['attacktype1','targsubtype1'], as_index = False)['nkill'].sum()
        nwound = data.groupby(['attacktype1','targsubtype1'], as_index = False)['nwound'].sum()
        for j in range(len(nkill)):
                if attack['targsubtype1'][j] > con[num]:
                        continue
                else:
                        attack_list[int(attack['attacktype1'][j])-1][int(attack['targsubtype1'][j])-1] = attack['Count'][j]
                        nkill_list[int(nkill['attacktype1'][j])-1][int(nkill['targsubtype1'][j])-1] = nkill['nkill'][j]
                        nwound_list[int(nwound['attacktype1'][j]) - 1][int(nwound['targsubtype1'][j]) - 1] = nwound['nwound'][j]
        num += 1
#print("37hang:".format(attack_list, nkill_list, nwound_list))
        print(attack_list[:, 37:38])
        print(nkill_list[:, 37:38])

sa = np.zeros((9, 113), dtype = float)
sk = np.zeros((9, 113), dtype = float)
sw = np.zeros((9, 113), dtype = float)
for item in range(9):
        attack_total = np.sum(attack_list, axis=0)
        nkill_total = np.sum(nkill_list, axis=0)
        nwound_total = np.sum(nwound_list, axis=0)
        print(attack_total[37])
        print(nkill_total[37])
        for j in range(111):
                if attack_total[j] == 0:
                        sa[item][j] = attack_list[item][j]
                        sk[item][j] = nkill_list[item][j]
                        sw[item][j] = nwound_list[item][j]
                else:
                        sa[item][j] = attack_list[item][j] / attack_total[j]
                        sk[item][j] = nkill_list[item][j] / nkill_total[j]
                        sw[item][j] = nwound_list[item][j] / nwound_total[j]
'''print(sa)
print(sk)
print(sw)'''

target =['油气公司','餐厅/酒吧/咖啡厅','银行/贸易','跨国公司','工业/纺织/工厂','医疗/制药','零售/食品杂货/面包店','酒店/度假村','农场/牧场','采矿','娱乐/文化/体育场/赌场','建筑','私人安全公司/公司',
         '法官/律师/法院','政治家或政党运动/会议/集会','皇室','国家元首','政府人员（不包括警察、军队','与选举相关','情报机构','政府大楼/设施/办公室',
         '警察大楼（总部/车站/学校）','警察巡逻（车辆和车队）','警察检查点','警察安全部队/军官','监狱',
         '军事/基地/总部/检查站','军事征兵展/学院','军事单位/巡逻/护航','海军','空军','海岸警卫队','国民警卫队','军事人员（士兵、部队、军官）','军事运输车辆','军事检查站','与北大西洋公约组织有关','海军陆战队','准军事部队',
         '诊所','人员',
         '飞机','航空公司官员/人员','机场',
         '外交人员','大使馆/领事馆','国际组织（维和、援助机构、复合）',
         '教师/教授/讲师','学校/大学/教育大楼','其他人员',
         '食品供应','供水',
         '新闻记者/工作人员/设施','广播记者/工作人员/设施','电视记者/工作人员/设施','其他（包括网络新闻机构）',
         '民事海事','商业海运','游轮','港口',
         '国内非政府组织','国际非政府组织',
         '救护车','消防战士/卡车','难民营','非军事区',
         '无名平民','有名字的平民','确定的宗教','学生','确定的种族/族裔','农民','车辆/运输','市场/广场','乡村/城市/小镇/郊区','房子/公寓/住宅','劳动者/确定的职业','游行/集会','公共区域','纪念馆/公墓/纪念碑','博物馆/文化中心/文化馆','与工会相关的','抗议者','政党成员/集会',
         '宗教人物','礼拜场所','附属机构',
         '收音机','电视','电话/电报','互联网基础设施','多个电信目标',
         '恐怖组织','非州立民兵组织',
         '旅行社','旅行车/货车/车辆','游客','其他设施',
         '巴士','火车/火车轨道/小车','巴士站','地铁','桥/汽车隧道','公路/道路/交通信号','出租车/人力车',
         '天燃气','电力','石油',
         '政党的官员/候选人/其他人员','政党的办公室/设施','集会']
scene = ['商业','政府','警察','军事','流产有关','机场和飞机','政府（外交）','教育机构','事务或水供应','新闻记者','海事','非政府组织','其他','公民自身和私有财产',
         '宗教人物/机构','电信','恐怖分子/非州立民兵组织','游客','运输','公用事业','暴力政党']
attack_type = ['暗杀','武装袭击','轰炸/爆炸','劫持','劫持人质（路障事件）','绑架','设施/基础设施攻击','徒手攻击','未知']
km = KMeans(n_clusters=3)
label_list = [[] for i in range(111)]
label_type = ['低','中','高']
for i in range(111):
        kdata = np.hstack((sa[0:9, i:i+1], sk[0:9, i:i+1]))
        kdata = np.hstack((kdata, sw[0:9, i:i+1]))
        print(i)
        #print(kdata)
        #print("\n")
        estum = km.fit(kdata)
        label = estum.fit_predict(kdata)
        label_list[i].append(label)
        print(label)
        for j in range(9):
                print("针对%s目标的%s攻击类型的风险等级为%s\n" % (target[i], attack_type[j], label_type[label[j]]))

#km = KMeans(n_clusters=3)
#estum = km.fit(data)
#label = estum.fit_predict(data)
#print(label)
#center = estum.cluster_centers_
#print(center)


