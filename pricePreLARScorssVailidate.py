import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from pandas import DataFrame

import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
import sys
'''价格预测类读取历史数据展示输出进行预测'''

class pricePiredict:


    def __init__(self,filename):
        file=open(filename,encoding='GBK',errors='ignore')

        lines=file.readlines()
        self.data = [[] for i in range(len(lines))]
        file.close()
        i=0
        for line in lines:

            cells=line.strip('\n').split('\t')  #去掉换行符，再以制表符切割
            for cellIndex in range(len(cells)):
                if i !=0  and cellIndex not in [0,9]:
                    self.data[i].append(float((cells[cellIndex])))   #尽量将字符转换成数字
                else :
                    self.data[i].append(cells[cellIndex])  #最后出价时间
            i += 1

    def setColumn(self,key,columnIndex):
        columnData=[]
        columnData = [self.data[i][columnIndex] for i in range(1, len(self.data))
           if self.data[i][0][0:4] == key]  # 默认读进来是string类型，不会按数字大小
        return columnData

    '''数据归集函数,按照年份返回dict，key对应列数据
    Param Def:
    col int 数据列
    year string 提取数据年份'''
    def dataManage(self):

        y={}

        y.setdefault('2015',{})
        y.setdefault('2016',{})
        y.setdefault('2017',{})
        y.setdefault('2018',{})
        y.setdefault('2019', {})

        for key in y.keys():

            y[key].setdefault('Month', [])
            y[key].setdefault('Goals',[])
            y[key].setdefault('Candidates', [])
            y[key].setdefault('First Mean', [])
            y[key].setdefault('Second Mean', [])
            y[key].setdefault('Final Lowest', [])
            y[key].setdefault('Final Mean', [])
            y[key].setdefault('Final Done', [])

            #注意，用数组方式截取 是截取0到4之前的，即只截取0,1,2,3位置上的'''
            y[key]['Month'] = [int(self.data[i][0][5:7]) for i in range(1, len(self.data))
             if self.data[i][0][0:4] == key]
            y[key]['Goals'] = self.setColumn(key,1)
            y[key]['Candidates'] = self.setColumn(key,2)
            y[key]['First Mean'] = self.setColumn(key,3)
            y[key]['Second Mean'] = self.setColumn(key,4)
            y[key]['Final Lowest'] = self.setColumn(key,5)
            y[key]['Final Mean'] = self.setColumn(key,6)
            y[key]['Final Done'] = self.setColumn(key,11)


        return y

    '''
    归集为一个矩阵
    '''
    def dataListManage(self):
        headersInex=[0,1,2,3,4,5,6]
        dataList=[]

        for i in range(1,len(self.data)):
            xrow=[]
            for j in headersInex:
                if j==0:
                    try:
                        xrow.append(int(self.data[i][j][5:7]))   #第一列只要月月份
                    except:
                        print('except: i %s j %s'% (i,j) )
                else:
                    xrow.append(self.data[i][j])
            dataList.append(xrow)
        return  dataList








#test module
pricePridictTest=pricePiredict('./pricePridict.TXT')
#x=[i for i in range(1,len(pricePridictTest.data),1)]
dataall=[]
dataall=pricePridictTest.dataManage()
np.random.seed(101)
years=['2015','2016','2017','2018','2019']
jet = cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=len(years))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# 0,1,2,3,4，
#lable1 5 lowest  6 final mean
names = ['Month', 'Goals', 'Candidates', 'First Mean', 'Second Mean','Years']
dataList=pricePridictTest.dataListManage()
print(dataList)
lablesLowest = []  # ,'Final Lowest','Final Mean'
lablesMean = []

data=DataFrame(dataList)
print(data.describe())
dataT=data.T
print(data.shape)
corrMat=DataFrame(data.corr())
print(names)
plt.figure()
# for i in range(len(dataList)):
#     colorVal = scalarMap.to_rgba(i)
#     plt.plot(dataList[i], c=colorVal)
#     plt.grid(True)
#     plt.legend(loc='upper right')


i=0
for key  in dataall.keys():
    colorVal = scalarMap.to_rgba(i)
    i += 1
    plt.plot(dataall[key]['Month'],dataall[key]['Final Lowest'], c=colorVal,markerfacecolor=colorVal,marker='o')

    if key != '2019':
        for a, b in zip(dataall[key]['Month'], dataall[key]['Final Lowest']):
            #plt.text(a, b+10, b, ha='center', va='bottom', fontsize=10)
            plt.annotate('%s' %  b, xy=(a, b), xytext=(-5, 5),textcoords='offset points',fontsize=10)
    else :
        for a, b in zip(dataall[key]['Month'], dataall[key]['Final Lowest']):
            #plt.text(a, b-100, b, ha='center', va='bottom', fontsize=10)
            plt.annotate('%s' % b, xy=(a, b), xytext=(-5, -10), textcoords='offset points',fontsize=10)
plt.title("Price / Month")
plt.legend(dataall.keys(), loc='upper left')
plt.xticks(range(1,13))
plt.yticks(range(10000,110000,10000))
plt.grid(True,linestyle='-.')
plt.xlabel('月份')
plt.ylabel('最低价')

#
plt.figure()
i=0
for key  in dataall.keys():
    colorVal = scalarMap.to_rgba(i)
    i += 1
    plt.plot(dataall[key]['Month'],dataall[key]['Final Mean'], c=colorVal,markerfacecolor=colorVal,marker='o')

    if key != '2019':
        for a, b in zip(dataall[key]['Month'], dataall[key]['Final Mean']):
            #plt.text(a, b+10, b, ha='center', va='bottom', fontsize=10)
            plt.annotate('%s' %  b, xy=(a, b), xytext=(-5, 5),textcoords='offset points',fontsize=10)
    else :
        for a, b in zip(dataall[key]['Month'], dataall[key]['Final Mean']):
            #plt.text(a, b-100, b, ha='center', va='bottom', fontsize=10)
            plt.annotate('%s' % b, xy=(a, b), xytext=(-5, -10), textcoords='offset points',fontsize=10)
plt.title("Price / Month")
plt.legend(dataall.keys(), loc='upper left')
plt.xticks(range(1,13))
plt.yticks(range(10000,110000,10000))
plt.grid(True,linestyle='-.')
plt.xlabel('月份')
plt.ylabel('均价')

#报价人数
plt.figure()
i=0
for key  in dataall.keys():
    colorVal = scalarMap.to_rgba(i)
    i += 1
    plt.plot(dataall[key]['Month'],dataall[key]['Candidates'], c=colorVal,markerfacecolor=colorVal,marker='o')

    if key != '2019':
        for a, b in zip(dataall[key]['Month'], dataall[key]['Candidates']):
            #plt.text(a, b+10, b, ha='center', va='bottom', fontsize=10)
            plt.annotate('%s' %  b, xy=(a, b), xytext=(-5, 5),textcoords='offset points',fontsize=10)
    else :
        for a, b in zip(dataall[key]['Month'], dataall[key]['Candidates']):
            #plt.text(a, b-100, b, ha='center', va='bottom', fontsize=10)
            plt.annotate('%s' % b, xy=(a, b), xytext=(-5, -10), textcoords='offset points',fontsize=10)
plt.title("Price / Month")
plt.legend(dataall.keys(), loc='upper left')
plt.xticks(range(1,13))
plt.yticks(range(10000,40000,2500))
plt.grid(True,linestyle='-.')
plt.xlabel('月份')
plt.ylabel('报价人数')
plt.show()
# plt.figure()
# plt.pcolor(corrMat)   #画相关性热力图
# plt.colorbar()  #展示色条
# plt.figure()
# plt.boxplot(dataT)
#
# plt.show()
'''
 使用LARS惩罚线性回归，解决此问题
1.将beta都初始化为0
2.决定哪个属性与残差有最大关联
3.如果关联为正，小幅度增加关联系数，关联为负，小幅减小关联系数，（即加入属性后是可逆的）
'''
# nrows=len(dataList)
# ncols=len(dataList[0])
# xMeans=list(data.mean())
# xSD=list(data.std())
#
# #use calculate mean and standard deviation to normalize xList  数据归一化处理
# xNormalized = []
# for i in range(nrows):
#     rowNormalized = [(dataList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
#     xNormalized.append(rowNormalized)
# print(xNormalized)
#
# labelNormalized=[ xNormalized[i][5] for i in range(nrows)]
# labelNormalized2=[ xNormalized[i][6] for i in range(nrows)]
# arr=np.array(xNormalized)
# xNormalized=arr[0:,:5]
# print('Lowest normalized \n')
# print(labelNormalized)
# print('final mean normalized\n')
# print(xNormalized)
# nrows=len(xNormalized)
# ncols=len(xNormalized[0])
# #initialize a vector of coefficients beta
# beta = [0.0] * ncols
# #initialize matrix of betas at each step
# betaMat = []
# betaMat.append(list(beta))
#
#
# #number of steps to take
# nSteps = 1000
# stepSize = 0.004
# nzList = []
#
# #初始化残差矩阵
# errors=[ ]
# for n in range(nSteps):
#     initial=[]
#     errors.append(initial)
#
# #初始化训练数据集测试数据集，
# nxval=4 #分为五组数据，每次留一组做测试
# for ixval in range(nxval):
#     idxTest=[a for a in range(nrows) if a % nxval == 0]   #能整除的作为测试集，其他做为训练集
#     idxTrain = [a for a in range(nrows) if a % nxval != 0]
#
# xTest=[xNormalized[r] for r in idxTest]
# xTrain=[xNormalized[r] for r in idxTrain]
#
# labelTest=[labelNormalized[r] for r in idxTest]
# labelTrain=[labelNormalized[r] for r in idxTrain]
#
# nrowsTest=len(xTest)
# nrowsTrain=len(xTrain)
#
# for istep in range(nSteps):
#     #calculate residuals 用训练集
#     residuals = [0.0] * nrows
#     for j in range(nrowsTrain):
#         labelsHat = sum([xTrain[j][k] * beta[k] for k in range(ncols)])
#         residuals[j] = labelTrain[j] - labelsHat
#
#     #calculate correlation between attribute columns from normalized wine and residual
#     corr = [0.0] * ncols
#
#     for j in range(ncols):
#         corr[j] = sum([xNormalized[k][j] * residuals[k] for k in range(nrowsTrain)]) / nrowsTrain
#
#     iStar = 0
#     corrStar = corr[0]
#
#     for j in range(1, (ncols)):
#         if abs(corrStar) < abs(corr[j]):
#             iStar = j; corrStar = corr[j]
#
#     beta[iStar] += stepSize * corrStar / abs(corrStar)
#     betaMat.append(list(beta))
#
#     #使用本轮迭代计算的beta，在测试集上，计算残差
#     for j in range(nrowsTest):
#         labelsHat = sum([xTest[j][k] * beta[k] for k in range(ncols)])
#         err = labelTest[j] - labelsHat
#         errors[istep].append(err)
#
#     #记录特征加入顺序
#     nzBeta = [index for index in range(ncols) if beta[index] != 0.0]
#     for q in nzBeta:
#         if (q in nzList) == False:
#             nzList.append(q)
#
# cvCurve=[]
# #算出每一次迭代上的测试后平均残差
# for errVect in errors:
#     mse=sum(x*x for x in errVect )/len(errVect)
#     cvCurve.append(mse)
#
# minMse=min(cvCurve)
# minMseIndex=[index for index in range(len(cvCurve)) if cvCurve[index]==minMse]
#
#
# print('add in List:')
# print(nzList)
# nameList = [names[nzList[i]] for i in range(len(nzList))]
#
# print(nameList)
# plt.figure()
# for i in range(ncols):
#     #plot range of beta values for each attribute
#     coefCurve = [betaMat[k][i] for k in range(nSteps)]
#     xaxis = range(nSteps)
#     plt.plot(xaxis, coefCurve,label=names[i])
#
# plt.legend(loc='upper right')
# plt.xlabel("Steps Taken")
# plt.ylabel(("Coefficient Values"))
#
# print("Minimum Mean Square error:",minMse)
# print("Minimum Mean Square error's stepindex:",minMseIndex)
# print("Minimum Mean Square erroe 's Beta :" ,betaMat[minMseIndex[0]])
# xaxis=range(len(cvCurve))
# plt.figure()
# plt.plot(xaxis,cvCurve)
# plt.xlabel("Steps Taken")
# plt.ylabel("Mean Square Error!")
# plt.show()

# #经过LARS回归，初步获得特征重要程度为，
# add in List:
# [4, 3, 2, 0, 1]
# ['Second Mean', 'First Mean', 'Candidates', 'Month', 'Goals']

#采用交叉验证从350个迭代中选取最佳解 分为5组数据
#
#        [0.0, -0.20400000000000015, 0.0, 0.0, 0.008]
#names = ['Month', 'Goals', 'Candidates', 'First Mean', 'Second Mean']
# mean    6.289474  3291.684211   9816.342105  28897.605263  30944.500000
# std     3.525259  1254.700483   5094.551251  12193.252584  13442.545206
# mean   35892.105263  42363.526316
# std    17892.371643  17924.919552
#278.464-601.392=-322.298

#[0.0, -0.35200000000000026, 0.0, 0.0, 0.0]

#[-0.6377579664291363, -0.27020962299697826, 2.5539887710667655, 0.188306668935181, 0.2834898360549172, 1.701291162703833, 1.575190279888436]]