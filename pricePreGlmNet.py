import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from pandas import DataFrame

import sys
#定义S曲线函数
def S(z, gamma):
    if gamma >= abs(z):
        return 0.0
    return (z/abs(z))*(abs(z) - gamma)

'''价格预测类读取历史数据展示输出进行预测'''


class pricePiredict:


    def __init__(self,filename):
        file=open(filename,encoding='GBK',errors='ignore')

        lines=file.readlines()
        self.data = [[] for i in range(len(lines))]
        file.close()
        i=0
        for line in lines:

            cells=line.strip('\n').split('\t')  #去掉换行符，再以制表符副歌
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
    def dataListManage(self):
        headersInex=[0,1,2,3,4,5,6]
        dataList=[]

        for i in range(1,len(self.data)):
            xrow=[]
            for j in headersInex:
                if j==0:
                    xrow.append(int(self.data[i][j][5:7]))   #第一列只要月月份
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
years=['2015','2016','2017','2018']
jet = cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=len(years))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# 0,1,2,3,4，
#lable1 5 lowest  6 final mean
names = ['Month', 'Goals', 'Candidates', 'First Mean', 'Second Mean']
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
# plt.figure()
# for i in range(len(dataList)):
#     colorVal = scalarMap.to_rgba(i)
#     plt.plot(dataList[i], c=colorVal)
#     plt.grid(True)
#     plt.legend(loc='upper right')


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
nrows=len(dataList)
ncols=len(dataList[0])
xMeans=list(data.mean())
xSD=list(data.std())

#use calculate mean and standard deviation to normalize xList  数据归一化处理
xNormalized = []
for i in range(nrows):
    rowNormalized = [(dataList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
    xNormalized.append(rowNormalized)
print(xNormalized)

labelNormalized=[ xNormalized[i][5] for i in range(nrows)]
labelNormalized2=[ xNormalized[i][6] for i in range(nrows)]
arr=np.array(xNormalized)
xNormalized=arr[0:,:5]
print('Lowest normalized \n')
print(labelNormalized)
print('final mean normalized\n')
print(xNormalized)
nrows=len(xNormalized)
ncols=len(xNormalized[0])





#初始化训练数据集测试数据集，
nxval=4 #分为五组数据，每次留一组做测试
for ixval in range(nxval):
    idxTest=[a for a in range(nrows) if a % nxval == 0]   #能整除的作为测试集，其他做为训练集
    idxTrain = [a for a in range(nrows) if a % nxval != 0]

xTest=[xNormalized[r] for r in idxTest]
xTrain=[xNormalized[r] for r in idxTrain]

labelTest=[labelNormalized[r] for r in idxTest]
labelTrain=[labelNormalized[r] for r in idxTrain]

nrowsTest=len(xTest)
nrowsTrain=len(xTrain)

#select value for alpha parameter
alpha = 1.0

#make a pass through the data to determine value of lambda that
# just suppresses all coefficients.
#start with betas all equal to zero.
xy = [0.0]*ncols
for i in range(nrows):
    for j in range(ncols):
        xy[j] += xNormalized[i][j] * labelNormalized[i]

maxXY = 0.0
for i in range(ncols):
    val = abs(xy[i])/nrows
    if val > maxXY:
        maxXY = val

#calculate starting value for lambda
lam = maxXY/alpha

#this value of lambda corresponds to beta = list of 0's
#initialize a vector of coefficients beta
beta = [0.0] * ncols
#initialize matrix of betas at each step
betaMat = []
betaMat.append(list(beta))

#begin iteration
nSteps = 150
lamMult = 0.93 #100 steps gives reduction by factor of 1000 in
               # lambda (recommended by authors)

nzList = []  #非0 属性值

#初始化残差矩阵
errors=[ ]
for n in range(nSteps):
    initial=[]
    errors.append(initial)


for iStep in range(nSteps):
    #make lambda smaller so that some coefficient becomes non-zero
    lam = lam * lamMult

    deltaBeta = 100.0
    eps = 0.01
    iterStep = 0
    betaInner = list(beta)
    while deltaBeta > eps:
        iterStep += 1
        if iterStep > 100:         break   #循环的两个退出条件，1，beta变化值小于eps  2.达到迭代步数

        #cycle through attributes and update one-at-a-time
        #record starting value for comparison
        betaStart = list(betaInner)
        for iCol in range(ncols):

            xyj = 0.0
            for i in range(nrowsTrain):
                #calculate residual with current value of beta
                labelHat = sum([xTrain[i][k]*betaInner[k]
                                for k in range(ncols)])
                residual = labelNormalized[i] - labelHat

                xyj += xTrain[i][iCol] * residual

            uncBeta = xyj/nrows + betaInner[iCol]
            betaInner[iCol] = S(uncBeta, lam * alpha) / (1 +
                                            lam * (1 - alpha))

        sumDiff = sum([abs(betaInner[n] - betaStart[n])
                       for n in range(ncols)])
        sumBeta = sum([abs(betaInner[n]) for n in range(ncols)])
        deltaBeta = sumDiff/sumBeta
    print(iStep, iterStep)
    beta = betaInner

    #add newly determined beta to list
    betaMat.append(beta)

    #计算残差
    for j in range(nrowsTest):
        labelsHat = sum([xTest[j][k] * beta[k] for k in range(ncols)])
        err = labelTest[j] - labelsHat
        errors[iStep].append(err)

    #keep track of the order in which the betas become non-zero
    nzBeta = [index for index in range(ncols) if beta[index] != 0.0]
    for q in nzBeta:
        if (q in nzList) == False:
            nzList.append(q)


cvCurve=[]
#算出每一次迭代上的测试后平均残差
for errVect in errors:
    mse=sum(x*x for x in errVect )/len(errVect)
    cvCurve.append(mse)

#print out the ordered list of betas
nameList = [names[nzList[i]] for i in range(len(nzList))]
print(nameList)
plt.figure()
nPts = len(betaMat)
for i in range(ncols):
    #plot range of beta values for each attribute
    print(names[i])
    coefCurve = [betaMat[k][i] for k in range(nPts)]
    xaxis = range(nPts)
    plt.plot(xaxis, coefCurve,label=names[i])

plt.xlabel("Steps Taken")
plt.ylabel(("Coefficient Values"))
plt.legend(loc="upper right")

minMse=min(cvCurve)
minMseIndex=[index for index in range(len(cvCurve)) if cvCurve[index] == minMse]

print("Minimum Mean Square error:",minMse)
print("Minimum Mean Square error's stepindex:",minMseIndex)
print("Minimum Mean Square erroe 's Beta :" ,betaMat[minMseIndex[0]])
xaxis=range(len(cvCurve))
plt.figure()
plt.plot(xaxis,cvCurve)
plt.xlabel("Steps Taken")
plt.ylabel("Mean Square Error!")

for i in range(nrows):
    #用最好的beta解，预测最后一次
    print("Actual value：", dataList[i][5])
    bataPick=betaMat[minMseIndex[0]]
    predictVaule= sum(bataPick[k]*xNormalized[i][k] for k in range(len(bataPick))) * xSD[5] + xMeans[5]
    print("Predict value：", predictVaule)
#特征加入顺序
#['Second Mean', 'Candidates', 'First Mean', 'Month', 'Goals']
plt.show()