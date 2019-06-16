import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from pandas import DataFrame

from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
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

X=np.array(xNormalized)
Y=np.array(labelNormalized)

# #Call LassoCV from sklearn.linear_model
# priceModel = LassoCV(cv=10).fit(X, Y)
#
# plt.figure()
# plt.plot(priceModel.alphas_, priceModel.mse_path_, ':')
# plt.plot(priceModel.alphas_, priceModel.mse_path_.mean(axis=-1),
#          label='Average MSE Across Folds', linewidth=2)   #priceModel.mse_path_.mean(axis=-1)  最后一维求均值
# plt.axvline(priceModel.alpha_, linestyle='--',
#             label='CV Estimate of Best alpha')  #添加一条与axes垂直的线
#                                                     #添加一条水平线用axhline
# plt.text(priceModel.alpha_,3,"best alpha:"+str(priceModel.alpha_),fontdict={'size':10,'color':'b'})
# plt.semilogx()  #是x轴为log刻度
# plt.legend()
# ax = plt.gca()
# ax.invert_xaxis()  #反转轴
# plt.xlabel('alpha')
# plt.ylabel('Mean Square Error')
# plt.axis('tight')
# #print out the value of alpha that minimizes the Cv-error
# print("alpha Value that Minimizes CV Error  ",priceModel.alpha_)
# print("Minimum MSE  ", min(priceModel.mse_path_.mean(axis=-1)))
# plt.show()

# alpha Value that Minimizes CV Error   0.048203158061158054
# Minimum MSE   0.4654056539530398
alphas, coefs, _  = linear_model.lasso_path(X, Y,  return_models=False)

print("alpha:\n")
print(alphas)
print("coefs:\n")
print(coefs)
plt.plot(alphas,coefs.T)

plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.axis('tight')
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()


nattr, nalpha = coefs.shape

#find coefficient ordering
nzList = []
for iAlpha in range(1,nalpha):
    coefList = list(coefs[: ,iAlpha])
    nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
    for q in nzCoef:
        if not(q in nzList):
            nzList.append(q)

nameList = [names[nzList[i]] for i in range(len(nzList))]
print("Attributes Ordered by How Early They Enter the Model", nameList)

#find coefficients corresponding to best alpha value. alpha value corresponding to
#normalized X and normalized Y is 0.013561387700964642

alphaStar = 0.048203158061158054
indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
indexStar = max(indexLTalphaStar)

#here's the set of coefficients to deploy
coefStar = list(coefs[:,indexStar])
print("Best Coefficient Values ", coefStar)

#The coefficients on normalized attributes give another slightly different ordering

absCoef = [abs(a) for a in coefStar]

#sort by magnitude
coefSorted = sorted(absCoef, reverse=True)

#按照系数的绝对值大小，从大到小排序
idxCoefSize = [absCoef.index(a) for a in coefSorted if not(a == 0.0)]

namesList2 = [names[idxCoefSize[i]] for i in range(len(idxCoefSize))]

print("Attributes Ordered by Coef Size at Optimum alpha", namesList2)

for i in range(nrows):
    #用最好的beta解，预测最后一次
    print("Actual value：", dataList[i][5])
    bataPick=coefStar
    predictVaule= sum(bataPick[k]*xNormalized[i][k] for k in range(len(bataPick))) * xSD[5] + xMeans[5]
    print("Predict value：", predictVaule)


# print('dataList:')
# print(dataList)
# #加上待预测数据
# #-'Month', 'Goals', 'Candidates', 'First Mean', 'Second Mean']
# newrow=[5,2948.0,20000.0,33043.0,36573.0,0.0,0.0]
# dataListNew=dataList
# dataListNew.append(newrow)
# dataNew=DataFrame(dataListNew)
# xMeansNew=list(dataNew.mean())
# xSDNew=list(dataNew.std())
# print('dataListNew:')
# print(dataListNew)
# rowNormalizedNew = [(dataListNew[-1][j] - xMeansNew[j])/xSDNew[j] for j in range(ncols)]
# predictVauleNew=sum(bataPick[k]*rowNormalizedNew[k] for k in range(len(bataPick))) * xSD[5] + xMeans[5]
# print('last predict lowest:',predictVauleNew)

plt.show()