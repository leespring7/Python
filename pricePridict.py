import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
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

            cells=line.strip('\n').split('\t')  #去掉换行符，再以制表符副歌
            for cellIndex in range(len(cells)):
                if i !=0  and cellIndex not in [0,9]:
                    self.data[i].append(float((cells[cellIndex])))   #尽量将字符转换成数字
                else :
                    self.data[i].append(cells[cellIndex])  #最后出价时间

            print(self.data[i])   #打印每一行的各个栏位
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

i=1
for key in dataall['2015'].keys():
    if key != 'Month':
        #ax = plt.subplot(7,1,i)  #全部画在一张图上，内容太多，展示不清晰
        plt.figure()
        i += 1
        for index in range(len(years)):


            colorVal = scalarMap.to_rgba(index)
            plt.plot(dataall[years[index]]['Month'],dataall[years[index]].get(key), c=colorVal,marker='.',label=years[index])
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.title(key)
#flg.tight_layout()
plt.show()
'''
#x=np.linspace(1,len(pricePridictTest.data)) 
#画出均值曲线

y1=[pricePridictTest.data[i][6] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2015' ]  #默认读进来是string类型，不会按数字大小
     #按年份取出数据
###注意，用数组方式截取 是截取0到4之前的，即只截取0,1,2,3位置上的
x1=np.arange(1,len(y1)+1)
y2=[pricePridictTest.data[i][6] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2016' ]
x2=np.arange(1,len(y2)+1)
y3=[pricePridictTest.data[i][6] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2017' ]
x3=np.arange(1,len(y3)+1)
y4=[pricePridictTest.data[i][6] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2018' ]
x4=np.arange(1,len(y4)+1)

plt.figure()
plt.subplot(121)
plt.title("Mean")
l1=plt.plot(x1,y1,c='r',marker='o',label='2015')  #label用于添加legend
l2=plt.plot(x2,y2,c='g',marker='o',label='2016')
l3=plt.plot(x3,y3,c='y',marker='o',label='2017')
l4=plt.plot(x4,y4,c='b',marker='o',label='2018')
#plt.legend((l1,l2,l3,l4),('2015','2016','2017','2018'))
plt.legend(loc='upper left')
#plt.ylim((10000,100000))
#plt.scatter((1,1),(3,10000))


###最低价曲线
ylow1=[pricePridictTest.data[i][5] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2015' ]  #默认读进来是string类型，不会按数字大小
     #按年份取出数据
###注意，用数组方式截取 是截取0到4之前的，即只截取0,1,2,3位置上的
xlow1=np.arange(1,len(ylow1)+1)
ylow2=[pricePridictTest.data[i][5] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2016' ]
xlow2=np.arange(1,len(ylow2)+1)
ylow3=[pricePridictTest.data[i][5] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2017' ]
xlow3=np.arange(1,len(ylow3)+1)
ylow4=[pricePridictTest.data[i][5] for i in range(1,len(pricePridictTest.data))
    if pricePridictTest.data[i][0][0:4]=='2018' ]
xlow4=np.arange(1,len(ylow4)+1)
plt.subplot(122)
plt.title("Loewest")
l1=plt.plot(xlow1,ylow1,c='r',marker='o',label='2015')  #label用于添加legend
l2=plt.plot(xlow2,ylow2,c='g',marker='o',label='2016')
l3=plt.plot(xlow3,ylow3,c='y',marker='o',label='2017')
l4=plt.plot(xlow4,ylow4,c='b',marker='o',label='2018')

plt.legend(loc='upper left')

plt.show()
'''
