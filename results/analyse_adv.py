import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import csv
import seaborn as sns
sns.set()
palette = sns.color_palette("bright", 4)

x_maddpg = list()
y_maddpg = list()

maddpgFile = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/maddpg.csv')  # 打开csv文件
maddpgReader = csv.reader(maddpgFile)  # 读取csv文件
maddpgData = list(maddpgReader)  # csv数据转换为列表
length_zu = len(maddpgData)  # 得到数据行数
length_yuan = len(maddpgData[0])  # 得到每行长度

for i in range(1, length_zu):  # 从第二行开始读取
    x_maddpg.append(int(maddpgData[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_maddpg.append((float(maddpgData[i][2]) + float(maddpgData[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

sns.lineplot(x=x_maddpg, y=y_maddpg, palette=palette[0])
# for i in range(1, length_zu):  # 从第二行开始读取
#     x.append(int(maddpgData[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
#     y.append(float(maddpgData[i][3]))  # 将第二列数据从第二行读取到最后一行赋给列表

x_ddpg = list()
y_ddpg = list()

ddpgFile = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/ddpg.csv')  # 打开csv文件
ddpgReader = csv.reader(ddpgFile)  # 读取csv文件
ddpgData = list(ddpgReader)  # csv数据转换为列表

for i in range(1, length_zu):  # 从第二行开始读取
    x_ddpg.append(int(ddpgData[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_ddpg.append((float(ddpgData[i][2]) + float(ddpgData[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

sns.lineplot(x=x_ddpg, y=y_ddpg, palette=palette[1])

x_snn = list()
y_snn = list()

snn1File = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/snn1.csv')  # 打开csv文件
snn1Reader = csv.reader(snn1File)  # 读取csv文件
snn1Data = list(snn1Reader)  # csv数据转换为列表
length_zu = len(snn1Data)  # 得到数据行数
length_yuan = len(snn1Data[0])  # 得到每行长度

for i in range(1, length_zu):  # 从第二行开始读取
    x_snn.append(int(snn1Data[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_snn.append((float(snn1Data[i][2]) + float(snn1Data[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

snn2File = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/snn2.csv')  # 打开csv文件
snn2Reader = csv.reader(snn2File)  # 读取csv文件
snn2Data = list(snn2Reader)  # csv数据转换为列表
length_zu = len(snn2Data)  # 得到数据行数
length_yuan = len(snn2Data[0])  # 得到每行长度

for i in range(1, length_zu):  # 从第二行开始读取
    x_snn.append(int(snn2Data[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_snn.append((float(snn2Data[i][2]) + float(snn2Data[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

snn3File = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/snn3.csv')  # 打开csv文件
snn3Reader = csv.reader(snn3File)  # 读取csv文件
snn3Data = list(snn3Reader)  # csv数据转换为列表
length_zu = len(snn3Data)  # 得到数据行数
length_yuan = len(snn3Data[0])  # 得到每行长度

for i in range(1, length_zu):  # 从第二行开始读取
    x_snn.append(int(snn3Data[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_snn.append((float(snn3Data[i][2]) + float(snn3Data[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

sns.lineplot(x=x_snn, y=y_snn, palette=palette[2])


x_tom = list()
y_tom = list()

tom1File = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/tom1.csv')  # 打开csv文件
tom1Reader = csv.reader(tom1File)  # 读取csv文件
tom1Data = list(tom1Reader)  # csv数据转换为列表
length_zu = len(tom1Data)  # 得到数据行数
length_yuan = len(tom1Data[0])  # 得到每行长度

for i in range(1, length_zu):  # 从第二行开始读取
    x_tom.append(int(tom1Data[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_tom.append((float(tom1Data[i][2]) + float(tom1Data[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

tom2File = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/tom2.csv')  # 打开csv文件
tom2Reader = csv.reader(tom2File)  # 读取csv文件
tom2Data = list(tom2Reader)  # csv数据转换为列表
length_zu = len(tom2Data)  # 得到数据行数
length_yuan = len(tom2Data[0])  # 得到每行长度

for i in range(1, length_zu):  # 从第二行开始读取
    x_tom.append(int(tom2Data[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_tom.append((float(tom2Data[i][2]) + float(tom2Data[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

tom3File = open('/home/zhaozhuoya/maddpg-pytorch/results/adv/logs/tom3.csv')  # 打开csv文件
tom3Reader = csv.reader(tom3File)  # 读取csv文件
tom3Data = list(tom3Reader)  # csv数据转换为列表
length_zu = len(tom3Data)  # 得到数据行数
length_yuan = len(tom3Data[0])  # 得到每行长度

for i in range(1, length_zu):  # 从第二行开始读取
    x_tom.append(int(tom3Data[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y_tom.append((float(tom3Data[i][2]) + float(tom3Data[i][1]))*25)  # 将第二列数据从第二行读取到最后一行赋给列表

sns.lineplot(x=x_tom, y=y_tom, palette=palette[3])
plt.legend(labels=['MADDPG', 'DDPG', 'SNN', 'ToM'])
plt.xlabel('episode')
plt.ylabel('episode reward')
plt.show()  # 显示折线图


