import numpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# load the dataset 导入数据
dataframe = read_csv('As_train.csv', engine='python', skipfooter=0)

dataframe['date'] = pd.to_datetime(dataframe['date'])
# dataframe.set_index('date', inplace=True)
dataset = dataframe['As'].values
dataset = numpy.array(dataset)
dataset.resize(len(dataset), 1)
datetimeIndex = pd.DatetimeIndex(dataframe['date'])

'''
数据转化:

将一列变成两列，第一列是 t 月的乘客数，第二列是 t+1 列的乘客数
look_back 就是预测下一步所需要的 time steps：

timesteps 就是 LSTM 认为每个输入数据与前多少个陆续输入的数据
有联系。例如具有这样用段序列数据 “…ABCDBCEDF…”，当 timesteps
为 3 时，在模型预测中如果输入数据为“D”，那么之前接收的数据如果
为“B”和“C”则此时的预测输出为 B 的概率更大，之前接收的数据如果
为“C”和“E”，则此时的预测输出为 F 的概率更大。
'''


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

'''
当激活函数为 sigmoid 或者 tanh 时，
要把数据正则话，此时 LSTM 比较敏感
设定 70% 是训练数据，余下的是测试数据
'''
# split into train and test sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# X=t and Y=t+1 时的数据，并且此时的维度为 [samples, features]
# use this function to prepare the train and test datasets for modeling
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]，所以做一下变换
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

'''
建立 LSTM 模型：
输入层有 1 个input，隐藏层有 4 个神经元，输出层就是预测一个值，激活函数用 sigmoid，迭代 100 次，batch size 为 1
'''
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 预测：
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions 计算误差之前要先把预测数据转换成同一单位
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# 计算 mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.3f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) +
                1:len(dataset) - 1, :] = testPredict

# plot baseline and predictions
fig1 = plt.figure(figsize=(15, 8))
ax1 = fig1.add_subplot(111)
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 设置时间标签显示格式
# plt.xticks(rotation=45)#X轴文字旋转
# 画出结果
plt.plot(datetimeIndex, scaler.inverse_transform(dataset), color='blue')
plt.plot(datetimeIndex, trainPredictPlot, color='green')
plt.plot(datetimeIndex, testPredictPlot, color='red')
plt.show()
# print(create_dataset(dataset))
# plt.plot(dataset)
# plt.show()
# 参考 https://blog.csdn.net/aliceyangxi1987/article/details/73420583
