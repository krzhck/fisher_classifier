import numpy as np

from visualize import visualize_original_data, visualize_projected_data

np.random.seed(7)  # 固定随机种子

# 定义w1的均值和协方差矩阵
mu1 = np.array([1, 5])
cov1 = np.array([[4, 0],
                [0, 1]])
# 使用multivariate_normal函数生成w1
w1 = np.random.multivariate_normal(mu1, cov1, 100)  # ndarray:(100,2)

# 定义w2的均值和协方差矩阵
mu2 = np.array([5, -3])
cov2 = np.array([[3, 0],
                [0, 1]])
# 使用multivariate_normal函数生成w2
w2 = np.random.multivariate_normal(mu2, cov2, 100)

# 求w1、w2的均值，可以用np.mean()函数，注意是在axis=0这个维度上取平均，返回一个大小为(2,)的ndarray
w1_mean = w1.mean(axis=0)
w2_mean = w2.mean(axis=0)

# 此处需要计算S1,S2,Sw，此处需要同学们补全~~~~~
# 方法一：循环
# np.zeros((2,2))可以生成一个二维全零矩阵
# np.expand_dims(..., axis=1)可以将shape为(?,)的ndarray变为(?,1)，便于做矩阵乘法
# @:矩阵乘法，.T:转置

# 方法二：矩阵乘法
# .T:转置
# np.dot(A,B)可以实现矩阵乘法AB

# 方法三：np.cov函数
# np.cov函数返回协方差矩阵，计算的是各个特征之间的协方差

S1 = np.zeros((2, 2))
S2 = np.zeros((2, 2))

for i in range(w1.shape[0]):
    S1 += np.dot(np.expand_dims(w1[i] - w1_mean, axis=1), np.expand_dims(w1[i] - w1_mean, axis=1).T)
for i in range(w2.shape[0]):
    S2 += np.dot(np.expand_dims(w2[i] - w2_mean, axis=1), np.expand_dims(w2[i] - w2_mean, axis=1).T)

Sw = S1 + S2

# 计算投影向量w，此处需要同学们补全~~~~~
# 求逆：np.linalg.inv()
# np.dot(A,x)可以实现矩阵A与向量x的乘法:Ax，等价于A.dot(x)

w = np.dot(np.linalg.inv(Sw), w1_mean - w2_mean)

# 可视化原始数据w1, w2以及投影向量w
visualize_original_data(w1, w2, w)

# 计算分类阈值w0，此处需要同学们补全~~~~~
# np.dot()与np.mean()

w0 = np.dot(w.T, (w1_mean + w2_mean) / 2)

# 可视化投影后的数据以及分类阈值
visualize_projected_data(w1_projected=w1.dot(w),
                        w2_projected=w2.dot(w),
                        w0=w0)

# 计算分类准确率
w1_pred_correct_num = np.count_nonzero(w1.dot(w) > w0)
# w1.dot(w)是投影结果，大小为(100,)
# w1.dot(w) > w0返回一个大小为(100,)的布尔矩阵
# np.count_nonzero用于计算一个ndarray中True的个数
# w1_pred_correct_num：w1中被正确分类的元素数量

w2_pred_correct_num = np.count_nonzero(w2.dot(w) < w0)
accuracy = (w1_pred_correct_num + w2_pred_correct_num) / (w1.shape[0] + w2.shape[0])
print("分类准确率：", accuracy)
