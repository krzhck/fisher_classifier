import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def visualize_original_data(w1, w2, w):
    """
    :param w1: w1类别的原始数据，ndarray，(100,2)
    :param w2: w2类别的原始数据，ndarray，(100,2)
    :param w: 投影向量，ndarray，(2,)
    """
    # 绘制散点图，用不同颜色表示两类数据
    plt.scatter(w1[:, 0], w1[:, 1], color='red', label="w1")
    plt.scatter(w2[:, 0], w2[:, 1], color='blue', label="w2")

    # 绘制投影向量
    x = np.linspace(start=-0.9, stop=0.9, num=100)
    y = (w[1] * x) / w[0]
    plt.plot(x, y, linewidth='2', label="projection vector", color='black')

    # 添加标题和标签
    plt.title('visualization of w1 & w2 & projection vector')
    plt.xlabel('x')
    plt.ylabel('y')

    # 显示label
    plt.legend()

    # 显示图像
    plt.show()


def visualize_projected_data(w1_projected, w2_projected, w0):
    """
    :param w1_projected: 投影后的w1类别的数据，ndarray，(100,)
    :param w2_projected: 投影后的w2类别的数据，ndarray，(100,)
    :param w0: 分类阈值，a float number
    """
    plt.hist(w1_projected, color="red", label="w1")
    plt.hist(w2_projected, color="blue", label="w2")
    plt.vlines([w0], 0, 15, linestyles='dashed', colors='black', label="threshold")
    plt.legend(loc='upper center')

    plt.xlabel("投影后的值")
    plt.ylabel("频率")
    plt.title("投影后数据分布特点")

    plt.show()