
import numpy as np
import scipy
# 寻峰
def findPeaks(y, Peak_width, Peak_hight):
    '''
    功能：寻峰
    :param y:
    :param Peak_width: 设定峰宽
    :param Peak_hight: 周围6个像素点内的平均峰高，类似于计算峰面积来综合细高峰和矮宽峰
    :return:
    '''
    if (max(y[-15:]) - min(y)) > 0.1:
        Peak_hight = (max(y[-30:]) - min(y)) * 1.5

    # y = savgol_filter(data, 7, 3)
    peakind2 = scipy.signal.find_peaks_cwt(y, np.arange(1, Peak_width, 0.5))
    # plt.plot(x, y)
    # plt.plot(x[peakind2], y[peakind2], "*")
    # plt.show()
    if (peakind2[0] - 3 <= 0):
        peakind2 = peakind2[1:]
    if (peakind2[-1] + 3 >= len(y)):
        peakind2 = peakind2[:-1]

    n = peakind2.size
    # Peak_step = Peak_width // 2
    Peak_append = []
    # 创建一个列表，用于存放二次去毛刺删除的元素
    # if(peakind2[0]-3 < 0):

    for i in range(n):
        # # 修正最大值
        lift_right_3 = [y[peakind2[i] - 3], y[peakind2[i] - 2], y[peakind2[i] - 1], \
                        y[peakind2[i]], y[peakind2[i] + 1], y[peakind2[i] + 2], y[peakind2[i] + 3]]
        # # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
        # # operator.itemgetter()函数用于获取对象的哪些维的数据
        # # max_index, max_number = max(enumerate(lift_right_3), key=operator.itemgetter(1))
        # max_index = lift_right_3.index(max(lift_right_3))
        # peakind2[i] = peakind2[i] + max_index -3
        # peakind3 = peakind2

        high_Peakind = np.mean(lift_right_3)
        # print("@@@@",high_Peakind)
        if (high_Peakind < Peak_hight):
            Peak_append.append(i)

    peakind3 = np.delete(peakind2, Peak_append).tolist()
    # print("数组长度：",len(peakind3))
    # 删除异常峰
    # sorted(set(peakind3), key=peakind3.index),删除重复list中元素
    return sorted(set(peakind3), key=peakind3.index)