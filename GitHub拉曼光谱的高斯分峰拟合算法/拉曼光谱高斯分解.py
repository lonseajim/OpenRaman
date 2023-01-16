# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方法：基于高斯多峰拟合的拉曼光谱分解方法
作者: 李新立
时间：2023.01.01
E-mail：lixinli2017@126.com
"""
import math
# import csv
from matplotlib import pyplot as plt

pi = math.pi
e = math.e
import numpy as np
import pandas as pd
from scipy.optimize import minimize
#非线性规划优化函数


# 选择你想要分解光谱部分
section_start = 1089  # 峰的起始波数
points = 13  # 整个峰数据点的个数
points2 = 18 # 整个峰数据点跨越波数1106.27-1088.39

# 将选择峰分解为几部分组成
peak_number = 2 #这个必须小于或等于高斯拟合阶数
scale_peak = 1100 # 用于缩放的波数
x_0_guess_1 = [1096, 1100] #峰位点
lam_guess_1 = [5.9, 4] #半峰宽
y_guess = 0.2 #校正基数

A_guess_1 = lam_guess_1
A_guess_2 = A_guess_1

bnds = []
for i in range(peak_number):
    bnds.append((x_0_guess_1[i] - 2, x_0_guess_1[i] + 2))
    bnds.append((0.1, lam_guess_1[i] + 10))
    bnds.append((A_guess_1[i] / 50, A_guess_1[i] * 100))
bnds.append((0, 1))

# 如果算法不收敛，请尝试下面列表中的另一个算法
choices = ['SLSQP','L-BFGS-B', 'TNC', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'COBYLA']
algorithm = choices[0]

# 一条曲线的高斯分布函数
def gaus_dist(x, x_01, lam1, A1, y):
    return (y + A1 * (1 / lam1 / np.sqrt(2 * pi) * np.exp(-(x - x_01) ** 2 / 2 / (lam1 ** 2))))
# 多条曲线的高斯分布函数
def gaus_dist_mult(x, x_01, lam1, A1, y):
    ans = 0
    for i in range(len(lam1)):
        ans += y + (A1[i] * (1 / lam1[i] / np.sqrt(2 * pi) * np.exp(-(x - x_01[i]) ** 2 / 2 / (lam1[i] ** 2))))
    return (ans)


data = np.array(pd.read_csv("Mean_03h.txt", delim_whitespace=True, header=None))
wavenumbers = data[:,0]
activities = data[:,1]
activities_old = activities
# plt.plot(wavenumbers,activities,'b')
print("scale_peak:",scale_peak) # 数据规模
def scale(activities, wavenumbers):
    wave_mod = []
    for i in range(len(wavenumbers)):
        wave_mod.append(math.floor(wavenumbers[i]))
        #向下取整
    scale_index = wave_mod.index(scale_peak)
    scale_act = max(activities[scale_index - 5:scale_index + 5])
    for i in range(len(activities)):
        activities[i] = activities[i] / scale_act

    return activities,scale_act

# print("activities",activities)
activities ,scale_act= scale(activities, wavenumbers)
# print("activities",activities)
# print("scale_act",scale_act)

# 将数据切成你想要的任何部分
def section(wavenumbers, activities, section_start, points):
    wave_mod = []
    for i in range(len(wavenumbers)):
        wave_mod.append(math.floor(wavenumbers[i]))
    low_index = wave_mod.index(section_start)
    wavenumbers_mod = wavenumbers[low_index: low_index + points]
    activities_mod = activities[low_index: low_index + points]
    return (wavenumbers_mod, activities_mod)


# 调用
wavenumbers, activities = section(wavenumbers, activities, section_start, points)

# 将曲线拟合给定的部分到给定的曲线数目
x = np.array(wavenumbers)
y = np.array(activities)
# plt.plot(x, y)
# plt.show()

guess = []
for i in range(peak_number):
    guess.append(x_0_guess_1[i])
    guess.append(lam_guess_1[i])
    guess.append(A_guess_1[i])
guess.append(y_guess)
guess = np.array(guess)
# print("guess:",guess)


# 定义高斯函数的两个范数
def gaus_cauchy_dist_mult_two_norm(var, y, x):
    y_new = []
    for j in range(len(y)):
        ans = 0
        for i in range(peak_number):
            ans += var[-1] + (var[i * 3 + 2] * (1 / var[i * 3 + 1] / np.sqrt(2 * pi) \
                                                * np.exp(-(x - var[i * 3]) ** 2 / 2 / (var[i * 3 + 1] ** 2))))
        y_new.append(ans)

    y_new = np.array(y_new)
    res = np.linalg.norm(y - y_new)
    return (res)


# 使用sciPy优化函数拟合曲线
bnds = tuple(bnds)
params = minimize(gaus_cauchy_dist_mult_two_norm, guess, args=(y, x), \
                  method=algorithm, bounds=bnds, options={'maxiter': 1000})

params = params.x

# 存储参数
x_01 = []
lam1 = []
A1 = []
x_02 = []
lam2 = []
A2 = []
for i in range(peak_number):
    x_01.append(params[3 * i])
    lam1.append(params[3 * i + 1])
    A1.append(params[3 * i + 2])
y_0 = (params[-1])

#***********************************************************************************画图
plt.figure()
plt.plot(wavenumbers, activities * scale_act, 'ko', markersize = 5)
# plt.show()
wave_est_new = np.linspace(section_start - 10, section_start + 10 + points2, 500)
#创建一个由等差数列构成的一维数组
act_est = []
for i in wave_est_new:
    act_est.append(gaus_dist_mult(i, x_01, lam1, A1, y_0))

plt.plot(wave_est_new, np.array(act_est) * scale_act, 'k')
#高斯拟合
plt.show()

# plot the individual peaks，绘制单个峰值
for i in range(peak_number):
    act_est1 = []
    for j in wave_est_new:
        act_est1.append(gaus_dist(j, x_01[i], lam1[i], A1[i], y_0))
    plt.plot(wave_est_new, np.array(act_est1) * scale_act, )
legend_font = {"family" : "Times New Roman",'size': 20}
# plt.title(title_name, **{'size': '18', 'weight': 'bold', 'fontname': 'Arial'})
plt.xlabel('Wavenumbers (cm^-1)', **{'size': '20', 'fontname': 'Times New Roman'})
plt.ylabel('PeakFitting', **{'size': '20', 'fontname': 'Times New Roman'})
plt.legend(['Spectra', 'Combined Peaks', 'Individual Peaks'],prop=legend_font)
plt.show()


def error_analysis(activities_old, activities, wavenumbers, A1, x_01, lam1, y):
    # 取一段没有峰的光谱，然后用这个找到残留
    noise_points = activities_old[len(activities_old) - 100:len(activities_old)]
    noise_mean = np.mean(noise_points)
    residual = abs(noise_points - noise_mean)
    residual_mean = np.average(residual)
    print('Residual mean:', residual_mean) #输出拟合残差

    num_trial = 1000
    A_test_tethered = []
    A_test_free = []

    # 模拟数据
    for i in range(num_trial):
        act_mont = gaus_dist_mult(wavenumbers, x_01, lam1, A1, y)
        for j in range(len(act_mont)):
            act_mont[j] = act_mont[j] + residual_mean * np.random.normal()
            # 拟合模拟数据
        coeffs = minimize(gaus_cauchy_dist_mult_two_norm, guess, args=(act_mont, wavenumbers), \
                          method=algorithm, bounds=bnds, options={'maxiter': 1000})
        coeffs = coeffs.x
        A_test_tethered.append(coeffs[2])
        A_test_free.append(coeffs[5])

    cation_sim = []
    for i in range(len(A_test_free)):
        cation_sim.append(A_test_free[i] / (A_test_free[i] + A_test_tethered[i]))
    cation_error = np.std(cation_sim) * 2
    free_cation_sim = np.mean(cation_sim)

    free_cation = A1[1] / (A1[1] + A1[0])

    return free_cation, free_cation_sim, cation_error
if 1:
    free_cation, free_cation_sim, cation_error = \
        error_analysis(activities_old, activities, wavenumbers, A1, x_01, lam1, y_0)
    # 原光谱
    # print('Experimental percent of free Li: ' + str(free_cation * 100))
    # print('Simulation percent of free Li: ' + str(free_cation_sim * 100) \
    #       + ' Plus or Minus ' + str(cation_error * 100))
    # print('Percent Error: ' + str(cation_error / free_cation * 100))