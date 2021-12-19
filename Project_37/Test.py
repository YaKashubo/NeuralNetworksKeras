import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random as rm


def V(dp, L, R, x):
    v = (dp * (R * R - x * x)) / (4 * 0.001 * L)
    return v


def Re(v, D):
    return (1000 * v * D) / 0.001

    # Rmax=5[m]
    # Lmax=100[m]
    # dpmax=3.68*10**-6
    # vmax=2.4*10**-4


R = 5  # радиус трубы [м]
dp = 3.68 * 10 ** -6  # разность давлений [Па]
L = 100  # длина трубы [м]
# n = 50  # кол-во нейронов
maxim = 0

for i in range(L, 0, -1):
    v = V(dp, i, R, 0)
    print(v)
    if (Re(v, 2 * R) <= 2300) and v > maxim:
        maxim = v
print(maxim)

count = 0  # кол-во входных данных
counter = 0
X = [[0, 0, 0, 0] for j in range(count)]
Y = []

# while (counter != count):
#     R = rm.uniform(0, 5)
#     L = rm.uniform(0, 100)
#     dp = rm.uniform(0, 3.68 * 10 ** -6)
#     x = rm.uniform(0, R)
#     v = V(dp, L, R, x)
#     if (Re(v, 2 * R) <= 2300):
#         X[counter][0] = L
#         X[counter][1] = dp
#         X[counter][2] = R
#         X[counter][3] = x
#         Y.append(v)
#         counter += 1

# dL = L / (count // 4)
# while L > 0:
#     dp = 3.68 * 10 ** -6
#     ddp = dp / (count // 4)
#     while dp > 0:
#         R = 5
#         dR = R / (count // 4)
#         while R > 0:
#             dx = R/(count//4)*2
#             x = R
#             while x - dx >= -R:
#                 x -= dx
#                 v = V(dp, L, R, x)
#                 if Re(v, 2 * R) <= 2300:
#                     X[counter][0] = L
#                     X[counter][1] = dp
#                     X[counter][2] = R
#                     X[counter][3] = x
#                     Y.append(v)
#                     counter += 1
#             R -= dR
#         dp -= ddp
#     L -= dL
