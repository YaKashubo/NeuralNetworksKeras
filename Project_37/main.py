import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras.layers import Dense
import random as rm
import time as t
from sklearn.model_selection import train_test_split


def V(dp, L, R, x):
    v = (dp * (R * R - x * x)) / (4 * 0.001 * L)
    return v


def Re(v, D):
    return (1000 * v * D) / 0.001


start_time = t.time()

n = 150  # кол-во нейронов
count = 10000  # кол-во входных данных
counter = 0

X = [[0, 0, 0, 0] for j in range(count)]  # массив входных данных [[L, dp, R, x], [...]]
Y = []  # массив выходнох данных [v, ...]
while counter != count:  # цикл генерации входных и выходных данных
    R = rm.uniform(0, 5)  # радиус трубы (0,5.0] [м]
    L = rm.uniform(0, 100)  # длина трубы (0,100.0] [м]
    dp = rm.uniform(0, 3.68 * 10 ** -6)  # разность давлений (0,3.68*10^- [Па]
    x = rm.uniform(0, R)  # точки на параболе [-R,R) [м]
    v = V(dp, L, R, x)  # скорость в точке x [0;2,3*10^-4] [м/c]
    if Re(v, 2 * R) <= 2300:
        X[counter][0] = L
        X[counter][1] = dp
        X[counter][2] = R
        X[counter][3] = x
        Y.append(v)
        counter += 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=0.3,
                                                    shuffle=0)  # разделение на обучающую(30%) и тестовую(70%) выборку

model = keras.Sequential(
    [Dense(units=n, input_shape=(4,), activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation="tanh"),
     Dense(units=1, activation='sigmoid')])
model.compile(optimizer=keras.optimizers.RMSprop(0.00001), loss='mean_squared_error')  # создание НС
# Nadam
# RMSprop
history = model.fit(X_train, Y_train, epochs=550, verbose=1)  # обучение НС

# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.plot(history.history['loss'])
# plt.show()
s = 0
Y_predicted = model.predict(X_test)  # скорости, предсказанные НС
newarr = [i[0] for i in Y_predicted]  # магия для нормальной работы с массивом скоростей
Y_predicted = newarr  # продолжение магии
dif = []  # массив отклонений предсказанных значений от тестовых
for i in range(len(Y_test)):
    dif.append(abs(Y_predicted[i] - Y_test[i]))
    s += (abs(Y_predicted[i] - Y_test[i])) / abs(Y_test[i])
accur = s / len(Y_test)
# оценки работы программы
print("Среднее процентное отклонение", accur * 100)
print("Максимальное отклонение =", max(dif))
print("Время работы = %s seconds " % (t.time() - start_time))

# while str != 'exit':
#     x = float(input())
#     R = float(input())
#     print(model.predict(R, x), V(dp, L, R, x))
#     str = input("Продолжаем?")
