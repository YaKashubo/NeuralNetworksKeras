# учим нейроннку быть функцией f(x)=2x+5

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense


def V(dp, L, R, x):  # функция для подсчёта скорости
    v = (dp * (R * R - x * x)) / (4 * 0.001 * L)
    return v


def Re(v, D):  # функция для подсчёта числа Рейнольдса
    return (1000 * v * D) / 0.001


n = 20  # кол-во нейронов
R = 0.51  # радиус трубы [м]
L = 7  # длина трубы  [м]
dp = 0.0002  # разность давлений [Па]
x = R  # точки на параболе [м]
dx = R / 3000

X = []  # массив входных данных
Y = []  # массив выходнох данных
while x - dx >= -R:  # цикл генерации входных и выходных данных
    x -= dx
    v = V(dp, L, R, x)
    if Re(v, 2 * R) <= 2300:
        X.append(x)
        Y.append(v)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=0.3,
                                                    random_state=42)
model = keras.Sequential(
    [Dense(units=n, input_shape=(1,), activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=1, activation='linear')])
model.compile(optimizer=keras.optimizers.Adam(0.00001), loss='mean_squared_error')  # создание НС
history = model.fit(X, Y, epochs=70, verbose=0)  # обучение НС
R = 0.3
x = R
dx = R / 2000

Y_predicted = model.predict(X_test)  # скорости, предсказанные НС
newarr = [i[0] for i in Y_predicted]
Y_predicted = newarr
maxim = abs(Y_predicted[0] - Y_test[0])
index = 0
dif = []  # массив отклонений предсказанных значений от тестовых
for i in range(len(Y_test)):
    dif.append(abs(Y_predicted[i] - Y_test[i]))
    if abs(Y_predicted[i] - Y_test[i]) > maxim:
        index = i
print("Максимальное отклонение =", max(dif))
print("Тестовое значение скорости = ", Y_test[index], "Предугаданное нейронной сетью значение =", Y_predicted[index])

