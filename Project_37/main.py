import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras.layers import Dense
import random as rm
import time as t
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def V(dp, L, R, x):
    v = (dp * (R * R - x * x)) / (4 * 0.001 * L)
    return v


def Re(v, D):
    return (1000 * v * D) / 0.001


start_time = t.time()

n = 15  # кол-во нейронов
count = 13000  # кол-во входных данных
counter = 0

X = [[0, 0, 0, 0] for j in range(count)]  # массив входных данных [[L, dp, R, x], [...]]
Y = []  # массив выходнох данных [v, ...]
while counter != count:  # цикл генерации входных и выходных данных
    R = rm.uniform(0, 5)  # радиус трубы (0,5.0] [м]
    L = rm.uniform(0, 100)  # длина трубы (0,100.0] [м]
    dp = rm.uniform(0, 3.68 * 10 ** -6)  # разность давлений (0,3.68*10^-6] [Па]
    x = rm.uniform(-R, R)  # точки на параболе [-R,R) [м]
    v = V(dp, L, R, x)  # скорость в точке x [0;2,3*10^-4] [м/c]
    if Re(v, 2 * R) <= 2300:
        X[counter][0] = L
        X[counter][1] = dp
        X[counter][2] = R
        X[counter][3] = x
        Y.append(v)
        counter += 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=0.4,
                                                    shuffle=0)  # разделение на обучающую(40%) и тестовую(60%) выборку

model = keras.Sequential(
    [Dense(units=n, input_shape=(4,), activation='tanh'),
     Dense(units=7, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=7, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=7, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=n, activation='tanh'),
     Dense(units=7, activation='linear'),
     Dense(units=n, activation='tanh'),
     Dense(units=7, activation='tanh'),
     Dense(units=n, activation="tanh"),
     Dense(units=1, activation='linear')])
model.compile(optimizer=keras.optimizers.RMSprop(0.00001), loss='mean_squared_error')  # создание НС
history = model.fit(X_train, Y_train, epochs=50, verbose=1)  # обучение НС

Y_predicted = model.predict(X_test)  # скорости, предсказанные НС
newarr = [i[0] for i in Y_predicted]
Y_predicted = newarr
dif = []  # массив отклонений предсказанных значений от тестовых
for i in range(len(Y_test)):
    dif.append(abs(Y_predicted[i] - Y_test[i]))
print("Максимальное отклонение =", max(dif))
print("Минимальное отклонение =", min(dif))
print("Время работы = %s seconds " % (t.time() - start_time))
Y_test1 = []
Y_predicted1 = []
for i in range(100):
    Y_test1.append(Y_test[i])
    Y_predicted1.append(Y_predicted[i])
plt.plot(list(range(0, 100)), Y_test1, 'b', label='Test')
plt.plot(list(range(0, 100)), Y_predicted1, 'r', label='Predicted')
plt.show()
