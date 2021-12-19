# учим нейроннку быть функцией f(x)=2x+5

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.layers import Dense


def f(x):
    return 2 * x + 5


X = []
Y = []
x = -1
while x <= 1:
    x += 0.2
    X.append(x)
    Y.append(f(x))

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))
model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mean_squared_error')

model.fit(X, Y, epochs=100, verbose=1)

print(model.predict([10]))  # должна вернуть f(10)=25


#   -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
#   3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8,

# , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#    5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8,
