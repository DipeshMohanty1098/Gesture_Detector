import tensorflow as tf
from tensorflow import keras
import numpy as np
from test import final_test
from test import size
import matplotlib.pyplot as plt
from test import final_train
import pickle

train = final_train
test = final_test

x_train = np.array([i[0] for i in train])
y_train = np.array([i[1] for i in train])

print(y_train.shape)

x_test = np.array([i[0] for i in test])
y_test = np.array([i[1] for i in test])

x_train = x_train/255.0
x_test = x_test/255.0

print(x_train.shape)

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(50,50)),
	keras.layers.Dense(325, activation="relu"),
	keras.layers.Dense(2, activation="softmax")
	])

#mathematics
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#feeding it data, epochs = 6 to change the order of data
model.fit(x_train, np.array(y_train), epochs=9)


test_loss, test_acc = model.evaluate(x_test, y_test)

prediction = model.predict(x_test)

print('\nTest accuracy:', test_acc)
if test_acc> 0.93:
	with open("mymodel.pickle","wb") as f:
		pickle.dump(model, f)

classes = ['thumbs up', 'ok']

for i in range(10):
	plt.imshow(x_test[i])
	plt.xlabel("Actual: " + classes[y_test[i]])
	plt.title("Predicted: " + classes[np.argmax(prediction[i])])
	plt.show()