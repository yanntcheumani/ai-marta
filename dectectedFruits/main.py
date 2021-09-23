import os
from PIL import Image
from Marta.Network import Network
from Marta.Layer.FCLayer import FCLayer
from Marta.Layer.ActivationLayer import ActivationLayer
from Marta.Activation import AllActivation
import numpy as np
from keras.utils import np_utils


class Mangagement:
	def __init__(self, nbr_fruit=12):
		self.image_train = np.array([])
		self.image_test = np.array([])
		self.all_response = np.array([])
		self.response_test = np.array([])
		self.nbr_fruit = nbr_fruit

	def __load_train(self):
		tour = 0
		with os.scandir("dataset/train/") as it:
			for entry in it:
				good_place = [0 for i in range(self.nbr_fruit)]
				fruit_array = np.array([])
				if tour == self.nbr_fruit:
					break
				print(f"=======================[{entry.name}]=============================")
				with os.scandir("dataset/train/" + entry.name) as its:
					for file in its:
						print(file.__str__())
						image = Image.open("dataset/train/" + entry.name + '/' + file.name)
						new_image = image.resize((24, 24))
						image_sequence = new_image.getdata()
						image_array = np.array(image_sequence)
						fruit_array = np.array([*fruit_array, *image_array[:]], dtype=object)
				good_place[tour] = 1
				self.all_response = np.array([*self.all_response, *good_place[:]], dtype=object)
				self.image_train = np.array([*self.image_train, *fruit_array[:]], dtype=object)
				tour += 1

	def __load_test(self):
		tour = 0
		with os.scandir("dataset/test/") as it:
			for entry in it:
				good_place = [0 for i in range(self.nbr_fruit)]
				fruit_array = np.array([])
				if tour == self.nbr_fruit:
					break
				print(f"=======================[{entry.name}]=============================")
				with os.scandir("dataset/train/" + entry.name) as its:
					for file in its:
						print(file.__str__())
						image = Image.open("dataset/train/" + entry.name + '/' + file.name)
						new_image = image.resize((24, 24))
						image_sequence = new_image.getdata()
						image_array = np.array(image_sequence)
						fruit_array = np.array([*fruit_array, *image_array[:]], dtype=object)
				good_place[tour] = 1
				self.response_test = np.array([*self.response_test, *good_place[:]], dtype=object)
				self.image_test = np.array([*self.image_test, *fruit_array[:]], dtype=object)
				tour += 1

	def load_data(self):
		self.__load_train()
		self.__load_test()
		return (self.image_train, self.all_response), (self.image_test, self.response_test)


manager = Mangagement(1)
(x_train, y_train), (x_test, y_test) = manager.load_data()
print(x_train)
print(x_train[0].size)
# x_train = x_train.reshape(x_train.shape[0], 1, 24*24)
x_train = x_train[0].astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
# x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test[0].astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


# Network
net = Network()
net.add(FCLayer(24*24, 576))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(AllActivation.TANH))
net.add(FCLayer(576, 288))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(AllActivation.TANH))
net.add(FCLayer(288, 144))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(AllActivation.TANH))
net.add(FCLayer(144, 1))
net.add(ActivationLayer(AllActivation.TANH))

# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
net.fit(x_train[0:900], y_train[0:900], epochs=100, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
for line in out:
    print(np.rint(line), end="\n")
print("true values : ")
print(y_test[0:3])
