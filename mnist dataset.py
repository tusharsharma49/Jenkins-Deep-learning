from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
train , test = dataset
X_train , y_train = train
X_test , y_test = test
#plt.imshow(img1 , cmap='gray')
#img1_1d = img1.reshape(28*28)
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=512, input_dim=28*28, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
         metrics=['accuracy']
         )
h = model.fit(X_train, y_train_cat, epochs=3)
scores = model.evaluate(X_test, y_test_cat, verbose=0)
print(scores[1]*100 , file = open("/home/cnn/output1.txt","a"))
if scores[1]*100>=90:
         model.save("/home/cnn/mnist.h5")