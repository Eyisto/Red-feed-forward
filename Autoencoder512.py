# TensorFlow y tf.keras
#import tensorflow as tf
from tensorflow import keras
from matplotlib.pyplot import figure
# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

#from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Reviso la primera figura
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#preproceso las imagenes

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

LATENT_SIZE = 32

encoder = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(512),
    Dropout(0.1)
    #Dense(256),
    #LR(),
    #Dropout(0.1),
    #Dense(128),
    #LR(),
    #Dropout(0.1),
    #Dense(64),
    #LR(),
    #Dropout(0.1),
    #Dense(LATENT_SIZE),
    #LR()
])

decoder = Sequential([
    Dense(512, input_shape = (512,)),
    Dense(784),
    Activation("relu"),
    Reshape((28, 28))
])

img = Input(shape = (28, 28))
latent_vector = encoder(img)
output = decoder(latent_vector)

model = Model(inputs = img, outputs = output)
#model.compile("SGD", loss = "binary_crossentropy")
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['MeanSquaredError'])

history=model.fit(train_images, train_images,
                epochs=1000,
                batch_size=60,
                shuffle=False,
                validation_data=(test_images, test_images))

encoded_imgs = encoder.predict(test_images)
decoded_imgs = decoder.predict(encoded_imgs)

n = 5  # How many digits we will display
fig = plt.figure()
#plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#fig.savefig('EJEMPLO2.png', dpi = 1000)
fig.savefig('EJEMPLO512.png', dpi = 1000)
plt.show()


fig1 = plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
filename = 'loss512.txt'
file=open(filename,"w")
np.savetxt(filename , np.column_stack([history.history['loss'],history.history['val_loss']]), fmt=['%lf','%lf'])
file.close()
plt.title('Función de perdida para 512 neuronas de capa oculta')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig1.savefig('errovsepoca512.png', dpi = 1000)
plt.show()
#EPOCHS = 50
#Only do plotting if you have IPython, Jupyter, or using Colab

#for epoch in range(EPOCHS):
 #   fig, axs = plt.subplots(4, 4)
  #  rand = test_images[np.random.randint(0, 10000, 16)].reshape((4, 4, 1, 28, 28))


   # for i in range(4):
    #    for j in range(4):
     #       axs[i, j].imshow(model.predict(rand[i, j])[0], cmap = "gray")
     #       axs[i, j].axis("off")

    #plt.subplots_adjust(wspace = 0, hspace = 0)
   # plt.show()
    #print("-----------", "EPOCH", epoch, "-----------")
    #model.fit(train_images, train_images)
