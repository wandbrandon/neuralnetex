import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# handwritten digits from 0 - 9
mnist = tf.keras.datasets.mnist 

# load the data into variables
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data between zero and one
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# compile the model
model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
        )

# fit the model
model.fit(x_train, y_train, epochs=2)

# get the facts
val_loss, val_acc = model.evaluate(x_test, y_test)

plt.imshow(x_test[3])
plt.show()

predictions = model.predict(x_test)

print("Predicted number by the neural network:")
print(np.argmax(predictions[3]))
