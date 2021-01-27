import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt


def celsius_to_fahrenheit():
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    celsius_q = np.array([-40, -30, -17.78, -10, 0, 8, 15, 22, 38, 50, 62, 75, 81, 82],  dtype=float)
    fahrenheit_a = np.array([-40, -22, -0.004, 14, 32, 46, 59, 72, 100, 122, 143.6, 167, 177.8, 179.6],  dtype=float)

    for i, c in enumerate(celsius_q):
        print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10, input_shape=[1]),  # Layer 0
        tf.keras.layers.Dense(units=10),  # Layer 1
        tf.keras.layers.Dense(units=1)  # Layer 2
    ])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

    history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
    print("Finished training the model")

    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])
    plt.show()
    print(model.predict([100.0]))


celsius_to_fahrenheit()
