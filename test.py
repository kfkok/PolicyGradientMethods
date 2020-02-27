import tensorflow.compat.v1 as tf
import numpy as np

# turn off eager execution
tf.disable_eager_execution()

# build graph
input = tf.placeholder(tf.float32, (None, 3), name="input")
hidden = tf.keras.layers.Dense(3, activation="tanh")(input)
output = tf.keras.layers.Dense(2, activation="sigmoid")(hidden)

# gradient operator
trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
gradients = tf.gradients(output, trainable_variables)

# create the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1):
    state = np.random.rand(3).reshape(1, 3)
    result, grad = sess.run([output, gradients], feed_dict={input:state})
    print(grad)



