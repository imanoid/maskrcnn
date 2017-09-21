import numpy as np
import tensorflow as tf
if __name__ == "__main__":
    g = tf.Graph()
    with g.as_default():
        inp = tf.constant(np.random(4, 5, 5, 8), tf.float32)
        f = tf.Variable(np.random(3, 3, 8, 4))
        c_out = tf.depthwise_conv2d_native(
            inp,
            f,
            [1, 1, 1, 1],
            "VALID"
        )
        
        print(inp.shape)
        print(c_out.shape)
