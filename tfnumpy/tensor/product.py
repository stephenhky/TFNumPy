
import tensorflow as tf


def khatrirao_product(tensor1, tensor2):
    i0 = tf.constant(1)
    prod0 = tf.multiply(tensor1[0, :], tensor2)
    _, khprod = tf.while_loop(lambda i, m: tf.less(i, tf.shape(tensor1)[0]),
                              lambda i, m: [i+1, tf.concat([m, tf.multiply(tensor1[i, :], tensor2)], 0)],
                              loop_vars=[i0, prod0],
                              shape_invariants=[i0.get_shape(), tf.TensorShape([None, None])])
    return khprod

