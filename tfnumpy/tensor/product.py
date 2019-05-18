
import tensorflow as tf


def tf_khatrirao_product(tensor1, tensor2):
    i0 = tf.constant(1)
    prod0 = tf.multiply(tensor1[0, :], tensor2)
    _, khprod = tf.while_loop(lambda i, m: tf.less(i, tf.shape(tensor1)[0]),
                              lambda i, m: [i+1, tf.concat([m, tf.multiply(tensor1[i, :], tensor2)], 0)],
                              loop_vars=[i0, prod0],
                              shape_invariants=[i0.get_shape(), tf.TensorShape([None, None])])
    return khprod


def khatrirao_product(matrix1, matrix2, tfsess=None):
    if tfsess==None:
        sess = tf.Session()
    else:
        sess = tfsess

    tensor1 = tf.constant(matrix1)
    tensor2 = tf.constant(matrix2)

    if tfsess:
        init = tf.global_variables_initializer()
        sess.run(init)

    khresult = sess.run(tf_khatrirao_product(tensor1, tensor2))

    return khresult