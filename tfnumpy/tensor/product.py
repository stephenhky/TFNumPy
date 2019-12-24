
import tensorflow.compat.v1 as tf

# Kronecker Product
def tf_kronecker_product(tm1, tm2):
    # disable eager execution
    tf.disable_eager_execution()

    kprod0 = tf.multiply(tm1[0, 0], tm2)

    j0 = tf.constant(1)
    _, kprod = tf.while_loop(lambda j, m: tf.less(j, tm1.shape[1]),
                             lambda j, m: [j + 1, tf.concat([m, tf.multiply(tm1[0, j], tm2)], axis=1)],
                             loop_vars=[j0, kprod0],
                             shape_invariants=[j0.get_shape(), tf.TensorShape([tm2.shape[0], None])])

    for i in range(1, tm1.shape[0]):
        kprod0 = tf.multiply(tm1[i, 0], tm2)

        j0 = tf.constant(1)
        _, kprod1 = tf.while_loop(lambda j, m: tf.less(j, tm1.shape[1]),
                                  lambda j, m: [j + 1, tf.concat([m, tf.multiply(tm1[i, j], tm2)], axis=1)],
                                  loop_vars=[j0, kprod0],
                                  shape_invariants=[j0.get_shape(), tf.TensorShape([tm2.shape[0], None])])

        kprod = tf.concat([kprod, kprod1], axis=0)

    return kprod


def kronecker_product(matrix1, matrix2, tfsess=None):
    if tfsess==None:
        sess = tf.Session()
    else:
        sess = tfsess

    tm1 = tf.constant(matrix1)
    tm2 = tf.constant(matrix2)

    if tfsess:
        init = tf.global_variables_initializer()
        sess.run(init)

    kresult = sess.run(tf_kronecker_product(tm1, tm2))

    return kresult



# Khatri-Rao Product
def tf_khatrirao_product(tensor1, tensor2):
    # disable eager execution
    tf.disable_eager_execution()

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