
import numpy as np
import tensorflow.compat.v1 as tf
from . import tf_khatrirao_product


def rank3tensor_decomposition_ALS(matrix, k, alpha=0.01, nbiter=1000):
    # disable eager execution
    tf.disable_eager_execution()

    dim0, dim1, dim2 = matrix.shape

    X = tf.placeholder(tf.float32, shape=(dim0, dim1, dim2), name='X')

    X0 = tf.reshape(tf.transpose(X, perm=(0, 2, 1)), shape=(dim0, dim1 * dim2))
    X1 = tf.reshape(tf.transpose(X, perm=(1, 2, 0)), shape=(dim1, dim0 * dim2))
    X2 = tf.reshape(tf.transpose(X, perm=(2, 0, 1)), shape=(dim2, dim0 * dim1))

    A = tf.Variable(initial_value=tf.random_normal([dim0, k]), name='A')
    B = tf.Variable(initial_value=tf.random_normal([dim1, k]), name='B')
    C = tf.Variable(initial_value=tf.random_normal([dim2, k]), name='C')

    costA = tf.reduce_sum(tf.abs(X0 - tf.matmul(A, tf.transpose(tf_khatrirao_product(C, B)))))
    costB = tf.reduce_sum(tf.abs(X1 - tf.matmul(B, tf.transpose(tf_khatrirao_product(C, A)))))
    costC = tf.reduce_sum(tf.abs(X2 - tf.matmul(C, tf.transpose(tf_khatrirao_product(B, A)))))

    trainA = tf.train.GradientDescentOptimizer(alpha).minimize(costA)
    trainB = tf.train.GradientDescentOptimizer(alpha).minimize(costB)
    trainC = tf.train.GradientDescentOptimizer(alpha).minimize(costC)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for _ in range(nbiter):
        sess.run(trainA, feed_dict={X: matrix})
        sess.run(trainB, feed_dict={X: matrix})
        sess.run(trainC, feed_dict={X: matrix})

    return sess.run((A, B, C), feed_dict={X: matrix})

