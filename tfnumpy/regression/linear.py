
import numpy as np
import tensorflow as tf


def fit_linear_regression(trainX, trainY,
                          learning_rate=0.01,
                          ridge_alpha=0.0,
                          lasso_alpha=0.0,
                          max_iter=1000,
                          display_step=50,
                          converged_tol=1e-8,
                          to_print=False):
    # check dimensions
    assert trainX.shape[0] == trainY.shape[0]   # number of training data the same

    # placeholder
    X = tf.constant(trainX, dtype='float', name='X')
    Y = tf.constant(trainY, dtype='float', name='Y')

    # Dimension placeholder
    nbtrain = X.shape[0]
    nbfeatures = X.shape[1]

    # fitting parameters
    theta = tf.Variable(np.random.uniform(size=nbfeatures), name='theta', dtype='float')
    b = tf.Variable(np.random.uniform(), name='b', dtype='float')

    # fitting function
    pred_Y = tf.multiply(theta, X) + b

    # cost function
    cost = tf.reduce_sum(tf.square(pred_Y - Y)) / nbtrain.value \
                + ridge_alpha * (tf.reduce_sum(tf.square(theta)) + tf.square(b)) \
                + lasso_alpha * (tf.reduce_sum(tf.abs(theta)) + tf.abs(b))

    # training the machine
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)

        old_cost = sess.run(cost)

        # Fit all training data
        for epoch in range(max_iter):
            sess.run(optimizer)

            if to_print:
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    c = sess.run(cost)
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                          "theta=", sess.run(theta), "b=", sess.run(b))

            if converged_tol is not None:
                new_cost = sess.run(cost)
                if abs(new_cost - old_cost) < converged_tol:
                    break
                else:
                    old_cost = new_cost

        if to_print:
            print("Optimization Finished!")

        # extract value
        training_cost = sess.run(cost)
        trained_theta = sess.run(theta)
        trained_b = sess.run(b)

    return {'theta': trained_theta, 'b': trained_b, 'cost': training_cost, 'epoch': epoch,
            'nbfeatures': nbfeatures.value, 'nbtrain': nbtrain.value}