
import numpy as np
import tensorflow as tf
from .. import SupervisedClassifier
from .. import ModelNotTrainedError


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
    nbtrain = trainX.shape[0]
    nbfeatures = trainX.shape[1]

    # placeholder
    X = tf.placeholder(tf.float32, shape=(None, nbfeatures), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, 1), name='Y')

    # fitting parameters
    theta = tf.Variable(np.random.uniform(size=(nbfeatures, 1)), name='theta', dtype='float')
    b = tf.Variable(np.random.uniform(), name='b', dtype='float')

    # fitting function
    pred_Y = tf.matmul(X, theta) + b

    # cost function
    cost = tf.reduce_mean(tf.square(pred_Y - Y))
    # regularization
    if ridge_alpha is not None and ridge_alpha!=0:
        cost += 0.5 * ridge_alpha * (tf.reduce_sum(tf.square(theta)) + tf.square(b))
    if lasso_alpha is not None and lasso_alpha!=0:
        cost += lasso_alpha * (tf.reduce_sum(tf.abs(theta)) + tf.abs(b))

    # training the machine
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    initializer = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(initializer)

    old_cost = sess.run(cost, feed_dict={X: trainX, Y: trainY})

    # Fit all training data
    for epoch in range(max_iter):
        sess.run(optimizer, feed_dict={X: trainX, Y: trainY})

        if to_print:
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: trainX, Y: trainY})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                      "theta=", sess.run(theta), "b=", sess.run(b))

        if converged_tol is not None:
            new_cost = sess.run(cost, feed_dict={X: trainX, Y: trainY})
            if abs(new_cost - old_cost) < converged_tol:
                break
            else:
                old_cost = new_cost

    if to_print:
        print("Optimization Finished!")

    # extract value
    training_cost = sess.run(cost, feed_dict={X: trainX, Y: trainY})
    trained_theta = sess.run(theta)
    trained_b = sess.run(b)

    fitted_params = {'theta': trained_theta,
                     'b': trained_b,
                     'cost': training_cost,
                     'nbepoch': epoch,
                     'nbfeatures': nbfeatures,
                     'nbtrain': nbtrain}
    tf_sess = {'session': sess, 'inputs': X, 'outputs': pred_Y}

    return fitted_params, tf_sess


class TFLinearRegression(SupervisedClassifier):
    def __init__(self, learning_rate=0.01,
                 ridge_alpha=0.0,
                 lasso_alpha=0.0,
                 max_iter=1000,
                 converged_tol=1e-8):
        self.learning_rate = learning_rate
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.max_iter = max_iter
        self.convered_tol = converged_tol
        self.trained = False

    def train(self, trainX, trainY, to_print=False, display_step=50):
        fitted_param, tf_sess = fit_linear_regression(trainX, trainY,
                                                      learning_rate=self.learning_rate,
                                                      ridge_alpha=self.ridge_alpha,
                                                      lasso_alpha=self.lasso_alpha,
                                                      max_iter=self.max_iter,
                                                      display_step=display_step,
                                                      converged_tol=self.convered_tol,
                                                      to_print=to_print
                                                      )

        self.fitted_param = fitted_param
        self.tf_sess = tf_sess
        self.trained = True

    def predict(self, testX):
        if not self.trained:
            raise ModelNotTrainedError()
        sess = self.tf_sess['session']
        X = self.tf_sess['inputs']
        Y = self.tf_sess['outputs']
        return sess.run(Y, feed_dict={X: testX})


