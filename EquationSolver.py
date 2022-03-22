import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import numpy as np
import time

from DataSet import DataSet
from net import Net
from ModelTrain import Train
from Plotting import Plotting

np.random.seed(1234)
tf.set_random_seed(1234)

def main():
    y_range = np.array((-np.pi, np.pi))
    x_range = np.array((-np.pi, np.pi))
    t_range = np.array((0., 2.))
    NX = 30
    Ny = 30
    Nt = 10
    N_bc = 30

    data = DataSet(x_range, y_range, t_range, NX, Ny, Nt, N_bc)
    # input data
    x_data, y_data, t_data, x_ini, y_ini, t_ini, x_b, y_b, t_b, u_b, v_b, rou_b, u_ini, v_ini, rou_ini = data.Data_Generation()
    # size of the DNN
    layers_eq = [3] + 4 * [50] + [3]
    layers_neq = [3] + 4 * [50] + [9]
    # definition of placeholder
    [x_train, y_train, t_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
    [x_ini_train, y_ini_train, t_ini_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
    [x_bc_train, y_bc_train, t_bc_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
    [rou_train, u_train, v_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
    [rou_ini_train, u_ini_train, v_ini_train] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
    # definition of nn
    net_eq = Net(x_data, y_data, t_data, layers=layers_eq)
    net_neq = Net(x_data, y_data, t_data, layers=layers_neq)

    [rou_pre, u_pre, v_pre] = net_eq(x_train, y_train, t_train)
    fneq_pre = net_neq(x_train, y_train, t_train) / 1e4

    [rou_bc_pre, u_bc_pre, v_bc_pre] = net_eq(x_bc_train, y_bc_train, t_bc_train)
    fneq_bc_pre = net_neq(x_bc_train, y_bc_train, t_bc_train) / 1e4

    [rou_ini_pre, u_ini_pre, v_ini_pre] = net_eq(x_ini_train, y_ini_train, t_ini_train)
    fneq_ini_pre = net_neq(x_ini_train, y_ini_train, t_ini_train) / 1e4

    bgk = data.bgk(fneq_pre, rou_pre, u_pre, v_pre, x_train, y_train, t_train)
    bgk_bc = data.bgk(fneq_bc_pre, rou_bc_pre, u_bc_pre, v_bc_pre, x_bc_train, y_bc_train, t_bc_train)
    fneq_bc = data.fBC(fneq_bc_pre, rou_train, u_train, v_train, x_b, y_b, t_b)

    Eq_res = data.Eq_res(fneq_pre, rou_pre, u_pre, v_pre, x_train, y_train, t_train)

    # loss
    loss = tf.reduce_mean(tf.square(u_ini_pre - u_ini_train)) + \
           tf.reduce_mean(tf.square(v_ini_pre - v_ini_train)) + \
           tf.reduce_mean(tf.square(rou_ini_pre - rou_ini_train)) + \
           tf.reduce_mean(tf.square(u_bc_pre - u_train)) + \
           tf.reduce_mean(tf.square(v_bc_pre - v_train)) + \
           tf.reduce_mean(tf.square(rou_bc_pre - rou_train)) + \
           tf.reduce_mean(bgk) + \
           tf.reduce_mean(bgk_bc) + \
           tf.reduce_mean(fneq_bc) * 1e8
    # tf.reduce_mean(fneq_ini_pre - fneq_ini_train) + \

    start_lr = 1e-3
    learning_rate = tf.train.exponential_decay(start_lr, global_step=5e4, decay_rate=1-5e-3, decay_steps=500)
    train_adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                         method="L-BFGS-B",
                                                         options={'maxiter': 50000,
                                                                  'maxfun': 70000,
                                                                  'maxcor': 100,
                                                                  'maxls': 100,
                                                                  'ftol': 10.0 * np.finfo(float).eps
                                                                  }
                                                         )
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tf_dict = {x_train: x_data, y_train: y_data, t_train: t_data,
               x_bc_train: x_b, y_bc_train: y_b, t_bc_train: t_b,
               x_ini_train: x_ini, y_ini_train: y_ini, t_ini_train: t_ini,
               u_train: u_b, v_train: v_b, rou_train: rou_b,
               u_ini_train: u_ini, v_ini_train: v_ini, rou_ini_train: rou_ini}

    Model = Train(tf_dict)
    start_time = time.perf_counter()
    Model.ModelTrain(sess, loss, train_adam, train_lbfgs)
    #Model.LoadModel(sess)
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds' % (stop_time - start_time))

    NX_test = 125
    NY_test = 125
    Plotter = Plotting(x_range, NX_test, y_range, NY_test, t_range, Nt, sess)
    Plotter.Saveplot(u_pre, v_pre, fneq_pre, Eq_res, x_train, y_train, t_train)

if __name__ == '__main__':
    main()
