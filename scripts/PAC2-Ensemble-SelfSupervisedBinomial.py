import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import tensorflow_probability as tfp



def PAC2Ensemble(dataSource=tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=0, num_ensemble_models=20, batch_size=100, num_epochs=50, num_hidden_units=20):
    """ Run experiments for Ensemble, PAC^2-Ensemble and PAC^2_T-Ensemble algorithms for the self-supervised classification task with a Categorical data model.
        Args:
            dataSource: The data set used in the evaluation.
            NPixels: The size of the images: NPixels\times NPixels.
            algorithm: Integer indicating the algorithm to be run.
                0- Ensemble Learning [As derived for a first-order PAC-Bayes bound. No change in performance when using several models.]
                1- PAC^2-Ensemble Learning
                2- PAC^2_T-Ensemble Learning
            num_ensemble_models: Number of models in the ensemble.
            batch_size: Size of the batch.
            num_epochs: Number of epochs.
            num_hidden_units: Number of hidden units in the MLP.
        Returns:
            NLL: The negative log-likelihood over the test data set.
            :param algorithm:
            :param algorithm:
    """

    np.random.seed(1)
    tf.set_random_seed(1)

    K=num_ensemble_models

    sess = tf.Session()


    (x_train, y_train), (x_test, y_test) = dataSource.load_data()

    if (dataSource.__name__.__contains__('cifar')):
        x_train=sess.run(tf.cast(tf.squeeze(tf.image.rgb_to_grayscale(x_train)),dtype=tf.float32))
        x_test=sess.run(tf.cast(tf.squeeze(tf.image.rgb_to_grayscale(x_test)),dtype=tf.float32))

    x_train = (x_train < 128).astype(np.int32)
    x_test = (x_test < 128 ).astype(np.int32)

    NPixels = np.int(NPixels/2)

    y_train = x_train[:, NPixels:]
    x_train = x_train[:, 0:NPixels]

    y_test = x_test[:, NPixels:]
    x_test = x_test[:, 0:NPixels]

    NPixels= NPixels * NPixels * 2






    N = x_train.shape[0]
    M = batch_size

    x_batch = tf.placeholder(dtype=tf.float32, name="x_batch", shape=[None, NPixels])
    y_batch = tf.placeholder(dtype=tf.float32, name="y_batch", shape=[None, NPixels])


    def model(NHIDDEN, x):
        W = tf.Variable(tf.random_normal([NPixels, NHIDDEN], 0.0, 0.1, dtype=tf.float32, seed=1))
        b = tf.Variable(tf.random_normal([1, NHIDDEN], 0.0, 0.1, dtype=tf.float32, seed=1))

        W_out = tf.Variable(tf.random_normal([NHIDDEN, 2 * NPixels], 0.0, 0.1, dtype=tf.float32, seed=1))
        b_out = tf.Variable(tf.random_normal([1, 2 * NPixels], 0.0, 0.1, dtype=tf.float32, seed=1))

        hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
        out = tf.matmul(hidden_layer, W_out) + b_out
        y = ed.Categorical(logits=tf.reshape(out, [tf.shape(x_batch)[0], NPixels, 2]), name="y")

        ###Prior
        normal = tf.distributions.Normal(0., 1.)
        logpiror = tf.math.reduce_sum(normal.log_prob(W)) + \
                   tf.math.reduce_sum(normal.log_prob(b)) + \
                   tf.math.reduce_sum(normal.log_prob(W_out)) + \
                   tf.math.reduce_sum(normal.log_prob(b_out))

        return x, y, logpiror

    t = []
    tpy = []
    logprior = tf.constant(0.)
    for i in range(K):
        px,py, logp = model(num_hidden_units, x_batch)
        t.append(tf.expand_dims(tf.reduce_sum(py.distribution.log_prob(y_batch),axis=1),1))
        tpy.append(py)
        logprior = logprior + logp

    probs = tf.math.softmax(tf.Variable(tf.ones([K], dtype=tf.float32), trainable=False, name='probs'))


    ensemble = tf.concat(t,1)

    if K>1:
        max = tf.stop_gradient(tf.math.reduce_max(ensemble,axis=1))
        logmean = tf.stop_gradient(tf.math.reduce_logsumexp(ensemble + tf.reshape(tf.tile(tf.log(probs), [batch_size]), [batch_size, K]), axis=1) - tf.log(K + 0.0))
        varlist = []

        #####
        inc = logmean-max
        if (algorithm==2):
            hmax = 2*tf.stop_gradient(inc/tf.math.pow(1-tf.math.exp(inc),2) + tf.math.pow(tf.math.exp(inc)*(1-tf.math.exp(inc)),-1))
        else:
            hmax = 1.
        #####


        for i in range(K):
            vari = 0.5*(tf.reduce_mean(tf.exp(2*ensemble[:,i]-2*max)*hmax,axis=0))
            for j in range(K):
                vari = vari - 0.5*tf.reduce_sum(tf.reduce_mean(tf.exp(ensemble[:,i] + ensemble[:,j] - 2*max)*hmax,axis=0))*probs[j]
            varlist.append(vari)



        var=tf.stack(varlist,0)
    else:
        var=tf.constant(0.)

    dataenergy = tf.reduce_mean(ensemble,axis=0)

    if (algorithm==1 or algorithm==2):
        elboEnsemble = dataenergy + var
        elbo = tf.reduce_sum(tf.math.multiply(elboEnsemble, probs))
        elbo = elbo + 2 * tf.reduce_sum(tf.math.multiply(probs, tf.log(probs)))/N + logprior/N
    elif (algorithm == 0):
        elboEnsemble = dataenergy
        elbo = tf.reduce_sum(tf.math.multiply(elboEnsemble,probs))
        elbo = elbo + tf.reduce_sum(tf.math.multiply(probs,tf.log(probs)))/N + logprior/N


    verbose=True
    sess = tf.Session()
    optimizer = tf.train.AdamOptimizer(0.001)
    t = []
    train = optimizer.minimize(-elbo)
    init = tf.global_variables_initializer()
    sess.run(init)




    for i in range(num_epochs+1):
        perm = np.random.permutation(N)
        x_train = np.take(x_train, perm, axis=0)
        y_train = np.take(y_train, perm, axis=0)

        x_batches = np.array_split(x_train, N / M)
        y_batches = np.array_split(y_train, N / M)

        for j in range(N // M):
            batch_x = np.reshape(x_batches[j], [x_batches[j].shape[0], -1]).astype(np.float32)
            batch_y = np.reshape(y_batches[j],[y_batches[j].shape[0],-1]).astype(np.float32)

            value, _ = sess.run([elbo, train],feed_dict={x_batch: batch_x, y_batch: batch_y})
            t.append(value)
            if verbose:
                if j % 1000 == 0: print(".", end="", flush=True)
                if i%50==0 and j % 1000 == 0:
                    print("\nEpoch: " + str(i))
                    str_elbo = str(-t[-1])
                    print("\n" + str(j) + " epochs\t" + str_elbo, end="", flush=True)
                    print("\n" + str(j) + " data\t" + str(sess.run(dataenergy,feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    if K>1: print("\n" + str(j) + " var\t" + str(sess.run(var,feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    if K>1: print("\n" + str(i) + " hmax\t" + str(sess.run(tf.reduce_mean(hmax),feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)




    M=1000
    N=x_test.shape[0]
    x_batches = np.array_split(x_test, N / M)
    y_batches = np.array_split(y_test, N / M)

    NLL = 0

    for j in range(N // M):
        batch_x = np.reshape(x_batches[j], [x_batches[j].shape[0], -1]).astype(np.float32)
        batch_y = np.reshape(y_batches[j], [y_batches[j].shape[0], -1]).astype(np.float32)
        y_pred_list = []
        for i in range(K):
            y_pred_list.append(tf.expand_dims(tf.reduce_sum(tpy[i].distribution.log_prob(y_batch), axis=1), 1))
        y_preds = tf.concat(y_pred_list, axis=1)
        score = tf.reduce_sum(tf.math.reduce_logsumexp(y_preds, axis=1) - tf.log(K + 0.0))
        score = sess.run(score,feed_dict={x_batch: batch_x, y_batch: batch_y})
        NLL = NLL + score
        if verbose:
            if j % 1 == 0: print(".", end="", flush=True)
            if j % 1 == 0:
                str_elbo = str(score)
                print("\n" + str(j) + " epochs\t" + str_elbo, end="", flush=True)

    print("\nNLL: "+str(NLL))

    return NLL



iter=100
batch=100
text_file = open("./results/output-PAC2-Ensemble-SelfSupervisedBinomial.txt", "w")

text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=0, num_ensemble_models=1, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=1, num_ensemble_models=2, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=2, num_ensemble_models=2, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=1, num_ensemble_models=3, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=2, num_ensemble_models=3, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()


text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=0, num_ensemble_models=1, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=1, num_ensemble_models=2, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=2, num_ensemble_models=2, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=1, num_ensemble_models=3, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2Ensemble(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=2, num_ensemble_models=3, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()


text_file.close()