import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed


def PAC2VI(dataSource=tf.keras.datasets.fashion_mnist, NPixels=14, algorithm=0, PARTICLES=20, batch_size=100, num_epochs=50, num_hidden_units=20):
    """ Run experiments for MAP, Variational, PAC^2-Variational and PAC^2_T-Variational algorithms for the self-supervised classification task with a Normal data model.
        Args:
            dataSource: The data set used in the evaluation.
            NPixels: The size of the images: NPixels\times NPixels.
            algorithm: Integer indicating the algorithm to be run.
                0- MAP Learning
                1- Variational Learning
                2- PAC^2-Variational Learning
                3- PAC^2_T-Variational Learning
            PARTICLES: Number of Monte-Carlo samples used to compute the posterior prediction distribution.
            batch_size: Size of the batch.
            num_epochs: Number of epochs.
            num_hidden_units: Number of hidden units in the MLP.
        Returns:
            NLL: The negative log-likelihood over the test data set.
    """

    np.random.seed(1)
    tf.set_random_seed(1)

    sess = tf.Session()

    (x_train, y_train), (x_test, y_test) = dataSource.load_data()
    if (dataSource.__name__.__contains__('cifar')):
        x_train=sess.run(tf.cast(tf.squeeze(tf.image.rgb_to_grayscale(x_train)),dtype=tf.float32))
        x_test=sess.run(tf.cast(tf.squeeze(tf.image.rgb_to_grayscale(x_test)),dtype=tf.float32))

    NPixels = np.int(NPixels/2)

    y_train = x_train[:, NPixels:]
    x_train = x_train[:, 0:NPixels]

    y_test = x_test[:, NPixels:]
    x_test = x_test[:, 0:NPixels]

    NPixels= NPixels * NPixels * 2

    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train / 255.0, y_test / 255.0




    N = x_train.shape[0]
    M = batch_size

    x_batch = tf.placeholder(dtype=tf.float32, name="x_batch", shape=[None, NPixels])
    y_batch = tf.placeholder(dtype=tf.float32, name="y_batch", shape=[None, NPixels])

    def model(NHIDDEN, x):
        W = ed.Normal(loc=tf.zeros([NPixels, NHIDDEN]), scale=1., name="W")
        b = ed.Normal(loc=tf.zeros([1, NHIDDEN]), scale=1., name="b")

        W_out = ed.Normal(loc=tf.zeros([NHIDDEN, NPixels]), scale=1., name="W_out")
        b_out = ed.Normal(loc=tf.zeros([1, NPixels]), scale=1., name="b_out")

        hidden_layer = tf.nn.relu(tf.matmul(x, W) + b)
        out = tf.matmul(hidden_layer, W_out) + b_out
        y = ed.Normal(loc=out, scale=1./255,name="y")

        return W, b, W_out, b_out, x, y



    def qmodel(NHIDDEN):
        W_loc = tf.Variable(tf.random_normal([NPixels, NHIDDEN], 0.0, 0.1, dtype=tf.float32))
        b_loc = tf.Variable(tf.random_normal([1, NHIDDEN], 0.0, 0.1, dtype=tf.float32))

        if algorithm==0:
            W_scale = 0.000001
            b_scale = 0.000001
        else:
            W_scale = tf.nn.softplus(tf.Variable(tf.random_normal([NPixels, NHIDDEN], -3., stddev=0.1, dtype=tf.float32)))
            b_scale = tf.nn.softplus(tf.Variable(tf.random_normal([1, NHIDDEN], -3., stddev=0.1, dtype=tf.float32)))

        qW = ed.Normal(W_loc, scale=W_scale, name="W")
        qW_ = ed.Normal(W_loc, scale=W_scale, name="W")

        qb = ed.Normal(b_loc, scale=b_scale, name="b")
        qb_ = ed.Normal(b_loc, scale=b_scale, name="b")

        W_out_loc = tf.Variable(tf.random_normal([NHIDDEN, NPixels], 0.0, 0.1, dtype=tf.float32))
        b_out_loc = tf.Variable(tf.random_normal([1, NPixels], 0.0, 0.1, dtype=tf.float32))
        if algorithm==0:
            W_out_scale = 0.000001
            b_out_scale = 0.000001
        else:
            W_out_scale = tf.nn.softplus(tf.Variable(tf.random_normal([NHIDDEN, NPixels], -3., stddev=0.1, dtype=tf.float32)))
            b_out_scale = tf.nn.softplus(tf.Variable(tf.random_normal([1, NPixels], -3., stddev=0.1, dtype=tf.float32)))

        qW_out = ed.Normal(W_out_loc, scale=W_out_scale, name="W_out")
        qb_out = ed.Normal(b_out_loc, scale=b_out_scale, name="b_out")

        qW_out_ = ed.Normal(W_out_loc, scale=W_out_scale, name="W_out")
        qb_out_ = ed.Normal(b_out_loc, scale=b_out_scale, name="b_out")

        return qW, qW_, qb, qb_, qW_out, qW_out_, qb_out, qb_out_


    W,b,W_out,b_out,x,y = model(num_hidden_units, x_batch)

    qW,qW_,qb,qb_,qW_out,qW_out_,qb_out,qb_out_ = qmodel(num_hidden_units)

    with ed.interception(ed.make_value_setter(W=qW,b=qb,W_out=qW_out,b_out=qb_out)):
        pW,pb,pW_out,pb_out,px,py = model(num_hidden_units, x)

    with ed.interception(ed.make_value_setter(W=qW_,b=qb_,W_out=qW_out_,b_out=qb_out_)):
        pW_,pb_,pW_out_,pb_out_,px_,py_ = model(num_hidden_units, x)


    pylogprob = tf.expand_dims(tf.reduce_sum(py.distribution.log_prob(y_batch),axis=1),1)
    py_logprob = tf.expand_dims(tf.reduce_sum(py_.distribution.log_prob(y_batch),axis=1),1)

    logmax = tf.stop_gradient(tf.math.maximum(pylogprob,py_logprob)+0.1)
    logmean_logmax = tf.math.reduce_logsumexp(tf.concat([pylogprob-logmax,py_logprob-logmax], 1),axis=1) - tf.log(2.)
    alpha = tf.expand_dims(logmean_logmax,1)

    if (algorithm==3):
        hmax = 2*tf.stop_gradient(alpha/tf.math.pow(1-tf.math.exp(alpha),2) + tf.math.pow(tf.math.exp(alpha)*(1-tf.math.exp(alpha)),-1))
    else:
        hmax=1.

    var = 0.5*(tf.reduce_mean(tf.exp(2*pylogprob-2*logmax)*hmax) - tf.reduce_mean(tf.exp(pylogprob + py_logprob - 2*logmax)*hmax))


    datalikelihood = tf.reduce_mean(pylogprob)


    logprior = tf.reduce_sum(pW.distribution.log_prob(pW.value)) + \
             tf.reduce_sum(pb.distribution.log_prob(pb.value)) + \
             tf.reduce_sum(pW_out.distribution.log_prob(pW_out.value)) + \
             tf.reduce_sum(pb_out.distribution.log_prob(pb_out.value))


    entropy = tf.reduce_sum(qW.distribution.log_prob(qW.value)) + \
              tf.reduce_sum(qb.distribution.log_prob(qb.value)) + \
              tf.reduce_sum(qW_out.distribution.log_prob(qW_out.value)) + \
              tf.reduce_sum(qb_out.distribution.log_prob(qb_out.value))

    entropy = -entropy

    KL = (- entropy - logprior)/N

    if (algorithm==2 or algorithm==3):
        elbo = datalikelihood + var - KL
    elif algorithm == 1:
        elbo = datalikelihood - KL
    elif algorithm == 0:
        elbo = datalikelihood + logprior/N

    verbose=True
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
            t.append(-value)
            if verbose:
                #if j % 1 == 0: print(".", end="", flush=True)
                if i%50==0 and j%1000==0:
                #if j >= 5 :
                    print("\nEpoch: " + str(i))
                    str_elbo = str(t[-1])
                    print("\n" + str(j) + " epochs\t" + str_elbo, end="", flush=True)
                    print("\n" + str(j) + " data\t" + str(sess.run(datalikelihood,feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    print("\n" + str(j) + " var\t" + str(sess.run(var,feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    print("\n" + str(j) + " KL\t" + str(sess.run(KL,feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    print("\n" + str(j) + " energy\t" + str(sess.run(logprior,feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    print("\n" + str(j) + " entropy\t" + str(sess.run(entropy,feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    print("\n" + str(j) + " hmax\t" + str(sess.run(tf.reduce_mean(hmax),feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    print("\n" + str(j) + " alpha\t" + str(sess.run(tf.reduce_mean(alpha),feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)
                    print("\n" + str(j) + " logmax\t" + str(sess.run(tf.reduce_mean(logmax),feed_dict={x_batch: batch_x, y_batch: batch_y})), end="", flush=True)





    M=1000


    N=x_test.shape[0]
    x_batches = np.array_split(x_test, N / M)
    y_batches = np.array_split(y_test, N / M)

    NLL = 0

    for j in range(N // M):
        batch_x = np.reshape(x_batches[j], [x_batches[j].shape[0], -1]).astype(np.float32)
        batch_y = np.reshape(y_batches[j], [y_batches[j].shape[0],-1]).astype(np.float32)
        y_pred_list = []
        for i in range(PARTICLES):
            y_pred_list.append(sess.run(pylogprob,feed_dict={x_batch: batch_x, y_batch: batch_y}))
        y_preds = np.concatenate(y_pred_list, axis=1)
        score = tf.reduce_sum(tf.math.reduce_logsumexp(y_preds,axis=1)-tf.log(np.float32(PARTICLES)))
        score = sess.run(score)
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
text_file = open("./results/output-PAC2-Variational-SelfSupervisedNormal.txt", "w")

text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=0, PARTICLES=1, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=1, PARTICLES=20, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=2, PARTICLES=20, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.fashion_mnist, NPixels=28, algorithm=3, PARTICLES=20, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()

text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=0, PARTICLES=1, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=1, PARTICLES=20, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=2, PARTICLES=20, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()
text_file.write(str(PAC2VI(dataSource= tf.keras.datasets.cifar10, NPixels=32, algorithm=3, PARTICLES=20, batch_size=batch, num_epochs=iter, num_hidden_units= 20)) + "\n")
text_file.flush()

text_file.close()

