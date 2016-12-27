---
layout: post
title: A Gentle Guide to Using Batch Normalization in Tensorflow
disqus: y
share: y
visible: 0
---

I recently made the switch to TensorFlow and am very happy with how easy it was to [get things done](https://github.com/RuiShu/vae-clustering) using this awesome library. Tensorflow has come a long way since I first [experimented with it](https://github.com/RuiShu/tensorflow-gp) in 2015, and I am happy to be back. 

Since I am getting myself re-acquainted with TensorFlow, I decided that I should write a post about how to do batch normalization in TensorFlow. It's kind of weird that batch normalization still presents such a challenge for new TnesorFlow users, especially since TensorFlow comes with invaluable functions like [`tf.nn.moments`](https://github.com/tensorflow/tensorflow/blob/40dcfc6f9287d360eead23f58d63d9627c075dc5/tensorflow/g3doc/api_docs/python/functions_and_classes/shard1/tf.nn.moments.md), [`tf.nn.batch_normalization`](https://github.com/tensorflow/tensorflow/blob/40dcfc6f9287d360eead23f58d63d9627c075dc5/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.batch_normalization.md), and even [`tf.contrib.layers.batch_norm`](https://github.com/tensorflow/tensorflow/blob/40dcfc6f9287d360eead23f58d63d9627c075dc5/tensorflow/g3doc/api_docs/python/functions_and_classes/shard4/tf.contrib.layers.batch_norm.md). One would think that using batch normalization in TensorFlow will be a cinch. But alas, confusion still crops up [from time to time](https://github.com/tensorflow/tensorflow/issues/4361), and the devil really lies in the details.

### Batch Normalization The Easy Way

Perhaps the easiest way to use batch normalization would be to simply use the `tf.contrib.layers.batch_norm` layer. So let's give that a go!

{% highlight python %}
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
x = Placeholder((None, 784), name='x')
y = Placeholder((None, 10), name='y')
phase = Placeholder(None, tf.bool, name='phase')

with tf.name_scope('nn/layer1'):
    l = tf.nn.fully_connected(x, 100, scope='dense1')
    l = tf.contrib.layers.batch_norm(l, center=True, scale=True, 
                                     scope='bn1', is_training=phase)
    l = tf.nn.relu(l, 'relu1')
    
with tf.name_scope('nn/layer2'):
    l = tf.nn.fully_connected(l, 100, scope='dense1')
    l = tf.contrib.layers.batch_norm(l, center=True, scale=True, 
                                     scope='bn1', is_training=phase)
    l = tf.nn.relu(l, 'relu1')

with tf.variable_scope('nn/logit'):
    logit = Dense(l, 10, 'dense')

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(y, 1), 
        tf.argmax(logit, 1)), 'float32'))

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, y))
{% endhighlight %}

Because batch normalization behaves different during training versus test time, `tf.contrib.layers.batch_norm` has kindly enabled us to pass in a `tf.bool` placeholder as the `is_training` argument.

{% highlight python %}
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
history = []
train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

iterep = 500
for i in range(iterep * 100):
    x_train, y_train = mnist.train.next_batch(100)
    sess.run(train_step,
             feed_dict={'x:0': x_train, 'y:0': y_train, 'phase:0': 1})
    progbar(i, iterep)
    if (i + 1) %  iterep == 0:
        tr = sess.run([loss, accuracy], feed_dict={'x:0': mnist.train.images, 'y:0': mnist.train.labels, 'phase:0': 1})
        t = sess.run([loss, accuracy], feed_dict={'x:0': mnist.test.images, 'y:0': mnist.test.labels, 'phase:0': 0})
        history += [tr + t]
{% endhighlight %}

Unfortunately, if you look at the training verus test performance over time, it looks like we have done something *very* wrong. Indeed, if we go back and read the `tf.contrib.layers.batch_norm` documentation a little more carefully, there's a pretty important note:

>Note: When is_training is True the moving_mean and moving_variance need to be updated, by default the update_ops are placed in tf.GraphKeys.UPDATE_OPS so they need to be added as a dependency to the train_op, example:
>
>update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
>if update_ops: updates = tf.group(*update_ops) 
>total_loss = control_flow_ops.with_dependencies([updates], total_loss)

Aha!  
