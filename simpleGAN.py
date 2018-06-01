import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
import os
import time
#tf.enable_eager_execution()
res  = tf.executing_eagerly()  
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

t_begin = time.time()

if (res):
    print("Executing Eargerly")

gen_weights = dict()
gen_weights['w1'] = tf.Variable(tf.random_normal([1,5]))
gen_weights['b1'] = tf.Variable(tf.random_normal([5]))
gen_weights['w2'] = tf.Variable(tf.random_normal([5,1]))
gen_weights['b2'] = tf.Variable(tf.random_normal([1]))

disc_weights = dict()
disc_weights['w1'] = tf.Variable(tf.random_normal([1,10]))
disc_weights['b1'] = tf.Variable(tf.random_normal([10]))
disc_weights['w2'] = tf.Variable(tf.random_normal([10,10]))
disc_weights['b2'] = tf.Variable(tf.random_normal([10]))
disc_weights['w3'] = tf.Variable(tf.random_normal([10,1]))
disc_weights['b3'] = tf.Variable(tf.random_normal([1]))



z_p = tf.placeholder('float',[None,1])
x_d = tf.placeholder('float',[None,1])
g_h = tf.nn.softplus(tf.add(tf.matmul(z_p,gen_weights['w1']),gen_weights['b1']))
x_g = tf.add(tf.matmul(g_h,gen_weights['w2']),gen_weights['b2'])

#%%
def discriminator(x):
    d_h1 = tf.nn.tanh(tf.add(tf.matmul(x, disc_weights['w1']),disc_weights['b1']))
    d_h2 = tf.nn.tanh(tf.add(tf.matmul(d_h1,disc_weights['w2']),disc_weights['b2']))
    score = tf.nn.sigmoid(tf.add(tf.matmul(d_h2,disc_weights['w3']),disc_weights['b3']))
    return score
    
x_data_score = discriminator(x_d)
x_gen_score = discriminator(x_g)


D_cost = -tf.reduce_mean(tf.log(x_data_score) + tf.log(1.0 - x_gen_score))
G_cost = tf.reduce_mean(tf.log(1.0 - x_gen_score))

#Оптимизаторы в TensorFlow всегда решают задачу МИНИМИЗАЦИИ


batch_size =  64
updates = 40000
learning_rate = 0.01
prior_mu = -2.5
prior_std = 0.5
noise_range = 5.0
#%%
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
D_optimizer =  optimizer.minimize(D_cost,var_list = list(disc_weights.values()))
G_optimizer = optimizer.minimize(G_cost, var_list =list(gen_weights.values()))

#%%
def sample_z(size = batch_size):
    return np.random.uniform(-noise_range,noise_range,size = [size,1])

def sample_x(size = batch_size,mu = prior_mu,std = prior_std ):
    return np.random.normal(mu,std,size=[size,1])   
    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for  i in range(updates):
    z_batch = sample_z()
    x_batch = sample_x()
    sess.run(D_optimizer,feed_dict = {z_p:z_batch,x_d:x_batch})
    z_batch = sample_z()
    sess.run(G_optimizer,feed_dict = {z_p:z_batch})


t_end = time.time()
print("end, time = ", t_end-t_begin)
    
#saver = tf.train.Saver()
#saves = saver.save(sess,'model.ckpt')