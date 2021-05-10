#!/usr/bin/env python
# coding: utf-8
'''
Author: Bouhlel Mohamed Amine

This script runs the airfoil shape analysis test case in both subsonic and transonic regimes for the paper:
Scalable gradient-enhanced artificial neural networks for airfoil shape design in the subsonic and transonic regimes

To run this script, the user should:
 - Adapt the file paths
 - Create a repository called saveModel in the same repository containing this file
'''

import  random
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

# Load both transonic and subsonic data
X_subTr = np.loadtxt('XTr.dat')
X_subV = np.loadtxt('XV.dat')
y_subCdTr = (np.loadtxt('yTr.dat')).reshape((-1,1))
y_subCdV = (np.loadtxt('yV.dat')).reshape((-1,1))
dy_subCdTr = np.loadtxt('dyTr.dat')
X_transTr = np.loadtxt('X_transTr.dat')
X_transV = np.loadtxt('X_transV.dat')
y_transCdTr = (np.loadtxt('y_transCdTr.dat')).reshape((-1,1))
y_transCdV = (np.loadtxt('y_transV.dat')).reshape((-1,1))
dy_transCdTr = np.loadtxt('dy_transCdTr.dat')

# Read dimensions of the data
nSubTr = X_subTr.shape[0]
nSubV = X_subV.shape[0]
dimSub = X_subV.shape[1]
nTransTr = X_transTr.shape[0]
nTransV = X_transV.shape[0]
dimTrans = X_transV.shape[1]

# Define the new design space
XTr = np.zeros((nTransTr+nSubTr,dimTrans + dimSub - 2))
XTr[:nTransTr,:dimTrans-2] = X_transTr[:,:dimTrans-2]
XTr[nTransTr:,dimTrans-2:] = X_subTr
XTr[:nTransTr,-2:] = X_transTr[:,-2:]
XV = np.zeros((nTransV+nSubV,dimTrans + dimSub - 2))
XV[:nTransV,:dimTrans-2] = X_transV[:,:dimTrans-2]
XV[nTransV:,dimTrans-2:] = X_subV
XV[:nTransV,-2:] = X_transV[:,-2:]
yTr = np.zeros((nSubTr+nTransTr,1))
yTr[:nTransTr,0] = y_transCdTr.T
yTr[nTransTr:,0] = y_subCdTr.T
yV = np.zeros((nSubV+nTransV,1))
yV[:nTransV,0] = y_transCdV.T
yV[nTransV:,0] = y_subCdV.T
dyTr = np.zeros((nTransTr+nSubTr,dimTrans + dimSub - 2))
dyTr[:nTransTr,:dimTrans-2] = dy_transCdTr[:,:dimTrans-2]
dyTr[nTransTr:,dimTrans-2:] = dy_subCdTr
dyTr[:nTransTr,-2:] = dy_transCdTr[:,-2:]
dyV = np.zeros((nTransV+nSubV,dimTrans + dimSub - 2))

# Set up NN parameters
INPUT_DIM = dimTrans+dimSub-2
OUTPUT_DIM = 1
NUM_SAMPLES =  int(XTr.shape[0]/3)
NUM_TRAINING = XTr.shape[0]
NUM_VALIDATING = XV.shape[0]
NUM_HIDDEN = 100

# Define the SN class
class SobolevNetwork:
    def __init__(self, input_dim, num_hidden,init = None):
        self.input_dim = input_dim 
        self.num_hidden = num_hidden
        self.W1 = tf.Variable(tf.random_normal([self.input_dim, self.num_hidden],stddev=0.1))
        self.b1 = tf.Variable(tf.ones([self.num_hidden]))
        self.W2 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden],stddev=0.1))
        self.b2 = tf.Variable(tf.ones([self.num_hidden]))
        self.W3 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden],stddev=0.1))
        self.b3 = tf.Variable(tf.ones([self.num_hidden]))
        self.W4 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden],stddev=0.1))
        self.b4 = tf.Variable(tf.ones([self.num_hidden]))
        self.W5 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden],stddev=0.1))
        self.b5 = tf.Variable(tf.ones([self.num_hidden]))
        self.W6 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden],stddev=0.1))
        self.b6 = tf.Variable(tf.ones([self.num_hidden]))        
        self.W7 = tf.Variable(tf.random_normal([self.num_hidden, 1],stddev=0.1))
        self.b7 = tf.Variable(tf.ones([1]))
        self.weights = [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3),(self.W4, self.b4), (self.W5, self.b5), (self.W6, self.b6),(self.W7, self.b7)]
        

    def forward(self, X):
        #Input layer
        out = X
        #Hidden layers
        W,b = self.weights[0]
        out = tf.nn.tanh(tf.matmul(out, W) + b)
        W,b = self.weights[1]
        out = tf.nn.tanh(tf.matmul(out, W) + b)
        W,b = self.weights[2]
        out = tf.nn.sigmoid(tf.matmul(out, W) + b)
        W,b = self.weights[3]
        out = tf.nn.sigmoid(tf.matmul(out, W) + b)
        W,b = self.weights[4]
        out = tf.nn.leaky_relu(tf.matmul(out, W) + b)
        W,b = self.weights[5]
        out = tf.nn.relu(tf.matmul(out, W) + b)
        #Output layer
        W,b = self.weights[-1]
        out = tf.matmul(out, W) + b
        return out

# Define tensors and operations
X = tf.placeholder(tf.float32, shape=[None, INPUT_DIM],name='X')
y = tf.placeholder(tf.float32, shape=[None,OUTPUT_DIM],name='y')
y_der = tf.placeholder(tf.float32, shape=[None, INPUT_DIM],name='dydX')

model = SobolevNetwork(INPUT_DIM, NUM_HIDDEN) 
y_p = model.forward(X)
predict_named = tf.identity(y_p, "prediction")
dy = tf.gradients(y_p, X)
predict_named = tf.identity(dy, "gradient")
optimizer = tf.train.AdamOptimizer()

yLossLambda = dict()
for lam in range(11):
    yLossLambda[lam] = tf.reduce_mean(tf.pow(y_p - tf.reshape(y,[NUM_SAMPLES,1]), 2) + lam/10. * tf.reshape(tf.reduce_sum(tf.pow(dy - y_der, 2),2),[NUM_SAMPLES,1]))

# Train storage
optim = dict()
timer = dict()
resTrain = dict()
for lam in range(11):
    optim[lam] = optimizer.minimize(yLossLambda[lam])
    timer[lam] = []
    resTrain[lam] = []

# Validation storage
resValid = []
resValid1 = []
resValid2 = []

# Saved model storage
saver = tf.train.Saver(max_to_keep=100)
errModel = np.zeros((1000,1000))

# Sobolev optimizer
tstart = time.time()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

X_valid = XV;y_valid = yV;dy_valid = dyV
valid_dict = {X: X_valid, y: y_valid, y_der: dy_valid}    
X_valid1 = XV[:nTransV,:];y_valid1 = yV[:nTransV,:];dy_valid1 = dyV[:nTransV,:]
valid_dict1 = {X: X_valid1, y: y_valid1, y_der: dy_valid1}
X_valid2 = XV[nTransV:,:];y_valid2 = yV[nTransV:,:];dy_valid2 = dyV[nTransV:,:]
valid_dict2 = {X: X_valid2, y: y_valid2, y_der: dy_valid2}
best_sub = 1000
best_trans = 1000
iter = 0
for epoch_num in range(1000):
    batch_samples = random.sample(xrange(NUM_TRAINING),NUM_SAMPLES)
    X_train = XTr[batch_samples,:]
    y_train = yTr[batch_samples,:]
    dy_train = dyTr[batch_samples,:]
    train_dict = {X: X_train, y: y_train, y_der: dy_train}

    for lam in range(1,11):
        print 'lam: ', lam/10.
        t1 = time.time()
        _, current_loss, ytrain_loss, ydytrain_loss = sess.run([optim[lam],yLossLambda[lam], yLossLambda[0],yLossLambda[10]], feed_dict=train_dict)
        iter += 1
        timer[lam].append(time.time()-t1)
        resTrain[lam].append(current_loss)
        resTrain[0].append(ytrain_loss)
        resTrain[10].append(ydytrain_loss)
        ypreds = sess.run([y_p], feed_dict=valid_dict)
        re = np.linalg.norm(ypreds-y_valid)/np.linalg.norm(y_valid)*100
        resValid.append(re)
        ypreds1 = sess.run([y_p], feed_dict=valid_dict1)
        re1 = np.linalg.norm(ypreds1-y_valid1)/np.linalg.norm(y_valid1)*100
        resValid1.append(re1)
        ypreds2 = sess.run([y_p], feed_dict=valid_dict2)
        re2 = np.linalg.norm(ypreds2-y_valid2)/np.linalg.norm(y_valid2)*100
        resValid2.append(re2)
        if re1 < best_trans and re2 < best_sub:
            saver.save(sess, '/home/mbouhlel/hg/SBO/ANN/SubTrans/clear/old_modes_combine_trans_sub/final_run/cd/saveModel/my_test_model_'+str(iter))
            best_trans = re1
            best_sub = re2
        print("Epoch: %d,Current Loss: %f , ydy Loss: %f, y Loss: %f" % (epoch_num,current_loss,ydytrain_loss,ytrain_loss*1e4))
        print("y valid: %f, yT valid %f,yS valid %f" % (re,re1,re2))

        for i in range(10):
            t1 = time.time()
            _, ytrain_loss, ydytrain_loss = sess.run([optim[0],yLossLambda[0],yLossLambda[10]], feed_dict=train_dict)
            iter += 1
            timer[0].append(time.time()-t1)
            resTrain[0].append(ytrain_loss)
            resTrain[10].append(ydytrain_loss)
            ypreds = sess.run([y_p], feed_dict=valid_dict)
            re = np.linalg.norm(ypreds-y_valid)/np.linalg.norm(y_valid)*100
            resValid.append(re)
            ypreds1 = sess.run([y_p], feed_dict=valid_dict1)
            re1 = np.linalg.norm(ypreds1-y_valid1)/np.linalg.norm(y_valid1)*100
            resValid1.append(re1)
            ypreds2 = sess.run([y_p], feed_dict=valid_dict2)
            re2 = np.linalg.norm(ypreds2-y_valid2)/np.linalg.norm(y_valid2)*100
            resValid2.append(re2)
            if re1 < best_trans and re2 < best_sub:
                saver.save(sess, '/home/mbouhlel/hg/SBO/ANN/SubTrans/clear/old_modes_combine_trans_sub/final_run/cd/saveModel/my_test_model_'+str(iter))
                best_trans = re1
                best_sub = re2
            print("Epoch: %d,Current Loss: %f , ydy Loss: %f, y Loss: %f" % (epoch_num,current_loss,ydytrain_loss,ytrain_loss*1e4))
            print("y valid: %f, yT valid %f,yS valid %f" % (re,re1,re2))

            t1 = time.time()
            _, ytrain_loss, ydytrain_loss = sess.run([optim[10],yLossLambda[0],yLossLambda[10]], feed_dict=train_dict)
            iter += 1
            timer[10].append(time.time()-t1)
            resTrain[0].append(ytrain_loss)
            resTrain[10].append(ydytrain_loss)
            ypreds = sess.run([y_p], feed_dict=valid_dict)
            re = np.linalg.norm(ypreds-y_valid)/np.linalg.norm(y_valid)*100
            resValid.append(re)
            ypredsTr = sess.run([y_p], feed_dict=train_dict)
            ypreds1 = sess.run([y_p], feed_dict=valid_dict1)
            re1 = np.linalg.norm(ypreds1-y_valid1)/np.linalg.norm(y_valid1)*100
            resValid1.append(re1)
            ypreds2 = sess.run([y_p], feed_dict=valid_dict2)
            re2 = np.linalg.norm(ypreds2-y_valid2)/np.linalg.norm(y_valid2)*100
            resValid2.append(re2)
            if re1 < best_trans and re2 < best_sub:
                saver.save(sess, '/home/mbouhlel/hg/SBO/ANN/SubTrans/clear/old_modes_combine_trans_sub/final_run/cd/saveModel/my_test_model_'+str(iter))
                best_trans = re1
                best_sub = re2
            print("Epoch: %d,Current Loss: %f , ydy Loss: %f, y Loss: %f" % (epoch_num,current_loss,ydytrain_loss,ytrain_loss*1e4))
            print("y valid: %f, yT valid %f,yS valid %f" % (re,re1,re2))
    
tfinal1 = time.time() - tstart
print 'Final time: '+str(tfinal1)

with open('resValidCd.pkl', 'wb') as f:
   pickle.dump(resValid, f)

with open('resValidTransCd.pkl', 'wb') as f:
   pickle.dump(resValid1, f)

with open('resValidSubCd.pkl', 'wb') as f:
   pickle.dump(resValid2, f)
