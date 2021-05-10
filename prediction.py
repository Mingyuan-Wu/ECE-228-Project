#!/usr/bin/env python
# coding: utf-8
'''
Author: Bouhlel Mohamed Amine

This script runs prediction of airfoil shape analysis test case in both subsonic and transonic regimes for the paper:
Scalable gradient-enhanced artificial neural networks for airfoil shape design in the subsonic and transonic regimes

To run this script, the user should:
 - Adapt the file paths
 - Put the number containing the last model
'''

import numpy as np
import tensorflow as tf

# Load both transonic and subsonic data
X_subV = np.loadtxt('XV.dat')
X_subT = np.loadtxt('XT.dat')
y_subV = (np.loadtxt('yV.dat')).reshape((-1,1))
dy_subV = (np.loadtxt('dyV.dat'))

y_subT = (np.loadtxt('yT.dat')).reshape((-1,1))
X_transV = np.loadtxt('X_transV.dat')
X_transT = np.loadtxt('X_transT.dat')
y_transV = (np.loadtxt('y_transV.dat')).reshape((-1,1))
dy_transV = (np.loadtxt('dy_transCdV.dat'))
y_transT = (np.loadtxt('y_transT.dat')).reshape((-1,1))

# Read dimensions of the data
nSubV = X_subV.shape[0]
nSubT = X_subT.shape[0]
dimSub = X_subV.shape[1]
nVTrans = X_transV.shape[0]
nTTrans = X_transT.shape[0]
dimTrans = X_transV.shape[1]

# Define the new design space
XV = np.zeros((nSubV+nVTrans,dimTrans + dimSub - 2))
XV[:nVTrans,:dimTrans-2] = X_transV[:,:dimTrans-2]
XV[nVTrans:,dimTrans-2:] = X_subV
XV[:nVTrans,-2:] = X_transV[:,-2:]
XT = np.zeros((nSubT+nTTrans,dimTrans + dimSub - 2))
XT[:nTTrans,:dimTrans-2] = X_transT[:,:dimTrans-2]
XT[nTTrans:,dimTrans-2:] = X_subT
XT[:nTTrans,-2:] = X_transT[:,-2:]

yV = np.zeros((nSubV+nVTrans,1))
yV[:nVTrans,0] = y_transV.T
yV[nVTrans:,0] = y_subV.T

yT = np.zeros((nSubT+nTTrans,1))
yT[:nTTrans,0] = y_transT.T
yT[nTTrans:,0] = y_subT.T

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('/home/mbouhlel/hg/SBO/ANN/SubTrans/clear/old_modes_combine_trans_sub/final_run/cd/saveModel/my_test_model_204763.meta',clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint('/home/mbouhlel/hg/SBO/ANN/SubTrans/clear/old_modes_combine_trans_sub/final_run/cd/saveModel/'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
y = graph.get_tensor_by_name("y:0")
y_der = graph.get_tensor_by_name("dydX:0")

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("prediction:0")
op_to_restore1 = graph.get_tensor_by_name("gradient:0")

valid_dict = {X: XV}
ypV = sess.run(op_to_restore,feed_dict=valid_dict)
dypV = sess.run(op_to_restore1,feed_dict=valid_dict)

yVTrans = yV[:nVTrans,0].reshape((-1,1))
yVSub = yV[nVTrans:,0].reshape((-1,1))
ypVTrans = ypV[:nVTrans,0].reshape((-1,1))
ypVSub = ypV[nVTrans:,0].reshape((-1,1))
reV = np.linalg.norm(ypV-yV)/np.linalg.norm(yV)*100
print reV
reVSub = np.linalg.norm(ypVSub-yVSub)/np.linalg.norm(yVSub)*100
print reVSub
reVTrans = np.linalg.norm(ypVTrans-yVTrans)/np.linalg.norm(yVTrans)*100
print reVTrans

np.savetxt('ypVTransCd.dat',ypVTrans)
np.savetxt('ypVSubCd.dat',ypVSub)
test_dict = {X: XT}
ypT = sess.run(op_to_restore,feed_dict=test_dict)

yTTrans = yT[:nTTrans,0].reshape((-1,1))
yTSub = yT[nTTrans:,0].reshape((-1,1))
ypTTrans = ypT[:nTTrans,0].reshape((-1,1))
ypTSub = ypT[nTTrans:,0].reshape((-1,1))
reT = np.linalg.norm(ypT-yT)/np.linalg.norm(yT)*100
print reT
reTSub = np.linalg.norm(ypTSub-yTSub)/np.linalg.norm(yTSub)*100
print reTSub
reTTrans = np.linalg.norm(ypTTrans-yTTrans)/np.linalg.norm(yTTrans)*100
print reTTrans

np.savetxt('ypTTransCd.dat',ypTTrans)
np.savetxt('ypTSubCd.dat',ypTSub)
