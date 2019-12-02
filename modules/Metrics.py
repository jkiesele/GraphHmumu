
# Define custom metrics here and add them to the global_metrics_list dict (important!)
global_metrics_list = {}

import tensorflow as tf
import keras.backend as K
from tools import muP4


def resolutionImprovement(truth, pred, ptcenter=-1, returnOffset=False):
    
    correction_factors = pred[:,:3]
    gen_muon_p = truth[:,:3]
    recmup = correction_factors * truth[:,5:8]
    
    truep4 = muP4(truth)
    recop4 = muP4(truth[:,5:8])
    predp4 = muP4(recmup)
    
    truept = truep4.p()
    recopt = recop4.p()
    predpt = predp4.p()
    
    in_range_mask = None
    if ptcenter>0:
        in_range_mask = tf.where(tf.abs(truept - ptcenter) < 20, tf.zeros_like(truept) + 1., tf.zeros_like(truept) )
    else:
        in_range_mask = tf.zeros_like(truept) + 1.
    n_in_range = tf.cast(tf.count_nonzero(in_range_mask, axis=-1), dtype='float32')
    
        
    # mean/variance can be over full range because it will cancel in final ratui
    if returnOffset:
        recores = tf.reduce_mean(in_range_mask*recopt/truept, axis=0)
        predres = tf.reduce_mean(in_range_mask*predpt/truept, axis=0)
        return (predres / recores)*100.
    else:
        rmean = tf.reduce_mean(in_range_mask*recopt/truept, axis=0)
        ros   = tf.math.reduce_variance(in_range_mask* recopt/truept , axis=0)/rmean
        pmean = tf.reduce_mean(in_range_mask*predpt/truept, axis=0)
        pos   = tf.math.reduce_variance(in_range_mask* predpt/truept , axis=0)/pmean
        
        return (pos/ros)*100.
        

global_metrics_list['resolutionImprovement'] = resolutionImprovement

def resolutionImprovement20(truth, pred):
    return resolutionImprovement(truth,pred,20)
global_metrics_list['resolutionImprovement20'] = resolutionImprovement20

def resolutionImprovement70(truth, pred):
    return resolutionImprovement(truth,pred,70)
global_metrics_list['resolutionImprovement70'] = resolutionImprovement70

def resolutionImprovement150(truth, pred):
    return resolutionImprovement(truth,pred,150)
global_metrics_list['resolutionImprovement150'] = resolutionImprovement150


def resolutionImprovementOS20(truth, pred):
    return resolutionImprovement(truth,pred,20,True)
global_metrics_list['resolutionImprovementOS20'] = resolutionImprovement20

def resolutionImprovementOS70(truth, pred):
    return resolutionImprovement(truth,pred,70,True)
global_metrics_list['resolutionImprovementOS70'] = resolutionImprovement70

def resolutionImprovementOS150(truth, pred):
    return resolutionImprovement(truth,pred,150,True)
global_metrics_list['resolutionImprovementOS150'] = resolutionImprovement150


