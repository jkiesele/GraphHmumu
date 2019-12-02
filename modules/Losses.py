
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

import tensorflow as tf
import keras.backend as K
from tools import muP4

def muon_loss(truth, pred):
    '''
    truth: genmup,muonp, B x 2xF, F = 5, px, py, pz, eta, phi
    pred: B x 3*2, correction factors and confidence
    
    '''
    
    correction_factors = pred[:,:3]
    
    #to be used later
    confidence = pred[:,3:]
    
    gen_muon_p = truth[:,:3]
    recmup = truth[:,5:8]
    
    #gen_muon_p= tf.Print(gen_muon_p,[gen_muon_p],'gen_muon_p ')
    #recmup= tf.Print(recmup,[recmup],'recmup ')
    
    p = muP4(gen_muon_p)
    pt = p.pt()
    
    #correction_factors = tf.Print(correction_factors,[correction_factors],'correction_factors ')
    
    weight = tf.exp(- 2.71828182 * tf.log(0.02 * pt)**2)
    
    pred_reco_muon_p = correction_factors*recmup
    
    
    #pred_reco_muon_p= tf.Print(pred_reco_muon_p,[pred_reco_muon_p],'pred_reco_muon_p ')
    
    #maybe we wanna weight the contributions differntly?
    rel_diff = (gen_muon_p - pred_reco_muon_p)**2  / (gen_muon_p + 5.)**2 # / confidence + sum(confidence)
    
    c = 0.5
    
    rel_diff = tf.where(rel_diff > 0.1, 
                        tf.zeros_like(rel_diff)+0.1**2+tf.log(c*rel_diff-c*0.1 +K.epsilon()+1.),
                        rel_diff)
    
    #rel_diff = tf.Print(rel_diff,[rel_diff, tf.shape(rel_diff)],'rel_diff ')
    
    rel_diff = tf.reduce_mean(rel_diff, axis=-1)
    
    rel_diff *= weight
    
    loss = tf.reduce_mean(rel_diff)
    #loss = tf.Print(loss,[loss],'loss ')
    
    return  loss# + tf.reduce_mean(confidence)#just to keep it down
    
    
global_loss_list['muon_loss']=muon_loss