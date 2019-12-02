
import tensorflow as tf
import keras.backend as K

class muP4(object):
    def __init__(self, predortruth):
        '''
        truth: genmup,muonp, B x 2xF, F = 5, px, py, pz, eta, phi
        '''
        self.px = predortruth[:,0]
        self.py = predortruth[:,1]
        self.pz = predortruth[:,2]
        self.E  = tf.sqrt(self.px**2 + self.py**2 + self.pz**2 + 0.1)
        
    def p(self):
        return tf.sqrt(self.px**2 + self.py**2 + self.pz**2 + K.epsilon())
    
    def pt(self):
        return tf.sqrt(self.px**2 + self.py**2 + K.epsilon())