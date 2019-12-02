
import DeepJetCore
from DeepJetCore.training.training_base import training_base
#import keras

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization #etc
from Layers import MergeActiveHits, ReduceMeanVertices, GravNet_simple, GlobalExchange, TestLayer

from Metrics import resolutionImprovement, resolutionImprovement20, resolutionImprovement70, resolutionImprovement150, resolutionImprovementOS20, resolutionImprovementOS70, resolutionImprovementOS150

from Losses import muon_loss

from DeepJetCore.DJCLayers import ScalarMultiply


def my_model(Inputs,momentum=0.6):
    
    feat, mask, hitmatched = Inputs[0],Inputs[1],Inputs[2]
    
    x = Concatenate()([feat, mask, hitmatched])
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Dense(32, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(32, activation='elu')(x)
    x = BatchNormalization(momentum=momentum)(x)
    #x = TestLayer()(x)
    allx = []
    for i in range(8):
        x = GravNet_simple(n_neighbours =12, 
                    n_dimensions =3, 
                    n_filters    =64, 
                    n_propagate  =24
                    )(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Dropout(0.05)(x)
        allx.append(x)
      
    x = Concatenate()(allx)          
    x = Dense(32, activation='tanh')(x)
    
    #predict corr factor for px, py, pz, so 3
    # correction factor is 1 + c, so ranges between 0 and 2
    correction = ScalarMultiply(.1)(x)
    correction = Dense(3, activation = 'sigmoid', use_bias=False)(correction)
    correction = ScalarMultiply(2.)(correction)
    confidence = Dense(3, activation = 'sigmoid')(x)
    
    correction = ReduceMeanVertices()(correction)
    confidence = ReduceMeanVertices()(confidence)
    
    predictions = Concatenate()([correction,confidence])
    return tf.keras.models.Model(inputs=Inputs, outputs=[predictions])


train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model)
    
    train.compileModel(learningrate=0.0001,
                   loss=muon_loss,
                   metrics = [resolutionImprovement, resolutionImprovement20, resolutionImprovement70, resolutionImprovement150,
                              resolutionImprovementOS20, resolutionImprovementOS70, resolutionImprovementOS150],
                   ) 


print(train.keras_model.summary())



model,history = train.trainModel(nepochs=150, 
                                 batchsize=450,
                                 checkperiod=2, # saves a checkpoint model every N epochs
                                 verbose=1)
                                 







