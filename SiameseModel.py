import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Input,Flatten
from tensorflow.keras.metrics import *
from tensorflow_addons.optimizers import MultiOptimizer
from itertools import chain, combinations
import os
os.environ["tf_gpu_allocator"]="cuda_malloc_async"
ACTIVATION=[
    "elu","exponential","gelu",
    "hard_sigmoid","linear","mish","relu","selu",
    "sigmoid","softmax","softplus",
    "swish","tanh"
]
#scale and resize
def preprocess(fileName):
    img=tf.io.decode_jpeg(
        tf.io.read_file(fileName)
    )
    return tf.image.resize(img,(105,105))/255.0

def preprocess_twin(inName,valName,label):return(preprocess(inName),preprocess(valName),label)

def manD(ie,ve):return tf.math.abs(ie-ve)

def powerset(lz):
    s=list(lz)
    return [list(s)for s in chain.from_iterable(combinations(s,r)for r in range(len(s)+1))]

#20231115 NOTE keras tuner
def mod_builder(hp=None):#take hyperparams 
    if not hp:
        ## embedding mod
        inputShape=(105,105,3)
        inp=Input(shape=inputShape,name="dump_image")
        d1=Dense(4096,activation="sigmoid",input_shape=inputShape)(
            Flatten()(
                Conv2D(256,(4,4),activation="relu",input_shape=inputShape)(#conv 4th
                    MaxPooling2D(64,(2,2),padding="same",input_shape=inputShape)(
                        Conv2D(128,(4,4),activation="relu",input_shape=inputShape)(#conv 3rd
                            MaxPooling2D(64,(2,2),padding="same",input_shape=inputShape)(
                                Conv2D(128,(7,7),activation="relu",input_shape=inputShape)(#conv 2nd
                                    MaxPooling2D(64,(2,2),padding="same",input_shape=inputShape)(
                                        Conv2D(64,(10,10),activation="relu",input_shape=inputShape)(inp)#conv 1st
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        embedding=Model(inputs=[inp],outputs=[d1],name="Embed")  
        val=Input(shape=inputShape,name="val_image")
        classifier=Dense(1,activation="sigmoid",input_shape=inputShape)(
            manD(
                embedding(inp),embedding(val)
                )#distance
        )
        siamese_model=Model(inputs=[inp,val],outputs=[classifier],name="SiameseArchitecture")
        opts=[
            tf.keras.optimizers.experimental.SGD(.15,.01,True,.9,use_ema=True),
            tf.keras.optimizers.AdamW(5E-2)
        ]
        ml=siamese_model.layers
        lenLayer=len(ml)
        model.compile(
            MultiOptimizer(
                [(opt,ml[j::lenLayer])for j,opt in enumerate(opts)]#partition layers for optimizers
                ),
            tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy",AUC(100,name="auc"),F1Score("macro",.5)]
            )
    else:
        d1unit=hp.Int("d1unit",256,4096,8,"linear",4096)
        c4filter=hp.Int("c4filter",64,d1unit,8,"linear",256)
        c4kernsize=hp.Int("c4kern",2,12,default=4)
        m3poolsize=hp.Int("m3pool",64,c4filter,8,default=64)
        m3stride=hp.Int("m3stride",2,10,default=2)
        c3filter=hp.Int("c3filter",32,256,8,default=128)
        c3kernsize=hp.Int("c3kern",2,10,default=4)
        m2poolsize=hp.Int("m2pool",32,256,8,default=64)
        m2stride=hp.Int("m2stride",2,10,default=2)
        c2filter=hp.Int("c2filter",32,256,8,default=128)
        c2kernsize=hp.Int("c2kern",2,10,default=7)
        m1poolsize=hp.Int("m1pool",32,256,8,default=64)
        m1stride=hp.Int("m1stride",2,15,default=2)
        c1filter=hp.Int("c1filter",32,256,8,default=64)
        c1kernsize=hp.Int("c1kern",2,20,default=10)
        
        d1act=hp.Choice("d1act",ACTIVATION)
        c4act=hp.Choice("d1act",ACTIVATION)
        m3pad=hp.Choice("m3pad",["same","valid"])
        c3act=hp.Choice("d1act",ACTIVATION)
        m2pad=hp.Choice("m2pad",["same","valid"])
        c2act=hp.Choice("d1act",ACTIVATION)
        m1pad=hp.Choice("m1pad",["same","valid"])
        c1act=hp.Choice("d1act",ACTIVATION)
        inputShape=(105,105,3)
        inp=Input(shape=inputShape,name="dump_image")
        d1=Dense(d1unit,activation=d1act,input_shape=inputShape)(
            Flatten()(
                Conv2D(c4filter,(c4kernsize,c4kernsize),activation=c4act,input_shape=inputShape)(#conv 4th
                    MaxPooling2D(m3poolsize,(m3stride,m3stride),padding=m3pad,input_shape=inputShape)(
                        Conv2D(c3filter,(c3kernsize,c3kernsize),activation=c3act,input_shape=inputShape)(#conv 3rd
                            MaxPooling2D(m2poolsize,(m2stride,m2stride),padding=m2pad,input_shape=inputShape)(
                                Conv2D(c2filter,(c2kernsize,c2kernsize),activation=c2act,input_shape=inputShape)(#conv 2nd
                                    MaxPooling2D(m1poolsize,(m1stride,m1stride),padding=m1pad,input_shape=inputShape)(
                                        Conv2D(c1filter,(c1kernsize,c1kernsize),activation=c1act,input_shape=inputShape)(inp)#conv 1st
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        embedding=Model(inputs=[inp],outputs=[d1],name="Embed")  
        val=Input(shape=inputShape,name="val_image")
        classifier=Dense(1,activation="sigmoid",input_shape=inputShape)(
            manD(
                embedding(inp),embedding(val)
                )#distance
        )
        siamese_model=Model(inputs=[inp,val],outputs=[classifier],name="SiameseArchitecture")
        opts=[
            tf.keras.optimizers.experimental.SGD(
                hp.Float("SGDlearn_rate",1E-3,5E-1,500,"reverse_log",.1),
                hp.Float("momentum",1E-5,2,500,"log",0),
                hp.Boolean("nesterov"),
                hp.Float("SGDweight_decay",0,.99,1E-2,"linear",0),
                use_ema=hp.Boolean("SGDuse_ema"),
                ema_momentum=hp.Float("ema_mom",.5,.99,1E-3,"linear",.99,"use_ema",[True])
                ),
            tf.keras.optimizers.AdamW(
                hp.Float("ADAMWlearn_rate",1E-3,5E-1,500,"reverse_log",.1),
                hp.Float("ADAMWweight_decay",0,.9,1E-2,"linear",.004),
                use_ema=hp.Boolean("ADAMWuse_ema"),
                ema_momentum=hp.Float("ema_mom",.5,.99,1E-3,"linear",.99,"ADAMWuse_ema",[True])
            )
        ]
        ml=siamese_model.layers
        lenLayer=len(ml)
        #partition layers for optimizers
        #use powerset of the layers to assign layers
        subLayers=hp.Choice("sublayerSGD",powerset(ml))
        siamese_model.compile(
            MultiOptimizer(
                [(opts[0],subLayers),(opts[1],[l for l in ml if l not in subLayers])]#partition layers for optimizers
                ),
            tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy",AUC(100,name="auc"),F1Score("macro",.5)]
            )
        siamese_model.compile(
            tf.keras.optimizers.experimental.SGD(hp.Float("learn_rate",1E-3,5E-1,500,"reverse_log",.1),
                                                 hp.Float("momentum",1E-5,2,500,"log",0),
                                                 hp.Boolean("nesterov"),
                                                 hp.Float("weight_decay",0,.99,1E-2,"linear",0),
                                                 use_ema=hp.Boolean("use_ema"),
                                                 ema_momentum=hp.Float("ema_mom",.5,.99,1E-3,"linear",.99,"use_ema",[True])
                                                 ),
            tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy",AUC(name="auc"),F1Score("macro",.5)]
            )
    return siamese_model

if __name__=="__main__":
    print("defaulting siamese_model")
    siamese_model=mod_builder()