import keras_tuner as kt #20231115 NOTE start tuning hp for SGD and embedding mod
from SiameseModel import *
import os, pickle, logging, itertools
logging.getLogger("tensorflow").setLevel(logging.WARNING)
os.environ["tf_gpu_allocator"]="cuda_malloc_async"
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.saving import *
from tensorflow.data.experimental import *
currentDir=os.getcwd()
modDir=os.path.join(currentDir,"siamese20231031v3")
hmDir=os.path.join(modDir,"hm")

CALLBACKS=[
    TensorBoard("/tmp/tb_logs"),
    EarlyStopping("val_loss",0,30),
    ModelCheckpoint(modDir,"val_auc",0,True,True)
    ]

with open("faceVerAPN0.pkl","rb")as f:
    anc,pos,neg=pickle.load(f)

megaAnc,megaPos,megaNeg=list(itertools.chain.from_iterable(anc)),list(itertools.chain.from_iterable(pos)),list(itertools.chain.from_iterable(neg))
megaRange=range(min(len(megaAnc),len(megaPos),len(megaNeg)))

tuner=kt.Hyperband(
    mod_builder,
    "val_loss",
    50,
    5,
    2,
    2023,
    directory=hmDir,
    project_name="hyperbandPostSiamsese"
)

def id2dat(a,p,n,_id):
    _length=len(_id)
    anchor,positive,negative=tf.data.Dataset.list_files([a[j]for j in _id]).map(preprocess,AUTOTUNE),\
        tf.data.Dataset.list_files([p[j]for j in _id]).map(preprocess,AUTOTUNE),\
        tf.data.Dataset.list_files([n[j]for j in _id]).map(preprocess,AUTOTUNE)
    positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(_length))))#zip tuple: cacheDataset x3
    negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(_length))))
    dat=positives.concatenate(negatives)
    _lenDat=len(dat)#NOTE len of dat= len(positives)+len(negatives)
    return dat.take(_lenDat).batch(4).prefetch(2).cache()

#cacheDs to eager tensor
try:
    with open("faceVerCache.pkl","wb")as f:
        pickle.dump(
            id2dat(megaAnc,megaPos,megaNeg,megaRange)
        )
except Exception as e:
    print(e)
finally:
    with open("faceVerCache.pkl","rb")as f:
        megaCache=pickle.load(f)

megaEager=[batch for batch in megaCache][0]
megaTuple,megaLabel=megaEager[:2],megaEager[2]
tuner.search(megaTuple,megaLabel,validation_split=.2,callbacks=CALLBACKS)

#get the best hyper model
save_model(
    tuner.get_best_models()[0],hmDir
)