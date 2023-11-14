from SiameseModel import *
import os, pickle, logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
import random as rand
from tensorflow.keras.callbacks import *
from tensorflow.keras.saving import *
from tensorflow.data.experimental import *
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
currentDir=os.getcwd()
modDir=os.path.join(currentDir,"siamese20231031v2")
ckptDir=os.path.join(modDir,"ckpt.ckpt")

with open("faceVerFileNames2.pkl","rb") as f:
        nameFileNames,uNameFileNames=pickle.load(f)

with open("faceVerFileNames3.pkl","rb") as f:
    nameGroupFileNames=pickle.load(f)

with open("faceVerAPN0.pkl","rb")as f:
    anc,pos,neg=pickle.load(f)

lengthU=len(uNameFileNames)
k=6
try:
    with open("faceVerhisTr.pkl","rb")as f:
        histories=pickle.load(f)
        _init=len(histories)//k
except Exception:
    histories,_init=[],0

try:
    siamese_model=tf.keras.models.load_model(modDir)
    try:
        siamese_model.load_weights(ckptDir)
    except Exception as e:
        print(e)
except Exception as e:
    print(e)

#NOTE extra pitstops: every few hundreds of iterations, save and load model
generator=((j,(u,a,p,n)) for j,(u,a,p,n) in enumerate(zip(uNameFileNames,anc,pos,neg)) if j>=_init)#NOTE pitstop1 for when we pause and restart
pitstops=list(filter(
    lambda ps: ps%500==0,list(range(lengthU))
))[1:]#exclude j==0
def id2dat(a,p,n,_id):
    _length=len(_id)
    anchor,positive,negative=tf.data.Dataset.list_files([a[j]for j in _id]).map(preprocess,AUTOTUNE),\
        tf.data.Dataset.list_files([p[j]for j in _id]).map(preprocess,AUTOTUNE),\
        tf.data.Dataset.list_files([n[j]for j in _id]).map(preprocess,AUTOTUNE)
    positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(_length))))#zip tuple: cacheDataset x3
    negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(_length))))
    dat=positives.concatenate(negatives)
    _lenDat=len(dat)#NOTE len of dat= len(positives)+len(negatives)
    return dat.take(_lenDat).batch(16).prefetch(8).cache()

for j,(u,a,p,n) in generator:
    print("starting {}/{};\nPerson's name: {};\nspan(anchor_person): {}".
            format(j+1,lengthU,u,len(nameGroupFileNames[j])))
    if j in pitstops:
        save_model(siamese_model,modDir)
        siamese_model=load_model(modDir)     
    lengthA=min(len(a),len(p),len(n))
    idLz=list(range(lengthA))
    rand.shuffle(idLz)
    idZSubLz=[idLz[j::k] for j in range(k)]
    for jFold,idSubLz in enumerate(idZSubLz):
        print("starting fold# {}/{}".format(jFold+1,k))
        idVal,idTrain=idSubLz,list(set(idLz).difference(idSubLz))
        dTr,dVal=id2dat(a,p,n,idTrain),id2dat(a,p,n,idVal)
        for batchT,batchV in zip(dTr,dVal):#cacheDs to eager tensor
            Xt,yt=batchT[:2],batchT[2]
            Xv,yv=batchV[:2],batchV[2]
            histories.append(
                siamese_model.fit(
                    Xt,yt,epochs=2,validation_data=(Xv,yv),
                    callbacks=[
                        TensorBoard("/tmp/tb_logs"),
                        EarlyStopping("val_auc",10),
                        ModelCheckpoint(ckptDir,"val_auc",0,True)
                    ]
                )
            )

save_model(siamese_model,os.path.join(currentDir,"siamese20231031v2"))

try:
    with open("faceVerhisTr.pkl","wb")as f:
        pickle.dump(histories,f)
except Exception: pass

