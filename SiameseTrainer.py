from SiameseModel import *
import os, pickle, logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
import random as rand
from tensorflow.keras.callbacks import *
from tensorflow.keras.saving import *
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
currentDir=os.getcwd()
modDir=os.path.join(currentDir,"siamese20231031v1")

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
        history=pickle.load(f)
        _init=len(history)//k
except Exception:
    history,_init=[],0

#NOTE extra pitstops: every few hundreds of iterations, save and load model
#NOTE extra pitstops: every few hundred iterations, save and load model
generator=((j,(u,a,p,n)) for j,(u,a,p,n) in enumerate(zip(uNameFileNames,anc,pos,neg)) if j>=500)#NOTE pitstop1 for when we pause and restart
pitstops=list(filter(
    lambda ps: ps%500==0,list(range(lengthU))
))[1:]#exclude j==0
for j,(u,a,p,n) in generator:
    print("starting {}/{};\nPerson's name: {};\nspan(anchor_person): {}".
            format(j+1,lengthU,u,len(nameGroupFileNames[j])))
    if j in pitstops:
        save_model(siamese_model,modDir)
        siamese_model=load_model(modDir)
    def id2dat(_id):
        _length=len(_id)
        anchor,positive,negative=tf.data.Dataset.list_files([a[j]for j in _id]),tf.data.Dataset.list_files([p[j]for j in _id]),\
            tf.data.Dataset.list_files([n[j]for j in _id])
        positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(_length))))
        negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(_length))))
        dat=positives.concatenate(negatives)
        return dat.map(preprocess_twin,num_parallel_calls=tf.data.experimental.AUTOTUNE).take(_length).batch(16).prefetch(8).cache()
     
    lengthA=len(a)
    idLz=list(range(lengthA))
    rand.shuffle(idLz)
    idZSubLz=[idLz[j::k] for j in range(k)]
    
    for jFold,idSubLz in enumerate(idZSubLz):
        print("starting fold# {}/{}".format(jFold+1,k))
        idVal,idTrain=idSubLz,list(set(idLz).difference(idSubLz))
        dTr,dVal=id2dat(idTrain),id2dat(idVal)
        for batchT,batchV in zip(dTr,dVal):
            Xt,yt=batchT[:2],batchT[2]
            Xv,yv=batchV[:2],batchV[2]
            history.append(
                siamese_model.fit(
                    Xt,yt,epochs=50,validation_data=(Xv,yv),
                    callbacks=[
                        TensorBoard("/tmp/tb_logs"),
                        EarlyStopping("val_loss",10)
                    ]
                )
            )

save_model(siamese_model,os.path.join(currentDir,"siamese20231031v1"))

try:
    with open("faceVerhisTr.pkl","wb")as f:
        pickle.dump(history,f)
except Exception: pass
