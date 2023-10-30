from SiameseModel import *
import os, pickle, logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
import random as rand
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

with open("faceVerFileNames2.pkl","rb") as f:
        nameFileNames,uNameFileNames=pickle.load(f)

with open("faceVerFileNames3.pkl","rb") as f:
    nameGroupFileNames=pickle.load(f)

with open("faceVerAPN0.pkl","rb")as f:
    anc,pos,neg=pickle.load(f)

lengthU=len(uNameFileNames)
k=6
sensDis,precDis,sensValDis,precValDis,history=[],[],[],[],[]

for j,(u,a,p,n) in enumerate(zip(uNameFileNames,anc,pos,neg)):
    print("starting {}/{};\nPerson's name: {};\nspan(anchor_person): {}".
            format(j+1,lengthU,u,len(nameGroupFileNames[j])))
    def id2dat(_id):
        _length=len(_id)
        anchor,positive,negative=tf.data.Dataset.list_files([a[j]for j in _id]),tf.data.Dataset.list_files([p[j]for j in _id]),\
            tf.data.Dataset.list_files([n[j]for j in _id])
        positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(_length))))
        negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(_length))))
        dat=positives.concatenate(negatives)
        dat=dat.map(preprocess_twin).cache.take(_length)
        return dat.batch(16).prefetch(8)
        
    lengthA=len(a)
    idLz=list(range(lengthA))
    rand.shuffle(idLz)
    idZSubLz=[idLz[j::k] for j in range(k)]
    for jFold,idSubLz in enumerate(idZSubLz):
        print("starting fold# {}/{}".format(jFold+1,k))
        idVal,idTrain=idSubLz,list(set(idLz).difference(idSubLz))
        lengthV,lengthT=len(idVal),len(idTrain)
        anchorTr,anchorVal=tf.data.Dataset.list_files([a[j] for j in idTrain]),tf.data.Dataset.list_files([a[j] for j in idVal])
        positiveTr,positiveVal=tf.data.Dataset.list_files([p[j] for j in idTrain]),tf.data.Dataset.list_files([p[j] for j in idVal])
        negativeTr,negativeVal=tf.data.Dataset.list_files([n[j] for j in idTrain]),tf.data.Dataset.list_files([n[j] for j in idVal])
        positivesTr=tf.data.Dataset.zip((anchorTr,positiveTr,tf.data.Dataset.from_tensor_slices(tf.ones(lengthT))))
        negativesTr=tf.data.Dataset.zip((anchorTr,negativeTr,tf.data.Dataset.from_tensor_slices(tf.zeros(lengthT))))
        positivesVal=tf.data.Dataset.zip((anchorVal,positiveVal,tf.data.Dataset.from_tensor_slices(tf.ones(lengthV))))
        negativesVal=tf.data.Dataset.zip((anchorVal,negativeVal,tf.data.Dataset.from_tensor_slices(tf.zeros(lengthV))))
        dTr0,dVal0=positivesTr.concatenate(negativesTr),positivesVal.concatenate(negativesVal)
        dTr0,dVal0=dTr0.map(preprocess_twin).cache().take(lengthT),dVal0.map(preprocess_twin).cache().take(lengthV)
        dTr,dVal=dTr0.batch(16).prefetch(8),dVal0.batch(16).prefetch(8)
        for batchT,batchV in zip(dTr,dVal):
            Xt,yt=batchT[:2],batchT[2]
            Xv,yv=batchV[:2],batchV[2]
            history.append(
                siamese_model.fit(
                    Xt,yt,epochs=50,validation_data=(Xv,yv),
                    callbacks=[TensorBoard("/tmp/tb_logs"),EarlyStopping("val_loss",10)]
                )
            )
        siamese_model.save("siamese20230925v01.h5")
        siamese_model=load_model("siamese20230925v01.h5")

    