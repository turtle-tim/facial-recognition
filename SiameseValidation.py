from SiameseModel import *
from SiameseTrainer import *
from tensorflow.keras.saving import *
import pickle, os, re, uuid
import random as rand
import numpy as np

with open("faceVerFileNames3.pkl","rb") as f:
    nameGroupFileNames=pickle.load(f)

currentDir=os.getcwd()   
siamese_model=load_model(os.path.join(currentDir,"siamese20231031v1"))
SIGNIFICANCE=.05
CONFIDENCE=1-SIGNIFICANCE
SAMPLE_SIZE=500
# experiment 1: randomly choose 500 images, one from each person
# compare them with themselves respectively
# test for false negative
dummyLz=[rand.choice(lz) for lz in rand.sample(nameGroupFileNames,k=SAMPLE_SIZE)]
falseNeg=np.mean(
    np.array(
        [siamese_model.predict(
            list(np.expand_dims([
                preprocess(path),preprocess(path)
                ],axis=1))
            )for path in dummyLz]
    )<CONFIDENCE
)
print("false negative rate when comparing a person's image with itself: {}".format(falseNeg))
# experiment 2: upload a image of a person novel to the lfw dataset;
#NOTE load your testing images into a folder, named testerSiamese, in the working directory;
# randomly choose 500 images
currentDir=os.getcwd()
testers=os.listdir(
    os.path.join(currentDir,"testerSiamese")
)
lenT=len(testers)
if lenT>=SAMPLE_SIZE:
    testerLz=rand.sample(testers,SAMPLE_SIZE)
else:
    testerLz=testers.copy()
    testerLz.append(
        rand.choices(testers,k=SAMPLE_SIZE-lenT)
    )
    
dummyLz=[rand.choice(lz) for lz in rand.sample(nameGroupFileNames,k=SAMPLE_SIZE)]
falsePos=np.mean(
    np.array(
        [siamese_model.predict(
            list(np.expand_dims([
                preprocess(pathDum),preprocess(pathTester)
                ],axis=1))
            )for pathDum,pathTester in zip(dummyLz,testerLz)]
    )>SIGNIFICANCE
)
print("false positive rate when comparing a person's image not in lfw with ones in lfw: {}".format(falsePos))
# experiment 3: in testerSiamese, group images by the names of their person- in the same way I generated nameGroupFileNames
# repeat the training algorithm with the testing dataset
nameFileNames=[re.split("\\\\([A-Za-z_\-]*)_.*$",fN)[1] for fN in testers]
uNameFileNames=list(set(nameFileNames))
nameGroupFileNames=[]
for uName in uNameFileNames:
    lzTemp=[]
    for id,name in enumerate(nameFileNames):
        if name==uName:
            temp=testers[id]
            t1,t2=os.path.split(temp)
            newPath=t1+"{}.{}".format(uuid.uuid4(),re.split("(\.\w+$)",t2)[-2])
            os.replace(
                temp,os.path.join(newPath)
            )
            lzTemp.append(newPath)
    nameGroupFileNames.append(lzTemp)
anc,pos,neg=[],[],[]
lengthN=list(range(len(nameGroupFileNames)))
idJackKnife=[[id for id in lengthN if id!=j]
             for j,_ in enumerate(nameGroupFileNames)]
for iD,uName in enumerate(nameGroupFileNames):
    lengthU=len(uName)//2
    for _ in range(924):#NOTE len(uName)=12c6->6360c3180
        stratSam=rand.sample(uName,lengthU)
        anc.append(stratSam)
        pos.append([un for un in uName if un not in stratSam])
        neg.append([rand.choice(nameGroupFileNames[idNeg]) 
                    for idNeg in rand.choices(idJackKnife[iD],k=lengthU)])
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
        return dat.map(preprocess_twin).take(_length).batch(16).prefetch(8).cache()
     
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
                        EarlyStopping("val_loss",10),
                        ModelCheckpoint(currentDir,"val_loss",1,True)
                    ]
                )
            )
        siamese_model.save("siamese20230925v01.h5")
        siamese_model=load_model("siamese20230925v01.h5")

try:
    with open("faceVerhisTs.pkl","wb")as f:
        pickle.dump(history,f)
except Exception: pass
