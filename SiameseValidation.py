from SiameseModel import *
from tensorflow.keras.saving import *
import tensorflow as tf
import pickle, os, re, uuid, shutil, math
import random as rand
import numpy as np
siamese_model=load_model("siamese20231031v2")
with open("faceVerFileNames3.pkl","rb") as f:
    nameGroupFileNames=pickle.load(f)

currentDir=os.getcwd()   
siamese_model=load_model(os.path.join(currentDir,"siamese20231031v2"))
SIGNIFICANCE=.05
CONFIDENCE=1-SIGNIFICANCE
SAMPLE_SIZE=500
# folder wrangling
#NOTE unzip your folder
tDir=os.path.join(currentDir,"testerSiamese")
if not os.path.exists(tDir):
    os.makedirs(tDir)

#clean up folder
try:
    dDir=os.path.join(currentDir,
    "Downloads\\Selfies ID Images dataset 2023 TrainingData\\Selfies ID Images dataset")
    subDir,subsubDir=os.listdir(dDir),[]
    for sb in os.listdir(dDir):
        sb1=os.path.join(dDir,sb)
        for ssb in os.listdir(sb1):
            subsubDir.append(os.path.join(sb1,ssb))

    nameGroupFileNames,uNameFileNames=[],[]
    for ssb in subsubDir:
        gTemp=[]
        uNameFileNames.append(re.split("[:\\\\A-Za-z0-9\-]*_(.*)",
                                    os.path.split(ssb)[1])[1])
        for fn in os.listdir(ssb):
            newPath=os.path.join(tDir,
                        "{}.{}".
                        format(uuid.uuid4(),re.split("\.(\w+$)",fn)[1]))
            shutil.copy2(os.path.join(ssb,fn),newPath)
            gTemp.append(newPath)
        nameGroupFileNames.append(gTemp)

    with open("faceVerTsFileNames.pkl","wb") as f:
        pickle.dump((nameGroupFileNames,uNameFileNames),f)#NOTE pitstop0
except Exception:pass

with open(os.path.join(currentDir,"faceVerTsFileNames.pkl"),"rb") as f:
    nameGroupFileNames,uNameFileNames=pickle.load(f)

# experiment 1: randomly choose 500 images, one from each person
# compare them with themselves respectively
# test for false negative
if len(nameGroupFileNames)>=SAMPLE_SIZE:
    dummyLz=[rand.choice(lz) for lz in rand.sample(nameGroupFileNames,k=SAMPLE_SIZE)]
else:
    dummyLz=[rand.choice(lz) for lz in rand.choices(nameGroupFileNames,k=SAMPLE_SIZE)]

falseNeg=np.mean(
    np.array(
        [siamese_model.predict(
            list(np.expand_dims([
                preprocess(path),preprocess(path)
                ],1))
            )for path in dummyLz]
    )<CONFIDENCE
)
print("false negative rate when comparing a person's image with itself: {}".format(falseNeg))
# experiment 2: upload a image of a person novel to the lfw dataset;
#NOTE load your testing images into a folder, named testerSiamese, in the working directory;
# randomly choose 500 images
tFN=os.listdir(tDir)
lenT,testers=len(tFN),[os.path.join(tDir,f)for f in tFN]
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
                ],1))
            )for pathDum,pathTester in zip(dummyLz,testerLz)]
    )>SIGNIFICANCE
)
print("false positive rate when comparing a person's image not in lfw with ones in lfw: {}".format(falsePos))
# experiment 3: using the images in testerSiamese, record testing accuracy
# for any instance of inaccuracy, repeat the training algorithm with the testing dataset
def apnV():
    anc,pos,neg=[],[],[]
    lenMinGp=min(len(gp)for gp in nameGroupFileNames)
    lengthN,numComb=list(range(lenMinGp)),math.comb(lenMinGp,lenMinGp//2)
    idJackKnife=[[id for id in lengthN if id!=j]
                for j,_ in enumerate(nameGroupFileNames)]
    for iD,uName in enumerate(nameGroupFileNames):
        lengthU=len(uName)//2
        for _ in range(numComb):#NOTE len(uName)=12c6->6360c3180
            stratSam=rand.sample(uName,lengthU)
            anc.append(stratSam)
            pos.append([un for un in uName if un not in stratSam])
            neg.append([rand.choice(nameGroupFileNames[idNeg]) 
                        for idNeg in rand.choices(idJackKnife[iD],k=lengthU)])
    return (anc,pos,neg)

try:
    with open("faceVerAPNV.pkl","wb") as f:
        pickle.dump(apnV(),f)
except Exception:
    with open("faceVerAPNV.pkl","rb")as f:
        anc,pos,neg=pickle.load(f)

lengthU=len(nameGroupFileNames)
k=6
try:
    with open("faceVerhisTr.pkl","rb")as f:
        histories=pickle.load(f)
        _init=len(histories)//k
except Exception:
    histories,_init=[],0

def id2dat(a,p,n,_id):
    _length=len(_id)
    anchor,positive,negative=tf.data.Dataset.list_files([a[j]for j in _id]).map(preprocess,AUTOTUNE),\
        tf.data.Dataset.list_files([p[j]for j in _id]).map(preprocess,AUTOTUNE),\
        tf.data.Dataset.list_files([n[j]for j in _id]).map(preprocess,AUTOTUNE)
    positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(_length))))#zip tuple: cacheDataset x3
    negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(_length))))
    dat=positives.concatenate(negatives) if rand.getrandbits(1) else negatives.concatenate(positives)
    _lenDat=len(dat)#NOTE len of dat= len(positives)+len(negatives)
    return dat.shuffle(_length*2).take(_lenDat).cache().batch(16).prefetch(8)

generator=((j,(u,a,p,n)) for j,(u,a,p,n) in enumerate(zip(uNameFileNames,anc,pos,neg)) if j>=_init)#NOTE pitstop1 for when we pause and restart
pitstops=list(filter(
    lambda ps: ps%500==0,list(range(lengthU))
))[1:]#exclude j==0
for j,(u,a,p,n) in generator:
    print("starting {}/{};\nPerson's name: {};\nspan(anchor_person): {}".
            format(j+1,lengthU,u,len(nameGroupFileNames[j])))
    predPos=[np.argmax(
        siamese_model.predict(
            list(np.expand_dims([
                preprocess(aTemp),preprocess(pTemp)
                ],1))
            ),-1
    )==0
        for aTemp,pTemp in zip(a,p)]
    predNeg=[np.argmax(
        siamese_model.predict(
            list(np.expand_dims([
                preprocess(aTemp),preprocess(nTemp)
                ],1))
            ),-1
    )==1
        for aTemp,nTemp in zip(a,n)]
    print("rate of <false negative,false positive>: <{},{}>".format(np.mean(predPos),np.mean(predNeg)))     
    lengthA=len(a)
    idLz=[_id for _id,t in enumerate(np.logical_or(predPos,predNeg)) if t]
    rand.shuffle(idLz)
    idZSubLz=[idLz[j::k] for j in range(k)]
    
    for jFold,idSubLz in enumerate(idZSubLz):
        print("starting fold# {}/{}".format(jFold+1,k))
        idVal,idTrain=idSubLz,list(set(idLz).difference(idSubLz))
        dTr,dVal=id2dat(a,p,n,idTrain),id2dat(a,p,n,idVal)
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
        siamese_model.save_model("siamese20230925v02Val.h5")
        siamese_model=load_model("siamese20230925v02Val.h5")

try:
    with open("faceVerhisVl.pkl","wb")as f:
        pickle.dump(history,f)
except Exception: pass
