import os, cv2, uuid, re, pickle, math, itertools
import random as rand
from tensorflow.image import stateless_random_brightness,stateless_random_contrast,\
    stateless_random_flip_left_right,stateless_random_flip_up_down,\
        stateless_random_jpeg_quality,\
        stateless_random_saturation,stateless_random_hue

currentDir=os.getcwd();dDir=os.path.join(currentDir,"dataSiamese")
if not os.path.exists(dDir):
    os.makedirs(dDir)
'''
NOTE 1: download lfw from https://vis-www.cs.umass.edu/lfw/
NOTE 2: in Command Prompt, run: tar -xf lfw.tgz
NOTE 3: alternatively, you may load any portraiture of your liking into dataSiamese
'''
#move images- jpg low-res (<20kb), small (105*105)
def loadImg(folderName):#fileName in currentDir
    fileNames,folderPath=[],os.path.join(currentDir,folderName)
    for directory in os.listdir(folderPath):
        subDir=os.path.join(folderPath,directory)
        for file in os.listdir(subDir):
            newDir=os.path.join(dDir,file)
            fileNames.append(newDir)
            os.replace(
                os.path.join(subDir,file),newDir
            )#move file
    return fileNames

try:
    with open("faceVerFileNames0.pkl","wb") as f:
        pickle.dump(loadImg("lfw"),f)#NOTE pitstop0
except Exception:pass

with open(os.path.join(currentDir,"faceVerFileNames0.pkl"),"rb") as f:
    fileNames=pickle.load(f)

#duplicate images with various image quality issues- contrast, saturation, brightness, orientation
def imgAug(img):
    imgLz=[]
    rows,cols,_=img.shape
    for _ in range(11):
        ranSeed=[rand.randint(0,10000) for _ in range(14)]
        imgTemp=cv2.warpAffine(img,
                               cv2.getRotationMatrix2D(
                                   (cols/2,rows/2),rand.uniform(0,360),1),
                               (cols,rows))
        imgLz.append(
            stateless_random_flip_left_right(
                stateless_random_flip_up_down(
                    stateless_random_jpeg_quality(
                        stateless_random_saturation(
                            stateless_random_brightness(
                                stateless_random_hue(
                                    stateless_random_contrast(
                                        imgTemp,.4,1,(ranSeed[0],ranSeed[1])
                                    ),.5,(ranSeed[2],ranSeed[3])
                                ),.4,(ranSeed[4],ranSeed[5])
                            ),.6,1,(ranSeed[6],ranSeed[7])
                        ),85,100,(ranSeed[8],ranSeed[9])
                    ),(ranSeed[10],ranSeed[11])
                ),(ranSeed[12],ranSeed[13])
            )
        )
    return imgLz

def augImg():
    fNTemp=[]
    for fileName in fileNames:
        img=cv2.imread(fileName)
        for dup in imgAug(img):
            dupDir=re.split("\.jpg",fileName)[0]+"_{}.jpg".format(uuid.uuid4())
            fNTemp.append(dupDir)
            cv2.imwrite(
                dupDir,dup.numpy()
            )
    return fNTemp

try:
    with open("faceVerFileNames1.pkl","wb") as f:#NOTE pitstop1
        pickle.dump(augImg(),f)
except Exception:pass

with open("faceVerFileNames1.pkl","rb") as f:
    fNTemp=pickle.load(f)
    
#group fileNames by person's name
## collect a full fileNames
fileNames1=fileNames+fNTemp; fileNames1.sort()
nameFileNames=[re.split("\\\\([A-Za-z_\-]*)_.*$",fN)[1] for fN in fileNames1]
uNameFileNames=list(set(nameFileNames))
## index fileNames1 whose images are of one person
try:
    with open("faceVerFileNames2.pkl","wb") as f:#NOTE pitstop2
        pickle.dump(
            (nameFileNames,uNameFileNames),f
        )
except Exception:pass

with open("faceVerFileNames2.pkl","rb") as f:
        nameFileNames,uNameFileNames=pickle.load(f)

try:
    with open("faceVerFileNames3.pkl","wb") as f:#NOTE pitstop3
        pickle.dump(
            [[fileNames1[id] for id,name in enumerate(nameFileNames) if name==uName]
                    for uName in uNameFileNames],f
        )
except Exception:pass

with open("faceVerFileNames3.pkl","rb") as f:
    nameGroupFileNames=pickle.load(f)
    
#stratified bootstrap
lenMinGp=min(len(ng)for ng in nameGroupFileNames)
numComb=math.comb(lenMinGp,lenMinGp//2)
def apn():
    anc,pos,neg=[],[],[]
    lengthN=list(range(len(nameGroupFileNames)))
    idJackKnife=[[id for id in lengthN if id!=j]
             for j,_ in enumerate(nameGroupFileNames)]
    for iD,nameGroup in enumerate(nameGroupFileNames):
        lengthU=len(nameGroup)//2
        ancTemp,posTemp,negTemp=[],[],[]
        for _ in range(numComb):#NOTE len(nameGroup)=12c6->6360c3180
            stratSam=rand.sample(nameGroup,lengthU)
            ancTemp.append(stratSam)
            posTemp.append([un for un in nameGroup if un not in stratSam])
            negTemp.append([rand.choice(nameGroupFileNames[idNeg]) 
                    for idNeg in rand.choices(idJackKnife[iD],k=lengthU)])
        anc.append(list(itertools.chain.from_iterable(ancTemp)))
        pos.append(list(itertools.chain.from_iterable(posTemp)))
        neg.append(list(itertools.chain.from_iterable(negTemp)))
    return (anc,pos,neg)

try:
    with open("faceVerAPN0.pkl","wb")as f:
        pickle.dump(apn(),f)
except Exception: pass

with open("faceVerAPN0.pkl","rb")as f:
    anc,pos,neg=pickle.load(f)
