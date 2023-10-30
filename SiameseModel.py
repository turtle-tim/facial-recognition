import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Input,Flatten
#scale and resize
def preprocess(fileName):
    img=tf.io.decode_jpeg(
        tf.io.read_file(fileName)
    )
    return tf.image.resize(img,(105,105))/255.0

def preprocess_twin(inName,valName,label):return(preprocess(inName),preprocess(valName),label)

def manD(ie,ve):return tf.math.abs(ie-ve)

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
siamese_model.compile(
    tf.keras.optimizers.experimental.SGD(),
    tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy",tf.keras.metrics.AUC()])