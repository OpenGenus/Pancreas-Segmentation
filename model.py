from blocks import conv_block, repeat_elem, att_block


def att_model(no_filter,input_shape=(224,224,3) ):
   #down-sampling process
   inputs = tf.keras.layers.Input(input_shape, dtype=tf.float32)
   x1= conv_block(inputs,no_filter,(3,3),BN = True,drop_out = 0)
   pool1= tf.keras.layers.MaxPooling2D(2,2)(x1)

   x2= conv_block(pool1,2*no_filter,(3,3),BN = True,drop_out = 0)
   pool2= tf.keras.layers.MaxPooling2D(2,2)(x2)

   x3= conv_block(pool2,4*no_filter,(3,3),BN = True,drop_out = 0)
   pool3= tf.keras.layers.MaxPooling2D(2,2)(x3)

   x4= conv_block(pool3,8*no_filter,(3,3),BN = True,drop_out = 0)
   pool4= tf.keras.layers.MaxPooling2D(2,2)(x4)
   
   #bottle-neck
   x5= conv_block(pool4,16*no_filter,(3,3),BN = True,drop_out = 0)

   #up-sampling layers
   
   x6= att_block(x4,x5, no_filter*2)
   u6= tf.keras.layers.UpSampling2D(2)(x5) 
   concate1 = tf.keras.layers.Concatenate()([x6,u6])
   conv6 = conv_block(concate1,8*no_filter,(3,3),BN = True,drop_out = 0) 


   x7= att_block(x3,conv6, no_filter*2)
   u7= tf.keras.layers.UpSampling2D(2)(conv6) 
   concate2 = tf.keras.layers.Concatenate()([x7,u7])
   conv7 = conv_block(concate2,4*no_filter,(3,3),BN = True,drop_out = 0) 
   

   x8= att_block(x2,conv7, no_filter*2)
   u8= tf.keras.layers.UpSampling2D(2)(conv7) 
   concate3 = tf.keras.layers.Concatenate()([x8,u8])
   conv8 = conv_block(concate3,2*no_filter,(3,3),BN = True,drop_out = 0) 



   x9= att_block(x1,conv8, no_filter*2)
   u9= tf.keras.layers.UpSampling2D(2)(conv8) 
   concate4 = tf.keras.layers.Concatenate()([x9,u9])
   conv8 = conv_block(concate4,no_filter,(3,3),BN = True,drop_out = 0) 

   
   conv_final = tf.keras.layers.Conv2D(1, kernel_size=(1,1))(conv8)
   conv_final =tf.keras.layers.BatchNormalization(axis=3)(conv_final)
   conv_final = tf.keras.layers.Activation('sigmoid')(conv_final)



   return tf.keras.Model(inputs,conv_final)


# Defining loss function suitable for semantic segmentation as Dice score.

import tensorflow.keras.backend as K

def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1. - K.mean(dice)
    return loss


model = att_model(32,input_shape=(224,224,3) )
model.compile(optimizer = "Adam",
loss=dice_loss)

