import tensorflow as tf
import tensorflow.keras.backend as K

#definig the convolution block that consist of 2 conv layers.

def conv_block(input,no_filter,f_size,BN = False,drop_out = 0):
'''
arg:
input--> the input to the block which is the output of the previous block
no_filter --> number of filters used in the conv layers
f_size --> size of the kernel
BN --> batch normalization layer (bolean)
drop_out --> dropout ratio for regularization
'''
  x = tf.keras.layers.Conv2D(no_filter,f_size,activation="relu",padding = "same")(input)
  if BN :
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(no_filter,f_size,activation="relu",padding = "same")(x)
  if BN :
    x = tf.keras.layers.BatchNormalization()(x)
  if drop_out > 0 :
    x = tf.keras.layers.Dropout(drop_out)(x)
  return x
  

#lambda function for repeating the result from AG 
def repeat_elem(tensor, rep):
   
     return tf.keras.layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

#defining the attention blocks that take 2 input and dimentionality 

def att_block(x,g, desired_dimensionality):
  '''
  x-->> the input from skip connection
  g-->> the input from next lower 
  '''
  x_shape = x.shape
  g_shape = g.shape
  #strides for xl should be 2 to equal the shapes before addition
  xl = tf.keras.layers.Conv2D(desired_dimensionality,(1,1),strides=(2,2),activation="relu",padding = "same")(x)
  gl = tf.keras.layers.Conv2D(desired_dimensionality,(1,1),activation="relu",padding = "same")(g)
  xg = tf.keras.layers.Add()([xl,gl])
  xg = tf.keras.layers.Activation("relu")(xg)
  xg = tf.keras.layers.Conv2D(1,(1,1),activation="sigmoid",padding = "same")(xg)
  xg_shape = xg.shape
  xg = tf.keras.layers.UpSampling2D((x_shape[1]//xg_shape[1],x_shape[2]//xg_shape[2]))(xg)
  #repetion for equal the dimensionality
  xg = repeat_elem(xg, x_shape[-1])
  output = tf.keras.layers.Multiply()([xg,x])
  return output

