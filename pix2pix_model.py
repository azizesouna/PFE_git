import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Activation,Concatenate,LeakyReLU,Dropout,BatchNormalization,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

#define descriminator model

def define_discriminator(image_shape): 
#define weights

  init=RandomNormal(stddev=0.02)
  source_image=Input(shape=image_shape)
  target_image=Input(shape=image_shape)
  concatenated=Concatenate()([source_image,target_image])
  #c64 layer
  l=Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(concatenated)
  l=LeakyReLU(alpha=0.02)(l)
  #c128 layer
  l=Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(l)
  l=BatchNormalization()(l)
  l=LeakyReLU(alpha=0.02)(l)
  #c256 layer
  l=Conv2D(256,(4,4),strides=(2,2),padding='same-',kernel_initializer=init)(l)
  l=LeakyReLU(alpha=0.02)(l)
  l=BatchNormalization()(l)
  #c512 layer
  l=Conv2D(512,(4,4),padding='same',kernel_initializer=init)(l)
  l=LeakyReLU(alpha=0.02)(l)
  l=BatchNormalization()(l)
  #patch output
  l=Conv2D(1,(4,4),padding='same',kernel_initializer=init)(l)
  patch_output=Activation('sigmoid')(l)
  #define the model
  model=Model([source_image,target_image],patch_output)
  opt=Adam(lr=0.0002,beta_1=0.5)
  model.compile(loss='binary_crossentropy',optimizer=opt,loss_weights=[0.5])
  return model

image_shape=(256,256,3)
model=define_discriminator(image_shape)
#model.summary()

#define the encoder block
def define_encoder_block(input_layer,filters,batchnorm=True):
  #the random vector
  init=RandomNormal(stddev=0.02)
  #the Conv layer
  l=Conv2D(filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(input_layer)
  if batchnorm:
    l=BatchNormalization()(l,training=True)
  l=LeakyReLU(alpha=0.2)(l)
  return l
#define the decoder block
def define_decoder_block(input_layer,filters,skip_con,dropout=True):
  #the random weights
  init=RandomNormal(stddev=0.02)
  #the conv2Dtranspose layer
  d=Conv2DTranspose(filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(input_layer)
  d=BatchNormalization()(d,training=True)
  if dropout:
    d=Dropout(0.5)(d,training=True)
  d=Concatenate()([d,skip_con])
  d=Activation('relu')(d)
  return d

#define the generator model
def define_generator(image_shape):
  input_image=Input(shape=image_shape)
  e1=define_encoder_block(input_image,64,batchnorm=False)
  e2=define_encoder_block(e1,128)
  e3=define_encoder_block(e2,256)
  e4=define_encoder_block(e3,512)
  e5=define_encoder_block(e4,512)
  e6=define_encoder_block(e5,512)
  e7=define_encoder_block(e6,512)
  #bottleneck
  init=RandomNormal(stddev=0.02)
  b=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(e7)
  b=Activation('relu')(b)
  #define the decoder
  d1=define_decoder_block(b,512,e7)
  d2=define_decoder_block(d1,512,e6)
  d3=define_decoder_block(d2,512,e5)
  d4=define_decoder_block(d3,512,e4,dropout=False)
  d5=define_decoder_block(d4,256,e3,dropout=False)
  d6=define_decoder_block(d5,128,e2,dropout=False)
  d7=define_decoder_block(d6,64,e1,dropout=False)
  #output layer
 
  o=Conv2DTranspose(3,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d7)
  output_image=Activation('tanh')(o)
  #define model
  model=Model(input_image,output_image)
  return model
  image_shape=(256,256,3)
  model=define_generator(image_shape)
  #model_summary()
  #tf.keras.utils.plot_model(model,to_file='generator_plot.png',show_shapes=True,show_layer_names=True)
##########################################
#define the combined model
def define_gan(generator,discriminator,image_shape):
  #initialize descriminator as not trainable
  src=Input(shape=image_shape)
  discriminator.trainable=False
  gen_out=generator(src)
  disc_output=discriminator([src,gen_out])
  model=Model(src,[disc_output,gen_out])
  opt=Adam(lr=0.0002,beta_1=0.5)
  model.compile(loss=['binary_crossentropy','mae'],optimizer=opt,loss_weights=[1,100])
  return model
#define image shape
image_shape=(256,256,3)
#define the descriminztor and the generator
discriminator=define_discriminator(image_shape)
generator=define_generator(image_shape)
gan_model=define_gan(generator,discriminator,image_shape)
#gan_model.summary()
#select random data from the data set
def generate_real_samples(dataset,n_samples,patch_shape):
  train_A,train_B=dataset
  x=randint(0,train_A.shape(0),1)
  X1=train_A[x]
  X2=train_B[x]
  y=ones(n_samples,patch_shape,patch_shape,1)
  return [X1,X2],y
#generate fake data
def generate_fake_data(g_model,samples,patch_shape):
  X=g_model.predict(samples)
  y=zeros(len(X),patch_shape,patch_shape,1)
  return X,y
#train the model 
def train_model(d_model,g_model,gan_model,dataset,n_epoches=100,n_batch=1,n_patch=16):
  trainA,trainB=dataset
  bat_per_epo=int(len(trainA)/n_batch)
  n_steps=bat_per_epo*n_epoches
  for i in range(n_steps):
    [X_realA,X_realB],y_real=generate_real_samples(dataset,n_batch,n_patch)
    X_fake,y_fake=generate_fake_samples(g_model,x_realA,n_patch)
    d_loss1=d_model.train_on_batch([X_realA,X_realB],y_real)
    d_loss2=d_model.train_on_batch([X_realA,X_fakeB],y_fake)
    g_loss,_,_=gan_model.train_on_batch(X_realA,[y_real,X_realB])
    print('>%d,d1[%.3f] d2[%.3f] g[%.3f]' %(i+1,d_loss1,d_loss2,g_loss))
#load data

#print images
train_model(discriminator,generator,gan_model,dataset)
  
  

 
   



