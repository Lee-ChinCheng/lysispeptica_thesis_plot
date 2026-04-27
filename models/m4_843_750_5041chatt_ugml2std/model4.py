import tensorflow as tf
#print(tf.__version__) #2.16.1
#print(tf.keras.layers.Lambda) #<class 'keras.src.layers.core.lambda_layer.Lambda'>

#from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from training_func.custom_model import CustomModel

# w/o prefix import -> y = PReLU()(y)
# w/  prefix import -> y = tf.keras.layers.PReLU()(y)
# --------------------------------------------------


class GlobalMinPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalMinPooling1D, self).__init__(**kwargs)

    def call(self, inputs):
        # reduce along the time/sequence axis
        return tf.reduce_min(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        # input_shape = (batch_size, steps, features)
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = super(GlobalMinPooling1D, self).get_config()
        return config




def chatt(x1, chatt_drop):
    ### channel_attention
    #avg_pool = tf.reduce_mean(x1, axis=1, keepdims=True)  #Sh: (batch, 1, channels)
    #max_pool = tf.reduce_max(x1, axis=1, keepdims=True)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)(x1) #Sh: (batch, channels)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(keepdims=True)(x1) 

    # Apply Shared MLP to both pooled outputs
    max_pool = tf.keras.layers.Conv1D(filters=80, kernel_size=1, use_bias=False)(max_pool) #relu?#Prelu?
    max_pool = tf.keras.layers.PReLU()(max_pool)
    max_pool = tf.keras.layers.Dropout(chatt_drop)(max_pool)
    max_pool = tf.keras.layers.Conv1D(filters=161, kernel_size=1, use_bias=False)(max_pool)

    avg_pool = tf.keras.layers.Conv1D(filters=80, kernel_size=1, use_bias=False)(avg_pool)
    avg_pool = tf.keras.layers.PReLU()(avg_pool)
    avg_pool = tf.keras.layers.Dropout(chatt_drop)(avg_pool)
    avg_pool = tf.keras.layers.Conv1D(filters=161, kernel_size=1, use_bias=False)(avg_pool)
  
    att = tf.keras.layers.Add()([avg_pool, max_pool])
    att = tf.keras.layers.Activation('sigmoid')(att)
    print( 'attention.shape', att.shape )
    x1 = tf.keras.layers.Multiply()([x1, att])
    print( 'x1.shape', x1.shape )
    return x1






def self_defined_model(dropout_rate, model_name=None):

    inputs = tf.keras.layers.Input(shape=(49, 161))    

    y = chatt(inputs, 0.2)

    y = tf.keras.layers.Flatten()( y )
 
    y = tf.keras.layers.Dense(1024)(y)
    y = tf.keras.layers.PReLU()(y)
    y = tf.keras.layers.Dropout(dropout_rate)(y)

    y = tf.keras.layers.Dense(256)(y)
    y = tf.keras.layers.PReLU()(y)
    y = tf.keras.layers.Dropout(dropout_rate)(y)

    y = tf.keras.layers.Dense(32)(y)
    y = tf.keras.layers.PReLU()(y)
    y = tf.keras.layers.Dropout(dropout_rate)(y)

    outputs = tf.keras.layers.Dense(2, activation='softmax')(y)

    # If there are miltiple inputs, the 'inputs' parameter below should be editted as expected.
    model = CustomModel(inputs=inputs, outputs=outputs, name=model_name)

    return model

