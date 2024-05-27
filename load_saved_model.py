from tensorflow.keras.models import load_model

# load model 
model = load_model('all_signal_ternary_classification_model.h5')

# check model info 
model.summary()