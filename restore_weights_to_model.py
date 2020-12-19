'''
    Script for restoring a pretrained weights to a Keras model
'''

# Due to different Keras versions on a training machine and at the
# Udacity workspace, it is impossible to save and load the whole
# model as .h5 file -- it fails across different versions.
# Luckily, the weight format remains the same, and so it is possible
# to just re-create the model and then load pretrained weights.
# See `restore_weights_to_model.py` for details.

from model import build_behavioral_model

restored_model = build_behavioral_model()
restored_model.summary()

restored_model.load_weights("saved_weights.h5")
restored_model.save("restored_model.h5")
