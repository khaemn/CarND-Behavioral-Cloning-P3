import numpy as np
import os.path

from model import build_behavioral_model

model = build_behavioral_model()
model.summary()

model.load_weights("saved_weights.h5")
model.save("restored_model.h5")
