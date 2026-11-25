import sys
print("Python:", sys.version.split()[0])

import numpy, pandas, sklearn, matplotlib, cv2, streamlit
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("opencv:", cv2.__version__)

import tensorflow as tf
print("tensorflow:", tf.__version__)
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(3,))])
print("keras test output shape:", model.predict(numpy.zeros((1, 3)), verbose=0).shape)

print("\nAll checks passed.")