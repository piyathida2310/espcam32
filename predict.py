import sys
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

CLASS_NAMES = ['dogs', 'not_dogs']
model = tf.keras.models.load_model('dog_classifier_final.h5')

image_path = sys.argv[1]
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
    sys.exit(1)

image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

predictions = model.predict(image, verbose=0)
prob = predictions[0] if predictions.shape == (1,) else predictions[0][0]
predicted_class = int(prob > 0.5)
predicted_label = CLASS_NAMES[predicted_class]

print(predicted_label)
sys.exit(0)
