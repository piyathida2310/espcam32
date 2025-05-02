import sys
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# ตั้งค่าการเข้ารหัสเป็น UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# คลาสที่โมเดลจำแนก
CLASS_NAMES = ['dogs', 'cats', 'humans']


# โหลดโมเดล
model = tf.keras.models.load_model('animal_model_3class.h5')

# อ่านไฟล์ภาพจาก argument
image_path = sys.argv[1]
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
    sys.exit(1)

# ปรับขนาดภาพให้ตรงกับที่โมเดลใช้ (160x160)
image = cv2.resize(image, (160, 160))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB
image = np.expand_dims(image, axis=0)  # เพิ่ม batch dimension
image = preprocess_input(image)  # Preprocess ตาม MobileNetV2

# ทำนาย
predictions = model.predict(image, verbose=0)  # ปิด progress bar
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_label = CLASS_NAMES[predicted_class]

# ส่งผลลัพธ์กลับ
print(predicted_label)
sys.exit(0)