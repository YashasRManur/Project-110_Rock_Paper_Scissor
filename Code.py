import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

vid = cv2.VideoCapture(0)

while True:
    lol, frame = vid.read()
    img = cv2.resize(frame, (224, 224))
    test = np.array(img, dtype = np.float32)
    test = np.expand_dims(test, axis = 0)
    norm = test/ 255.0
    predict = model.predict(norm)
    key = cv2.waitKey(1)

    print(predict)  
    cv2.imshow('input', frame )
    if key == 27:
        break

vid.release()

cv2.destroyAllWindows()