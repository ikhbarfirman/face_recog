import cv2
import numpy as np
from PIL import Image as imm
import mtcnn
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from playsound import playsound

with open('face_svc.pkl', 'rb') as file_1:
  model_svc = pickle.load(file_1)

with open('face_svc_ikh_toni.pkl', 'rb') as file_2:
  model_svc2 = pickle.load(file_2)

model_facenet = load_model('facenet_model/facenet_keras.h5')
detector = mtcnn.MTCNN()

def detect_face_cam(image,size=(160,160)):
    get_face = detector.detect_faces(image)
    if len(get_face) > 0:
        x1, y1, width, height = get_face[0]['box']
        x2, y2 = x1 + width, y1+ height
        face_array = image[y1:y2,x1:x2]
        image = imm.fromarray(face_array).resize(size)
        face_array_final = np.asarray(image)
        return face_array_final,x1,y1,x2,y2
    else: pass

def prepare_to_predict_cam(image,model):
    image_ori,x1,y1,x2,y2 = detect_face_cam(image)
    get_array = np.expand_dims(image_ori/255,axis=0)
    embed = model_facenet.predict(get_array)
    embed_scaled = Normalizer(norm='l2').transform(embed)
    preds = model.predict(embed_scaled)
    preds_proba = round(np.max(model.predict_proba(embed_scaled)),2)
    if preds_proba > 0.65:
        image = cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),8)
        image = cv2.putText(image, preds[0] + ' ' +str(preds_proba), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 3)
        playsound('beep-warning-6387.mp3')
        return image,preds[0]
    else: pass

cap = cv2.VideoCapture(0)
width_ = 640
height_ = 480
cap.set(3,width_) # adjust width
cap.set(4,height_) # adjust height


while True:
    success, img = cap.read()
    cv2.waitKey(1)
     # This will open an independent window
    cv2.imshow("Webcam", img)
    try:
        img2,nickname = prepare_to_predict_cam(img,model_svc2)
        print('hello')
        cv2.imshow("Webcam", img2)
        if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
            cap.release()
            cv2.destroyAllWindows()
            break
    except TypeError: pass
# cv2.destroyAllWindows() 
# cv2.waitKey(1) # normally unnecessary, but it fixes a bug on MacOS where the window doesn't close