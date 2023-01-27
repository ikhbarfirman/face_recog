import cv2
import os
import numpy as np
from PIL import Image as imm
import mtcnn
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from playsound import playsound
import pandas as pd
from datetime import datetime

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
    array = image
    image_ori,x1,y1,x2,y2 = detect_face_cam(image)
    get_array = np.expand_dims(image_ori/255,axis=0)
    embed = model_facenet.predict(get_array)
    embed_scaled = Normalizer(norm='l2').transform(embed)
    preds = model.predict(embed_scaled)
    preds_proba = round(np.max(model.predict_proba(embed_scaled)),2)
    if preds_proba > 0.725:
        image = cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),8)
        image = cv2.putText(image, preds[0] + ' ' +str(preds_proba), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 3)
        playsound('beep-warning-6387.mp3')
        return image,preds[0]
    else: pass

def add_attendance(nickname):
    data_karyawan = pd.read_csv('data_staff/data_karyawan.csv',sep=';')
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    date_att = now[:10]
    time_att = now[-8:]

    # data_attendance = pd.DataFrame({})
    # data_attendance.to_csv('data_staff/data_attendance.csv')
    data_attendance = pd.read_csv('data_staff/data_attendance.csv')

    if nickname in data_karyawan.nickname.unique():
        print(f'Welcome {nickname}, attendance time: {now}')
        data_nickname = data_karyawan.loc[data_karyawan.nickname == nickname,['id','fullname','role']]
        data_nickname['date'] = date_att
        data_nickname['time'] = time_att
        data_attendance = pd.concat([data_attendance,data_nickname],axis=0)
        data_attendance.id = data_attendance.id.astype('int64')
        data_attendance.reset_index(drop=True)
        data_attendance.to_csv('data_staff/data_attendance.csv',index=False)
        path_ = nickname+'_'+str(data_nickname.id.iloc[0])+'_'+now.replace(' ','_').replace(':','-').replace('/','-')
        return path_,date_att.replace('/','-')
    else:
        print(f'{nickname} tidak terdaftar !!')

webcam = cv2.VideoCapture(0)
width_ = 640
height_ = 480
webcam.set(3,width_) # adjust width
webcam.set(4,height_) # adjust height

# Read face_frame and resize
face_frame = cv2.imread('face_pos.png')
# face_frame = cv2.resize(face_frame, (width_, height_))
img2gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

while True:
    success, img = webcam.read()

    # Menambahkan face frame
    img_ = img.copy()
    roi = img[:,:]
    roi[np.where(mask)] = 0
    roi += face_frame
    try:
        cv2.imshow("Webcam", img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            try:
                img2,nickname = prepare_to_predict_cam(img_,model_svc2)
                img2.shape
                webcam.release()
                    # img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                cv2.imshow("Webcam", img2)
                path_,date_ = add_attendance(nickname)
                try:
                    image_path = 'data_staff/image_attendance/'+date_
                    os.makedirs(image_path)
                except FileExistsError: 
                    pass
                cv2.imwrite(filename=image_path+'/'+path_+'.jpg', img=img2)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                print("Image saved!")
                break
            except (TypeError,AttributeError) as e:
                img = cv2.putText(img, 'COBA LAGI YA :))', (100, 200-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (125,100,12), 2)
                cv2.imshow('Webcam',img)
                cv2.waitKey(1650)
                pass
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    except ValueError:
        img = cv2.putText(img, 'COBA LAGII', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 3)
        cv2.imshow('Webcam',img) 
        pass