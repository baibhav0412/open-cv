import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('/home/baibhav/Desktop/Project/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/baibhav/Desktop/Project/cv2/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle",'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

while True:
	
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame,1.5,5)
	for(x, y, w, h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h,x:x+w]
		roi_color = frame[y:y+h,x:x+w]
		#img_item = "7.png"
		#cv2.imwrite(img_item, roi_color)
		#recognize? deep learned model predict keras tensorflow pytorch scikit learn
		id_, conf = recognizer.predict(roi_gray)
		if conf < 100:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font, 1, color, stroke,cv2.LINE_AA)
		color = (0,0,255) #BGR
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,1)
		

	cv2.imshow("Video",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

