import cv2
import numpy as np
img = cv2.imread('kalabalik2.jpg')

yuz_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#haarcascade algoritmasını etkin hale getiriyoruz yüz tanıma için bu

griton = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

yüzler=yuz_casc.detectMultiScale(griton,1.1,4) #orda o yüz kaç kere var skala teyit gibiişlemleri gerçekleştirir

for (x,y,w,h) in yüzler:

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow('yuzler',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
