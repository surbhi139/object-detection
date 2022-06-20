#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install opencv-python


# In[1]:


import cv2


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


config_file= 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model= 'frozen_inference_graph.pb'


# In[4]:


model= cv2.dnn_DetectionModel(frozen_model, config_file)


# In[5]:


classlabels = []
file_name='coco.names'
with open(file_name,'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')


# In[6]:


print(classlabels)


# In[ ]:


print(len(classlabels))


# In[ ]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# In[ ]:


img=cv2.imread('man-tuxedo-car-style-wallpaper-preview.jpg')


# In[ ]:


plt.imshow(img)


# In[ ]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


classIndex,confidece, bbox = model.detect(img, confThreshold = 0.5)


# In[ ]:


print(classIndex)


# In[ ]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(), confidece.flatten(), bbox):
    #cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    #cv2.putText(img,text,(text_offset_x, text_offset_y), font, fontScale = font_scale, color = (0,0,0), thickness =1)
    cv2.rectangle(img,boxes,(255,0,0), 2)
    cv2.putText(img, classlabels[classInd -1], (boxes[0] + 10, boxes[1]+40), font, fontScale = font_scale, color = (0,255,0), thickness =3)


# In[ ]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


cap=cv2.VideoCapture('0')


# check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open video")
    
    
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    
    classIndex, confidece, bbox = model.detect(frame, confThreshold = 0.55)
    
    print(classIndex)
    if(len(classIndex)!=0):
        for classInd, conf, boxes in zip(classIndex.flatten(), confidece.flatten(), bbox):
            if(classInd <= 80):
                cv2.rectangle(frame, boxes,(255,0,0), 2)
                cv2.putText(frame, classlabels[classInd-1], (boxes[0]+10, boxes[1] + 40), font, fontScale = font_scale, color = (0, 255,0), thickness = 3)
    cv2.imshow("object detection", frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




