import cv2


# Create our body classifier
body_classifier = cv2.CascadeClassifier('C:/Users/pidik/Downloads/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C:/Users/pidik/Downloads/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    body =  body_classifier.detectMultiScale(grey,1.1,5)

    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in body:
        rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame.jpg',frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
