import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load the eye cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture("filename.mp4")

mean_values = []

count = 0

while cap.isOpened():

    count = count + 1

    re, img = cap.read()

    if (not re) :
        print ("Error reading capture device")
        break

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes_centres = []
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            e1x = int(ex+ew/2)
            e1y = int(ey+eh/2)
            eyes_centres.append((e1x, e1y))
        # Calculate the coordinates midpoint of the eyes
        if len(eyes_centres) == 2 :
            centre_x = int((eyes_centres[0][0] + eyes_centres[1][0])/2)
            centre_y = int((eyes_centres[0][1] + eyes_centres[1][1])/2)
            centre = (centre_x, centre_y)
        else :
            break
        # Now calculate distance between the eyes
        # 4d is the actual distance
        (x1, y1) = eyes_centres[0]
        (x2, y2) = eyes_centres[1]
        d = (((x1-x2)**2 + (y1-y2)**2)**0.5)/4
        # Now the calculate the dimension and position of rectangle representing forehead
        bx = int(centre_x - int(3*d/2))
        bw = int(3*d)
        by = int(centre_y - d)
        bh = int(3*d/2)
        # Highlight the forehead by the bounding box
        cv2.rectangle(roi_color,(bx,by),(bx+bw,by-bh),(0,0,255),2)
        forehead = roi_color[by:by+bh, bx:bx+bh]
        forehead_gray = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY)
        mean_values.append(np.mean(forehead_gray))
    cv2.imshow("video output", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(count)

plt.plot(mean_values)
plt.xlabel('frames')
plt.ylabel('Mean value of pixels in ROI')
plt.show()

n = len(mean_values)
fhat = np.fft.fft(mean_values, n)
PSD = fhat * np.conj(fhat) / n
freq = np.arange(n, dtype='int')
L = np.arange(1, np.floor(n/2), dtype='int')
plt.plot(freq[L], PSD[L])
plt.xlim(freq[L[0]], freq[L[-1]])
plt.show()

indices = PSD > 0
PSDclean = PSD * indices
fhat = fhat * indices
ffilt = np.fft.ifft(fhat)
plt.plot(ffilt)
plt.show()
