import cv2 as cv
import dlib
import numpy as np

cap = cv.VideoCapture(0)
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
dnn = cv.dnn.readNetFromCaffe(configFile, modelFile)

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

'''
Remember to give credits to
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
'''

while True:

    mask_on = False
    mask_off_msg = "Mask off"
    mask_on_msg = "Mask on"

    sat, frame = cap.read()

    frame = cv.resize(frame, (0, 0), fx=0.7, fy=0.7, interpolation=cv.INTER_AREA)
    temp = frame.copy()
    temp = cv.copyMakeBorder(temp, 5, 5, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 0))

    dlib_frame = temp.copy()
    haar_frame = temp.copy()
    dnn_frame = temp.copy()
    dnn_varied_frame=temp

    def dis_mask(frame, bottom_right, text):
        sub = lambda l1,l2 : tuple(map(lambda i, j: i + j, l1, l2))

        lil_h = 20
        lil_w = 80
        top_left = sub(bottom_right, (-lil_w, -lil_h) )
        cv.rectangle(frame, top_left, bottom_right, (0, 0, 0), thickness=-1)
        cv.rectangle(frame, top_left, bottom_right, (255,255,255), thickness=1)

        origin = sub(top_left,(lil_w//2-25,lil_h//2 + 4))
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (255, 255, 255)
        thickness = 1
        cv.putText(frame, text, origin, font, fontScale, color, thickness)

    def detect_dis_haar_mouth(detector, frame, face, x, y):
        lip_rect = detector.detectMultiScale(face, scaleFactor=1.03, minNeighbors=15)
        mouth = False
        for (x1, y1, w1, h1) in lip_rect:
            cv.rectangle(frame, (x1 + x, y1 + y), (x1 + x + w1, y1 + h1 + y), (0, 0, 255), thickness=2)
            mouth = True

        if mouth:
            return mask_off_msg , (0,0,255)
        return mask_on_msg, (0,255,0)


    # Dlib Dectector
    gframe = cv.cvtColor(dlib_frame, cv.COLOR_BGR2GRAY)
    faces = face_detector(gframe)
    for face in faces:
        landmarks = landmark_detector(gframe, face)
        # for i in range(68):
        #     landmark = landmarks.part(i)
        #     cv.circle(dlib_frame, (landmark.x, landmark.y), 5, (50, 50, 255), cv.FILLED)
        cv.rectangle(dlib_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
        dis_mask(dlib_frame, (face.right(), face.top()), mask_off_msg)

    # HAAR detector
    haar_face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    haar_lip = cv.CascadeClassifier("haarcascade_smile.xml")
    face_rect = haar_face.detectMultiScale(haar_frame, scaleFactor=1.05, minNeighbors=3)
    for (x, y, w, h) in face_rect:
        msg,color = detect_dis_haar_mouth(haar_lip, haar_frame, haar_frame[x:x + w, y:y + h], x, y)
        cv.rectangle(haar_frame, (x, y), (x + w, y + h), color, thickness=2)
        dis_mask(haar_frame, (x + w, y), msg)
    # try:
    #     if num_detect[0][0] > 0:
    #         mask_on = True
    #     else:
    #         mask_on = False
    # except Exception as e:
    #     mask_on = False

    # DNN detector
    blob = cv.dnn.blobFromImage(dnn_frame, 1.0, (300, 300), [104, 117, 123], False, False)
    dnn.setInput(blob)
    face = dnn.forward()
    h, w = dnn_frame.shape[:2]
    for i in range(face.shape[2]):
        confid = face[0, 0, i, 2]
        if confid > 0.5:
            x1 = int(face[0, 0, i, 3] * w)
            y1 = int(face[0, 0, i, 4] * h)
            x2 = int(face[0, 0, i, 5] * w)
            y2 = int(face[0, 0, i, 6] * h)

            offset = 10
            #With Haar
            msg,color = detect_dis_haar_mouth(haar_lip, dnn_frame, dnn_frame[x1:x2, y1:y2], x1, y1)
            # cv.imshow("sdf",dnn_frame[x1-offset:x2+offset, y1-offset:y2+offset])
            cv.rectangle(dnn_frame, (x1, y1), (x2, y2), color, thickness=2)
            dis_mask(dnn_frame, (x1+ abs(x2-x1), y1), msg)

            #With Dlib

            gframe = cv.cvtColor(dnn_varied_frame, cv.COLOR_BGR2GRAY)
            dfaces = face_detector(gframe)
            msg = mask_on_msg
            color = (0,255,0)
            for dface in dfaces:
                # landmarks = landmark_detector(gframe, face)
                msg = mask_off_msg
                color = (0, 0, 255)
                break
            cv.rectangle(dnn_varied_frame, (x1, y1), (x2, y2), color, thickness=2)
            dis_mask(dnn_varied_frame, (x1 + abs(x2 - x1), y1), msg)

    # display
    text_box = np.zeros((50, temp.shape[1], 3), np.uint8)
    def makeText(frame, text, img):
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2
        frame[:] = (0, 0, 0)
        (wid,hei),bas = cv.getTextSize(text,font,fontScale,thickness)
        hei += bas
        origin = ((frame.shape[1]-wid)//2 , (frame.shape[0]+hei)//2)
        cv.putText(frame, text, origin, font, fontScale, color, thickness)
        return np.concatenate((text_box, img), axis=0)


    dlib_frame = makeText(text_box, 'Dlib', dlib_frame)
    haar_frame = makeText(text_box, 'Haar with Haar', haar_frame)
    dnn_frame = makeText(text_box, 'DNN with Haar', dnn_frame)
    dnn_varied_frame = makeText(text_box, 'DNN with Dlib', dnn_varied_frame)

    line1 = np.concatenate((dlib_frame, haar_frame), axis=1)
    line2 = np.concatenate((dnn_frame, dnn_varied_frame), axis=1)
    final_frame = np.concatenate((line1, line2), axis=0)

    cv.imshow("Mask Detector", final_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()