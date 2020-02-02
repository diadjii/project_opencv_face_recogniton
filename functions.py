import face_recognition
import cv2
import time
import os
import numpy as np
import glob
from imutils import paths
from PIL import Image

#cette fonction devra etre supprimer
def alignFace(image, face_locations, face_landmarks, desiredFaceWidth, desiredFaceHeight):

    '''
    Let's find and angle of the face. First calculate
    the center of left and right eye by using eye landmarks.
    '''
    leftEyePts = face_landmarks[0]['left_eye']
    rightEyePts = face_landmarks[0]['right_eye']

    leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")

    leftEyeCenter = (leftEyeCenter[0],leftEyeCenter[1])
    rightEyeCenter = (rightEyeCenter[0],rightEyeCenter[1])

    # draw the circle at centers and line connecting to them
    cv2.circle(image, leftEyeCenter, 2, (255, 0, 0), 10)
    cv2.circle(image, rightEyeCenter, 2, (255, 0, 0), 10)
    cv2.line(image, leftEyeCenter, rightEyeCenter, (255,0,0), 10)

    # find and angle of line by using slop of the line.
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # to get the face at the center of the image,
    # set desired left eye location. Right eye location
    # will be found out by using left eye location.
    # this location is in percentage.
    desiredLeftEye=(0.35, 0.35)
    #Set the croped image(face) size after rotaion.
    desiredFaceWidth = 128
    desiredFaceHeight = 128

    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    (y2,x2,y1,x1) = face_locations[0]

    output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)

    return output

def getAllImages():
    """ Cette fonction permet d'aller recuperer la base de données des images
    """
    known_face_encodings = []
    known_face_names = []
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'images/')

    #make an array of all the saved jpg files' paths
    list_of_files = [f for f in glob.glob(path+'*.png')]
    #find number of known faces
    number_files = len(list_of_files)

    names = list_of_files.copy()

    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        if(len(face_recognition.face_encodings(globals()['image_{}'.format(i)])) > 0):
            globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
            known_face_encodings.append(globals()['image_encoding_{}'.format(i)])
        else :
            print("non")
        # Create array of known names
        # names[i] = names[i].replace("images/", "")
        known_face_names.append(names[i])

    return known_face_encodings, known_face_names

def detect(knownFace):
    # image = face_recognition.load_image_file(knownFace)
    face_locations = face_recognition.face_locations(knownFace, model="hog")
    face_landmarks = face_recognition.face_landmarks(knownFace)

    if len(face_locations) > 0:

        (top,right,bottom,left) = face_locations[0]
        desiredWidth = (right-left)
        desiredHeight = (bottom-top)

        # align_f = alignFace(image, face_locations, face_landmarks, desiredWidth,desiredHeight)

        if(len(face_recognition.face_encodings(knownFace, num_jitters=20)) > 0) :
            known_face_encoding = face_recognition.face_encodings(knownFace, num_jitters=10)[0]

            known_face_encodings, known_face_names = getAllImages()

            name = ""
            values = {}

            for im in known_face_names:
                d = get_distance(knownFace, im)

                if d != None:
                    values[d] =im.replace("images/","")

            ind = min(values)
            if ind != None:
                name = values[ind]
        else:
            name = "Inconnu"

        return name

def get_distance(knownFace,unknownFace):
    """Cette fonction permet de calculer la distance entre 2 images pour savoir
        si c'est la même personne ou pas
    """
    face_locations = face_recognition.face_locations(knownFace, model="hog")
    face_landmarks = face_recognition.face_landmarks(knownFace)

    if len(face_locations) > 0:#on teste s'il a trouve le visage d'une personne
        (top,right,bottom,left) = face_locations[0]
        desiredWidth = (right-left)
        desiredHeight = (bottom-top)

        # align_f = alignFace(image, face_locations, face_landmarks, desiredWidth,desiredHeight)

        if(len(face_recognition.face_encodings(knownFace, num_jitters=20)) > 0) :
            known_face_encoding = face_recognition.face_encodings(knownFace, num_jitters=10)[0]

            image = face_recognition.load_image_file(unknownFace)
            unknownFace_locations = face_recognition.face_locations(image, model="hog",number_of_times_to_upsample = 2)

            if( len(unknownFace_locations) > 0 ):
                face_landmarks = face_recognition.face_landmarks(image)

                (top,right,bottom,left) = unknownFace_locations[0]
                desiredWidth = (right-left)
                desiredHeight = (bottom-top)

                align_f = alignFace(image, face_locations, face_landmarks, desiredWidth, desiredHeight)

                #calculate face encodings of align face. It is array of 128 length.
                if(len(face_recognition.face_encodings(image, num_jitters=20)) > 0) :
                    unknown_face_encoding = face_recognition.face_encodings(image, num_jitters=10)[0]

                    distance = face_recognition.face_distance([known_face_encoding], unknown_face_encoding)[0]

                    return distance

def find_person(img):
    image = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(image, model="hog")
    font = cv2.FONT_HERSHEY_DUPLEX
    face_landmarks = face_recognition.face_landmarks(image)

    for (top,right,bottom,left), landmarks in zip(face_locations,face_landmarks):

        #enregistrer chaque visage comme une image et l'enregistrer
        crop_img = image[top:bottom+0, left:right+10]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        #recuperation du nom de la personne
        name =  detect(crop_img)
        if ".png" in name:
            name = name.split("/")[-1]

        cv2.rectangle(image,(left,bottom),(right,top),(255,0,0),2)

        cv2.putText(image, name, (right+10,top+50), font, 0.7, (248, 24, 148), 1)
    cv2.imshow("Faces found", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

