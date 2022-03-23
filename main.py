import math
from playsound import playsound
import _thread
from pygame import mixer
import time
import cv2
import cv2 as cv
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else 0)

mouse_x = 100
mouse_y = 400
mouse_radius = 30
score = 0

def hit(hit_mouse,ok):
    if hit_mouse:
        playsound('music/hit.mp3')
        playsound('music/score.mp3')
        print(ok)

def bgm(play,playing):
    if play:
        mixer.init()
        mixer.music.load('music/bgm.mp3')
        mixer.music.play()
        time.sleep(90)
        mixer.music.stop()
        # playsound('bgm.mp3')
        print(playing)

playing = 'playing bgm'
play = True
_thread.start_new_thread(bgm,(play,playing,))

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    Rhand = out[0, 4, :, :]
    _, conf, _, point = cv.minMaxLoc(Rhand)
    Rhand_x = int((frameWidth * point[0]) / out.shape[3])
    Rhand_y = int((frameHeight * point[1]) / out.shape[2])
    R = (Rhand_x,Rhand_y)

    Lhand = out[0, 7, :, :]
    _, conf, _, point = cv.minMaxLoc(Lhand)
    Lhand_x = int((frameWidth * point[0]) / out.shape[3])
    Lhand_y = int((frameHeight * point[1]) / out.shape[2])
    L = (Lhand_x, Lhand_y)
    cv2.circle(frame, R ,20, (255,0,0) , 2)
    cv2.circle(frame, L, 20, (255, 0, 0),2)
    points = []
    for i in range(len(BODY_PARTS)):

        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    #生成老鼠
    cv.putText(frame, 'score:%d' % score, (300, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (50, 25, 50))
    cv2.circle(frame,(mouse_x,mouse_y),mouse_radius,(0,0,255),cv.FILLED)
    R_in_mouse = math.sqrt((R[0]- mouse_x)*(R[0]- mouse_x)+(R[1]- mouse_y)*(R[1]- mouse_y)) < mouse_radius
    L_in_mouse = math.sqrt((L[0]- mouse_x)*(L[0]- mouse_x)+(L[1]- mouse_y)*(L[1]- mouse_y)) < mouse_radius
    hit_mouse = R_in_mouse or L_in_mouse
    if hit_mouse:
        cv2.circle(frame, (mouse_x, mouse_y), mouse_radius, (0, 255, 0), cv.FILLED)
        # mixer.init()
        # mixer.music.load('hit.mp3')
        # mixer.music.play()
        # time.sleep(0.1)
        # mixer.music.stop()
        ok = 'ok'
        _thread.start_new_thread(hit,(hit_mouse,ok,))
        score = score + 1
        mouse_x = random.randint(30, 610)




    cv.imshow('OpenPose using OpenCV', frame)