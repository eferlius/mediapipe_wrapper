# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:27:28 2023

@author: ichino
"""

import cv2
import os
import numpy as np
import mediapipe as mp
import mp_hands_functions
import mp_pose_functions
import basic_timer
import basic_utils

def videoOrLiveRecognition(video_path):
    try:
        video_path = int(video_path)
        return 'live'
    except ValueError:
        return 'video'
    
VIDEO_PATH = 0
# VIDEO_PATH = 1
# VIDEO_PATH = r'path to video'

video_live = videoOrLiveRecognition(VIDEO_PATH)

# in case of live view
LIVE_FREQ = 30 # int, estimation of the elapsed time between each iteration
MAX_LIVE_DURATION = 10 # seconds
MAX_LIVE_FRAMES = 2000 # int

# whether to analyze the hand or the pose or both
HAND = True
POSE = False

WRITE_ON_FRAME = False

DISPLAY_RAW_VIDEO = False
SAVE_RAW_VIDEO = False
SAVE_RAW_CSV = False
# with elaborations of mediapipe
DISPLAY_VIDEO = True
SAVE_VIDEO = False
SAVE_CSV = False

PRINT_EXECUTION = True

if video_live == 'live':
    FOLDER_SAVING = os.getcwd()
if video_live == 'video':
    FOLDER_SAVING = os.path.split(VIDEO_PATH)[0]

# ----------------------------- PARAMETERS
mp_hands = mp.solutions.hands
SIMh = False # static_image_mode
MNH = 2 # max_num_hands
MDCh = 0.1 # min_detection_confidence
MTCh = 0.5 # min_tracking_confidence

mp_pose = mp.solutions.pose
SIMp = False # static_image_mode
MC = 1 # model_complexity
MDCp = 0.1 # min_detection_confidence
MTCp = 0.5 # min_tracking_confidence

# to write on the image
STRING_FORMAT = 'frame: {:05d} time: {:03.3f}'
font = cv2.FONT_HERSHEY_SIMPLEX
origin = (20, 20)
fontScale = 0.5
color = (0, 0, 0) #BGR format
thickness = 1


if SAVE_RAW_VIDEO or SAVE_RAW_CSV or SAVE_VIDEO or SAVE_CSV:
    folder_saving = os.path.join(FOLDER_SAVING, 'mp execution ' + basic_utils.this_moment.this_moment())
    os.makedirs(folder_saving, exist_ok = True)
if SAVE_RAW_CSV:
    csv_raw = os.path.join(folder_saving, 'times.csv')
if SAVE_CSV:
    if HAND:
        csv_hand = os.path.join(folder_saving, 'hand.csv')
    if POSE:
        csv_pose = os.path.join(folder_saving, 'pose.csv')
        
if SAVE_RAW_VIDEO:
    video_path_raw = os.path.join(folder_saving, 'raw.mp4')    
if SAVE_VIDEO:
    if HAND:
        video_path_hand = os.path.join(folder_saving, 'hand.mp4')
    if POSE:
        video_path_pose = os.path.join(folder_saving, 'pose.mp4')


with mp_hands.Hands(static_image_mode=SIMh, max_num_hands=MNH, min_detection_confidence=MDCh, min_tracking_confidence=MTCh) as hands:
    with mp_pose.Pose(static_image_mode=SIMp, model_complexity=MC, min_detection_confidence=MDCp, min_tracking_confidence=MTCp) as pose:
        # initialize capture
        capture = cv2.VideoCapture(VIDEO_PATH)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        
        if video_live == 'live':
            video_freq = LIVE_FREQ 
            max_frames = MAX_LIVE_FRAMES 
            max_duration = MAX_LIVE_DURATION 
            
        elif video_live == 'video':
            video_freq = capture.get(cv2.CAP_PROP_FPS)
            max_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            max_frames += 3  # slightly increase values
            max_duration = max_frames / video_freq
        
        # initialize csv header
        init = ['frame', 'time']
        if SAVE_RAW_CSV:
            basic_utils.csv_ext.write_row_csv(csv_raw, init)
        if SAVE_CSV:
            if HAND:
                header = mp_hands_functions.create_header(init, hands_names = ['h{}'.format(i) for i in range(MNH)])
                basic_utils.csv_ext.write_row_csv(csv_hand, header)
            if POSE:
                header = mp_pose_functions.create_header(init)
                basic_utils.csv_ext.write_row_csv(csv_pose, header)
                
        # initialize video writers
        if SAVE_RAW_VIDEO:
            video_writer_raw = cv2.VideoWriter(video_path_raw, fourcc, video_freq, (frame_width, frame_height))
        if SAVE_VIDEO:
            if HAND:
                video_writer_hand = cv2.VideoWriter(video_path_hand, fourcc, video_freq, (frame_width, frame_height))
            if POSE:
                video_writer_pose = cv2.VideoWriter(video_path_pose, fourcc, video_freq, (frame_width, frame_height))            
        
        counter = -1
        elapsed = 0
        t = basic_timer.timer.Timer()
        
        while (elapsed <= max_duration and counter <= max_frames) :
            counter+=1
            ret, orig_frame = capture.read()
            if video_live == 'live':
                orig_frame = cv2.flip(orig_frame, 1)
                elapsed = t.elap(printTime=False)
                elapsed = np.around(elapsed, 3)
            else:
                elapsed = counter/video_freq

            if ret:
                if WRITE_ON_FRAME:
                    stringForImage =  STRING_FORMAT.format(counter, elapsed)
                    frame = cv2.putText(orig_frame, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                else:
                    frame = orig_frame
                if DISPLAY_RAW_VIDEO:    
                    cv2.imshow('video', frame)
                
                if SAVE_RAW_CSV:
                    basic_utils.csv_ext.write_row_csv(csv_raw, [counter, elapsed])
                if SAVE_RAW_VIDEO:
                    video_writer_raw.write(frame)
                
                if HAND:
                    results = hands.process(orig_frame)
                    frame_h = mp_hands_functions.draw_mp_on_image(frame, results)
                    if DISPLAY_VIDEO:
                        cv2.imshow('hands', frame_h)
                    
                    if SAVE_VIDEO:
                        video_writer_hand.write(frame_h)
                    if SAVE_CSV:
                        row_h = mp_hands_functions.from_results_to_list(results, init = [counter, elapsed], n_hands = MNH)
                        basic_utils.csv_ext.write_row_csv(csv_hand, row_h)
                if POSE:
                    results = pose.process(orig_frame)
                    frame_p = mp_pose_functions.draw_mp_on_image(frame, results)
                    if DISPLAY_VIDEO:
                        cv2.imshow('pose', frame_p)
                    if SAVE_VIDEO:
                        video_writer_pose.write(frame_p)
                    if SAVE_CSV:
                        row_p = mp_pose_functions.from_results_to_list(results, init = [counter, elapsed])
                        basic_utils.csv_ext.write_row_csv(csv_pose, row_p)
                        
            if PRINT_EXECUTION:
                print(STRING_FORMAT.format(counter, elapsed))
            
            # press esc to exit
            if cv2.waitKey(1) == 27:
                break

elapsed = t.stop()
print('{c:.0f} frames in {e:.2f} seconds'.format(c=counter, e=elapsed))
print('{f:.2f} Hz'.format(f=(counter/elapsed)))

capture.release()
if SAVE_RAW_VIDEO:
    video_writer_raw.release()
if SAVE_VIDEO:
    if HAND:
        video_writer_hand.release()
    if POSE:
        video_writer_pose.release()
 
cv2.destroyAllWindows()

