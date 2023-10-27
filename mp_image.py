# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:37:22 2023

@author: ichino
"""

import cv2
import os
import mediapipe as mp
import mp_hands_functions
import mp_pose_functions
import basic_utils

    
IMG_PATH = r'G:\My Drive\python projects\mediaPipeExecution\example image\example2.jpg'
IMG_PATH = r'G:\My Drive\python projects\mediaPipeExecution\example image\hand4.jpg'

FOLDER_NAME = os.path.split(IMG_PATH)[0]
FILE_COMPLETE_NAME = os.path.split(IMG_PATH)[1]
FILE_NAME = os.path.splitext(FILE_COMPLETE_NAME)[0]
FILE_EXT = os.path.splitext(FILE_COMPLETE_NAME)[1]

# whether to analyze the hand or the pose or both
HAND = True
POSE = True

# with elaborations of mediapipe
SAVE_CSV = True
SAVE_IMG = True

# ----------------------------- PARAMETERS
mp_hands = mp.solutions.hands
SIMh = True # static_image_mode
MNH = 2 # max_num_hands
MDCh = 0.1 # min_detection_confidence
MTCh = 0.1 # min_tracking_confidence

mp_pose = mp.solutions.pose
SIMp = True # static_image_mode
MC = 1 # model_complexity
MDCp = 0.1 # min_detection_confidence
MTCp = 0.5 # min_tracking_confidence

folder_saving = os.path.join(FOLDER_NAME, 'mp execution ' + basic_utils.this_moment.this_moment())
os.makedirs(folder_saving, exist_ok = True)
if SAVE_CSV:
    if HAND:
        csv_hand = os.path.join(folder_saving, 'hand.csv')
        header = mp_hands_functions.create_header(init = [], hands_names = ['h{}'.format(i) for i in range(MNH)])
        basic_utils.csv_ext.write_row_csv(csv_hand, header)
    if POSE:
        csv_pose = os.path.join(folder_saving, 'pose.csv')
        header = mp_pose_functions.create_header(init = [])
        basic_utils.csv_ext.write_row_csv(csv_pose, header)  

frame = cv2.imread(IMG_PATH)

if HAND:
    results = mp_hands_functions.find_results_on_img(frame, SIMh, MNH, MDCh, MTCh)
    frame_h = mp_hands_functions.draw_mp_on_image(frame, results)
    cv2.imshow('hand', frame_h)
    cv2.waitKey(1)
    if SAVE_IMG:
        cv2.imwrite(os.path.join(folder_saving, FILE_NAME+'_hand'+FILE_EXT), frame_h)
    if SAVE_CSV:
        row_h = mp_hands_functions.from_results_to_list(results, n_hands = MNH)
        basic_utils.csv_ext.write_row_csv(csv_hand, row_h)
if POSE:
    results = mp_pose_functions.find_results_on_img(frame, SIMp, MC, MDCp, MTCp)
    frame_p = mp_pose_functions.draw_mp_on_image(frame, results)
    cv2.imshow('pose', frame_p)
    cv2.waitKey(1)
    if SAVE_IMG:
        cv2.imwrite(os.path.join(folder_saving, FILE_NAME+'_pose'+FILE_EXT), frame_p)
    if SAVE_CSV:
        row_p = mp_pose_functions.from_results_to_list(results)
        basic_utils.csv_ext.write_row_csv(csv_pose, row_p)
            

                        




