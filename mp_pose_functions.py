# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 06:13:14 2023

@author: eferlius
"""
import mediapipe as mp
import numpy as np

# aliasing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def find_results_on_img(img, stat_im_mode = False, mod_compl = 2, min_det_conf = 0.1, min_track_conf = 0.5):
   with mp_pose.Pose(static_image_mode=stat_im_mode, model_complexity=mod_compl, min_detection_confidence=min_det_conf, min_tracking_confidence=min_track_conf) as pose:
        results = pose.process(img)
        return results

def create_header(init = ['frame', 'time'], variables = ['x','y','z', 'vis'], n_key_points = 33):
    init.extend(['{}{:02d}'.format(var,num) for num in range(n_key_points) for var in variables])   
    return init

def from_results_to_list(results, init = [], variables = ['x','y','z', 'visibility'], n_key_points = 33, decimals = 3, landmark_WorldLandmark = 'landmark'):
    return_list = init
    assert landmark_WorldLandmark == 'landmark' or landmark_WorldLandmark == 'WorldLandmark'
    if landmark_WorldLandmark == 'landmark':
        res = results.pose_landmarks
    elif landmark_WorldLandmark == 'WorldLandmark':
        res = results.pose_world_landmarks
    if res:
        for i in range(len(res.landmark)):
            for var in variables:
                if var == 'x':
                    value = res.landmark[i].x
                elif var == 'y':
                    value = res.landmark[i].y
                elif var == 'z':
                    value = res.landmark[i].z
                elif var == 'visibility':
                    value = res.landmark[i].visibility
                else:
                    break
                value = np.around(value, decimals)
                return_list.append(value)   
        return return_list
    else:
        return_list.extend([np.nan]*len(variables)*n_key_points)
    
    return return_list        
    
def draw_mp_on_image(img, results):
    annotated_image = img.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(annotated_image, 
            results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
