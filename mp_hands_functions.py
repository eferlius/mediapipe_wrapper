# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 06:12:35 2023

@author: eferlius
"""
import mediapipe as mp
import numpy as np

# aliasing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def find_results_on_img(img, stat_im_mode = False, max_n_hands = 2, min_det_conf = 0.1, min_track_conf = 0.5):
    with mp_hands.Hands(static_image_mode=stat_im_mode, max_num_hands=max_n_hands, min_detection_confidence=min_det_conf, min_tracking_confidence=min_track_conf) as hands:
        results = hands.process(img)
        return results

def create_header(init = ['frame', 'time'], hands_names = ['h0'],  variables = ['x','y','z'], n_key_points = 21,):
    for hand_name in hands_names:
        init.extend(['{}_label'.format(hand_name), '{}_score'.format(hand_name)])
        init.extend(['{}_{}{:02d}'.format(hand_name,var,num) for num in range(n_key_points) for var in variables])   
    return init

def from_results_to_list(results, init = [], variables = ['x','y','z'], n_key_points = 21, n_hands = 1, decimals = 3):
    nan_list =  [np.nan]*(n_hands*(2+len(variables)*n_key_points)) # 2 for label and score of each hand
    if not results.multi_hand_landmarks or not results.multi_handedness:
        init.extend(nan_list)
        return_list = init
    else:
        hands_list = init
        n_found_hands = len(results.multi_hand_landmarks)
        for i in range(min(n_hands, n_found_hands)):
            # label = left or right
            label = results.multi_handedness[i].classification[0].label
            score = results.multi_handedness[i].classification[0].score
            score = np.around(score, decimals)
            hand_list = [label, score]
            # extract all keypoint of the hand
            for index, landmark in enumerate(results.multi_hand_landmarks[i].landmark):
                for var in variables:
                    if var == 'x':
                        value = landmark.x
                    elif var == 'y':
                        value = landmark.y
                    elif var == 'z':
                        value = landmark.z
                    else:
                        break
                    value = np.around(value, decimals)
                    hand_list.append(value)
            hands_list.extend(hand_list)
            
        hands_list.extend([np.nan]*(len(nan_list)-len(hands_list)))
        return_list = hands_list
    return return_list
    
def draw_mp_on_image(img, results):
    annotated_image = img.copy()
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks,
            mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return annotated_image