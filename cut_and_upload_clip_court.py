import threading
import os
import cv2
import numpy as np
import time
import datetime
import pdb
import re
import ast
import sys
import heapq
import subprocess
import yaml

import split_clips_help as sc
from class_predict import *
from predict_code.frame_predict import frame_predict, crop_each_player

print('path ', os.getcwd())
sys.path.append('YOLOv6')
from YOLOv6.tools.a2pics import run
from end import process_frame
from YOLOv6.yolov6.core.inferer_img import Inferer
from multiprocessing import Process
from datetime import datetime, timedelta
from types import SimpleNamespace


webcam=False
webcam_addr=0
round_yaml='./YOLOv6/data/coco.yaml'
device='1'
half=False
weights = './YOLOv6/runs/train/exp17/weights/best_ckpt.pt'
img_size = [1600, 1600]
inferer = Inferer(webcam, webcam_addr, weights, device, round_yaml, img_size, half)

# 全局定义回合开始识别的参数

start_flag = None 
get_round_end = True

# 全局定义拍数相关变量
frame_idx_prev, x_prev, y_prev = 0, 0, 0   # 上一帧x y不为0的帧
frame_idx_prev_ball = 0
flag_same_idx  = 0
flag_already_crossed = False


################### ffmpeg剪辑 ####################################
def cut_clip_in_court(fps, folder_path, record_clip_in_court_list, Court_Name):

    all_files = os.listdir(folder_path)
    input_video_name = [file for file in all_files if file.endswith(('.mp4'))]
    input_video = os.path.join(folder_path, input_video_name[0])

    upload_file = []
    output_video_file = []

    for i in range(0, len(record_clip_in_court_list) - 1, 2) if len(record_clip_in_court_list) % 2 == 0 else range(0, len(record_clip_in_court_list) - 2, 2):
        p1 = None
        clip_start_time = (int(record_clip_in_court_list[i]))/fps 
        clip_end_time = (int(record_clip_in_court_list[i+1]))/fps
        clip_during_time = clip_end_time - clip_start_time 
        output_video = os.path.join(folder_path, f"{Court_Name}_{record_clip_in_court_list[i]}.mp4")
        output_video_file.append(output_video)
        if clip_during_time > 5:
            ffmpeg_command = ['ffmpeg', '-ss', str(clip_start_time), '-i', str(input_video), '-t', str(clip_during_time), '-c:v', 'copy', '-c:a', 'aac', str(output_video)]
            upload_file.append(output_video)
            p1 = subprocess.run(ffmpeg_command)
        
class MinHeap:
    def __init__(self):
        self.data = {}
        self.heap = []

    def update(self, key, value):
        self.data[key] = value
        # 维护堆，保留最大的十个值
        if len(self.heap) < 10:
            heapq.heappush(self.heap,(value, key))
        else:
            if value > self.heap[0][0]:
                heapq.heappushpop(self.heap, (value, key))

    def get_sorted_dict(self):
        sorted_dict = {key: self.data[key] for _, key in sorted(self.heap, reverse = True)}
        return sorted_dict
        
def split_clip(save_origin_video_file,Court_Name,polygon, net_ball_threshold, crop_frame):

    ball_num = 0
    frame_cal = 0
    frame_key = 0
    tmp_flag = True

    polygon = ast.literal_eval(polygon)
    record_clip_in_court_list = []
    global get_round_end
    video = cv2.VideoCapture(save_origin_video_file, cv2.CAP_FFMPEG)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    # fps = 25
    all_clip = MinHeap()  
    clip_test_num = 0
    round_start_flag = False
    round_end_flag = False

    frame_test_num = 0
    clip_output_video_path_dir = os.path.dirname(save_origin_video_file)
    clip_output_txt_path = os.path.join(clip_output_video_path_dir, f'{os.path.basename(save_origin_video_file)[:-4]}.txt')

    tracknet_list = []
    tracknet_list.append('Frame,Visibility,X,Y')
    if not video.isOpened():
        print("Unable to open video")
        exit()
    try:
        while True: 
            for i in range(fps - 1):
                ret, frame = video.read()
                frame_test_num += 1
                if ret:
                    crop_frame_list = re.findall(r'\d+', crop_frame)
                    numbers = [int(match) for match in crop_frame_list]
                    frame = frame[numbers[0]:numbers[1], numbers[2]:numbers[3], :]
                    tracknet_list.append(f"{frame_test_num},0,0,0")
                    if i == (fps - 1)// 2:
                        clip_test_num += 1

                    if i == (fps - 1) // 2 and not round_start_flag: 
                        cls, conf, xywhn, xywh, xyxy, boxes, orig_img, annotated_frame = frame_predict(frame)
                        xywh, xyxy, cls, conf, xywhn = select_player_in_court(orig_img, xywh, xyxy, cls, conf, xywhn,polygon)
                        human_rects = get_human_rects(cls, conf)
                        round_start_flag = sc.round_start_function(cls, xywhn, human_rects, clip_test_num)
                        first_start_flag = True

                    if round_start_flag and first_start_flag:
                        # clip_output_name = os.path.join(clip_output_video_path_dir, f'{Court_Name}_{frame_test_num}.mp4')
                        # record_clip_in_court_list.append(f"{frame_test_num}")

                        # 初始化变量
                        previous_x, previous_y = 0, 0  # 上一帧球的位置
                        zero_count = 0  # 连续检测到的x, y为0的帧数
                        seg = [] # 回合结束时间

                        started_append = False # 回合开始标志
                        first_start_flag = False
                        round_end_flag = True

                        crossings = 0
                        count_clip_run = 1
                        count_pause_run = 0
                        speed = 0
                        count_speed1 = 0
                        count_speed2 = 0
                        count_speed3 = 0
                        crossings_timestamps= []
                        predictions = []

                    
                    if round_end_flag:
                        frame_cal +=1
                        round_frame, round_x, round_y = run(inferer, frame)  
                        if tracknet_list:
                            last_item = tracknet_list[-1]
                            last_num_part = last_item.split(',')[0]
                            if int(last_num_part) == frame_test_num:
                                tracknet_list[-1] = f"{frame_test_num},1,{round_x},{round_y}"
                            else:
                                tracknet_list.append(f"{frame_test_num},1,{round_x},{round_y}")
                        else:
                            tracknet_list.append(f"{frame_test_num},1,{round_x},{round_y}")

                        if round_x != 0 or round_y != 0:
                            ball_num += 1
                        # draw_rectangle(frame, round_x, round_y, frame_test_num)
                        count_pause_run = 1 + count_pause_run                  
                        crossings, speed, crossings_timestamps, predictions = sc.record_total_crossings(crossings, crossings_timestamps, int(round_frame), round_x, round_y, predictions, fps, net_ball_threshold)
                        if 144 > speed >= 120:
                            count_speed1 = count_speed1 + 1
                        elif 160 > speed >= 144:
                            count_speed2 = count_speed2 + 1
                        elif speed >= 160:
                            count_speed3 = count_speed3 + 1
                        # if crossings != 0:
                        #     count_clip_run = count_clip_run + 1
                        #     count_pause_run = -1 + count_pause_run  
                        count_clip_run = count_clip_run + 1 
                        started_append, zero_count, previous_x, previous_y, seg = process_frame(round_frame, round_x, round_y, started_append, zero_count, previous_x, previous_y, seg)
                        if started_append and tmp_flag:
                            frame_key = frame_test_num
                            record_clip_in_court_list.append(f"{frame_test_num-70}")
                            clip_output_name = os.path.join(clip_output_video_path_dir, f'{Court_Name}_{frame_test_num-50}.mp4')

                            tmp_flag = False

                        if seg:
                            # 回合结束
                            # if crossings > 4 :
                            all_clip.update(f"{clip_output_name}__{ball_num}__{frame_cal}__{frame_key}", crossings *160 / count_clip_run + 0.6 * count_speed2 + count_speed3)
                            if ball_num > 40 :
                                record_clip_in_court_list.append(f"{frame_test_num}")
                            else:
                                record_clip_in_court_list.append(f"{frame_key}")
                            ball_num = 0
                            frame_cal = 0
                            tmp_flag = True
                            frame_key = 0
                            round_end_flag = False
                            round_start_flag = False
                            get_round_end = True
                else:
                    break
            ret, frame = video.read()

            if ret:                 
                try:
                    crop_frame_list = re.findall(r'\d+', crop_frame)
                    numbers = [int(match) for match in crop_frame_list]
                    frame = frame[numbers[0]:numbers[1], numbers[2]:numbers[3], :]

                    
                    cls, conf, xywhn, xywh, xyxy, boxes, orig_img, annotated_frame = frame_predict(frame)
                    xywh, xyxy, cls, conf, xywhn = select_player_in_court(orig_img, xywh, xyxy, cls, conf, xywhn, polygon)
                    human_rects = get_human_rects(cls,conf)
                    clip_test_num += 1
                    frame_test_num += 1

                    # draw_pic(cls, xywhn, annotated_frame, play_flag, record_flag, session_num, clip_test_num, window)

                    # 如果场次开始，则进行回合开始判断
                    if not round_start_flag: 
                        round_start_flag = sc.round_start_function(cls, xywhn, human_rects, clip_test_num)
                        first_start_flag = True

                    if round_start_flag and first_start_flag:
                        # clip_output_name = os.path.join(clip_output_video_path_dir, f'{Court_Name}_{frame_test_num}.mp4')
                        # record_clip_in_court_list.append(f"{frame_test_num}")
                        # 初始化变量
                        previous_x, previous_y = 0, 0  # 上一帧球的位置
                        zero_count = 0  # 连续检测到的x, y为0的帧数
                        seg = [] # 回合结束时间

                        started_append = False # 回合开始标志
                        first_start_flag = False
                        round_end_flag = True

                        crossings = 0
                        count_clip_run = 1
                        count_pause_run = 0
                        flag_correct_start = True
                        speed = 0
                        count_speed1 = 0
                        count_speed2 = 0
                        count_speed3 = 0
                        crossings_timestamps= []
                        predictions = []

                    
                    if round_end_flag:
                        frame_cal += 1
                        round_frame, round_x, round_y = run(inferer, frame)  
                        if round_x != 0 or round_y != 0:
                            ball_num += 1
                        # draw_rectangle(frame, round_x, round_y, frame_test_num)
                        count_pause_run = 1 + count_pause_run                  
                        crossings, speed, crossings_timestamps, predictions = sc.record_total_crossings(crossings, crossings_timestamps, int(round_frame), round_x, round_y, predictions, fps, net_ball_threshold)
                        if 144 > speed >= 120:
                            count_speed1 = count_speed1 + 1
                        if 160 > speed >= 144:
                            count_speed2 = count_speed2 + 1
                        elif speed >= 160:
                            count_speed3 = count_speed3 + 1
                        # if crossings != 0:
                        #     count_clip_run = count_clip_run + 1
                        #     count_pause_run = -1 + count_pause_run    
                        count_clip_run = count_clip_run + 1
                        started_append, zero_count, previous_x, previous_y, seg = process_frame(round_frame, round_x, round_y, started_append, zero_count, previous_x, previous_y, seg)
                        if started_append and tmp_flag:
                            frame_key = frame_test_num
                            record_clip_in_court_list.append(f"{frame_test_num-70}")
                            clip_output_name = os.path.join(clip_output_video_path_dir, f'{Court_Name}_{frame_test_num-50}.mp4')
                            tmp_flag = False

                        if seg:
                            # 回合结束
                            # if crossings > 4 :
                            all_clip.update(f"{clip_output_name}__{ball_num}__{frame_cal}__{frame_key}", crossings *160 / count_clip_run + 0.6 * count_speed2 + count_speed3)
                            if ball_num > 40 :
                                record_clip_in_court_list.append(f"{frame_test_num}")
                            else:
                                record_clip_in_court_list.append(f"{frame_key}")
                            ball_num = 0
                            frame_cal = 0
                            tmp_flag = True
                            frame_key = 0
                            round_end_flag = False
                            round_start_flag = False
                            get_round_end = True
                            

                finally:
                        pass
            else:
                break
    finally:
        with open(clip_output_txt_path, "a") as file:
            if len(record_clip_in_court_list) != 0:
                for i in record_clip_in_court_list[:-1]:
                    file.write(f"{i}\n")
                file.write(f"{record_clip_in_court_list[-1]}")


        video_dir, video_filename = os.path.split(clip_output_txt_path)
        video_basename, video_extension = os.path.splitext(video_filename)
        txt_filename = f"{video_basename}_tracknet.txt"
        txt_file_path = os.path.join(video_dir, txt_filename)
        with open(txt_file_path, 'w') as txt_file:
            for item in tracknet_list:
                txt_file.write(item + '\n') 

        with open("all_clip.txt", "a") as file:
            file.write(f"{all_clip.data}\n")

        fancy_clip = all_clip.get_sorted_dict()
        with open("fancy_clip.txt", "w") as file:
            file.write(f"\n{fancy_clip}")
        video.release()
        cut_clip_in_court(fps, clip_output_video_path_dir, record_clip_in_court_list, Court_Name)

if __name__ == "__main__":
    import sys
    current_path = os.getcwd()
    uploader_clip_video_python_path = os.path.join(current_path, 'uploader_sub.py')
    uploader_origin_video_python_path = os.path.join(current_path, 'uploader_ori.py')


    save_origin_video_file = '/ssd2/wmx/court/save_video/2024-05-14_19-38-39_the_28_court/2024-05-14_19-38-39_the_28_court.mp4'
    session_uuid = "1"
    session_event_timestamp = 11
    session_start_time = 0
    session_end_time = 30
    # Court_Name = "2024-05-21_21-21-44_the_42_court"
    Court_Name = save_origin_video_file.split('/')[-1][:-4]
    court_int_num = 2
    venue_id_num = 9
    polygon = "[(640, 340), (405, 1015), (1590, 1015), (1360,340)]"
    net_ball_threshold = 400
    crop_frame = "[0:1080, 0:1920 , :]"
    uploader_url = "https://haoqiuwa-test-75067-5-1318337180.sh.run.tcloudbase.com/videos/event/v1"
    uploader_surroundings = "cloud://test-5guvnyfne292a8b9.7465-test-5guvnyfne292a8b9-1318337180"
    cloudEnvId = "test-5guvnyfne292a8b9" 
    print(Court_Name)

    split_clip(save_origin_video_file,Court_Name, polygon, net_ball_threshold, crop_frame)
    
