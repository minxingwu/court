import threading
from class_predict import *
from predict_code.frame_predict import frame_predict, crop_each_player
from ym_round import ym_move_judge_single, ym_move_judge_double, get_player
# from track4txt import predict4txt,caldis
import os
import cv2
import numpy as np
import time
import datetime
import pdb
import re
import ast

from types import SimpleNamespace
import yaml
# from boxmot.utils import BOXMOT
# from boxmot.trackers.ocsort.ocsort import OCSort
# from home.sportvision.research.badminton_det.YOLOv6.tools import 
import sys
print('path ', os.getcwd())
sys.path.append('YOLOv6')
from YOLOv6.tools.a2pics import run
from end import process_frame
from YOLOv6.yolov6.core.inferer_img import Inferer
import heapq
import subprocess
from multiprocessing import Process
from datetime import datetime, timedelta

webcam=False
webcam_addr=0
round_yaml='./YOLOv6/data/coco.yaml'
device='0'
half=False
weights = './YOLOv6/runs/train/exp17/weights/best_ckpt.pt'
img_size = [1600, 1600]
inferer = Inferer(webcam, webcam_addr, weights, device, round_yaml, img_size, half)


# 全局定义回合开始识别的参数
last_result = []
last_time = []
frame_result = []
start_flag = None 
get_round_end = True

# 全局定义拍数相关变量
frame_idx_prev, x_prev, y_prev = 0, 0, 0   # 上一帧x y不为0的帧
frame_idx_prev_ball = 0
flag_same_idx  = 0
flag_already_crossed = False


################### ffmpeg剪辑 ####################################
def parse_timestamp(timestamp):
    return datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")

def format_timedelta(delta):
    seconds = delta.total_seconds()
    return str(timedelta(seconds=seconds))

def cut_clip_in_court(fps, folder_path, record_clip_in_court_list, court_int_num, venue_id_num, uploader_clip_video_python_path , session_uuid, Court_Name , uploader_url, uploader_surroundings, cloudEnvId, event_timestamp):
    # txt_file = os.path.join(folder_path, f'{os.path.basename(save_origin_video_file)[:-4]}.txt')

    all_files = os.listdir(folder_path)
    input_video_name = [file for file in all_files if file.endswith(('.mp4'))]
    input_video = os.path.join(folder_path, input_video_name[0])


    # with open(os.path.join(folder_path, txt_file), 'r') as file:
    #     record_clip_in_court_list = [line.strip() for line in file.readlines()]

    upload_file = []
    output_video_file = []
    for i in range(0, len(record_clip_in_court_list) - 1, 2) if len(record_clip_in_court_list) % 2 == 0 else range(0, len(record_clip_in_court_list) - 2, 2):
        p1 = None
        clip_start_time = (int(record_clip_in_court_list[i]))/fps 
        clip_end_time = (int(record_clip_in_court_list[i+1]))/fps

        # clip_start_time = timedelta(hours=(timestamp_start/2)//3600, minutes=(timestamp_start/2)//60, seconds=timestamp_start/2 - (timestamp_start/2)//3600 - (timestamp_start/2)//60) - bias1
        # clip_end_time = timedelta(hours=(timestamp_end/2)//3600, minutes=(timestamp_end/2)//60, seconds=timestamp_end/2 - (timestamp_end/2)//3600 - (timestamp_end/2)//60) - bias2
  
        clip_during_time = clip_end_time - clip_start_time 
        # h, m, s = str(clip_during_time).split(':')
        # during_time = int(h*3600 + m*60 + s*1)
        # start_time = '0' + str(clip_start_time)
        output_video = os.path.join(folder_path, f"{Court_Name}_{record_clip_in_court_list[i]}.mp4")
        # clip_start_time = '0' + str(clip_start_time)
        print(clip_during_time)
        print(clip_start_time)
        output_video_file.append(output_video)
        if clip_during_time > 5:
            ffmpeg_command = ['ffmpeg', '-ss', str(clip_start_time), '-i', str(input_video), '-t', str(clip_during_time), '-c:v', 'h264_nvenc', '-c:a', 'aac', str(output_video)]
            # ffmpeg_command = f'ffmpeg -ss {clip_start_time} -i {input_video} -t {clip_during_time} -c:v h264_nvenc -c:a aac {output_video}'
            upload_file.append(output_video)
            p1 = subprocess.run(ffmpeg_command)
        # p1.wait()
    # for output_video in upload_file:
    #     command = [
    #         "python",
    #         uploader_clip_video_python_path,
    #         output_video,
    #         str(event_timestamp),
    #         str(session_uuid),
    #         str(court_int_num),
    #         str(venue_id_num),
    #         str(uploader_url),
    #         str(uploader_surroundings),
    #         str(cloudEnvId)
    #     ]
    #     clip_upload_subprocess = subprocess.run(command)
        
#### 场次视频上传
def rename_and_upload_video(video_path, uploader_origin_video_python_path, event_timestamp, session_uuid, session_start_time, session_end_time, Court_Name,court_int_num, venue_id_num, uploader_url, uploader_surroundings, cloudEnvId):
    try:
        command = [
            "python",
            uploader_origin_video_python_path,
            video_path,
            event_timestamp,
            session_uuid,
            session_start_time,
            session_end_time,
            Court_Name,
            court_int_num, 
            venue_id_num,
            str(uploader_url),
            str(uploader_surroundings),
            str(cloudEnvId)
            # video_time,
        ]
        ori_upload_subprocess = subprocess.Popen(command, stderr=subprocess.PIPE)

        output, error = ori_upload_subprocess.communicate()
        if error:
            print("子进程错误输出：")
            print(error.decode('utf-8'))
        else:
            print('upload...')

    except Exception as e:
        print(f"Upload failed: {e}")
    finally:
        pass


################### ffmpeg剪辑 ####################################
               
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
        

def round_start_function(cls, xywhn, human_rects, clip_test_num):

    global frame_result, last_result, last_time, start_flag, get_round_end
    cls_reshaped = cls.reshape((-1, 1))
    tmp_result = np.concatenate((xywhn, cls_reshaped), axis=1)
    filtered_tmp_result = tmp_result
    tmp = [row[:-1] for row in filtered_tmp_result]
    if len(filtered_tmp_result) == 4 or len(filtered_tmp_result) == 2:
        for i in range(len(tmp)):
            frame_result.append([tmp[i][0]+tmp[i][2]/2, tmp[i][1]+tmp[i][3]/2, tmp[i][0]+tmp[i][2]/2, tmp[i][1]+tmp[i][3]])  # 中心的x,y 底线的x,y
        if last_result == []:
            last_result.append(frame_result)
            last_time.append(clip_test_num)
        elif frame_result != [] and len(last_result[0]) == len(frame_result):
            last_result.append(frame_result)
            last_time.append(clip_test_num)
        elif len(last_result[0]) != len(frame_result):
            last_time = []
            last_result = []
            last_result.append(frame_result)
            last_time.append(clip_test_num)
        # tmp = []

    frame_result = []
    
    if len(last_result)==3 :
        if (last_time[-1] - last_time[0]) == 2:
            if human_rects == 2:
                start_flag = ym_move_judge_single(last_result, clip_test_num)
            if human_rects == 4:
                start_flag = ym_move_judge_double(last_result, clip_test_num)

            if start_flag and get_round_end :
                # 回合开始
                get_round_end = False
                # with open("./results-output_10.txt", "a", newline='') as d:
                #     # d.write(f"{hour:02}:{minute:02}:{second:02}")
                #     d.write(f"{clip_test_num}")
                #     d.write("\n")
                # d.close()
                

                last_result = []
                last_time = []
                return True
            else:
                last_result.pop(0)
                last_time.pop(0)
        else:
            last_result.pop(0)
            last_time.pop(0)
    return False


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_crossing( frame_idx_prev, x_prev, y_prev, frame_idx_current, x_current, y_current, pixel_to_meter_ratio, fps):
    if (x_current != 0 or y_current != 0) and abs(y_current - y_prev) < 90 :

        distance = euclidean_distance(x_prev, y_prev, x_current, y_current) * pixel_to_meter_ratio
        time_interval =  (frame_idx_current - frame_idx_prev) / fps if fps else 0

        speed = distance / time_interval if time_interval != 0 else 0

        if speed < 90:
            speed *= 2
        else:
            speed *= 0.9
        
        return True, speed

    return False, 0

def process_crossing(frame_idx_current, speed, crossings, crossings_timestamps):
    crossings += 1
    crossings_timestamps.append((frame_idx_current, speed))
    return crossings, crossings_timestamps

def record_total_crossings(crossings, crossings_timestamps, frame_idx, x, y,  predictions, fps, threshold , pixel_to_meter_ratio = 0.1):
    global frame_idx_prev, x_prev, y_prev, frame_idx_prev_ball
    global flag_same_idx, flag_already_crossed
    speed = 0

    predictions.append((frame_idx, x, y))

    if len(predictions) > 5:
        predictions.pop(0)

        # 超过三帧没有检测出球过网
    if (frame_idx - frame_idx_prev_ball) > 3:   # 特殊情况是球被打出画面范围
        flag_already_crossed = False

    if len(predictions) > 4 and frame_idx_prev != 0 and (x!=0 or y!=0):
        # 索引x_prev, y_prev的值
        frame_idx_current, x_current, y_current = predictions[-1]
        count_pos, count_neg = 0, 0  # 连续五帧是否过网

        # 处理具有相同 frame_idx 的帧 （检测结果不只一个球）
        if frame_idx_current == frame_idx_prev:
            # is_crossing()会检查 y 的差异是否较大，如果是，则舍弃这一帧
            is_cross, speed = is_crossing(
                frame_idx_prev, x_prev, y_prev, frame_idx_current, x_current, y_current, pixel_to_meter_ratio, fps
                )
            if is_cross and flag_same_idx == 0 and flag_already_crossed == False:
                flag_same_idx = 1
                                
                crossings, crossings_timestamps = process_crossing(frame_idx_current, speed, crossings, crossings_timestamps)
                # 设置已经判断为过网
                flag_already_crossed = True
                frame_idx_prev_ball = frame_idx_current


        # 比较连续帧的 x 和 y 值
        else:
            for i in range(2, len(predictions) + 1):
                frame_idx_prev_i, x_prev_i, y_prev_i = predictions[-i]
                frame_idx_current_i, x_current_i, y_current_i = predictions[-1]

                if (x_prev_i!=0 or y_prev_i!=0):
                    if (y_prev_i > threshold and y_current_i <= threshold) or (y_prev_i <= threshold and y_current_i > threshold):
                        count_pos += 1
                    else:
                        count_neg += 1

            if count_pos > count_neg and flag_same_idx == 0 and flag_already_crossed == False:
                is_cross, speed = is_crossing( 
                    frame_idx_prev, x_prev, y_prev, frame_idx_current, x_current, y_current, pixel_to_meter_ratio, fps
                    )
                if is_cross:
                    flag_same_idx = 0

                    crossings, crossings_timestamps = process_crossing(frame_idx_current, speed, crossings, crossings_timestamps)
                    flag_already_crossed = True
                    frame_idx_prev_ball = frame_idx_current


        frame_idx_prev, x_prev, y_prev = frame_idx_current, x_current, y_current

    else: #初始化条件
        if (x!=0 or y!=0):
            frame_idx_prev, x_prev, y_prev = predictions[-1]

    return crossings, speed, crossings_timestamps, predictions

def draw_rectangle(frame, center_x, center_y, test_frame_num, box_width = 12, box_height = 12):
    left_top = (int(center_x - box_width), int(center_y - box_height ))
    right_bottom = (int(center_x + box_width ), int(center_y + box_height))

    cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), 3)  
    cv2.line(frame, (1350, 0), (1350, 1000), (255, 0, 0), 3)
    cv2.line(frame, (150, 0), (150, 1000), (255, 0, 0), 3)

    cv2.imwrite( (os.getcwd() +  f'/run_img/run_test_{test_frame_num}.jpg' ) , frame)

def split_clip(save_origin_video_file, uploader_origin_video_python_path, uploader_clip_video_python_path, session_event_timestamp, session_uuid, 
               session_start_time,session_end_time, Court_Name, court_int_num, venue_id_num, polygon, net_ball_threshold, crop_frame, uploader_url, uploader_surroundings, cloudEnvId):

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
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    round_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    round_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_clip = MinHeap()  
    clip_test_num = 0
    round_start_flag = False
    round_end_flag = False

    frame_test_num = 0
    clip_output_video_path_dir = os.path.dirname(save_origin_video_file)
    
    clip_output_txt_path = os.path.join(clip_output_video_path_dir, f'{os.path.basename(save_origin_video_file)[:-4]}.txt')
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

                    if i == (fps - 1)// 2:
                        clip_test_num += 1

                    if i == (fps - 1) // 2 and not round_start_flag: 
                        cls, conf, xywhn, xywh, xyxy, boxes, orig_img, annotated_frame = frame_predict(frame)
                        xywh, xyxy, cls, conf, xywhn = select_player_in_court(orig_img, xywh, xyxy, cls, conf, xywhn,polygon)
                        human_rects = get_human_rects(cls, conf)
                        round_start_flag = round_start_function(cls, xywhn, human_rects, clip_test_num)
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
                        frame_cal +=1
                        round_frame, round_x, round_y = run(inferer, frame)  
                        if round_x != 0 or round_y != 0:
                            ball_num += 1
                        draw_rectangle(frame, round_x, round_y, frame_test_num)
                        count_pause_run = 1 + count_pause_run                  
                        crossings, speed, crossings_timestamps, predictions = record_total_crossings(crossings, crossings_timestamps, int(round_frame), round_x, round_y, predictions, fps, net_ball_threshold)
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
                            record_clip_in_court_list.append(f"{frame_test_num-50}")
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
                        round_start_flag = round_start_function(cls, xywhn, human_rects, clip_test_num)
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
                        draw_rectangle(frame, round_x, round_y, frame_test_num)
                        count_pause_run = 1 + count_pause_run                  
                        crossings, speed, crossings_timestamps, predictions = record_total_crossings(crossings, crossings_timestamps, int(round_frame), round_x, round_y, predictions, fps, net_ball_threshold)
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
                            record_clip_in_court_list.append(f"{frame_test_num-50}")
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


        with open("all_clip.txt", "a") as file:
            file.write(f"{all_clip.data}\n")

        fancy_clip = all_clip.get_sorted_dict()
        with open("fancy_clip.txt", "w") as file:
            file.write(f"\n{fancy_clip}")
        video.release()
        cut_clip_in_court(fps, clip_output_video_path_dir, record_clip_in_court_list, court_int_num, venue_id_num, uploader_clip_video_python_path , session_uuid, Court_Name , uploader_url, uploader_surroundings, cloudEnvId, event_timestamp = int(time.time()* 1000))

        # uploader_ori_thread = threading.Thread(
        #                             target=rename_and_upload_video,
        #                             args=(
        #                                 save_origin_video_file,
        #                                 uploader_origin_video_python_path,
        #                                 str(session_event_timestamp),
        #                                 str(session_uuid),
        #                                 str(session_start_time),
        #                                 str(session_end_time),
        #                                 Court_Name,
        #                                 str(court_int_num), 
        #                                 str(venue_id_num),
        #                                 str(uploader_url), 
        #                                 str(uploader_surroundings), 
        #                                 str(cloudEnvId)
        #                             ),
        #                         )
        # uploader_ori_thread.start()
# if __name__ == "__main__":
#     import sys
#     # save_origin_video_file = sys.argv[1]
#     save_origin_video_file = '/home/sportvision/test_data_Lindan/2023-12-27_19-25-50_the_1_court.mp4'
#     split_clip(save_origin_video_file)

if __name__ == "__main__":
    import sys
    current_path = os.getcwd()
    uploader_clip_video_python_path = os.path.join(current_path, 'uploader_sub.py')
    uploader_origin_video_python_path = os.path.join(current_path, 'uploader_ori.py')

    # save_origin_video_file = sys.argv[1]
    # uploader_origin_video_python_path = sys.argv[2]
    # session_event_timestamp = sys.argv[3]
    # session_uuid = sys.argv[4]
    # session_start_time = sys.argv[5]
    # session_end_time = sys.argv[6]
    # Court_Name = sys.argv[7]
    # court_int_num =int(sys.argv[8])
    # venue_id_num = int(sys.argv[9])
    # polygon = sys.argv[10]
    # net_ball_threshold = int(sys.argv[11]) 
    # crop_frame = sys.argv[12]
    # uploader_url = sys.argv[13]
    # uploader_surroundings = sys.argv[14]
    # cloudEnvId = sys.argv[15]

    save_origin_video_file = '/home/sportvision/court_code_new_round_test/save_video/2024-05-21_21-35-46_the_44_court/2024-05-21_21-35-46_the_44_court.mp4'
    session_uuid = "1"
    session_event_timestamp = 11
    session_start_time = 0
    session_end_time = 30
    Court_Name = "2024-05-21_21-35-46_the_44_court"
    court_int_num = 2
    venue_id_num = 9
    polygon = "[(640, 340), (405, 1015), (1590, 1015), (1360,340)]"
    net_ball_threshold = 400
    crop_frame = "[0:1080, 0:1920 , :]"
    uploader_url = "https://haoqiuwa-test-75067-5-1318337180.sh.run.tcloudbase.com/videos/event/v1"
    uploader_surroundings = "cloud://test-5guvnyfne292a8b9.7465-test-5guvnyfne292a8b9-1318337180"
    cloudEnvId = "test-5guvnyfne292a8b9" 


    split_clip(save_origin_video_file, uploader_origin_video_python_path, uploader_clip_video_python_path, session_event_timestamp, session_uuid, 
               session_start_time,session_end_time, Court_Name, court_int_num, venue_id_num, polygon, net_ball_threshold, crop_frame, uploader_url, uploader_surroundings, cloudEnvId)
    
    # uploader_ori_thread = threading.Thread(
    #                                 target=rename_and_upload_video,
    #                                 args=(
    #                                     save_origin_video_file,
    #                                     uploader_origin_video_python_path,
    #                                     str(session_event_timestamp),
    #                                     str(session_uuid),
    #                                     str(session_start_time),
    #                                     str(session_end_time),
    #                                     Court_Name,
    #                                     str(court_int_num), 
    #                                     str(venue_id_num),
    #                                     str(uploader_url), 
    #                                     str(uploader_surroundings), 
    #                                     str(cloudEnvId)
    #                                 ),
    #                             )
    # uploader_ori_thread.start()


    # list = [240,
    #         711,
    #         1110,
    #         1252,
    #         1395,
    #         1650,
    #         1725,
    #         2136,
    #         2190,
    #         2732,
    #         2865,
    #         3074,
    #         3120,
    #         3296
    #         ]
    # cut_clip_in_court(30, "/Main/save_video/2024-01-08_10-31-26_the_615_court/", list, 10, uploader_clip_video_python_path,  "1704681086552",  "5f41f661-6820-4833-a130-f4d46fabc4b3")


