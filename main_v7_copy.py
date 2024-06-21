import threading
from class_predict import *
from predict_code.frame_predict_c import frame_predict, crop_each_player
from ym_round import ym_move_judge_single, ym_move_judge_double, get_player
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
# from track4txt import predict4txt,caldis
import os
import cv2
import numpy as np
import time
import datetime

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

print('path ', os.getcwd())
sys.path.append('..')
print('path ', os.getcwd())


import subprocess
from multiprocessing import Process, Value
import signal
import argparse

import uuid
import pdb

import heapq

s = time.time()

lock = threading.Lock()
sub_lock = threading.Lock()

def make_parser():
    parser = argparse.ArgumentParser('main v5')

    parser.add_argument('--court_rtsp_url',
                        default=None,
                        type=str,
                        help="court_rtsp_url")
    parser.add_argument('--main_rtsp_url',
                        default=None,
                        type=str,
                        help='main_rtsp_url',)
    parser.add_argument('--clip_rtsp_url',
                        default=None,
                        type=str,
                        help='clip_rtsp_url',)
    parser.add_argument('--audio_rtsp_url',
                        default=None,
                        type=str,
                        help='audio_rtsp_url',)
    parser.add_argument('--court_int_num',
                        default=None,
                        type=str,
                        help='court_int_num',)
    parser.add_argument('--venue_id_num',
                        default=None,
                        type=str,
                        help='venue_id_num',)

    return parser

def main():
    current_path = os.getcwd()
    # /home/sportvision/Main_Session
    save_origin_video_path = os.path.join(current_path, 'save_video') 

    save_origin_video_python_path = os.path.join(current_path, 'save_origin_video.py')
    uploader_origin_video_python_path = os.path.join(current_path, 'uploader_ori.py')
    uploader_clip_video_python_path = os.path.join(current_path, 'uploader_sub.py')

    args = make_parser().parse_args()
    court_rtsp_url = args.court_rtsp_url
    main_rtsp_url = args.main_rtsp_url
    clip_rtsp_url = args.clip_rtsp_url
    audio_rtsp_url = args.audio_rtsp_url
    court_int_num = args.court_int_num
    venue_id_num = args.venue_id_num


    # court_rtsp_url = (
    #     # "rtsp://admin:xu123456@10.10.26.251:554/cam/realmonitor?channel=1&subtype=0"
    #     "rtsp://admin:xu123456@192.168.31.90:554/cam/realmonitor?channel=1&subtype=0"

    # )
    # ''' main_rtsp:v8 / v6跑算法用的流，视频，算法源 '''
    # main_rtsp_url = (
    #     "rtsp://admin:xu123456@192.168.31.90:554/cam/realmonitor?channel=1&subtype=2"

    # )
    # clip_rtsp_url = (
    #     "rtsp://admin:xu123456@192.168.1.251:554/cam/realmonitor?channel=1&subtype=0"
    #     # "rtsp://admin:xu123456@10.10.26.251:554/cam/realmonitor?channel=1&subtype=0"
        
    # )
    # audio_rtsp_url = (
    #     "rtsp://admin:xu123456@192.168.31.8:554/cam/realmonitor?channel=1&subtype=1"
    # )

    
    process(
        clip_rtsp_url=clip_rtsp_url,
        main_rtsp_url=main_rtsp_url,
        court_rtsp_url=court_rtsp_url,
        audio_rtsp_url=audio_rtsp_url,
        court_int_num = court_int_num,
        venue_id_num = venue_id_num,
        save_origin_video_path=save_origin_video_path,
        save_origin_video_python_path=save_origin_video_python_path,
        uploader_clip_video_python_path=uploader_clip_video_python_path,
        uploader_origin_video_python_path=uploader_origin_video_python_path,
    )


def process(
    clip_rtsp_url,
    main_rtsp_url,
    court_rtsp_url,
    audio_rtsp_url,
    court_int_num,
    venue_id_num,
    save_origin_video_path,
    save_origin_video_python_path,
    uploader_clip_video_python_path,
    uploader_origin_video_python_path,
):
    
    if not os.path.exists(save_origin_video_path):
        os.makedirs(save_origin_video_path)

    # ----------------------------------场次识别----------------------------------
    p1 = None
    p2 = None
    p3 = None
    current_path = os.getcwd()
    split_clip_path = os.path.join(current_path, "cut_and_upload_clip_court.py")
    status = 2
    play_num = 0
    none_num = 0
    pass_num = 0
    change = False
    session_num = 0
    play_flag = 0
    get_fps_frame = 0
    First_frame = True
    last_status = status
    Session_skip_frames = 0
    Origin_Video_Name = None
    uploader_ori_thread = None
    now_save_court_video_path = None
    Court_Name = None
    logo_file = r"logo.png"
    ano_logo_file = r"logo_ano.png"
    
    status_list = ["Playing", "Passing", "Freeing"]
    window = [0] * 4
    Continuous_two = False
    Continuous_four = False
    CutTheCourt = False
    session_uuid = None
    session_end_time = None
    session_start_time = None
    session_event_timestamp = None
    PLAY_START_TIME = None
    PLAY_END_TIME = None
    # ----------------------------------场次识别----------------------------------


    # ----------------------------------回合判断----------------------------------
    kp = False
    status_count = 0
    count_two = 0
    count_four = 0
    First_round = False
    score_flag = False
    save_sub_video = None
    cut_video_thread = None
    round_event_timestamp = None
    round_start_flag = False
    round_end_flag = False
    # get_round_end = True # 用于判断是否是回合结束，只有结束了才会开始判断是不是回合开始
    # ----------------------------------回合判断----------------------------------
    webcam=False
    webcam_addr=0
    round_yaml='./YOLOv6/data/coco.yaml'
    device='0'
    half=False
    weights = './YOLOv6/runs/train/exp17/weights/best_ckpt.pt'
    img_size = [1600, 1600]
    inferer = Inferer(webcam, webcam_addr, weights, device, round_yaml, img_size, half)
    # ----------------------------------精彩回合----------------------------------


    # ----------------------------------跑动距离----------------------------------
    # def write_to_file(file_path, new_data):
    #     with open(file_path, 'a') as file:
    #         file.write(new_data + '\n')
    # file_path_det = '/home/sportvision/Main_Session/detect/'
    # frame_id = 1 
    # track_path = '/home/sportvision/Main_Session/track/'
    # distance_path = '/home/sportvision/Main_Session/distance/'
    # track_path_court = ''
    # distance_path_court = ''

    # def get_tracker_config(tracker_type):
    #     tracking_config = \
    #         BOXMOT /\
    #         'configs' /\
    #         (tracker_type + '.yaml')
    #     return tracking_config
    # tracker_config = get_tracker_config('ocsort')
    # with open(tracker_config, "r") as f:
    #     cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    # cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']

    
    uuid_court_name = "court_1"
    uuid_combine_name = None
    uuid_namespace = "新羽胜羽毛球馆"
    namespace_uuid = uuid.uuid3(uuid.NAMESPACE_DNS, uuid_namespace)

    # ----------------------------------得分判断----------------------------------
    xywh_list = []
    cls_list = []
    score_A = 0
    score_B = 0
    # ----------------------------------得分判断----------------------------------
    def signal_handler(signal, frame):
        os.killpg(os.getpgid(p2.pid), 9)

    def terminate_process(process):
        process.terminate()
        process.join()


    def terminate_process_new(process):
        # 发送SIGINT信号（相当于键盘上的Ctrl+C）给进程
        process.send_signal(signal.SIGINT)
        process.wait()
        # 终止进程（如果仍在运行）
        if process.poll() is None:
            process.terminate()

    # 辅码流
    outWriter = None

    # 是否录制的全局变量
    record_flag = False
    frame_number = 0


    test_num = 0
    test_frame_num = 0
    court_test_num = 0

    video = cv2.VideoCapture(main_rtsp_url, cv2.CAP_FFMPEG)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # pdb.set_trace()

    try:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = 25  # 每秒一帧
        frame_count = 0  # 当前帧计数
        if not video.isOpened():
            print("Unable to open video")
            exit()
        

        global get_round_end
        while True: 

            for i in range(fps - 1):
                ret, frame = video.read()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:19]
            ret, frame = video.read()
            # frame = frame[:1000, 60:1560 , :]
            if ret: 
                    
                try:
                    frame = frame[100:1440, 300:1600 , :]
                    cls, conf, xywhn, xywh, xyxy, boxes, orig_img, annotated_frame = frame_predict(frame)
                    xywh, xyxy, cls, conf, xywhn = select_player_in_court(orig_img, xywh, xyxy, cls, conf, xywhn)

                    
                    human_rects = get_human_rects(cls, conf)
                    print('pe is', human_rects)
                    test_num += 1
                    
                    play_flag = if_playing_new(
                        cls=cls,
                        xywhn=xywhn,
                        person_num=human_rects,
                    )
                    if human_rects in [2,3,4]:
                        window.pop(0)
                        window.append(human_rects)

                    draw_pic(cls, xywhn, annotated_frame, play_flag, record_flag, session_num, test_num, window)

                    
                    # if Continuous_two is True and all(num==2 for num in window) :
                    #     Continuous_two = False
                    #     CutTheCourt = True
                    # if window == [4,4,4,2]:
                    #     Continuous_two = True
                    #     count_two = 0
                    # if Continuous_two is True:
                    #     if human_rects == 2:
                    #         count_two += 1
                    #     else:
                    #         Continuous_two = False
                    #     if count_two == 4 and all(num==2 for num in window):
                    #         Continuous_two = False
                    #         CutTheCourt = True
                    #     if count_two == 4:
                    #         Continuous_two = False
                   

                    if play_flag == 2 or play_flag ==3 or play_flag == 4: 
                        continue
                    if play_flag == 0 : 
                        pass_num += 1
                        play_num = 0
                        if pass_num > 4:
                            status = 1
                            # status = 2
                    if play_flag == 1 :
                        print("i am here")
                        play_num += 1
                        pass_num = 0
                        if play_num > 2:
                            status = 0
                    # if play_flag == 3 :
                    #     pass_num += 1
                    #     play_num = 0
                    #     if pass_num > 1:
                    #         status = 1
                    # if play_flag == 4:
                    #     pass_num += 1
                    #     play_num = 0
                    #     if pass_num > 2:
                    #         status = 1

                    # 两个人到四个人的切割

                    
                    if window == [2,2,2,4] :
                        Continuous_four = True
                        count_four = 0
                    
                    if Continuous_four is True :
                        if human_rects == 4:
                            count_four += 1
                        else:
                            Continuous_four = False
                        if count_four == 4 and all(num==4 for num in window):
                            Continuous_four = False
                            CutTheCourt = True
                        if count_four == 4:
                            Continuous_four = False
                    
                    # 真正场次开始判断 
                    if (status_list[status] == "Playing") and (record_flag == False):
                        # pdb.set_trace()
                        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                        with open("log/change_status.txt", "a", newline="") as d:
                            d.write(
                                f"{formatted_time}: last_status:{status_list[last_status]} ————> status:{status_list[status]}"
                            )
                            d.write("\n")
                            d.write(f"{window}")
                            d.write("\n")
                        d.close()
                        # if ym_move_judge_box:
                        #     if PLAY_START_TIME == True:
                        #         return False
                        # width, height, _ = frame.shape
                        # frame[x:w, y:h, :]

                        court_test_num += 1
                        session_num += 1

                        ffmpeg_clip_num = 0
                        # pdb.set_trace()

                        session_start_time = int(time.time()* 1000)
                        session_event_timestamp = int(time.time()* 1000)
                        
                        time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        Court_Name = f"{time_string}_the_{session_num}_court"
                        Court_File_Name = f"{time_string}_the_{session_num}_court_{test_num}/"
                        Origin_Video_Name = Court_Name + '.mp4'
                        Origin_Txt_Name = Court_Name + '.txt'
                        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                        uuid_combine_name = uuid_court_name + Court_Name
                        session_uuid = str(uuid.uuid3(namespace_uuid, uuid_combine_name))
                        with open(
                            "log/Session_save_video.txt", "a", newline=""
                        ) as d:
                            d.write(
                                f"{formatted_time}: 开始录制：{Origin_Video_Name}"
                            )
                            d.write("\n")
                            d.write(f"start,{window}")
                            d.write("\n")
                        d.close()
                        record_flag = True


                        # ocsort = OCSort(
                        #     # per_class,
                        #     det_thresh=cfg.det_thresh,
                        #     max_age=cfg.max_age,
                        #     min_hits=cfg.min_hits,
                        #     iou_threshold=cfg.iou_thresh,
                        #     delta_t=cfg.delta_t,
                        #     asso_func=cfg.asso_func,
                        #     inertia=cfg.inertia,
                        #     use_byte=cfg.use_byte,
                        # )


                        # track_path_court = track_path + Court_Name + '.txt'
                        # distance_path_court = distance_path + Court_Name + '.txt'
                        current_path = os.getcwd()
                        save_origin_video_path = os.path.join(current_path, r"save_video")
                        save_origin_video_path = os.path.join(save_origin_video_path, Court_File_Name)
                        os.makedirs(save_origin_video_path, exist_ok=True)

                        now_save_court_video_path = os.path.join(
                            save_origin_video_path, Origin_Video_Name
                        )

                        now_save_court_txt_path = os.path.join(
                            save_origin_video_path, Origin_Txt_Name
                        )

                        

                        # recording_time = 600
                        crop_each_player(boxes, xyxy, orig_img, save=True, name=Court_Name, current_path=current_path, session_uuid=Court_Name)
                        ffmpeg_cmd_court = ['ffmpeg',
                            '-rtsp_transport',  'tcp',
                            '-i', court_rtsp_url,
                            # '-rtsp_transport',  'tcp',
                            # '-i', audio_rtsp_url,
                            '-i', logo_file,
                            '-i', ano_logo_file,
                            '-filter_complex', '[0:v]crop=2560:1440:0:0[v],[v][1:v]overlay=67.5:30[v],[v][2:v]overlay=1682.5:30',
                            # '-filter_complex', '[0:v][1:a]concat=n=2:v=1:a=1',
                            # '-map', '0:v',
                            '-af', 'afftdn,equalizer=f=200:t=h:w=200:g=-5,equalizer=f=400:t=h:w=100:g=10,equalizer=f=550:t=h:w=50:g=4,equalizer=f=800:t=h:w=200:g=-20,equalizer=f=4000:t=h:w=3500:g=-50',
                            # '-map', '1:a',
                            # '-c:a', 'aac',
                            # '-c:v', 'libx264',
                            # '-crf', '23',
                            # '-preset', 'veryfast',
                            # '-strict', 'experimental',
                            now_save_court_video_path]


                        def f_court(n,ffmpeg_cmd_court):
                            proc = subprocess.Popen(ffmpeg_cmd_court)
                            n.value = os.getpgid(proc.pid)
                        if p1 is None:
                            process_num_p1 = Value('i', 0)
                            p1 = Process(target=f_court, args=(process_num_p1,ffmpeg_cmd_court))
                            p1.start()
                            p1.join()
                            print("process_num_p1 is ", process_num_p1)
                        else:  
                            if p1.is_alive():
                                process_num_p3 = Value('i', 0)
                                p3 = Process(target=f_court, args=(process_num_p3,ffmpeg_cmd_court))
                                p3.start()
                                p3.join()
                                print("process_num_p3 is ", process_num_p3)
                        PLAY_START_TIME = datetime.datetime.now()
                        
                        # cap_cmd   = [
                        #         "python",
                        #         save_sub_video_python_path,
                        #         session_uuid,
                        #     ]
                        
                        # def cap_video(cap_cmd):
                        #     proc = subprocess.Popen(cap_cmd)

                        # if p2 is None:
                        #     p2 = Process(target=cap_video, args=(cap_cmd,))
                        #     p2.start()
                        #     p2.join()

                        
                        First_round = True

                        last_status = status

                    if record_flag is True:
                        cls_reshaped = cls.reshape((-1, 1))
                        conf_reshaped = conf.reshape((-1, 1))
                        result = np.concatenate((xyxy, conf_reshaped, cls_reshaped), axis=1)
                        filtered_result = result[result[:, 5] == 1]
                        # predict4txt(filtered_result, track_path_court, ocsort)

                    # 场次结束
                    if ( ((status_list[status] == "Freeing") or (status_list[status] == "Passing")) and (record_flag==True) ) or ((CutTheCourt == True) and (record_flag==True)):

                        get_round_end = True
                        record_flag = False
                        round_end_flag = False
                        round_start_flag = False
                        start_flag = None
                        CutTheCourt = False
                        # pdb.set_trace()

                        if p1 or p3:
                            if p1 is not None:
                                # p1_thread = threading.Thread(target=terminate_process, args=(p1,))
                                # p1_thread.start()
                                print(process_num_p1.value)
                                kill_process_court(now_save_court_video_path)
                                p1.terminate()
                                p1.kill()
                                p1 = None
                        
                            if p3 is not None:
                                print(process_num_p3.value)
                                kill_process_court(now_save_court_video_path)
                                p3.terminate()
                                p3.kill()
                                p3 = None
                            # def signal_handler(signal, frame, process):
                            #     os.killpg(os.getpgid(p1.pid), 9)
                            # signal.signal(signal.SIGINT, signal_handler)
                            # p1.wait()
                            # p1.terminate()
                            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                            with open(
                                "log/Session_save_video.txt", "a", newline=""
                            ) as d:
                                d.write(
                                    f"{formatted_time}: 关闭录制：{Origin_Video_Name}"
                                )
                                d.write("\n")
                                d.write(f"end,{window}")
                                d.write(f"end,{status}")

                                d.write("\n")
                            d.close()
                            PLAY_END_TIME = datetime.datetime.now()
                            time_diff = PLAY_END_TIME - PLAY_START_TIME
                            PLAY_START_TIME = datetime.datetime.now()
                            
                            # split_cmd = ['python', split_clip_path, str(now_save_court_video_path), session_uuid]
                            # subprocess.Popen(split_cmd)
                   
                            if time_diff > datetime.timedelta(minutes=3) :
                                session_end_time = int(time.time()* 1000)

                                # court_upload_args=[
                                #         now_save_court_video_path,
                                #         uploader_origin_video_python_path,
                                #         str(session_event_timestamp),
                                #         session_uuid,
                                #         str(session_start_time),
                                #         str(session_end_time),
                                #         Court_Name,
                                #         court_int_num,
                                #         venue_id_num
                                # ]
                                
                                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                                with open(
                                    "log/Session_uploader.txt", "a", newline=""
                                ) as d:
                                    d.write(
                                        f"{formatted_time}: 开始上传：{Origin_Video_Name}"
                                    )
                                    d.write("\n")
                                d.close()


                                # else:
                                #     print('有问题呀~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                                    
                                # if p2:    
                                #     if p2 is not None:
                                #         kill_capture()
                                #         kill_origin_ffmpeg()
                                #         p2.terminate()
                                #         p2.kill()
                                #         p2 = None           
                                window = [0] * 4
                                
                                # caldis(track_path_court, distance_path_court)
                                track_path_court = ''
                                distance_path_court = ''
                                
                                formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))

                                with open("log/score.txt", "a", newline="") as d:
                                    d.write(
                                        f"{formatted_time}: A队得分: {score_A}, B队得分: {score_B}"
                                    )
                                    d.write("\n")
                                d.close()
                                score_A = 0
                                score_B = 0

                                split_cmd = ['python', split_clip_path, str(now_save_court_video_path), 
                                             uploader_origin_video_python_path, str(session_event_timestamp), session_uuid, str(session_start_time), str(session_end_time),Court_Name, court_int_num, venue_id_num]

                                subprocess.Popen(split_cmd)
                    
                    
                        

                finally:
                    pass
                print(f"status: {status_list[status]}")
                print('playing flag is ', play_flag)
                print('record flag is ', record_flag)
                if ((status_list[status] == "Freeing") or (status_list[status] == "Passing")):
                    if status_count > 900:
                        status_count = 0
                    status_count += 1
            else:
                if video:
                    video.release()
                    video = cv2.VideoCapture(main_rtsp_url, cv2.CAP_FFMPEG)
                print("No frame...")
                continue
            del ret
            del frame
            
    except KeyboardInterrupt:
        if video:
            video.release()
    finally:
        if uploader_ori_thread:
            uploader_ori_thread.join()
        if p1:
            if p1 is not None:
                kill_process_court()
                p1.terminate()
                p1.kill()

        if video:
            video.release()
        e = time.time()
        print("time is :", e - s)

def round_start_function(cls, xywhn, human_rects, test_num):

    global frame_result, last_result, last_time, start_flag, get_round_end
    cls_reshaped = cls.reshape((-1, 1))
    tmp_result = np.concatenate((xywhn, cls_reshaped), axis=1)
    filtered_tmp_result = tmp_result[tmp_result[:, 4] == 1]
    tmp = [row[:-1] for row in filtered_tmp_result]
    if len(filtered_tmp_result) == 4 or len(filtered_tmp_result) == 2:
        for i in range(len(tmp)):
            frame_result.append([tmp[i][0]+tmp[i][2]/2, tmp[i][1]+tmp[i][3]/2, tmp[i][0]+tmp[i][2]/2, tmp[i][1]+tmp[i][3]])  # 中心的x,y 底线的x,y
        if last_result == []:
            last_result.append(frame_result)
            last_time.append(test_num)
        elif frame_result != [] and len(last_result[0]) == len(frame_result):
            last_result.append(frame_result)
            last_time.append(test_num)
        elif len(last_result[0]) != len(frame_result):
            last_time = []
            last_result = []
            last_result.append(frame_result)
            last_time.append(test_num)
        # tmp = []

    frame_result = []
    
    if len(last_result)==3 :
        if (last_time[-1] - last_time[0]) == 2:
            if human_rects == 2:
                start_flag = ym_move_judge_single(last_result, test_num)
            if human_rects == 4:
                start_flag = ym_move_judge_double(last_result, test_num)

            if start_flag and get_round_end :
                # 回合开始
                get_round_end = False
                # with open("./results-output_10.txt", "a", newline='') as d:
                #     # d.write(f"{hour:02}:{minute:02}:{second:02}")
                #     d.write(f"{test_num}")
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


def draw_rectangle(frame, center_x, center_y, test_frame_num, box_width = 12, box_height = 12):
    left_top = (int(center_x - box_width), int(center_y - box_height ))
    right_bottom = (int(center_x + box_width ), int(center_y + box_height))

    cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), 3)  
    cv2.line(frame, (1350, 0), (1350, 1000), (255, 0, 0), 3)
    cv2.line(frame, (150, 0), (150, 1000), (255, 0, 0), 3)

    cv2.imwrite( (os.getcwd() +  f'/run_img/run_test_{test_frame_num}.jpg' ) , frame)

def draw_pic(cls, xywhn, annotated_frame, play_flag, record_flag, session_num, test_num, window):
    hei, wid, _ = annotated_frame.shape
    print('wid, hei is', wid, hei)
    line_color = (0, 0, 255)  # 线的颜色 (BGR 格式)
    line_thickness = 2  # 线的粗细
    start_point = (0, int(hei*0.3))  # 起始点坐标 (x, y)
    end_point = (wid, int(hei*0.3))  # 终止点坐标 (x, y)
    cv2.line(annotated_frame, start_point, end_point, line_color, line_thickness)

    cv2.line(annotated_frame, (0, 160), (1000,160), line_color, line_thickness)
    text_result, _ = get_human_xy(cls=cls, xywhn=xywhn)
    print("text_result is ", text_result)
    
    tennis_racket_rects = get_racket_rects(cls)
    
    # for box in list(xywh):
    #     x, y, w, h = box
    #     x1, y1 = int(x - w / 2), int(y - h/2)
    #     x2, y2 = int(x + w / 2), int(y + h/2)

    #     # 画矩形框
    #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (0, 255, 0)  # 文本颜色 (BGR 格式)
    line_thickness = 3
    text = str(play_flag)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_thickness)
    text_position = (10, annotated_frame.shape[0] - 10) 
    cv2.putText(annotated_frame, text, text_position, font, font_scale, font_color, line_thickness, cv2.LINE_AA)
    for point in text_result:
        x, y, _, _ = point
        x = int(x * annotated_frame.shape[1])
        y = int(y * annotated_frame.shape[0])
        cv2.circle(annotated_frame, (x, y), 10, (255, 0, 0), -1)
    font_scale = 0.5
    font_color = (0, 0, 0)  # 文本颜色 (BGR 格式)
    line_thickness = 1
    text_result = str(text_result)
    text_position = (10, annotated_frame.shape[0] - 40)
    
    cv2.putText(annotated_frame, text_result, text_position, font, font_scale, font_color, line_thickness, cv2.LINE_AA)
    
    polygon = np.array( [(437, 105), (83, 875), (1235, 847 ), (854,101)] , np.int32)
    # [(437, 105), (83, 875), (1235, 847 ), (854,101)]  # 右边
    cv2.polylines(annotated_frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    
    
    if record_flag:
        cv2.imwrite( (os.getcwd() +  f'/test_img/test_{test_num}_{window}.jpg' ) , annotated_frame)
    else:
        cv2.imwrite(os.getcwd() + f'/test_img/test_{session_num}_{test_num}_000_{window}.jpg', annotated_frame)




def is_file_write_complete(file_path):
    try:
        output = subprocess.check_output(['lsof', file_path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return True
    
    if len(output.strip()) == 0:
        return True
    else:
        return False

def kill_process_court(kill_file_name = "None"):
    command = "ps aux | grep "  + str(kill_file_name) + " | awk '{print $2}'"
    output = subprocess.check_output(command, shell=True)
    pids = output.decode().split("\n")
    for pid in pids:
        if pid.strip() != "":
            try:
                os.kill(int(pid), signal.SIGINT)
                # subprocess.Popen(f"kill -9 {pid}", shell=True)
                print(f"进程 {pid} 被成功杀死")
            except OSError:
                print(f"无法杀死进程 {pid}")

def kill_process_clip(kill_file_name = "None"):
    command = "ps aux | grep "  + str(kill_file_name) + " | awk '{print $2}'"
    output = subprocess.check_output(command, shell=True)
    pids = output.decode().split("\n")
    for pid in pids:
        if pid.strip() != "":
            try:
                os.kill(int(pid), signal.SIGINT)
                # subprocess.Popen(f"kill -9 {pid}", shell=True)
                print(f"进程 {pid} 被成功杀死")
            except OSError:
                print(f"无法杀死进程 {pid}")


def get_time_in_video(video):
    timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return time


def cut_video(
    input_file,
    output_file,
    uploader_clip_video_python_path,
    round_event_timestamp,
    session_uuid,
    duration_to_keep=2,
):
    video_clip = VideoFileClip(input_file)
    video_time = video_clip.duration
    total_duration = video_clip.duration

    if total_duration < duration_to_keep:
        print("视频长度不足，无法剪切。")
        return
    start_time = total_duration - duration_to_keep
    ffmpeg_extract_subclip(
        input_file, start_time, total_duration, targetname=output_file
    )
    file_name_with_extension = os.path.basename(output_file)
    with sub_lock:
        try:
            command = [
                "python",
                uploader_clip_video_python_path,
                file_name_with_extension,
                round_event_timestamp,
                session_uuid,
                video_time,
            ]
            sub_upload_process = subprocess.Popen(command)
            sub_upload_process.wait()
        except Exception as e:
            print(f"Upload failed: {e}")
        # os.remove(input_file)


def create_output_directory(base_directory, prefix='res'):
    # 获取已存在的目录列表
    existing_directories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    # 从已存在的目录中提取索引
    existing_indices = [int(d[len(prefix):]) for d in existing_directories if d.startswith(prefix)]
    
    # 找到最大的索引，如果没有目录则从1开始
    start_index = max(existing_indices) + 1 if existing_indices else 1
    
    output_dir = os.path.join(base_directory, f"{prefix}{start_index}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

if __name__ == "__main__":
    with open("all_clip.txt", "a") as file:
        file.write(f"\n")
    main()

