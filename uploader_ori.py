import os
import time
from threading import Thread, Event
from queue import Queue, Empty
import eventlet

import requests
import json
import re
import uuid
# from detection.tool.utils import raw_file_structure

time_out_num = 500

class Uploader(Thread):
    def __init__(self, cfg, upload_queue):
        # multi-threading
        Thread.__init__(self)
        self.upload_queue = upload_queue
        self.quit = Event()

        # client-side rate limiting for video upload requests
        self.upload_interval = 0.5
        self.get_interval = 5

        # Setup cloud environment.
        self.cfg = cfg
        self.court_id = cfg["court_id"]
        self.APPID = cfg['weApp_info']["appId"]
        self.APPSECRET = cfg['weApp_info']["appSecret"]
        self.CLOUD_ENV_ID = cfg['weApp_info']["cloudEnvId"]

        self.token = self.get_token_from_secret()

    def get_token_from_secret(self):
        # if self.token:
        #     return self.token
        token_url = f'https://api.weixin.qq.com/cgi-bin/token'
        data = {
            'grant_type': 'client_credential',
            'appid': self.APPID,
            'secret': self.APPSECRET
        }
        response = requests.get(token_url, params=data)
        self.token = response.json()['access_token']
        print("[Uploader] Get token from secret")
        return self.token

    def get_upload_token(self, target_path):
        url = f'https://api.weixin.qq.com/tcb/uploadfile?access_token={self.token}'
        data_url = {
            'env': self.CLOUD_ENV_ID,
            'path': target_path
        }
        data_url = json.dumps(data_url)  # 一定要把参数转为json格式，不然会请求失败
        while True:
            print("[Uploader] IN THE LOOP. Try to get access for uploading video now.")
            try:
                with eventlet.Timeout(time_out_num, True):
                    resp_url = requests.post(url, data_url)
                    resp = resp_url.json()
                    if resp["errcode"] == 0 \
                            and 'authorization' in resp \
                            and 'token' in resp \
                            and 'cos_file_id' in resp:
                        print("[Uploader] Get authorization successfully.")
                        break
                    else:
                        print("[Uploader] Get authorization error, ", resp["errmsg"])
                        print("[Uploader] 'authorization' in resp: ", 'authorization' in resp)
                        print("[Uploader] 'token' in resp: ", 'token' in resp)
                        print("[Uploader] 'cos_file_id' in resp: ", 'cos_file_id' in resp)
                        print("[Uploader] Update token and retry after 5 seconds...")
                        self.token = self.get_token_from_secret()
                        url = f'https://api.weixin.qq.com/tcb/uploadfile?access_token={self.token}'
                        time.sleep(time_out_num)
                        continue
            except eventlet.timeout.Timeout:
                print("[Uploader] Get authorization timeout, retry after 5 seconds...")
                eventlet.sleep(time_out_num)
                continue
            except Exception as e:
                print("[Uploader] Get authorization exception, ", e)
                print("[Uploader] Retry after 5 seconds...")
                eventlet.sleep(time_out_num)
                continue
        return resp

    def upload_from_url_video(self, video_path, target_path, resp):
        print('here is video_path',video_path)
        data_vdo = {
            "Content-Type": (None, 'video/mp4'),  # 此处为上传文件类型
            "key": (None, target_path),
            "Signature": (None, resp['authorization']),
            'x-cos-security-token': (None, resp['token']),
            'x-cos-meta-fileid': (None, resp['cos_file_id']),
            'file': (video_path, open(video_path, 'rb'))
        }
        while True:
            print("[Uploader] IN THE LOOP. Try to upload video now.")
            try:
                with eventlet.Timeout(time_out_num, True):
                    resp_vdo = requests.post(resp['url'], files=data_vdo)
                    video_info = {
                        'download_url': resp['cos_file_id'],
                        'path': target_path
                    }
                    success = resp_vdo.status_code == 204
                    print("[uploader] Complete uploading to cloud {}. Success: {}".format(target_path, success))
                    if success:
                        break
                    else:
                        print("[Uploader] Upload error, ", resp_vdo.status_code)
                        eventlet.sleep(time_out_num)
                        continue
            except eventlet.timeout.Timeout:
                print("[Uploader] Upload timeout, retry after 2 seconds...")
                eventlet.sleep(time_out_num)
                continue
            except Exception as e:
                print("[Uploader] Upload error, ", e)
                print("[Uploader] Retry after 2 seconds...")
                eventlet.sleep(time_out_num)
                continue
        return success, video_info


    def upload_from_url_picture(self, video_path, target_path, resp):
        data_vdo = {
            "Content-Type": (None, 'image/png'),  # 此处为上传文件类型
            "key": (None, target_path),
            "Signature": (None, resp['authorization']),
            'x-cos-security-token': (None, resp['token']),
            'x-cos-meta-fileid': (None, resp['cos_file_id']),
            'file': (video_path, open(video_path, 'rb'))
        }
        while True:
            print("[Uploader] IN THE LOOP. Try to upload video now.")
            try:
                with eventlet.Timeout(time_out_num, True):
                    resp_vdo = requests.post(resp['url'], files=data_vdo)
                    video_info = {
                        'download_url': resp['cos_file_id'],
                        'path': target_path
                    }
                    success = resp_vdo.status_code == 204
                    print("[uploader] Complete uploading to cloud {}. Success: {}".format(target_path, success))
                    if success:
                        break
                    else:
                        print("[Uploader] Upload error, ", resp_vdo.status_code)
                        eventlet.sleep(time_out_num)
                        continue
            except eventlet.timeout.Timeout:
                print("[Uploader] Upload timeout, retry after 2 seconds...")
                eventlet.sleep(time_out_num)
                continue
            except Exception as e:
                print("[Uploader] Upload error, ", e)
                print("[Uploader] Retry after 2 seconds...")
                eventlet.sleep(time_out_num)
                continue
        return success, video_info

    def upload_video_to_cloud(self, video_path, target_path):
        print("[uploader] uploading video file {} to cloud".format(video_path))
        resp = self.get_upload_token(target_path)
        success, video_info = self.upload_from_url_video(video_path, target_path, resp)
        return success, video_info

    def upload_picture_to_cloud(self, picture_path, target_path):
        print("[uploader] uploading video file {} to cloud".format(picture_path))
        resp = self.get_upload_token(target_path)
        success, picture_info = self.upload_from_url_picture(picture_path, target_path, resp)
        return success, picture_info

    def upload_picture_top_to_cloud(self, pic_top, target_path):
        print("[uploader] uploading video file {} to cloud".format(pic_top))
        resp = self.get_upload_token(target_path)
        success, picture_info = self.upload_from_url_picture(pic_top, target_path, resp)
        return success, picture_info
    
    def upload_picture_bottom_to_cloud(self, pic_bottom, target_path):
        print("[uploader] uploading video file {} to cloud".format(pic_bottom))
        resp = self.get_upload_token(target_path)
        success, picture_info = self.upload_from_url_picture(pic_bottom, target_path, resp)
        return success, picture_info


    def upload_video(self, task):
        type, path = task
        # upload video address should follow this format:
        # type/yyyymmdd/hhmm/starttime_endtime_id.mp4
        # Example path in the cloud:
        # highlight/20221217/1830/20221217183000_20221217183030_00.mp4
        # time_start, time_end, cid = raw_file_structure(path)
        # year, month, day = time_start[:4], time_start[4:6], time_start[6:8]
        # hour, minute, second = time_start[8:10], time_start[10:12], time_start[12:14]
        # minute_round = (int(minute) // 30) * 30  # round min to 00 or 30.
        # fname = f'{type}/{self.court_id}/{str(year) + str(month) + str(day)}/{str(hour)}' + \
        #         '{:02d}/n'.format(minute_round) + \
        #         path.split('/')[-1]
        # print(type)
        success, video_info = self.upload_video_to_cloud(type, str(venue_id_num)+ '/'+ 'record/'+ court_num + '/' + folder_name_datatime_str + '/' + video_name)
        # success, video_info = self.upload_video_to_cloud(path)

        if not success:
            print("Unsuccessful request. url: ", path)

        # # remove the clip
        # # os.unlink(path)
        # print("[Uploader] completed upload task {}: response code = {}, response = {}".format(task, response.status_code, response.json()))
        print("[Uploader] completed upload task {}: response = {}".format(path, video_info['path']))

    def upload_picture(self, task):
        type, path = task
        success, picture_info = self.upload_picture_to_cloud(type, str(venue_id_num)+ '/'+ 'record/'+ court_num + '/' + folder_name_datatime_str + '/' + picture_name)
        if not success: 
            print("Unsuccessful request. url: ", path)

    
    def upload_picture_top(self, task):
        type, path = task
        success, picture_info = self.upload_picture_to_cloud(type, str(venue_id_num)+ '/'+ 'record/'+ court_num + '/' + folder_name_datatime_str + '/' + pic_top_name)
        if not success: 
            print("Unsuccessful request. url: ", path)
    

    def upload_picture_bottom(self, task):
        type, path = task
        success, picture_info = self.upload_picture_to_cloud(type, str(venue_id_num)+ '/'+ 'record/'+ court_num + '/' + folder_name_datatime_str + '/' + pic_bottom_name)
        if not success: 
            print("Unsuccessful request. url: ", path)

    def run_video(self):
        while not self.quit.is_set():
            try:
                task = self.upload_queue.get(block=True, timeout=self.get_interval)
            except Empty:
                continue

            self.upload_video(task)
            time.sleep(self.upload_interval)
            if self.upload_queue.empty():
                break

        # process the remaining tasks in the queue
        while True:
            try:
                task = self.upload_queue.get_nowait()
            except Empty:
                break

            self.upload_video(task)


    def run_picture(self):
        while not self.quit.is_set():
            try:
                task = self.upload_queue.get(block=True, timeout=self.get_interval)
            except Empty:
                continue

            self.upload_picture(task)
            time.sleep(self.upload_interval)
            if self.upload_queue.empty():
                break

        # process the remaining tasks in the queue
        while True:
            try:
                task = self.upload_queue.get_nowait()
            except Empty:
                break

            self.upload_picture(task)

    def run_picture_top(self):
        while not self.quit.is_set():
            try:
                task = self.upload_queue.get(block=True, timeout=self.get_interval)
            except Empty:
                continue

            self.upload_picture_top(task)
            time.sleep(self.upload_interval)
            if self.upload_queue.empty():
                break

        # process the remaining tasks in the queue
        while True:
            try:
                task = self.upload_queue.get_nowait()
            except Empty:
                break

            self.upload_picture_top(task)

    def run_picture_bottom(self):
        while not self.quit.is_set():
            try:
                task = self.upload_queue.get(block=True, timeout=self.get_interval)
            except Empty:
                continue

            self.upload_picture_bottom(task)
            time.sleep(self.upload_interval)
            if self.upload_queue.empty():
                break

        # process the remaining tasks in the queue
        while True:
            try:
                task = self.upload_queue.get_nowait()
            except Empty:
                break

            self.upload_picture_bottom(task)



if __name__ == "__main__":
    # 创建一个上传队列
    upload_queue = Queue()

    # 年月日文件夹
    import datetime
    tz = datetime.timezone(datetime.timedelta(hours=8)) 
    # 构造东0区时区对象 原八区北京时间，解决算法延时情况
    now = datetime.datetime.now(tz)  
    folder_name_datatime_str = now.strftime('%Y%m%d')  

    # 场地信息
    # court_int_num = 2

    # changguan
    # venue_id_num = 2

    import sys
    # 输入参数需要有一个文件名
    if len(sys.argv) < 11:
        print("Usage: python script.py input_file.mp4")
        exit()

    ori_video_name = sys.argv[1]
    event_timestamp = sys.argv[2]
    session_uuid = sys.argv[3]
    session_start_time = sys.argv[4]
    session_end_time = sys.argv[5]
    Court_Name = sys.argv[6]
    court_int_num = int(sys.argv[7])
    venue_id_num = int(sys.argv[8])
    uploader_url = sys.argv[9]
    uploader_surroundings = sys.argv[10]
    cloudEnvId = sys.argv[11]
    # video_time = sys.argv[6]
    # video_name = ori_video_name[55:]
    court_num = 'court' + str(court_int_num)
    video_name = ori_video_name.rsplit('/', 1)[1]
    picture_name = video_name.replace(".mp4", ".png")
    print('after name:', picture_name)
    print('video name ', video_name)
    hour = 19
    import cv2


    video_file = ori_video_name
    print('video_file is here :', video_file)

    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    suc, img = cap.read()
    width = 450
    height = 300
    if suc:
        resize_img  = cv2.resize(img, (width, height))
        img = resize_img[30:200, 100:350, :]

    
    duration = frame_count / fps
    # text = "{:.0f}s".format(duration)
    text = "{:02d}:{:02d}".format(int(duration//60), int(duration%60))
    text_file_name = "{:}".format(video_name)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    # cv2.rectangle(img, (img.shape[1] - text_size[0] - 10, img.shape[0] - text_size[1] - 10),
    #               (img.shape[1] - 10, img.shape[0] - 10), (255, 255, 255), -1)
    # cv2.putText(img, text_file_name, (20 , img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 25, 25), thickness=1, lineType=cv2.LINE_AA)
    # cv2.putText(img, text, (img.shape[1] - text_size[0] + 45 , img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 25, 25), thickness=1, lineType=cv2.LINE_AA)

    picture_file = os.path.join(ori_video_name.rsplit('/', 1)[0], picture_name)
    pic = cv2.imwrite(picture_file ,img)
    pic_top_name = 'crop_image_top-' + Court_Name + '.jpg'
    pic_bottom_name = 'crop_image_bottom-' + Court_Name + '.jpg'

    pic_top = os.getcwd() + '/img/' + pic_top_name
    pic_bottom = os.getcwd() + '/img/' + pic_bottom_name


    import time
    time.sleep(3)
    # 将要上传的视频信息添加到队列中
    print(video_file, picture_file)
    task = (video_file,  video_name)
    task_pic = (picture_file, picture_name)
    task_pic_top = (pic_top, pic_top_name)
    task_pic_bottom = (pic_bottom, pic_bottom_name)

    cfg = {
        "court_id": f"{court_int_num}",
        "weApp_info": {
            "appId": "wx69e45cc989de661d",
            "appSecret": "39374863c500287baba74aab0a5d1e91",
            # "cloudEnvId": "prod-2gicsblt193f5dc8"
            "cloudEnvId": f"{cloudEnvId}"
        }
    }
    upload_queue.put(task)
    uploader = Uploader(cfg, upload_queue)
    uploader.run_video()

    upload_queue.put(task_pic)
    uploader = Uploader(cfg, upload_queue)
    uploader.run_picture()
    upload_queue.put(task_pic_top)
    uploader = Uploader(cfg, upload_queue)
    uploader.run_picture_top()
    upload_queue.put(task_pic_bottom)
    uploader = Uploader(cfg, upload_queue)
    uploader.run_picture_bottom()


    url = uploader_url
    # url = "https://golang-m7vn-53609-9-1318337180.sh.run.tcloudbase.com/videos/event/v1"
    surroundings = uploader_surroundings
    # surroundings = 'cloud://prod-2gicsblt193f5dc8.7072-prod-2gicsblt193f5dc8-1318337180'


    payload = {
        "request_id":str(uuid.uuid4()),
        "event_type":1,
        "request_timestamp":int(time.time()* 1000),
        "event_timestamp":int(event_timestamp),
        "data":{
            "uuid":session_uuid,
            "court":court_int_num,
            "venue_id":venue_id_num,
            "file_path":surroundings + '/' + str(venue_id_num)+ '/' + 'record' + '/' + court_num + '/' + folder_name_datatime_str + '/' + video_name,
            "file_name":video_name,
            "hour":hour,
            "date":int(folder_name_datatime_str),
            "team_a_img_path":surroundings + '/' + str(venue_id_num)+ '/' + 'record' + '/' + court_num + '/' + folder_name_datatime_str + '/' + pic_top_name,
            "team_b_img_path":surroundings + '/' + str(venue_id_num)+ '/' + 'record' + '/' + court_num + '/' + folder_name_datatime_str + '/' + pic_bottom_name,
            "hover_img_path":surroundings + '/' + str(venue_id_num)+ '/' + 'record' + '/' + court_num + '/' + folder_name_datatime_str + '/' + picture_name,
            "start_timestamp":int(session_start_time),
            "end_timestamp":int(session_end_time),
            "time":int(duration) # video time - total_frame / fps = &&s
        }
    }
    response = requests.post(url,data=json.dumps(payload))
    print(response)
    # 打印请求信息
    print("Request URL:", response.request.url)
    print("Request Method:", response.request.method)
    print("Request Headers:", response.request.headers)
    print("Request Body:", response.request.body)

    # 打印响应信息
    print("Response Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Content:", response.content.decode('utf-8'))

