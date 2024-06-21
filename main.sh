# !/bin/bash

sleep 2
# cd /home/sportvision/court_3/
python main_v7.py \
    --court_rtsp_url "rtsp://admin:xu123456@192.168.6.240:554/cam/realmonitor?channel=1&subtype=0" \
    --main_rtsp_url "/home/sportvision/record_code/record/outrtsp.mp4" \
    --clip_rtsp_url "rtsp://admin:xu123456@192.168.110.65:554/cam/realmonitor?channel=1&subtype=0" \
    --audio_rtsp_url "rtsp://admin:jy123456@192.168.110.65:554/cam/realmonitor?channel=1&subtype=1" \
    --court_int_num "2" \
    --venue_id_num "9" \
    --polygon "[(640, 340), (405, 1015), (1590, 1015), (1360,340)]" \
    --net_y_ratio "0.56"  \
    --net_ball_threshold "400" \
    --crop_frame "[0:1080, 0:1920 , :]" \
    --uploader_url "https://golang-m7vn-53609-9-1318337180.sh.run.tcloudbase.com/videos/event/v1" \
    --uploader_surroundings "cloud://prod-2gicsblt193f5dc8.7072-prod-2gicsblt193f5dc8-1318337180" \
    --cloudEnvId "prod-2gicsblt193f5dc8" \
# --main_rtsp_url "rtsp://admin:xu123456@192.168.110.65:554/Streaming/Channels/103" \
# --uploader_url "https://haoqiuwa-test-75067-5-1318337180.sh.run.tcloudbase.com/videos/event/v1" \
    # --uploader_surroundings "cloud://test-5guvnyfne292a8b9.7465-test-5guvnyfne292a8b9-1318337180" \
    # --cloudEnvId "test-5guvnyfne292a8b9" 
