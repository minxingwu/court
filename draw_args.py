import os
import cv2
import numpy as np

def draw_pic(annotated_frame, polygon, net_y_ratio, net_ball_threshold):
    hei, wid, _ = annotated_frame.shape
    print('裁剪后的宽，高：', wid, hei)
    line_color = (0, 0, 255)  # 线的颜色 (BGR 格式)
    line_thickness = 2  # 线的粗细

    cv2.line(annotated_frame, (0, int(hei * net_y_ratio)), (wid, int(hei * net_y_ratio)), line_color, line_thickness)    # 场地线比例
    cv2.line(annotated_frame, (0, net_ball_threshold), (1000, net_ball_threshold), line_color, line_thickness)     # 球网线值

    polygon = np.array(polygon, np.int32)
    cv2.polylines(annotated_frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    
    save_path = os.path.join(os.getcwd(), 'court_img.jpg')
    cv2.imwrite(save_path, annotated_frame)


# 取流 裁剪 画一张图片找到球场参数
if __name__=="__main__":
    cap = cv2.VideoCapture("rtsp://admin:xu123456@192.168.6.240:554/cam/realmonitor?channel=1&subtype=2")
    ret, frame = cap.read()

    # frame = frame[100:1080, 300:1600 , :]
    polygon = [(640, 340), (405, 1015), (1590, 1015), (1360,340)]  
    net_y_ratio = 0.55
    net_ball_threshold = 410

    ## polygon值与位置的对应
    #   (455, 105)     (870,105)

    # (83, 875)          (1250, 875 )

    draw_pic(frame, polygon, net_y_ratio , net_ball_threshold)
    cap.release()
