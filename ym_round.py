import cv2
import numpy as np
from predict_code.frame_predict import frame_predict, crop_each_player, calculate_iou
import pdb
# from inference_ym import key_point

# def get_player(Frame):
#     cls, xywhn, boxes, orig_img, annotated_frame = frame_predict(path=Frame)
#     result = []
#     for i in range(0, len(cls)):
#         if int(cls[i]) != 1:
#             continue
#         result.append([xywhn[i][0]+xywhn[i][2]/2, xywhn[i][1]+xywhn[i][3]/2, xywhn[i][0]+xywhn[i][2]/2, xywhn[i][1]+xywhn[i][3]])  # 中心的x,y 底线的x,y
#     print(result)
#     return [], result, boxes, orig_img

def get_player(Frame, kp=False):
    # cls, xywhn, boxes, orig_img, annotated_frame = frame_predict(path=Frame)
    cls, conf, xywhn, xywh, xyxy, boxes, orig_img, annotated_frame = frame_predict(path=Frame)
    kp_list, result = [], []
    box_list = calculate_iou(boxes, iou_ratio=0.5)
    l1, l2 = crop_each_player(box_list, orig_img, save=False, name = 'test')
    for i in range(len(l2)):
        result.append([xywhn[i][0]+xywhn[i][2]/2, xywhn[i][1]+xywhn[i][3]/2, xywhn[i][0]+xywhn[i][2]/2, xywhn[i][1]+xywhn[i][3]])  # 中心的x,y 底线的x,y
    # if kp:
        # for item in l1:
            # kp_list.append(key_point(item, debug=10))
    print(result)
    print(kp_list)
    pdb.set_trace()
    return kp_list, result, boxes, orig_img, annotated_frame


def rmse(a,b):
    if len(a) == 4:
        a,b = match_boxes(a,b)
    res = np.sqrt(np.square(np.subtract(a,b))).mean(axis=1)
    return res 

def rmse_kp(a,b):
    res = np.sqrt(np.square(np.subtract(a,b))).mean()
    return res 

def match_boxes(boxes1, boxes2):
    # boxes1, boxes2 : shape=[4,2] , shape=[4,2]
    up_1, up_2 = boxes1[:2], boxes2[:2]
    down_1, down_2 = boxes1[2:], boxes2[2:]
    
    up_dis = np.square(np.array([up_1[0]-up_2[0],up_1[0]-up_2[1],up_1[1]-up_2[0],up_1[1]-up_2[1]])).mean(axis=1)
    flag = np.argmin(up_dis)
    if flag==1 or flag==2:
       boxes2[[0,1],:] = boxes2[[1,0],:] 
    
    down_dis = np.square(np.array([down_1[0]-down_2[0],down_1[0]-down_2[1],down_1[1]-down_2[0],down_1[1]-down_2[1]])).mean(axis=1)      
    flag = np.argmin(down_dis)
    if flag==1 or flag==2:
       boxes2[[2,3],:] = boxes2[[3,2],:]

    return boxes1, boxes2
    
def ym_move_judge_kp(last_kp, thr_high=10):  
    assert len(last_kp)==3, print("error in the length of last_result")
    start_xy_result = np.array(last_kp[0])  # 4 x [17,2]
    mid_xy_result = np.array(last_kp[1])
    new_xy_result = np.array(last_kp[2]) 
    
    s, m, e = start_xy_result[:,:,:2], mid_xy_result[:,:,:2], new_xy_result[:,:,:2]

    s2m = rmse(s,m)
    m2n = rmse(m,e)
    flag = m2n > thr_high + s2m
    with open("kp_dis-test.txt", "a", newline='') as d:
        if flag:
            d.write(f"s2m:{s2m:.5f}, m2n:{m2n:.5f}, True")
        else:
            d.write(f"s2m:{s2m:.5f}, m2n:{m2n:.5f}")
        d.write("\n")
    return flag
    pass

def ym_move_judge_single(last_result, test_num, thr_low = 0.009 , thr_high=0.013):  
    # 如果两个判断为相似，则认为是在发球 return True 
    # 否则判断为大球中， return False
    assert len(last_result)==3, print("error in the length of last_result")
    
    # start_xy_result, mid_xy_result, new_xy_result= match_boxes(np.array(last_result[0]), np.array(last_result[1]), np.array(last_result[2]))
    start_xy_result = np.array(sorted(last_result[0], key=lambda x:x[1]))
    mid_xy_result = np.array(sorted(last_result[1], key=lambda x:x[1]))
    new_xy_result = np.array(sorted(last_result[2], key=lambda x:x[1]))   #按y值从小到大排列
    
    
    center_s, below_s = start_xy_result[:,0:2], start_xy_result[:,2:4]
    center_m, below_m = mid_xy_result[:,0:2], mid_xy_result[:,2:4]
    center_n, below_n = new_xy_result[:,0:2], new_xy_result[:,2:4]
    
    c_s, c_m, c_n = center_s[:,0:1], center_m[:,0:1], center_n[:,0:1]
    
    c_s2m = rmse(center_s,center_m).mean()
    b_s2m = rmse(below_s, below_m).mean()
    c_m2n = rmse(center_m,center_n).mean()
    b_m2n = rmse(below_m, below_n).mean()
    
    flag1 = (b_s2m < thr_low or c_s2m < thr_low) and (b_m2n > thr_high or c_m2n > thr_high)
    flag2 = ((c_s<0.53).sum() == 1) or ((c_m<0.53).sum() == 1)  # 发球时球员在场地左右
    flag = flag1 and flag2
    
    with open("round_dis-test.txt", "a", newline='') as d:
        if flag:
            d.write(f"True, c_s2m:{c_s2m:.5f}, b_s2m:{b_s2m:.5f}, c_m2n:{c_m2n:.5f}, b_m2n:{b_m2n:.5f}, test_num:{test_num}")
            d.write("\n")
            d.write(f"c_s:{c_s}, c_m:{c_m}")
        else:
            d.write(f"c_s2m:{c_s2m:.5f}, b_s2m:{b_s2m:.5f}, c_m2n:{c_m2n:.5f}, b_m2n:{b_m2n:.5f}, test_num:{test_num}")  
        d.write("\n")
    return flag

# def ym_move_judge_double(last_result, test_num, thr_low = 0.0089 , thr_high=0.013):  
def ym_move_judge_double(last_result, test_num, thr_low = 0.0096 , thr_high=0.0091):
    # 如果两个判断为相似，则认为是在发球 return True 
    # 否则判断为大球中， return False
    assert len(last_result)==3, print("error in the length of last_result")
    
    # start_xy_result, mid_xy_result, new_xy_result= match_boxes(np.array(last_result[0]), np.array(last_result[1]), np.array(last_result[2]))
    start_xy_result = np.array(sorted(last_result[0], key=lambda x:x[1]))
    mid_xy_result = np.array(sorted(last_result[1], key=lambda x:x[1]))
    new_xy_result = np.array(sorted(last_result[2], key=lambda x:x[1]))   #按y值从小到大排列
    
    
    center_s, below_s = start_xy_result[:,0:2], start_xy_result[:,2:4]
    center_m, below_m = mid_xy_result[:,0:2], mid_xy_result[:,2:4]
    center_n, below_n = new_xy_result[:,0:2], new_xy_result[:,2:4]
    
    c_s2m = rmse(center_s,center_m).mean()
    b_s2m = rmse(below_s, below_m).mean()
    c_m2n = rmse(center_m,center_n).mean()
    b_m2n = rmse(below_m, below_n).mean()
    
    # f1 = (c_s2m < thr_low).sum()<=1 # <=1
    # f2 = (b_s2m < thr_low).sum()<=1
    # f3 = (c_m2n-c_s2m > thr_high).sum()>=2
    # f4 = (b_m2n-b_s2m > thr_high).sum()>=2 # >=2
    # flag = (f1 and f2) and (f3 and f4)
    
    flag1 = (b_s2m < thr_low or c_s2m < thr_low) and (b_m2n > thr_high or c_m2n > thr_high)
    # flag2 = (b_s2m > thr_low and b_m2n > 3*b_s2m) or (c_s2m > thr_low and c_m2n > 3*c_s2m)  增加更多的输出
    # flag3 = (b_s2m > thr_low and c_s2m > thr_low) and (b_m2n < thr_high and c_m2n < thr_high)
    # flag = flag3

    #####################
    flag3 = (b_s2m < thr_low and c_s2m < thr_low) and (b_m2n < thr_low and c_m2n < thr_low)
    flag = flag3
    
    
    with open("round_dis-test.txt", "a", newline='') as d:
        # d.write(f"c_s2m:{f1}, b_s2m:{f2}, c_m2n:{f3}, b_m2n:{f4}\n")
        if flag:
            # d.write(f"c_s2m:{c_s2m:.5f}, b_s2m:{b_s2m:.5f}, c_m2n:{c_m2n:.5f}, b_m2n:{b_m2n:.5f}, True")
            d.write(f"True, c_s2m:{c_s2m:.5f}, b_s2m:{b_s2m:.5f}, c_m2n:{c_m2n:.5f}, b_m2n:{b_m2n:.5f}, test_num:{test_num}")
        else:
            # d.write(f"c_s2m:{c_s2m:.5f}, b_s2m:{b_s2m:.5f}, c_m2n:{c_m2n:.5f}, b_m2n:{b_m2n:.5f}")
            d.write(f"c_s2m:{c_s2m:.5f}, b_s2m:{b_s2m:.5f}, c_m2n:{c_m2n:.5f}, b_m2n:{b_m2n:.5f}, test_num:{test_num}")
        d.write("\n")
    return flag

    
def round_judge(video_path, frame_extract_ratio=1, kp=False):
    last_kp, last_result, last_time = [], [], []

    person_num = None
    count_two = 0
    count_four = 0

    skip_frames = True
    count = 0
    return_time = None
    return_frame_img = None
    return_body_img_list = None
    return_boxes = None
    return_frame_id = None

    result = []

    try:
        video_capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_no = min(frame_count, 50)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)

        if not video_capture.isOpened():
            print("Can't open video!")
            exit()
        for i in range(int(frame_count)):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, i*fps//frame_extract_ratio) # (i*fps)  # 1s取一次
            ret, frame = video_capture.read()

            if not ret:
                print("No frame...")
                break
            kp_res, coords_result, boxes, orig_img, annotated_frame = get_player(frame, kp=False)   # 获得每个frame下的检测框位置
            if person_num is None:
                if len(coords_result) == 2:
                    count_two += 1
                elif len(coords_result) == 4:
                    count_four += 1
                if count_two > 10:
                    person_num = 2
                    break
                elif count_four > 10:
                    person_num = 4
                    break            

        for i in range(int(frame_count)):  
            print(len(last_result))
            print(len(last_time))
                
            if person_num is None:
                print("person_num is None...")
                break
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, i*fps//frame_extract_ratio) # (i*fps)  # 1s取一次
            
            current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            current_second = current_frame / fps

            ret, frame = video_capture.read()
            if not ret:
                print("No frame...")
                break
            cv2.imwrite(('./img_data/img_'+ str(i) +'.jpg'), frame)

            kp_res, coords_result, boxes, orig_img, annotated_frame = get_player(frame, kp=False)   # 获得每个frame下的检测框位置
            
            
            
            if len(coords_result) == person_num:
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)
                # cv2.waitKey(1)
                last_kp.append(kp_res)
                last_result.append(coords_result)
                last_time.append(current_second)
            else:
                continue
            
            # with open("frames_output_1.txt", "a", newline='') as d:
            #     d.write(f"last_result: {str(last_result)}")
            #     d.write("\n")
            #     d.write(f"coords_result: {str(coords_result)}")
            #     d.write("\n")
            
            
            if len(last_result)==3 and len(last_time)==3:   # start, mid, end
                # # if ym_move_judge_kp(last_kp): # 关键点
                # if ym_move_judge_box(last_result):  # 检测框
                if kp:
                    start_flag = ym_move_judge_kp(last_kp)
                elif person_num==2:
                    start_flag = ym_move_judge_single(last_result)
                else:
                    start_flag = ym_move_judge_double(last_result)
                if start_flag:
                    hour = int(last_time[0] / 3600)
                    minute = int((last_time[0] % 3600) / 60)
                    second = int(last_time[0] % 60)
                    return_time = f"{hour:02}:{minute:02}:{second:02}"
                    return_frame_img = orig_img
                    return_body_img_list = crop_each_player(boxes=boxes, orig_img=orig_img)
                    return_boxes = boxes
                    return_frame_id = i
                    count = 0
                    information = {'time':return_time, 'frame_img':return_frame_img, 'body_img_list':return_body_img_list, 'boxes':return_boxes, 'frame_id': return_frame_id}
                    cv2.imwrite(('./move_data/move_'+ str(i) +'.jpg'), return_frame_img)
                    result.append(information)
                    print("Time:" + f"{hour:02}:{minute:02}:{second:02}")
                    with open("./results-output_10.txt", "a", newline='') as d:
                        d.write(f"{hour:02}:{minute:02}:{second:02}")
                        d.write("\n")
                    d.close()
                    last_kp, last_result, last_time=[], [], []  #置空
                else:
                    return_time = None
                    return_frame_img = None
                    return_body_img_list = None
                    return_boxes = None
                    return_frame_id = None
                    last_kp.pop(0)
                    last_result.pop(0)
                    last_time.pop(0)
                    continue
            else:
                continue                 
        return result
    except KeyboardInterrupt:
        pass
    finally:
        video_capture.release()



if __name__ == '__main__':
    v_path = "/home/sportvision/court_data/output_9.mp4"
    # v_path = r"F:/ym/action localization/test_video_10_12/单打测试1/17-16.mp4"
    # v_path = r"../10.8/V15-08.mp4"
    round_judge(v_path, frame_extract_ratio=1, kp=False) # 1s 2帧
    
    
# "F:\ym\person_detect\tracknet_test\cam1_2023-07-20_21-57-14.mp4"