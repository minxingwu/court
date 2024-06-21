import numpy as np
from ym_round import ym_move_judge_single, ym_move_judge_double, get_player


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

    frame_result = []
    
    if len(last_result)==3 :
        if (last_time[-1] - last_time[0]) == 2:
            if human_rects == 2:
                start_flag = ym_move_judge_single(last_result, clip_test_num)
            if human_rects == 4:
                start_flag = ym_move_judge_double(last_result, clip_test_num)

            if start_flag and get_round_end :
                get_round_end = False
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
                _, x_prev_i, y_prev_i = predictions[-i]
                _, _, y_current_i = predictions[-1]

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
