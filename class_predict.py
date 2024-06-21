from predict_code.frame_predict import crop_each_player, calculate_iou
# from predict_code.inference_ym import key_point
import numpy as np
import pdb

def check_point_in_quadrilateral(points, test_point):
    x, y = zip(*points)
    x_test, y_test = test_point

    # 计算四边形的四个边的方程系数
    a = [y[i] - y[i-1] for i in range(4)]
    b = [x[i-1] - x[i] for i in range(4)]
    c = [x[i] * y[i-1] - x[i-1] * y[i] for i in range(4)]

    # 判断随机点是否在四边形内部
    for i in range(4):
        if a[i] * x_test + b[i] * y_test + c[i] < 0:
            return False
    return True

    ## 筛选场地区域的锚框
def select_player_in_court(orig_img, xywh, xyxy, cls, conf, xywhn, polygon):
    height, width, channels = orig_img.shape

    if xywh.size > 0:
    ## 底部中心点
        bottom_center_points = np.array([(box[0], box[1] + box[3]/2) for box in xywh])

        # condition = (bottom_center_points[:, 0] >= 100) & (bottom_center_points[:, 0] <= 1600) & (bottom_center_points[:, 1] >= 280) & (bottom_center_points[:, 1] <= 1150)
        condition = [i for i, point in enumerate(bottom_center_points) if check_point_in_quadrilateral(polygon, point)]
        xywh = np.array(xywh[condition])
        xyxy = np.array(xyxy[condition])
        cls = np.array(cls[condition])
        conf = np.array(conf[condition])
        xywhn = np.array(xywhn[condition])
    return xywh, xyxy, cls, conf, xywhn

def get_human_rects(cls, conf):

    cls_reshaped = cls.reshape((-1, 1))
    conf_reshaped = conf.reshape((-1, 1))
    result = np.concatenate((conf_reshaped, cls_reshaped), axis=1)
    # pdb.set_trace()
    # filtered_result = result[result[:, 1] == 1]
    filtered_result = result[(result[:, 0] > 0.3)]
    # for i in range(0, len(cls)):
        # if int(cls[i]) == 1:
            # count += 1
    return len(filtered_result)


def get_human_xy(cls, xywhn):
    result = []
    ori_result = []
    for i in range(0, len(cls)):
        x_img = xywhn[i][0]
        y_img = xywhn[i][1]
        w_img = xywhn[i][2]
        h_img = xywhn[i][3]
        # 用左下角的点
        y = y_img + h_img/2
        result.append((x_img, y, w_img, h_img))
        ori_result.append((x_img, y_img, w_img, h_img))
    return result, ori_result


def if_playing_new(cls, xywhn, person_num, net_y_ratio):
    # 0:5s cut ; 1:playing ; 2:waiting ; 3:player:02/20 1s cut; 4: player:13/31 2s cut;
    if person_num == 1 : return 0
    if person_num == 0 : return 3
    result, _ = get_human_xy(cls=cls, xywhn=xywhn)
    print('result is ', result)
    def is_above(y):
        if y < net_y_ratio:
            return True
        return False

    def is_below(y):
        if y > net_y_ratio:
            return True
        return False
    
    above_count = 0
    below_count = 0 
    # pdb.set_trace()
    for xy in result:
        x, y, _, _ = xy
        if is_above(y):
            above_count += 1
        if is_below(y):
            below_count += 1
    if person_num == 4:
        if above_count == 2 and below_count == 2: return 1
        if above_count == 3 and below_count == 1: return 4
        if above_count == 1 and below_count == 3: return 4
    if person_num == 2:
        if above_count == 1 and below_count == 1: return 1
        if above_count == 2 and below_count == 0: return 3
        if above_count == 0 and below_count == 2: return 3
    if person_num == 3:
        if (above_count == 2 and below_count == 1) or (above_count == 1 and below_count == 2): return 2
    if person_num == 5:
        if (above_count == 2 and below_count == 3) or (above_count == 3 and below_count == 2): return 2
    return 0
