
from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import save_one_box
# from predict_code.inference_ym import key_point
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
import torch
import pdb

# 
# model = YOLO("predict_code/model_training/yolov8s_p_r.yaml")  # build a new model from scratch
# model = YOLO(r'predict_code/runs/detect/train_p_r/weights/last.pt')  # load a pretrained model (recommended for training)

# model = YOLO("predict_code/bmt/yolov8s_bmt.yaml")
# model = YOLO(r'predict_code/bmt/best.pt')

model = YOLO(r'predict_code/bmt/yolov8s.pt')

def frame_predict(path, model=model):
    # path = "path/to/image"
    # pdb.set_trace()
    # results = model.predict(path, max_det=40, iou=0.7, classes=[0,1,2,3], show=False, show_labels=False, save_conf=True)

    # results = model.predict(path, max_det=4, iou=0.4, classes=1)

    results = model.predict(path, max_det=20, iou=0.4, classes=0)
    boxes = results[0].boxes # boxes
    conf = results[0].boxes.conf.cpu().numpy() 
    cls = results[0].boxes.cls.cpu().numpy()     # class 类别
    xywhn = results[0].boxes.xywhn.cpu().numpy() # x, y, width, height
    xywh = results[0].boxes.xywh.cpu().numpy() 
    xyxy = results[0].boxes.xyxy.cpu().numpy() 
    orig_img  = results[0].orig_img  # 原始图片
    annotated_frame = results[0].plot(probs=False) # 画框图
    # pdb.set_trace()

    return cls, conf, xywhn, xywh, xyxy, boxes, orig_img, annotated_frame


   
def crop_each_box(boxes, orig_img):
    cropped_img_list = []
    for i in boxes:
        cropped_img = save_one_box(i.xyxy, orig_img, BGR=True, save=False)
        cropped_img_list.append(cropped_img)
    return cropped_img_list

def predict_ball(path, model=model):
    # results = model.predict(path, max_det=1, classes=0)
    results = model.predict(path, classes=0)
    boxes = results[0].boxes # boxes
    cls = results[0].boxes.cls.cpu().numpy()     # class 类别
    xywhn = results[0].boxes.xywhn.cpu().numpy() # x, y, width, height
    orig_img  = results[0].orig_img  # 原始图片
    annotated_frame = results[0].plot(probs=False) # 画框图

    
    return cls, xywhn, boxes, orig_img, annotated_frame
    
    
def crop_each_player(boxes, xyxy, orig_img, save=False, name=None, current_path='/home/sportvision/Main_Session', session_uuid='None'):
    # path = "path/to/image"    
    # results = model.predict(path)  # 
    res = boxes
    orig_img  = orig_img 
    player = torch.from_numpy(xyxy)
    # annotated_frame = results[0].plot(probs=False)
    # if type(res) == list:
    #     for item in res:
    #         xyxy = item
    #         # player.append([xyxy[0], xyxy[1], xyxy[2], (xyxy[1]+xyxy[3])//2])
    #         player.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
    # else:
    #     for item in res:
    #         if item.cls==0:  # only get person detect boxes
    #             xyxy = item.xyxy[0].cpu()
    #             # player.append([xyxy[0], xyxy[1], xyxy[2], (xyxy[1]+xyxy[3])//2])
    #             player.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
    #     player.sort(key = lambda x:x[1])
   
    cropped_img_box_list = []
    crop_im_list = []
    if len(player)==2: # single player
        for i in range(len(player)):
            cropped_img_box_list.append(player[i])
            crop_im_list.append(save_one_box(player[i], orig_img, BGR=True, save=False))    
        if save:
            a, b = half_player(crop_im_list[0]), half_player(crop_im_list[1])
            top_path = current_path + '/img/crop_image_top-'
            bottom_path = current_path + '/img/crop_image_bottom-'
            cv2.imwrite(( top_path + session_uuid + '.jpg'), a)
            cv2.imwrite(( bottom_path + session_uuid + '.jpg'), b)
            # cv2.imwrite("crop_image_bottom.jpg", b)          
    elif len(player)==4:  # double player
        for i in range(len(player)):
            cropped_img_box_list.append(player[i])
            crop_im_list.append(save_one_box(player[i], orig_img, BGR=True, save=False))
        if save:
            #保存半身像
            # half_list = crop_im_list
            a, b, c, d = half_player_ano(crop_im_list[0]), half_player_ano(crop_im_list[1]), half_player_ano(crop_im_list[2]), half_player_ano(crop_im_list[3])
            # a, b, c, d = crop_im_list[0], crop_im_list[1], crop_im_list[2], crop_im_list[3]
            a_rows, a_cols = a.shape[:2]
            b_rows, b_cols = b.shape[:2]
            min_rows = min(a_rows, b_rows)
            min_cols = min(a_cols, b_cols)
            a = a[:min_rows, :min_cols]
            b = b[:min_rows, (b_cols-min_cols):b_cols]

            c_rows, c_cols = c.shape[:2]
            d_rows, d_cols = d.shape[:2]
            min_rows = min(c_rows, d_rows)
            min_cols = min(c_cols, d_cols)
            c = c[:min_rows, :min_cols]
            d = d[:min_rows, (d_cols-min_cols):d_cols]
            # b = cv2.resize(b, (a.shape[1],a.shape[0])) 
            # d = cv2.resize(d, (c.shape[1],c.shape[0]))
            c_top = cv2.hconcat([a,b])
            c_below = cv2.hconcat([c,d])
            # cv2.imwrite("crop_image_top.jpg", c_top)
            top_path = current_path + '/img/crop_image_top-'
            bottom_path = current_path + '/img/crop_image_bottom-'
            cv2.imwrite(( top_path + session_uuid + '.jpg'), c_top)
            cv2.imwrite(( bottom_path + session_uuid + '.jpg'), c_below)
            # cv2.imwrite("crop_image_double_top_1.jpg", crop_im_list[0])
            # cv2.imwrite("crop_image_double_top_2.jpg", crop_im_list[1])
            # cv2.imwrite("crop_image_double_bottom_1.jpg", crop_im_list[2])
            # cv2.imwrite("crop_image_double_bottom_2.jpg", crop_im_list[3])
    return crop_im_list, cropped_img_box_list


def half_player(img):
    # img.shape = h,w,c 
    h,w,c = img.shape
    aim_h = int(h*0.6)
    return img[:aim_h,:,:]
    
def half_player_ano(img):
    # img.shape = h,w,c 
    h,w,c = img.shape
    aim_h = int(h*0.8)
    return img[:aim_h,:,:]

def calculate_iou(boxes, iou_ratio):
    box_list = []
    for item in boxes:
        box_list.append(item.xywh[0])
    
    box_list.sort(key = lambda x:x[1])
    
    if len(boxes)==4:    
        iou_up = bbox_iou(box1=box_list[0],box2=box_list[1])
        iou_below = bbox_iou(box1=box_list[2],box2=box_list[3])
        # print(box_list)
        # print(iou_up)
        # print(iou_below)
        while iou_up>iou_ratio:
            print("iou_up:", iou_up)
            box_list[0][2]-=10
            box_list[0][3]-=10
            box_list[1][2]-=10
            box_list[1][3]-=10
            iou_up = bbox_iou(box1=box_list[0],box2=box_list[1])
        while iou_below>iou_ratio:
            print("iou_below:", iou_below)
            box_list[2][2]-=10
            box_list[2][3]-=10
            box_list[3][2]-=10
            box_list[3][3]-=10
            iou_below = bbox_iou(box1=box_list[2],box2=box_list[3]) 
    xyxy = []
    for item in box_list:
        xyxy.append(xywh2xyxy(item))
    # print(xyxy)
    return xyxy


if __name__ == '__main__':
    img_path = r"../test_video/output_9.mp4"
    cls, conf, xywhn, xywh, xyxy, boxes, orig_img, annotated_frame = frame_predict(img_path)
    # cropped_img_list = crop_each_box(boxes, orig_img)
    # for i in range(0, len(cropped_img_list)):
    #     cv2.imwrite(str(i) + ".jpg", cropped_img_list[i])
