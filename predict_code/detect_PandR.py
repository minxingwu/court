from ultralytics import YOLO

# Load a model
import time

s = time.time()
model = YOLO("predict_code/model_training/yolov8s_p_r.yaml")  # build a new model from scratch
model = YOLO(r'predict_code/runs/detect/train_p_r/weights/last.pt')  # load a pretrained model (recommended for training)

e = time.time()
print("load model time:", e-s)  # 0.2s

# image
model.predict(source="./cam1_2023-07-20_20-57-14.mp4", save=True, save_txt=False, save_crop=False, device=[0]) # , max_det=4, iou=0.4)
