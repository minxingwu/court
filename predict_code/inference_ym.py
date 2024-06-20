import os
import cv2
import torch
from movenet_master.movenet.opts import opts
from movenet_master.movenet.detectors.detector_factory import detector_factory

image_ext = ["jpg", "jpeg", "png", "webp"]
video_ext = ["mp4", "mov", "avi", "mkv"]
time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge"]


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind(".") + 1 :].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]

    for image_name in image_names:
        ret = detector.run(image_name)
        # print("ret['results']", ret['results'])
        # time_str = ""
        # for stat in time_stats:
        #     time_str = time_str + "{} {:.3f}s |".format(stat, ret[stat])
        # print(time_str)
    return ret['results']


def key_point(img, debug=2):
    # debug=10，不显示
    # debug = 2 显示
    # debug = 4, 保存
    opt = opts().init()
    opt.debug=debug
    opt.load_model = "movenet_master/weights/movenet.pth"
    opt.demo = img
    res = demo(opt)
    return res

if __name__ == "__main__":
    res = key_point("2.jpg")
    print(res)
    

    
    
    

# python demo.py single_pose --dataset active --arch movenet --demo data/crop_image_double_top_1.jpg --load_model ./weights/movenet.pth --K 1 --gpus -1 --debug 10
