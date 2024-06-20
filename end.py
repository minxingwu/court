import os
import numpy as np
import pdb

# def process_frame(frame_number, x, y, started_append, zero_count, previous_x, previous_y, seg):
#     # 第一次检查到球时开始计算
#     if x != 0 and y != 0 and not started_append:
#         zero_count += 1
#         # 如果一共20帧检测到球
#         if zero_count >= 20:
#             zero_count = 0
#             started_append = True
        
        
#     # 是否为异常检测(prev_x<150 or prev_x>1350)
    
#     # 检测是否为吊高球(prev_y<100)
#     if previous_y > 100 and x == 0 and y == 0 and started_append:
#         zero_count += 1
#         # frames_with_zero_detection.append(frame_number)
        
#         # 检查前连续40帧的检测结果是否都为0
#         if zero_count >= 34:
#             # 输出当前时间帧
#             seg.append(frame_number)
        
#         # 不更新前置位置
#         previous_x = previous_x
#         previous_y = previous_y
        
#     # 如果是吊高球
#     elif previous_y < 100 and x == 0 and y == 0 and started_append:
#         previous_x = previous_x
#         previous_y = previous_y

#     # 其中漏检情况和正常击打
#     elif started_append and (150 < x < 1350):
#         zero_count = 0 # 重置连续检测到的x, y为0的帧数
#         # 更新上一帧的位置
#         previous_x = x
#         previous_y = y  # 假设x为球的位置
    
#     elif started_append: # 异常检测(prev_x<150 or prev_x>1350)
#         zero_count += 1 # 设置为空
#         # 更新上一帧的位置
#         previous_x = previous_x
#         previous_y = previous_y  # 假设x为球的位置
    
#     else:    
#         pass
    
#     return started_append, zero_count, previous_x, previous_y, seg


def process_frame(frame_number, x, y, started_append, zero_count, previous_x, previous_y, seg):
    # 第一次检查到球时开始计算
    if x != 0 and y != 0 and not started_append:
        zero_count += 1
        # 如果一共20帧检测到球
        if zero_count >= 15:
            zero_count = 0
            started_append = True
        
        
    # 是否为异常检测(prev_x<150 or prev_x>1350)
    
    # 检测是否为吊高球(prev_y<100)
    if previous_y > 100 and x == 0 and y == 0 and started_append:
        zero_count += 1
        # frames_with_zero_detection.append(frame_number)
        
        # 检查前连续40帧的检测结果是否都为0
        if zero_count >= 30:
            # 输出当前时间帧
            seg.append(frame_number)
        
        # 不更新前置位置
        previous_x = previous_x
        previous_y = previous_y
        
    # 如果是吊高球
    elif previous_y < 100 and x == 0 and y == 0 and started_append:
        previous_x = previous_x
        previous_y = previous_y

    # 其中漏检情况和正常击打
    elif started_append and (150 < x < 1350):
        zero_count = 0 # 重置连续检测到的x, y为0的帧数
        # 更新上一帧的位置
        previous_x = x
        previous_y = y  # 假设x为球的位置
    
    elif started_append: # 异常检测(prev_x<150 or prev_x>1350)
        zero_count += 1 # 设置为空
        # 更新上一帧的位置
        previous_x = previous_x
        previous_y = previous_y  # 假设x为球的位置
    
    else:    
        pass
    
    return started_append, zero_count, previous_x, previous_y, seg



def main():
    # 初始化变量
    previous_x, previous_y = 0, 0  # 上一帧球的位置
    zero_count = 0  # 连续检测到的x, y为0的帧数
    seg = [] # 回合结束时间
    started_append = False # 回合开始标志
    
    # path = 'd:/desktop/end'
    # files = os.listdir(path)
    # for file in files:
    #     if file.endswith('.txt'):
    #         previous_position = 0  # 上一帧球的位置
    #         zero_count = 0  # 连续检测到的x, y为0的帧数
    #         seg = [] # 回合结束时间
    #         started_append = False # 回合开始标志
    #         video_name = os.path.join(path, os.path.basename(file)[:-4] + '.mp4')
    #         video_name_new = os.path.join(path, os.path.basename(file)[:-4] + '_01.mp4')
    #         with open(os.path.join(path, file), 'r')as f:
    #             for line in f:
    #                 if seg:
    #                     print(seg)
    #                     with open('d:/desktop/end/end_time.txt', 'a') as final:
    #                         final.write(file+' '+str(seg)+'\n')
    #                         cmd = f'ffmpeg -i {video_name} -frames:v {seg[0]} {video_name_new}' 
    #                         os.system(cmd)
    #                     break
    #                 else:
    #                     frame, x, y = line.strip('\n').split(' ')
    #                     frame = int(frame)
    #                     x = float(x)
    #                     y = float(y)
    #                     started_append, zero_count, previous_position, seg = process_frame(frame, x, y, started_append, zero_count, previous_position, seg)
            # with open('d:/desktop/end/end_time.txt', 'a') as final:
            #     final.write(file+' '+str(seg)+'\n')
            # cmd = f'ffmpeg -i {video_name} -frames:v {seg[0]} {video_name_new}' 
            # os.system(cmd)

    # 在每一帧处理时调用 process_frame 函数，并传入帧数、检测到的 x 和 y 坐标
    with open('D:/desktop/end/258.txt', 'r')as f:
        for line in f:
            if seg:
                # pdb.set_trace()
                print(seg)
                with open('d:/desktop/end/end_time.txt', 'a') as final:
                    final.write('258.txt'+' '+str(seg)+'\n')
                break
            else:
                frame, x, y = line.strip('\n').split(' ')
                frame = int(frame)
                x = float(x)
                y = float(y)
                started_append, zero_count, previous_x, previous_y, seg = process_frame(frame, x, y, started_append, zero_count, previous_x, previous_y, seg)
    cmd = f'ffmpeg -i D:/desktop/end/258.mp4 -frames:v {seg[0]} D:/desktop/end/258_01.mp4' 
    os.system(cmd)
        # pdb.set_trace()
        
if __name__ == '__main__':
    main()