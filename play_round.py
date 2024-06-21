# import time
# import os
# from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
# from PIL import Image
# import numpy as np

# dict_file_path = 'all_clip_sorted_test.txt' 

# # 初始化上次检查的最后修改时间
# last_check_time = os.path.getmtime(dict_file_path) - 0.5


# with open(dict_file_path, 'r') as file:
#     content = file.read()
#     # 将字符串转换成字典
#     my_dict = eval(content)

# # 生成文件列表
# filelist = [video_name + '.mp4' for video_name in my_dict.keys()]

# image_path = 'pause.jpg' 
# image = Image.open(image_path)

# # 调整图片大小以匹配视频帧大小
# resized_image = image.resize([1500,1000])

# # # 定义图片显示的时间（秒）
# image_duration = 0.5

# # 创建一个 MoviePy ImageClip，固定显示时间
# image_clip = ImageClip(np.array(resized_image)).set_duration(image_duration)

# while True:
#     current_time = os.path.getmtime(dict_file_path)

#     if current_time > last_check_time:
#         video_clips = []

#         for video_file in filelist:
#             video_path = os.path.join('test_clip_video2/res10', video_file)  
#             clip = VideoFileClip(video_path)

#             video_clips.append(clip)

#             video_clips.append(image_clip)

#         final_clip = concatenate_videoclips(video_clips)

#         output_path = 'output_video_test_court1_filter40.mp4'
#         final_clip.write_videofile(output_path, codec='libx264', fps=24)
#         print("finish")

#         # 更新最后检查时间
#         last_check_time = current_time

#     # 休眠一段时间后再次检查
#     time.sleep(1)  


# import time
# import os
# import subprocess
# from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
# from PIL import Image
# import numpy as np
# import pdb
# import signal
# dict_file_path = 'all_clip_sorted_test.txt' 

# # 初始化上次检查的最后修改时间
# last_check_time = os.path.getmtime(dict_file_path) - 0.5

# with open(dict_file_path, 'r') as file:
#     content = file.read()
#     # 将字符串转换成字典
#     my_dict = eval(content)

# # 生成文件列表
# filelist = [video_name + '.mp4' for video_name in my_dict.keys()]
# # pdb.set_trace()

# logo_path = 'logo.mp4'
# image_path = 'pause.jpg' 
# image = Image.open(image_path)

# # 调整图片大小以匹配视频帧大小
# resized_image = image.resize([1500,1000])

# # 定义图片显示的时间（秒）
# image_duration = 2

# image_clip = ImageClip(np.array(resized_image)).set_duration(image_duration)

# while True:
#     current_time = os.path.getmtime(dict_file_path)

#     if current_time > last_check_time:
#         playlist = []

#         for video_file in filelist:
#             video_path = os.path.join('test_clip_video2/res10', video_file)

#             ffplay_command1 = ['ffplay', '-autoexit', '-hide_banner', '-fs',  video_path]

#             subprocess.run(ffplay_command1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#             ffplay_command2 = ['ffplay', '-autoexit', '-fs', '-hide_banner', logo_path]

#             subprocess.run(ffplay_command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#         last_check_time = current_time

#     # 休眠一段时间后再次检查
#     time.sleep(1)



# import time
# import os
# import subprocess
# from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
# from PIL import Image
# import numpy as np
# import pdb
# import signal
# dict_file_path = 'all_clip_sorted_test.txt' 

# # 初始化上次检查的最后修改时间
# last_check_time = os.path.getmtime(dict_file_path) - 0.5

# while True:
#     current_time = os.path.getmtime(dict_file_path)

#     with open(dict_file_path, 'r') as file:
#         content = file.read()
#         my_dict = eval(content)

#     # 生成文件列表
#     filelist = [video_name + '.mp4' for video_name in my_dict.keys()]
#     logo_path = 'logo.mp4'
#     playlist = []

#     if current_time > last_check_time:
#         for video_file in filelist:
#             video_path = os.path.join('test_clip_video2/res10', video_file)

#             playlist.append(video_path)

#             playlist.append(logo_path)

#             ffplay_command1 = ['ffplay', '-autoexit', '-hide_banner', '-fs',  video_path]
#             subprocess.run(ffplay_command1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#             ffplay_command2 = ['ffplay', '-autoexit', '-fs', '-hide_banner', logo_path]
#             subprocess.run(ffplay_command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
#         # ffplay_command = ['ffplay', ' -f concat', '-autoexit', '-hide_banner', '-fs', ] + sum([['-i', file] for file in playlist], [])
#         # ffplay_command = [
#         #     'ffplay', '-autoexit', '-protocol_whitelist', 'file,concat',
#         #     '-f', 'concat', '-i', 'concat:' + '|'.join(playlist)
#         # ]
#         # subprocess.run(ffplay_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#         last_check_time = current_time

#     # 休眠一段时间后再次检查
#     time.sleep(1)


# import time
# import os
# import subprocess
# from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip
# from PIL import Image
# import numpy as np
# import pdb
# import signal
# dict_file_path = 'all_clip_sorted_test.txt' 

# # 初始化上次检查的最后修改时间
# # last_check_time = os.path.getmtime(dict_file_path)
# current_time = os.path.getmtime(dict_file_path)

# while True:
#     last_check_time = current_time
    
#     with open(dict_file_path, 'r') as file:
#         content = file.read()
#         my_dict = eval(content)

#     # 生成文件列表
#     filelist = [video_name + '.mp4' for video_name in my_dict.keys()]
#     logo_path = 'logo.mp4'
#     playlist = []

#     while current_time == last_check_time:
#         for video_file in filelist:
#             video_path = os.path.join('test_clip_video2/res10', video_file)

#             ffplay_command1 = ['ffplay', '-autoexit', '-hide_banner', '-fs',  video_path]
#             subprocess.run(ffplay_command1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#             ffplay_command2 = ['ffplay', '-autoexit', '-fs', '-hide_banner', logo_path]
#             subprocess.run(ffplay_command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#             current_time = os.path.getmtime(dict_file_path)

#             if current_time != last_check_time:
#                 break


import os
import time
import subprocess
import numpy as np
import pdb


dict_file_path = 'fancy_clip.txt' 
logo_path = 'logo.mp4'
playlist = []
# 初始化上次检查的最后修改时间
# last_check_time = os.path.getmtime(dict_file_path)
current_time = os.path.getmtime(dict_file_path)

with open(dict_file_path, 'r') as file:
    content = file.read().strip()
    # pdb.set_trace()
    my_dict = eval(content)
    filelist = [video_name for video_name in my_dict.keys()]
    

while True:
    last_check_time = current_time
    
    if os.path.getsize(dict_file_path) != 1:
        # 如果新生成的文件大小为1("\n")，  出现空场   继续播放上一个txt的内容
        with open(dict_file_path, 'r') as file:
            content = file.read().strip()
            my_dict = eval(content)
        # 生成文件列表
        filelist = [video_name for video_name in my_dict.keys()]


    while current_time == last_check_time and all(os.path.exists(video_path) for video_path in filelist):
        for video_path in filelist:

            ffplay_command1 = ['ffmpeg.ffplay', '-vcodec', 'h264_cuvid', '-autoexit', '-hide_banner', '-fs',  video_path]
            subprocess.run(ffplay_command1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            ffplay_command2 = ['ffmpeg.ffplay', '-autoexit', '-fs', '-hide_banner', logo_path]
            subprocess.run(ffplay_command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            current_time = os.path.getmtime(dict_file_path)

            if current_time != last_check_time:
                break

    time.sleep(1)
