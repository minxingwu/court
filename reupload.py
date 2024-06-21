import os
import subprocess
import uuid
import time

script_path = './uploader_ori_re.py'
script_ano_path = './uploader_sub_re.py'
input_folder = '/home/sportvision/court_code_new_round_test/save_video/2024-05-18_19-16-48_the_5_court'
# input_ano_folder = '/home/sportvision/highlight_code/output/clip_video'
folder_name = input_folder.split('/')[-1]
pattern = folder_name + '.mp4'
print('pattern is ',pattern)
session_uuid = str(uuid.uuid4())
# session_uuid = '2bf9079d-2450-4acf-967e-96e427c9b59f'
session_event_timestamp = int(time.time()* 1000)
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
    
        if filename == pattern:
            file_path = os.path.join(input_folder, filename)
            # filename, _ = filename.split('.')
            session_start_time = int(time.time()* 1000)
            session_end_time = int(time.time()* 1000)
            p = subprocess.Popen(['python', script_path, 
                                file_path,
                                str(session_event_timestamp),
                                session_uuid, 
                                str(session_start_time),
                                str(session_end_time),
                                filename[:-4],
                                '2',
                                '9'
                                ],
            )
            p.wait()
        else:
            file_path = os.path.join(input_folder, filename)

            p1 = subprocess.Popen(['python', script_ano_path, 
                                file_path,
                                str(session_event_timestamp),
                                session_uuid,
                                '2'
                                ],
            )
            p1.wait()
