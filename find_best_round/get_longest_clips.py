import argparse
import os 
import numpy as np



def get_longest_clips(time_file_path,longest_coefficent=0.2):
    """
    get the longest clips based on the start and ending time of clips

    Args:   
        time_file_path (string): the files stores the start and ending frame of 
            clips
            format:
            1000 -> start frame of first clip
            2000 -> end frame of first frame 
            3000 > start frame of second frame

            the input path should be specific to fille
            format:
            /ssd2/wmx/court_code_new_round_test/save_video/2024-05-17_20-32-52_
            the_13_court/2024-05-17_20-32-52_the_13_court.txt 

        longest_coefficent (float): this coefficent the proportion of clips 
            will be returned. 0.2 will mean this function will return 20% 
            longest clips
    
    Returns:
        list: the list contain basic information for longest clips
            format:
            [[clip_name,start_time,end_time,duration],....]
    """

    
    try:
        start_end_time = []
        with open(time_file_path, 'r') as file:
            for line in file:
                # Strip any whitespace characters and convert to integer
                start_end_time.append(int(line.strip()))


        start_time = [start_end_time[i] for i in range(0, len(start_end_time),2)]
        end_time =  [start_end_time[i] for i in range(1, len(start_end_time),2)]
        duration = [end_time[i]- start_time[i] for i in range(0,len(end_time))]




        # print("start_time:")
        # print(start_time)
        # print("end_time:")
        # print(end_time)
        # print("duration")
        # print(duration)
        
        # get sorted index of longest clips
        duration_np = np.array(duration)
        duration_sorted_indices = np.argsort(duration_np)[::-1]
        # print("duration index from short to long:")
        # print(duration_sorted_indices)


        # for index in duration_sorted_indices:
        #     if(duration[index] > 150):
        #         print("start:" + str(start_time[index])+ " end:" + str(end_time[index])+' duration:'+ str(duration[index]))


        # for index in duration_sorted_indices:
        #     if(duration[index] > 150):
        #         clip_name = last_folder_name + str(start_time[index]) + '.mp4'
        #         print(clip_name)

            
    
        longest_indexes = int(longest_coefficent * len(duration_sorted_indices))



        # get last folder name 
        longest_clips = []
        dirname = os.path.dirname(time_file_path)
        last_folder_name =  os.path.basename(dirname)        

        
        for index in range(longest_indexes):
            # duration_sorted_indices[index] is the index of long video
            indexx = duration_sorted_indices[index]
            if(duration[indexx] > 150):
                
                clip_name = last_folder_name + '_' + str(start_time[indexx])
                #print(clip_name)
                #print("start:" + str(start_time[indexx])+ " end:" + str(end_time[indexx])+' duration:'+ str(duration[indexx])) 
                longest_clips.append([clip_name,start_time[indexx],end_time[indexx],duration[indexx]])

        return longest_clips


    except FileNotFoundError:
        print(f"{time_file_path} The file does not exist.")
        return None
    except IOError:
        print("An error occurred while reading the file.")
        return None







if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("path", type=str, help="path to video time file")
    args = parser.parse_args()

    # print 20% longest video
    longest_coefficent = 0.2

    time_file_postfix = ".txt"

    path = args.path
    _, last_folder_name = os.path.split(path)
    time_file_path = path + '/' +last_folder_name + time_file_postfix

    print("longest_clips")
    print(get_longest_clips(time_file_path,longest_coefficent))
