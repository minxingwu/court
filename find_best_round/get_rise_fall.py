import argparse
import os 
from process_x_y import get_consecutive_value
from get_longest_clips import get_longest_clips
from get_clips_x_y import read_clips_x_y



def get_rise_falls(path = "/ssd2/wmx/court/save_video/2024-05-17_10-45-55_the_75_court", 
                            longest_coefficent=0.2, time_file_postfix = ".txt",
                            x_y_file_posfix = "_tracknet.txt"):
    
    _, last_folder_name = os.path.split(path)
    time_file = path + '/' + last_folder_name + time_file_postfix
    x_y_file = path + '/' + last_folder_name +  x_y_file_posfix

    longest_clips = get_longest_clips(time_file,longest_coefficent)
    x_ys = read_clips_x_y(longest_clips,x_y_file)
    for clip_name, x_y in x_ys.items():
        counting_rise_fall(x_y)
        print(clip_name)
        print(x_y)



def counting_rise_fall(clip):

    '''
    counting the number of fall and rise in the array, rise are defined as contiuns 
    increase of multiple elements. record the start and ending time.

    Args:  
        clips(df): must contain [clip_name, start_time, end_time, duration] for one clip

    Return:


    '''

    print(clip)

    if len(clip) < 2:
        return 0, 0

    rises_count = 0
    # store the start and ending frame
    rises = []
    falls_count = 0
    falls = []
    i = 1

    while i < len(clip):
        # Detect a rise
        round = []
        print(i)
        if clip.loc[i,'Y'] > clip.loc[i-1,'Y']:
            round.append(clip.loc[i-1,'name'])

            while i < len(clip) and clip.loc[i,'Y'] > clip.loc[i-1,'Y']:
                i += 1

            rises_count += 1
            round.append(clip.loc[i-2,'name'])
            rises.append(round)

        # Detect a fall
        elif clip.loc[i,'Y'] < clip.loc[i-1,'Y']:
            round.append(clip.loc[i-1,'name'])

            while i < len(clip) and clip.loc[i,'Y'] < clip.loc[i-1,'Y']:
                i += 1

            falls_count += 1
            round.append(clip.loc[i-2,'name'])
            falls.append(round)
        else:
            i += 1

    return rises, falls
    




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="calculate the rounds number of every clips inside a game")
    parser.add_argument("--path", type=str, default= "/ssd2/wmx/court/save_video/2024-05-17_10-45-55_the_75_court", help="path to where the video and relative data are stored")
    parser.add_argument("--longest_coefficent", type=float, default= 0.2 ,help="proportion of longests clips will be used to calculate rounds")
    parser.add_argument("--time_file_postfix", type=str, default= ".txt",  help = "postfix to file store the start and end time of clip.")
    parser.add_argument("--x_y_file_posfix", type=str, default= "_tracknet.txt",help = "postfix")
    args = parser.parse_args()
   
    path = args.path
    longest_coefficent = args.longest_coefficent
    time_file_postfix = args.time_file_postfix
    x_y_file_posfix = args.x_y_file_posfix

    get_rise_falls()




