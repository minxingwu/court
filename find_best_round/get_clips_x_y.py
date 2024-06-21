from get_longest_clips import get_longest_clips
import argparse
import numpy as np
import pandas as pd
import os 

"""
a good impletation to read the x_y data
"""



def filter_frames(file_name, frame_start, frame_end):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Filter the DataFrame for frames between frame_start and frame_end (inclusive)
    filtered_df = df[(df['Frame'] >= frame_start) & (df['Frame'] <= frame_end)]

    # Convert the filtered DataFrame to a list of dictionaries
     # filtered_list = filtered_df.to_dict(orient='records')

    #return filtered_list
    
    return filtered_df



def read_clips_x_y(longest_clips,x_y_file):
    """
    read the x_y datas from the csv file.

    Parameters:
    longest_clips(list):   [clip_name, start_time, end_time, duration]
    path(string): path to the csv file.


    Returns:
    {clip_name : df frame} : 
        '2024-05-21_21-27-54_the_43_court_10876':        
        Frame  Visibility    X    Y    
    """
    x_y_data = {}

    for longest_clip in longest_clips: 
   
        
        x_y_data[longest_clip[0]] = filter_frames(x_y_file,longest_clip[1],longest_clip[2])

    return x_y_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("path", type=str, help="path to video")
    args = parser.parse_args()

    path = args.path
    # print 20% longest video
    longest_coefficent = 0.2
    # the postfix would add to the path name of the txt files.
    time_file_postfix = ".txt"
    x_y_file_posfix = "_tracknet.txt"


    last_folder_name = os.path.split(path)[-1]
    time_file = path + '/' + last_folder_name + time_file_postfix
    x_y_file = path + '/' + last_folder_name +  x_y_file_posfix


    # [clip_name,start_time,end_time,duration]
    longest_clips = get_longest_clips(time_file,longest_coefficent)

    print(read_clips_x_y(longest_clips,x_y_file))








    




    