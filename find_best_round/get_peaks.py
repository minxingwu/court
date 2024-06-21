import argparse
import os 
from get_longest_clips import get_longest_clips
from get_clips_x_y import read_clips_x_y
from process_x_y import get_consecutive_value



def counting_peak(value):
    """
    rounds is calculated by counting the peak inside the data.

    Arg:
        value(List[int]): the value used to calculate peak.
    
    Return: 
        number of peak
    """
    count = 0
    i = 1
    n = len(value)

    # Loop through the array to find increasing then decreasing sequences
    while i < n - 1:
        # Find the increasing sequence
        while i < n and value[i] > value[i - 1]:
            i += 1
        # Check for a peak (increasing then decreasing)
        if i < n and value[i] < value[i - 1]:
            count += 1
            # Find the decreasing sequence
            while i < n and value[i] < value[i - 1]:
                i += 1
        else:
            i += 1

    if count == 0:
        count += 1 
    return count


def get_clips_peaks(x_ys):
    """
    Args: 
        x_ys(dict): 
            dict contain df for longest clips.
            The key is the longest clip name.
            value the if df which contain x_y values
    
    Return:
        int: the calculated rounds number inside clip. the minium
            number is 1.
        
    """
    # filter size 
    frame = 3

    # distance to find outliter
    max_diff = 150

    # distance to find outliter
    min_diff = 5

    
    # store the y_s of each clip
    clips = []

    for _, x_y in x_ys.items():

        y_values = x_y['Y'].tolist()
        valid_y = get_consecutive_value(y_values,frame,max_diff)
        clips.append(counting_peak(valid_y))

    return clips


def get_peaks_of_longest_clips(path, longest_coefficent=0.2, time_file_postfix = ".txt", x_y_file_posfix = "_tracknet.txt"):
    """
    Get rounds number of the selected longest clips. the number of selected 
    clips in decide by the longest_coefficent. process the folder path to the file path.

    Args:
        path (string):
            complete path to the game folder which contain the x_y datas files
            and start-end time flies of clips.

        longest_coefficent(float): this coefficent the proportion of clips will be
            returned. 0.2 will mean this function will return 20% longest clips

        time_file_postfix(string): postfix to time files' name
            the name of times' file which store the 
            start and end time of clips is constructed by adding the path and
            this postfix 

        x_y_file_posfix(string): postfix to x_y_files.
    """

    # generate the path to x_y file and start,end files
    # the file name prefix : 2024-05-21_21-27-54_the_43_court_2877
    _, last_folder_name = os.path.split(path)
    time_file = path + '/' + last_folder_name + time_file_postfix
    x_y_file = path + '/' + last_folder_name +  x_y_file_posfix

    longest_clips = get_longest_clips(time_file,longest_coefficent)
    x_ys = read_clips_x_y(longest_clips,x_y_file)

    rounds = get_clips_peaks(x_ys)

    scores = []

    # add one column to the clips df
    for index,clip in enumerate(longest_clips):
        clip.append(rounds[index])
        scores.append(clip[3]/rounds[index])
        clip.append(scores[index])

    print(f"{'clip_name': <40} {'round':<8} {'duration':<10} {'speed_based_score':<40}")
    for longest_clip in longest_clips:
        print(f"{longest_clip[0]:<40} {longest_clip[-2]:<9}{longest_clip[-3]:<10} {longest_clip[-1]:.2f}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate the rounds number of every clips inside a game")
    parser.add_argument("path", type=str, help="path to where the video and relative data are stored")
    parser.add_argument("longest_coefficent", type=float, default= 1.0 ,help="proportion of longests clips will be used to calculate rounds")
    parser.add_argument("time_file_postfix", type=str, default= ".txt",  help = "postfix to file store the start and end time of clip.")
    parser.add_argument("x_y_file_posfix", type=str, default= "_tracknet.txt",help = "postfix")
    args = parser.parse_args()
   
    path = args.path
    longest_coefficent = args.longest_coefficent
    time_file_postfix = args.time_file_postfix
    x_y_file_posfix = args.x_y_file_posfix

    get_peaks_of_longest_clips(path, longest_coefficent, time_file_postfix, x_y_file_posfix)









