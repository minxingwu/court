
import os 






def get_consecutive_value(value,size,max_diff):
    """
    filter all the list which is sparse and contain outlier.

    Args:
        value(list[int]): the value to be filter
        size(int): size of the filter
        max_diff(int): the max value to detect outlier


    Return:
        list[int]
    """
    valid_value = []
    # fliter to select the consecutive labeled frames.
    # calculate the diff between two frames for filter size's frames.
    for index in range(len(value)-size-1):
        diffs = []
        for i in range(size):
            diffs.append(abs(value[index + i + 1] - value[index + i]))
        
        # the diff between two consecutive y value must inside certain range
        valid = True
        for diff in diffs:
            if (diff > max_diff):
                valid = False
                break

        if valid == True:
            valid_value.append(value[index])

    return valid_value




