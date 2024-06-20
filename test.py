import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def filter_frames(file_name, frame_start, frame_end):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Filter the DataFrame for frames between frame_start and frame_end (inclusive)
    filtered_df = df[(df['Frame'] >= frame_start) & (df['Frame'] <= frame_end)]

    # Convert the filtered DataFrame to a list of dictionaries
    filtered_list = filtered_df.to_dict(orient='records')

    return filtered_list

def create_video(frames_dir, video_path, fps):
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])

    # Determine the width and height from the first frame
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # Create a video writer object with the specified FPS
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Filter frames from a CSV file and create a video.')
    parser.add_argument('--file_name', type=str,default="./save_video/2024-05-21_21-27-54_the_43_court/2024-05-21_21-27-54_the_43_court_tracknet.csv",help='Path to the CSV file.')
    parser.add_argument('--frame_start', type=int, default=1, help='Start frame number (default: 100).')
    parser.add_argument('--frame_end', type=int, default=12299, help='End frame number (default: 200).')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second for the output video (default: 30).')

    # Parse the arguments
    args = parser.parse_args()

    # Filter frames based on the input arguments
    filtered_list = filter_frames(args.file_name, args.frame_start, args.frame_end)

    # Directory to save frames
    frames_dir = 'frames'
    os.makedirs(frames_dir, exist_ok=True)

    # Generate and save frames with trace
    for i, row in enumerate(filtered_list):
        plt.figure()
        plt.scatter(row['X'], row['Y'], c='red')
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        plt.title(f'Frame {row["Frame"]}')
        frame_path = os.path.join(frames_dir, f'frame_{i}.png')
        plt.savefig(frame_path)
        plt.close()

    # Create the video from the saved frames
    video_path = 'output_video.mp4'
    create_video(frames_dir, video_path, args.fps)

    print(f"Video saved at {video_path}")

if __name__ == "__main__":
    main()
