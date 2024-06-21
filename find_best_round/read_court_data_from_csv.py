import pandas as pd
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Filter frames from a CSV file.')
parser.add_argument('--file_name', type=str, default="./save_video/2024-05-21_21-27-54_the_43_court/2024-05-21_21-27-54_the_43_court_tracknet.csv", help='Path to the CSV file.')
parser.add_argument('--frame_start', type=int, default=100, help='Start frame number.')
parser.add_argument('--frame_end', type=int, default=200, help='End frame number.')
args = parser.parse_args()

# Read the CSV file into a DataFrame
df = pd.read_csv(args.file_name)
start = args.frame_start
end = args.frame_end

# Filter the DataFrame for frames between 100 and 200 (inclusive)
filtered_df = df[(df['Frame'] >= start) & (df['Frame'] <= end)]

# Convert the filtered DataFrame to a list of dictionaries
filtered_list = filtered_df.to_dict(orient='records')

x_axis = []
y_axis = []
frames = []

# Print the filtered list
for row in filtered_list:
    x_axis.append(row['X'])
    y_axis.append(row['Y'])
    frames.append([row['X'], row['Y']])

print(x_axis)
print(y_axis)
print(x_y_axis)


frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# Generate and save frames
for i, (x, y) in enumerate(frames):
    plt.figure()
    plt.scatter(x, y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Frame {i}')
    frame_path = os.path.join(frames_dir, f'frame_{i}.png')
    plt.savefig(frame_path)
    plt.close()

# Create a video from the saved frames
video_path = 'output_video.mp4'
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])

# Determine the width and height from the first frame
frame = cv2.imread(frame_files[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    video.write(frame)

cv2.destroyAllWindows()
video.release()

print(f"Video saved at {video_path}")
