import pandas as pd
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def filter_frames(file_name, frame_start, frame_end):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Filter the DataFrame for frames between frame_start and frame_end (inclusive)
    filtered_df = df[(df['Frame'] >= frame_start) & (df['Frame'] <= frame_end)]

    # Convert the filtered DataFrame to a list of dictionaries
    filtered_list = filtered_df.to_dict(orient='records')

    return filtered_list

def plot_points(filtered_list, output_file):
    x_coords = [row['X'] for row in filtered_list]
    y_coords = [row['Y'] for row in filtered_list]

    # Create a scatter plot of all points
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, c='red', label='Object Movement')
    plt.title('Object Movement from Frame {} to Frame {}'.format(filtered_list[0]['Frame'], filtered_list[-1]['Frame']))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image file
    plt.savefig(output_file)
    plt.close()

def plot_3d_points(filtered_list, output_file):
    x_coords = [row['X'] for row in filtered_list]
    y_coords = [row['Y'] for row in filtered_list]
    z_coords = [row['Frame'] for row in filtered_list]

    


    # Create a 3D scatter plot of all points
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')


    scale_factor = 10  # Adjust this factor as needed
    scaled_z_coords = [z * scale_factor for z in z_coords]
    ax.scatter(x_coords, y_coords, scaled_z_coords, c='red', label='Object Movement')
    ax.set_title('3D Object Movement from Frame {} to Frame {}'.format(filtered_list[0]['Frame'], filtered_list[-1]['Frame']))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Frame')

    ax.set_xlim(min(x_coords) - 1, max(x_coords) + 1)
    ax.set_ylim(min(y_coords) - 1, max(y_coords) + 1)
    ax.set_zlim(min(scaled_z_coords) - 1, max(scaled_z_coords) + 1)

    ax.legend()
    plt.grid(True)
    
    # Save the plot as an image file
    plt.savefig(output_file)
    plt.close()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Filter frames from a CSV file and plot points.')
    parser.add_argument('--file_name', type=str,default="./save_video/2024-05-21_21-27-54_the_43_court/2024-05-21_21-27-54_the_43_court_tracknet.csv",help='Path to the CSV file.')
    parser.add_argument('--frame_start', type=int, default=8654, help='Start frame number (default: 100).')
    parser.add_argument('--frame_end', type=int, default=8750, help='End frame number (default: 200).')
    parser.add_argument('--output_file', type=str, default='output_plot.png', help='Output image file (default: output_plot.png).')

    # Parse the arguments
    args = parser.parse_args()

    # Filter frames based on the input arguments
    filtered_list = filter_frames(args.file_name, args.frame_start, args.frame_end)

    # Plot all points on one plot
    plot_3d_points(filtered_list, args.output_file)

    print(f"Plot saved as {args.output_file}")

if __name__ == "__main__":
    main()
