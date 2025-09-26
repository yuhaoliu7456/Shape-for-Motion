


import os
import cv2
import sys

def extract_frames(video_file, frame_rate=2, output_directory="output_frames", num_frames=100):
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_file}")
        return

    # Get the original frame rate of the video
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS) + 1
    if original_frame_rate == 0:
        print("Error: Cannot determine the frame rate of the video.")
        return
    print(f"Original frame rate: {original_frame_rate}")

    # Calculate the interval between frames to capture
    frame_interval = int(original_frame_rate / frame_rate)
    if frame_interval == 0:
        frame_interval = 1  # Ensure at least one frame is captured per second

    print(f"Frame interval: {frame_interval}", f"Frame rate: {frame_rate}")
    frame_count = 0
    saved_frame_count = 0

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_file = f"{output_directory}/{saved_frame_count:05}.png"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")
            saved_frame_count += 1
                
        frame_count += 1
        if frame_count == num_frames:
            break

    cap.release()

if __name__ == "__main__":
    video_file = sys.argv[1]
    output_directory = sys.argv[2]
    frame_rate = float(sys.argv[3])
    num_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None

    extract_frames(video_file, frame_rate=frame_rate, output_directory=output_directory, 
                   num_frames=num_frames if num_frames is not None else 100)