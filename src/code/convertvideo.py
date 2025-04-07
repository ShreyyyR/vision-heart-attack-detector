import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Function to select video file
def select_video():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return video_path

# Function to create directories
def create_directories(base_dir):
    emergency_dir = os.path.join(base_dir, "emergency")
    non_emergency_dir = os.path.join(base_dir, "non_emergency")
    
    if not os.path.exists(emergency_dir):
        os.makedirs(emergency_dir) 
    if not os.path.exists(non_emergency_dir):
        os.makedirs(non_emergency_dir)
    
    return emergency_dir, non_emergency_dir

# Function to extract frames and save them
def save_frames(video_path, emergency_dir, non_emergency_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_name = os.path.basename(video_path).split('.')[0]
    
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"{base_name}_frame_{frame_num}.jpg"
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Ask for user input to label the frame
        print(f"Processing frame {frame_num}/{frame_count}")
        label = input("Label this frame as 'e' for emergency, 'n' for non-emergency, 'q' to quit: ").lower()
        
        if label == 'e':
            frame_path = os.path.join(emergency_dir, frame_filename)
        elif label == 'n':
            frame_path = os.path.join(non_emergency_dir, frame_filename)
        elif label == 'q':
            break
        else:
            print("Invalid input, skipping this frame.")
            continue
        
        cv2.imwrite(frame_path, frame)
        frame_num += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    video_path = select_video()
    if not video_path:
        print("No video file selected.")
        return
    
    base_dir = os.path.join(os.getcwd(), "labeled_frames")
    emergency_dir, non_emergency_dir = create_directories(base_dir)
    
    save_frames(video_path, emergency_dir, non_emergency_dir)
    print(f"Frames saved in '{emergency_dir}' and '{non_emergency_dir}'")

if __name__ == "__main__":
    main()
