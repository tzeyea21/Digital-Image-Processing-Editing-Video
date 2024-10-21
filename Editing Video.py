import cv2
import numpy as np

def detect_daytime_nighttime(frame, brightness_threshold_factor=0.8): 
    # Process a single frame, adjusting brightness if necessary
    # Convert the frame to grayscale and calculate average brightness
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    threshold = avg_brightness * brightness_threshold_factor
    
    BRIGHTNESS_INCREASE_BASE = 400  # Base value
    BRIGHTNESS_SCALE_FACTOR = 70  # Scale factor
    
    # Calculate brightness threshold and adjust brightness if necessary
    # Threshold: to determine whether the frame is too dark
    if avg_brightness < threshold:
        brightness_increase = int(BRIGHTNESS_INCREASE_BASE + BRIGHTNESS_SCALE_FACTOR * (threshold - avg_brightness))
        frame = increase_brightness(frame, brightness_increase)

    return frame
    
def increase_brightness(frame, brightness_increase):
    # Increase the brightness of a frame
    return np.clip(frame + brightness_increase, 0, 255).astype(np.uint8)

def blur_faces(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame #Scale Factor, minNeighbors
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    # Blur the faces based on temporal information
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (15, 15), 10)
        frame[y:y+h, x:x+w] = blurred_face
          
    return frame
    
def add_border(frame, border_size = 3):
    # Add border to the overlay video
    return cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)

def resize_video(overlay_video, background_width, background_height):
    # Get overlay video properties
    overlay_width = int(overlay_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    overlay_height = int(overlay_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate scale factor (maintain original height and width while scaling down)
    scale_factor = min(background_width / (5 * overlay_width), background_height / (5 * overlay_height))
    
    # Resize the overlay video
    resized_width = int(overlay_width * scale_factor)
    resized_height = int(overlay_height * scale_factor)
   
    return (resized_width, resized_height)

def overlay_video(background_path):
    # Hardcoded path to the overlay video
    overlay_video_path = 'talking.mp4'
    
    # Load the overlay and background video
    overlay_video = cv2.VideoCapture(overlay_video_path)
    background_video = cv2.VideoCapture(background_path)
    
    #Get the properties of background video and resize
    background_width = int(background_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    background_height = int(background_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resized_overlay_dimensions = resize_video(overlay_video, background_width, background_height)
    
    # Initialize an empty list to store processed frames
    processed_frames = []
    position = (250, 150)

    while True:
        # Read frames from both videos
        background_ret, background_frame = background_video.read()
        overlay_ret, overlay_frame = overlay_video.read()
        
        if not background_ret or not overlay_ret:
            break

        # Apply blur_faces only to the background frame
        background_frame = detect_daytime_nighttime(background_frame)
        background_frame = blur_faces(background_frame)
        
        # Resize both frame
        resized_overlay_frame = cv2.resize(overlay_frame, resized_overlay_dimensions)
        bordered_overlay_frame = add_border(resized_overlay_frame)

        overlay_start_y = position[1]
        overlay_end_y = position[1] + bordered_overlay_frame.shape[0]
        overlay_start_x = position[0]
        overlay_end_x = position[0] + bordered_overlay_frame.shape[1]

        # Overlay the bordered overlay frame on top of the background frame
        background_frame[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x] = bordered_overlay_frame

        processed_frames.append(background_frame)

    background_video.release()
    overlay_video.release()

    return processed_frames


def watermark(frames):
    # Load the watermarks
    watermark1 = cv2.imread('watermark1.png')
    watermark2 = cv2.imread('watermark2.png')
    
    fps = 30  
    duration = int(fps) * 5  # 5 seconds for each watermark
    
    watermarked_frames = []
    for frame_number, frame in enumerate(frames):
        # Determine which watermark to show based on the frame number
        watermark = watermark1 if frame_number % (2 * duration) < duration else watermark2

        # Add the watermark to the frame
        overlay = np.zeros_like(frame)
        overlay[:watermark.shape[0], :watermark.shape[1]] = watermark
        frame_with_watermark = cv2.addWeighted(frame, 1, overlay, 0.5, 0)
        watermarked_frames.append(frame_with_watermark)

    return watermarked_frames

def add_endscreen(processed_frames, output_video_path):
    # Hardcoded path to the endscreen video
    endscreen_video_path = 'endscreen.mp4'
    
    # Load the endscreen video
    endscreen_video = cv2.VideoCapture(endscreen_video_path)
    
    # Save the original number of frames
    num_original_frames = len(processed_frames)
    
    # Initialize the output_video variable
    output_video = None

    # Append frames from the endscreen video
    while True:
        success_endscreen, endscreen_frame = endscreen_video.read()
        if not success_endscreen:
            break
        processed_frames.append(endscreen_frame)
    
    # Save the final video with the endscreen
    if processed_frames[num_original_frames:]:
        fps = 30  # Assuming a default fps
        height, width = processed_frames[0].shape[:2]
        output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, output_fourcc, fps, (width, height))

        for frame in processed_frames:
            output_video.write(frame)
        
   # Release the video objects and close the output video
    if output_video is not None:
        output_video.release()
    
    return processed_frames
    
def play_video(video_path):
    processed_frames = overlay_video(video_path)  
    watermarked_frames = watermark(processed_frames)
    output_video = add_endscreen(watermarked_frames, 'final_video.mp4')

    # Display the processed video
    for frame in output_video:
        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(30) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()

# Demo
play_video("traffic.mp4")
