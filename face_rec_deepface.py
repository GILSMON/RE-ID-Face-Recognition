import cv2
from deepface import DeepFace
import os



script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd() # Get the current directory of the script

# Define the folder where the video and image are located.
# Change this line to point to the 'resources' folder
video_folder = os.path.join(script_dir, 'resources') 

# Now the paths will be constructed correctly
video_path = os.path.join(video_folder, 'mi_video2.mp4')
tom_cruise_image_path = os.path.join(video_folder, 'tom_cruise_ref.jpg')
output_video_path = os.path.join(video_folder, 'mi_video2_deepface_identified.mp4')

print(f"Input Video: {video_path}")
print(f"Tom Cruise Reference Image: {tom_cruise_image_path}")
print(f"Output Video: {output_video_path}")



video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    print("Warning: FPS is 0 or less, setting to 30 for output video.")
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Video properties: Width={frame_width}, Height={frame_height}, FPS={fps}")
print(f"Output video writer initialized for: {output_video_path}")

# --- 5. Process Video Frame by Frame with DeepFace ---
frame_count = 0
print("Starting frame processing for identification...")

# Cache to store face bounding box for a few frames to improve performance
last_known_face = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processing frame {frame_count}...")

    # DeepFace's analysis is resource-intensive. 
    # Process only a few frames to maintain speed.
    if frame_count % 5 == 0:  # Process every 5th frame
        try:
            # Use DeepFace to detect faces in the current frame.
            # `enforce_detection=False` is crucial to prevent errors if no face is found.
            # `detector_backend='retinaface'` is a good choice for speed and accuracy.
            detections = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
            
            # Reset last_known_face if no faces are detected
            last_known_face = None 
            
            for detection in detections:
                x, y, w, h = detection['facial_area']['x'], detection['facial_area']['y'], detection['facial_area']['w'], detection['facial_area']['h']
                face_location = (y, x + w, y + h, x) # Convert to face_recognition's (top, right, bottom, left) format
                
                # Extract the face from the frame using the bounding box
                face_crop = frame[y:y+h, x:x+w]
                
                # Verify the detected face against the Tom Cruise reference image.
                # `enforce_detection=False` is important for the verification step as well.
                # `model_name` can be changed to 'VGG-Face', 'ArcFace', etc.
                result = DeepFace.verify(face_crop, tom_cruise_image_path, model_name='ArcFace', enforce_detection=False)
                
                # Check the verification result
                if result['verified']:
                    name = "Tom Cruise"
                else:
                    name = "Unknown"
                
                last_known_face = (face_location, name)
                
                # Break after the first face found to avoid processing multiple faces in the same frame
                # for simplicity, adjust as needed.
                break

        except Exception as e:
            # Handle cases where DeepFace might fail
            print(f"DeepFace analysis failed on frame {frame_count}: {e}")
            last_known_face = None
            
    # --- 6. Draw Bounding Boxes and Labels on Original Frame ---
    if last_known_face:
        (top, right, bottom, left), name = last_known_face
        
        box_color = (0, 255, 0) if name == "Tom Cruise" else (0, 0, 255)
        text_color = (255, 255, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, text_color, 1)

    # --- 7. Write Processed Frame to Output Video ---
    out.write(frame)

# Clean up
video_capture.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete.")