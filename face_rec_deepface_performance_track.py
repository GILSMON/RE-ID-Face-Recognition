import cv2
from deepface import DeepFace
import os
import time
import psutil
import threading
import numpy as np
from datetime import datetime

# --- Performance Monitoring Class ---
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = 0
        self.cpu_usage_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Frame timing
        self.frame_times = []
        self.detection_times = []
        self.verification_times = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.initial_memory
        self.monitoring = True
        
        # Start background thread to monitor CPU and memory
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        print(f"=== Performance Monitoring Started (NO TRACKING) ===")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Memory Usage: {self.initial_memory:.2f} MB")
        print(f"CPU Cores Available: {psutil.cpu_count()}")
        print("=" * 50)
    
    def _monitor_resources(self):
        """Background monitoring of CPU and memory"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.cpu_usage_samples.append(cpu_percent)
                
                # Memory usage
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                time.sleep(0.5)  # Sample every 0.5 seconds
            except:
                break
    
    def log_frame_time(self, frame_time):
        """Log time taken for frame processing"""
        self.frame_times.append(frame_time)
    
    def log_detection_time(self, detection_time):
        """Log time taken for face detection"""
        self.detection_times.append(detection_time)
    
    def log_verification_time(self, verification_time):
        """Log time taken for face verification"""
        self.verification_times.append(verification_time)
    
    def stop_monitoring(self):
        """Stop monitoring and generate report"""
        self.end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        total_time = self.end_time - self.start_time
        current_memory = self.process.memory_info().rss / (1024 * 1024)
        
        # Calculate averages
        avg_cpu = np.mean(self.cpu_usage_samples) if self.cpu_usage_samples else 0
        max_cpu = max(self.cpu_usage_samples) if self.cpu_usage_samples else 0
        
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        avg_verification_time = np.mean(self.verification_times) if self.verification_times else 0
        
        # Calculate FPS
        processed_frames = len(self.frame_times)
        fps_processed = processed_frames / total_time if total_time > 0 else 0
        
        report = {
            'total_time': total_time,
            'processed_frames': processed_frames,
            'fps_processed': fps_processed,
            'initial_memory_mb': self.initial_memory,
            'peak_memory_mb': self.peak_memory,
            'final_memory_mb': current_memory,
            'memory_increase_mb': current_memory - self.initial_memory,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'avg_verification_time_ms': avg_verification_time * 1000,
            'total_detections': len(self.detection_times),
            'total_verifications': len(self.verification_times)
        }
        
        return report
    
    def print_report(self, report):
        """Print formatted performance report"""
        print("\n" + "=" * 70)
        print("              PERFORMANCE REPORT (NO TRACKING)")
        print("=" * 70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Processing Time: {report['total_time']:.2f} seconds")
        print(f"Total Frames Processed: {report['processed_frames']}")
        print(f"Processing Speed: {report['fps_processed']:.2f} FPS")
        print()
        
        print("MEMORY USAGE:")
        print(f"  Initial Memory: {report['initial_memory_mb']:.2f} MB")
        print(f"  Peak Memory: {report['peak_memory_mb']:.2f} MB")
        print(f"  Final Memory: {report['final_memory_mb']:.2f} MB")
        print(f"  Memory Increase: {report['memory_increase_mb']:.2f} MB")
        print()
        
        print("CPU USAGE:")
        print(f"  Average CPU Usage: {report['avg_cpu_percent']:.1f}%")
        print(f"  Peak CPU Usage: {report['max_cpu_percent']:.1f}%")
        print()
        
        print("TIMING BREAKDOWN (Average per operation):")
        print(f"  Frame Processing: {report['avg_frame_time_ms']:.2f} ms")
        print(f"  Face Detection: {report['avg_detection_time_ms']:.2f} ms")
        print(f"  Face Verification: {report['avg_verification_time_ms']:.2f} ms")
        print()
        
        print("OPERATION COUNTS:")
        print(f"  Total Detection Operations: {report['total_detections']}")
        print(f"  Total Verification Operations: {report['total_verifications']}")
        print()
        
        # Calculate efficiency metrics
        detection_efficiency = report['total_detections'] / report['processed_frames'] * 100
        print("EFFICIENCY METRICS:")
        print(f"  Detection Efficiency: {detection_efficiency:.1f}% (detections/frames)")
        
        if report['avg_frame_time_ms'] > 0:
            real_time_factor = (1000/30) / report['avg_frame_time_ms']  # Assuming 30 FPS target
            print(f"  Real-time Factor: {real_time_factor:.2f}x")
            print(f"  {'✓ Real-time capable' if real_time_factor >= 1.0 else '✗ Not real-time capable'}")
        
        print("=" * 70)

# --- Original Code with Performance Monitoring ---
script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
video_folder = os.path.join(script_dir, 'resources')
video_path = os.path.join(video_folder, 'mi_video2.mp4')
tom_cruise_image_path = os.path.join(video_folder, 'tom_cruise_ref.jpg')
output_video_path = os.path.join(video_folder, 'mi_video2_deepface_performance_track.mp4')

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

# Get total frame count
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = "Unknown"

print(f"Video properties: Width={frame_width}, Height={frame_height}, FPS={fps}")
print(f"Total frames to process: {total_frames}")

# Initialize performance monitor
perf_monitor = PerformanceMonitor()
perf_monitor.start_monitoring()

# --- Process Video Frame by Frame with DeepFace ---
frame_count = 0
print("Starting frame processing for identification...")

# Cache to store face bounding box for a few frames to improve performance
last_known_face = None

while True:
    frame_start_time = time.time()
    
    ret, frame = video_capture.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processing frame {frame_count}/{total_frames} ({frame_count/int(total_frames)*100:.1f}%)")

    # DeepFace's analysis is resource-intensive. 
    # Process only a few frames to maintain speed.
    if frame_count % 10 == 0:  # Process every 10th frame
        detection_start_time = time.time()
        
        try:
            # Use DeepFace to detect faces in the current frame.
            detections = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
            
            detection_end_time = time.time()
            perf_monitor.log_detection_time(detection_end_time - detection_start_time)
            
            # Reset last_known_face if no faces are detected
            last_known_face = None 
            
            print(f"Frame {frame_count}: DeepFace found {len(detections)} faces")
            
            for i, detection in enumerate(detections):
                x, y, w, h = detection['facial_area']['x'], detection['facial_area']['y'], detection['facial_area']['w'], detection['facial_area']['h']
                
                # Convert to integers and validate
                x, y, w, h = int(x), int(y), int(w), int(h)
                x = max(0, min(x, frame.shape[1] - 10))
                y = max(0, min(y, frame.shape[0] - 10))
                w = max(10, min(w, frame.shape[1] - x))
                h = max(10, min(h, frame.shape[0] - y))
                
                face_location = (y, x + w, y + h, x)  # Convert to face_recognition's (top, right, bottom, left) format
                
                # Extract the face from the frame using the bounding box
                if w > 0 and h > 0:
                    face_crop = frame[y:y+h, x:x+w]
                    
                    # Verify the detected face against the Tom Cruise reference image.
                    verification_start_time = time.time()
                    try:
                        result = DeepFace.verify(face_crop, tom_cruise_image_path, model_name='ArcFace', enforce_detection=False)
                        
                        # Check the verification result
                        if result['verified']:
                            name = "Tom Cruise"
                            print(f"Frame {frame_count}: ✓ Tom Cruise verified")
                        else:
                            name = "Unknown"
                            print(f"Frame {frame_count}: ✗ Unknown person detected")
                        
                        verification_end_time = time.time()
                        perf_monitor.log_verification_time(verification_end_time - verification_start_time)
                        
                        last_known_face = (face_location, name)
                        
                        # Break after the first face found to avoid processing multiple faces in the same frame
                        break
                        
                    except Exception as verify_error:
                        print(f"Frame {frame_count}: Face verification failed: {verify_error}")
                        verification_end_time = time.time()
                        perf_monitor.log_verification_time(verification_end_time - verification_start_time)
                        
                        last_known_face = (face_location, "Unknown")
                        break

        except Exception as e:
            # Handle cases where DeepFace might fail
            print(f"DeepFace analysis failed on frame {frame_count}: {e}")
            detection_end_time = time.time()
            perf_monitor.log_detection_time(detection_end_time - detection_start_time)
            last_known_face = None
            
    # --- Draw Bounding Boxes and Labels on Original Frame ---
    if last_known_face:
        (top, right, bottom, left), name = last_known_face
        
        box_color = (0, 255, 0) if name == "Tom Cruise" else (0, 0, 255)
        text_color = (255, 255, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        
        # Label background
        label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (left, top - 30), (left + label_size[0] + 10, top), box_color, cv2.FILLED)
        
        # Label text
        cv2.putText(frame, name, (left + 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Write frame to output
    out.write(frame)
    
    # Log frame processing time
    frame_end_time = time.time()
    perf_monitor.log_frame_time(frame_end_time - frame_start_time)

# Stop monitoring and generate report
performance_report = perf_monitor.stop_monitoring()

# Clean up
video_capture.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete (No Tracking Method).")

# Print detailed performance report
perf_monitor.print_report(performance_report)

# Save performance report to file
report_file = os.path.join(video_folder, 'performance_report_no_tracking.txt')
with open(report_file, 'w') as f:
    f.write("DEEPFACE ONLY (NO TRACKING) - PERFORMANCE REPORT\n")
    f.write("=" * 55 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("SUMMARY:\n")
    f.write(f"Total Processing Time: {performance_report['total_time']:.2f} seconds\n")
    f.write(f"Frames Processed: {performance_report['processed_frames']}\n")
    f.write(f"Average Processing Speed: {performance_report['fps_processed']:.2f} FPS\n\n")
    
    f.write("RESOURCE USAGE:\n")
    f.write(f"Peak Memory Usage: {performance_report['peak_memory_mb']:.2f} MB\n")
    f.write(f"Memory Increase: {performance_report['memory_increase_mb']:.2f} MB\n")
    f.write(f"Average CPU Usage: {performance_report['avg_cpu_percent']:.1f}%\n")
    f.write(f"Peak CPU Usage: {performance_report['max_cpu_percent']:.1f}%\n\n")
    
    f.write("TIMING BREAKDOWN:\n")
    f.write(f"Average Frame Processing: {performance_report['avg_frame_time_ms']:.2f} ms\n")
    f.write(f"Average Face Detection: {performance_report['avg_detection_time_ms']:.2f} ms\n")
    f.write(f"Average Face Verification: {performance_report['avg_verification_time_ms']:.2f} ms\n\n")
    
    f.write("OPERATION COUNTS:\n")
    f.write(f"Total Detection Operations: {performance_report['total_detections']}\n")
    f.write(f"Total Verification Operations: {performance_report['total_verifications']}\n\n")
    
    f.write("METHOD: DeepFace Only (No Tracking)\n")
    f.write("DETECTION INTERVAL: Every 5 frames\n")
    f.write("FACE PERSISTENCE: Uses last known face location between detections\n")

print(f"\nDetailed performance report saved to: {report_file}")

# Print comparison summary
print("\n" + "=" * 50)
print("           QUICK COMPARISON METRICS")
print("=" * 50)
print(f"Method: DeepFace Only (No Tracking)")
print(f"Total Time: {performance_report['total_time']:.2f}s")
print(f"Processing FPS: {performance_report['fps_processed']:.2f}")
print(f"Memory Usage: {performance_report['peak_memory_mb']:.1f} MB")
print(f"Avg CPU: {performance_report['avg_cpu_percent']:.1f}%")
print(f"Detections: {performance_report['total_detections']}")
print(f"Verifications: {performance_report['total_verifications']}")
print("=" * 50)