import cv2
from deepface import DeepFace
import os
import numpy as np
from collections import OrderedDict
import time
import psutil
import threading
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
        self.tracking_times = []
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
        
        print(f"=== Performance Monitoring Started ===")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Memory Usage: {self.initial_memory:.2f} MB")
        print(f"CPU Cores Available: {psutil.cpu_count()}")
        print("=" * 45)
    
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
    
    def log_tracking_time(self, tracking_time):
        """Log time taken for tracking update"""
        self.tracking_times.append(tracking_time)
    
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
        avg_tracking_time = np.mean(self.tracking_times) if self.tracking_times else 0
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
            'avg_tracking_time_ms': avg_tracking_time * 1000,
            'avg_verification_time_ms': avg_verification_time * 1000,
            'total_detections': len(self.detection_times),
            'total_verifications': len(self.verification_times)
        }
        
        return report
    
    def print_report(self, report):
        """Print formatted performance report"""
        print("\n" + "=" * 60)
        print("                  PERFORMANCE REPORT")
        print("=" * 60)
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
        print(f"  Tracking Update: {report['avg_tracking_time_ms']:.2f} ms")
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
        
        print("=" * 60)
class DeepSORTTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, name):
        """Register a new object with given centroid"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'name': name
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracking with new detections
        detections: list of {'bbox': (x,y,w,h), 'name': str}
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Initialize arrays of input centroids for current frame
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        
        for (i, detection) in enumerate(detections):
            x, y, w, h = detection['bbox']
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)

        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for i, detection in enumerate(detections):
                centroid = input_centroids[i]
                object_id = self.register(centroid, detection['bbox'], detection['name'])

        # Otherwise, try to match existing objects to new detections
        else:
            # Get centroids of existing objects
            object_centroids = [obj['centroid'] for obj in self.objects.values()]

            # Compute distance matrix between existing and new centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Keep track of used row and column indices
            used_rows = set()
            used_cols = set()

            # Loop over (row, col) index tuples
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                # Update existing object
                object_id = list(self.objects.keys())[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = detections[col]['bbox']
                # Keep the original name unless it's unknown and new detection has a name
                if self.objects[object_id]['name'] == 'Unknown' and detections[col]['name'] != 'Unknown':
                    self.objects[object_id]['name'] = detections[col]['name']
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched detections and existing objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # If more existing objects than detections, mark as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # If more detections than existing objects, register new ones
            else:
                for col in unused_cols:
                    centroid = input_centroids[col]
                    self.register(centroid, detections[col]['bbox'], detections[col]['name'])

        return self.objects

# --- Face Processing Functions ---
def extract_faces_with_locations(frame):
    """Extract faces and return both crops and bounding boxes"""
    try:
        # Use DeepFace to detect faces
        detections = DeepFace.extract_faces(
            frame, 
            detector_backend='retinaface', 
            enforce_detection=False
        )
        
        faces_data = []
        print(f"DeepFace found {len(detections)} detections")
        
        if detections:
            for i, detection in enumerate(detections):
                region = detection['facial_area']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                print(f"Detection {i}: Raw coordinates x={x}, y={y}, w={w}, h={h}")
                
                # Convert to integers first
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Validate and fix coordinates
                x = max(0, min(x, frame.shape[1] - 10))
                y = max(0, min(y, frame.shape[0] - 10))
                w = max(10, min(w, frame.shape[1] - x))
                h = max(10, min(h, frame.shape[0] - y))
                
                print(f"Detection {i}: Corrected coordinates x={x}, y={y}, w={w}, h={h}")
                
                # Additional validation - ensure reasonable face size
                if w >= 30 and h >= 30 and w <= frame.shape[1] and h <= frame.shape[0]:
                    try:
                        face_crop = frame[y:y+h, x:x+w]
                        if face_crop.size > 0:
                            faces_data.append({
                                'crop': face_crop,
                                'bbox': (x, y, w, h)
                            })
                            print(f"Detection {i}: Successfully added face crop of size {face_crop.shape}")
                        else:
                            print(f"Detection {i}: Face crop is empty")
                    except Exception as crop_error:
                        print(f"Detection {i}: Failed to crop face: {crop_error}")
                else:
                    print(f"Detection {i}: Invalid dimensions, skipping")
        
        print(f"Returning {len(faces_data)} valid faces")
        return faces_data
    except Exception as e:
        print(f"Face extraction failed: {e}")
        return []

def verify_face(face_crop, reference_path):
    """Verify if face matches reference image"""
    try:
        if face_crop.size == 0:
            return False
        
        result = DeepFace.verify(
            face_crop, 
            reference_path, 
            model_name='ArcFace', 
            enforce_detection=False
        )
        return result.get('verified', False)
    except Exception as e:
        print(f"Face verification failed: {e}")
        return False

# --- Setup and Video Loading ---
script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
video_folder = os.path.join(script_dir, 'resources')
video_path = os.path.join(video_folder, 'mi_video2.mp4')
tom_cruise_image_path = os.path.join(video_folder, 'tom_cruise_ref.jpg')
output_video_path = os.path.join(video_folder, 'mi_video2_deepsort_performance.mp4')

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
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Get total frame count
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = "Unknown"

print(f"Video properties: Width={frame_width}, Height={frame_height}, FPS={fps}")
print(f"Total frames to process: {total_frames}")

# Initialize DeepSORT tracker and performance monitor
tracker = DeepSORTTracker(max_disappeared=10, max_distance=80)
perf_monitor = PerformanceMonitor()

# Start performance monitoring
perf_monitor.start_monitoring()

# --- Process Video Frame by Frame ---
frame_count = 0
detection_interval = 10  # Run detection every 10 frames for better tracking

print("Starting frame processing with DeepSORT tracking...")

while True:
    frame_start_time = time.time()
    
    ret, frame = video_capture.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processing frame {frame_count}/{total_frames} ({frame_count/int(total_frames)*100:.1f}%)")

    detections = []

    # Run face detection periodically
    if frame_count % detection_interval == 0 or frame_count == 1:
        detection_start_time = time.time()
        print(f"Running detection on frame {frame_count}")
        
        detected_faces = extract_faces_with_locations(frame)
        detection_end_time = time.time()
        perf_monitor.log_detection_time(detection_end_time - detection_start_time)
        
        for face_data in detected_faces:
            face_crop = face_data['crop']
            detected_bbox = face_data['bbox']
            
            # Verify identity
            verification_start_time = time.time()
            try:
                is_tom_cruise = verify_face(face_crop, tom_cruise_image_path)
                name = "Tom Cruise" if is_tom_cruise else "Unknown"
                print(f"Frame {frame_count}: Face verification result: {name}")
            except Exception as verify_error:
                print(f"Frame {frame_count}: Face verification failed: {verify_error}")
                name = "Unknown"
            
            verification_end_time = time.time()
            perf_monitor.log_verification_time(verification_end_time - verification_start_time)
            
            detections.append({
                'bbox': detected_bbox,
                'name': name
            })

    # Update tracker
    tracking_start_time = time.time()
    objects = tracker.update(detections)
    tracking_end_time = time.time()
    perf_monitor.log_tracking_time(tracking_end_time - tracking_start_time)
    
    # Draw tracking results
    for (object_id, obj) in objects.items():
        x, y, w, h = obj['bbox']
        name = obj['name']
        
        # Choose colors
        if name == "Tom Cruise":
            box_color = (0, 255, 0)  # Green
        else:
            box_color = (0, 0, 255)  # Red
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
        
        # Prepare label
        label = f"{name} ID:{object_id}"
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 30), (x + label_size[0] + 10, y), box_color, cv2.FILLED)
        
        # Label text
        cv2.putText(frame, label, (x + 5, y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Debug output
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Tracking {name} (ID: {object_id}) at ({x},{y},{w},{h})")

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

print("Video processing complete with DeepSORT tracking.")

# Print detailed performance report
perf_monitor.print_report(performance_report)

# Save performance report to file
report_file = os.path.join(video_folder, 'performance_report_deepsort.txt')
with open(report_file, 'w') as f:
    f.write("DEEPSORT FACE TRACKING - PERFORMANCE REPORT\n")
    f.write("=" * 50 + "\n")
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
    f.write(f"Average Face Verification: {performance_report['avg_verification_time_ms']:.2f} ms\n")
    f.write(f"Average Tracking Update: {performance_report['avg_tracking_time_ms']:.2f} ms\n\n")
    
    f.write("OPERATION COUNTS:\n")
    f.write(f"Total Detection Operations: {performance_report['total_detections']}\n")
    f.write(f"Total Verification Operations: {performance_report['total_verifications']}\n")

print(f"\nDetailed performance report saved to: {report_file}")