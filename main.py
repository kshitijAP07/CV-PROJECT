import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from time import time
import os
from sort import Sort

class BottleQCSystem:
    def __init__(self, args):
        # Configuration
        self.args = args
        self.class_colors = {
            'bottle': (0, 255, 0),     # Green
            'cap': (255, 0, 0),        # Blue
            'label': (0, 255, 255),    # Yellow
            'missing_label': (0, 0, 255),  # Red
            'damaged_label': (255, 0, 255)  # Magenta
        }
        
        # Initialize model
        print(f"Loading YOLOv8 model from {args.model}...")
        self.model = YOLO(args.model)
        
        # Initialize tracker
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
        # Initialize counters
        self.passed_bottles = []
        self.failed_bottles = []
        self.bottle_status = {}
        
        # Performance tracking
        self.prev_time = time()
        self.processing_fps = 0
        self.frame_count = 0
        
    def open_video(self):
        """Set up video capture and writer"""
        print(f"Opening video file: {self.args.video}")
        self.cap = cv2.VideoCapture(self.args.video)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.args.video}")
            return False
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define counting line
        self.line_x = int(self.width * self.args.line_position)
        self.line = [self.line_x, 0, self.line_x, self.height]
        
        print(f"Video info: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Set up video writer if saving
        self.out = None
        if self.args.save:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f'{output_dir}/bottle_qc_{time()}.mp4'
            self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
            print(f"Saving output to {output_path}")
            
        return True
    
    def detect_objects(self, frame):
        """Detect objects using YOLOv8"""
        results = self.model(frame, conf=self.args.conf)
        
        # Prepare detections for SORT tracker
        detections = np.empty((0, 5))
        
        # Store all detections by class for quality inspection
        all_detections = {
            'bottle': [],
            'cap': [],
            'label': [],
            'missing_label': [],
            'damaged_label': []
        }
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract coordinates, confidence and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Store all detections by class
                if class_name in all_detections:
                    all_detections[class_name].append((x1, y1, x2, y2, confidence))
                
                # Only track bottles
                if class_name == 'bottle':
                    current_detection = np.array([x1, y1, x2, y2, confidence])
                    detections = np.vstack((detections, current_detection))
        
        return detections, all_detections, results
    
    def check_quality(self, bottle_coords, all_detections):
        """Check bottle quality based on detected features"""
        x1, y1, x2, y2 = bottle_coords[:4]
        has_label = False
        has_quality_issue = False
        
        # Function to check if a detection overlaps with this bottle
        def has_overlap(det, iou_threshold=0.1):
            det_x1, det_y1, det_x2, det_y2, _ = det
            
            # Calculate intersection area
            inter_x1 = max(x1, det_x1)
            inter_y1 = max(y1, det_y1)
            inter_x2 = min(x2, det_x2)
            inter_y2 = min(y2, det_y2)
            
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                return False
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
            
            return inter_area / det_area > iou_threshold
        
        # Check for label
        for label_det in all_detections['label']:
            if has_overlap(label_det):
                has_label = True
                break
        
        # Check for quality issues (missing or damaged label)
        for issue_type in ['missing_label', 'damaged_label']:
            for issue_det in all_detections[issue_type]:
                if has_overlap(issue_det):
                    has_quality_issue = True
                    break
            if has_quality_issue:
                break
        
        # Determine bottle status
        if has_quality_issue or not has_label:
            return "FAIL", (0, 0, 255)  # Red for failed
        else:
            return "PASS", (0, 255, 0)  # Green for passed
    
    def draw_ui(self, frame, stats):
        """Draw UI elements on the frame"""
        # Draw counting line
        cv2.line(frame, (self.line[0], self.line[1]), (self.line[2], self.line[3]), 
                (0, 255, 255), 3)
        
        # Draw statistics panel
        cv2.rectangle(frame, (5, 5), (250, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (250, 160), (200, 200, 200), 1)
        
        # Display stats
        cv2.putText(frame, f"FPS: {self.processing_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Bottles: {stats['bottles_in_frame']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display counts in green/red
        cv2.putText(frame, f"PASSED: {len(self.passed_bottles)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FAILED: {len(self.failed_bottles)}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add progress bar
        progress = int(self.width * (self.frame_count / self.total_frames))
        cv2.rectangle(frame, (0, self.height - 20), (progress, self.height - 10), (0, 255, 0), -1)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Update performance metrics
        current_time = time()
        elapsed = current_time - self.prev_time
        self.prev_time = current_time
        self.processing_fps = 1.0 / elapsed if elapsed > 0 else 0
        
        # Create a copy for drawing
        annotated_frame = frame.copy()
        
        # Detect objects
        detections, all_detections, results = self.detect_objects(frame)
        
        # Draw object detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Draw detection with appropriate color
                color = self.class_colors.get(class_name, (255, 255, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Stats for this frame
        stats = {
            'bottles_in_frame': 0,
            'passed_in_frame': 0,
            'failed_in_frame': 0
        }
        
        # Process tracked bottles
        for tracked_obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, tracked_obj)
            obj_id_str = str(obj_id)
            
            # Count bottles
            stats['bottles_in_frame'] += 1
            
            # Calculate bottle center
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            
            # Check quality
            status, status_color = self.check_quality([x1, y1, x2, y2], all_detections)
            
            # Update bottle status history
            if obj_id_str not in self.bottle_status:
                self.bottle_status[obj_id_str] = status
            
            # Update frame stats
            if status == "PASS":
                stats['passed_in_frame'] += 1
            else:
                stats['failed_in_frame'] += 1
            
            # Draw tracking box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), status_color, 2)
            
            # Add minimal labeling
            label_text = f"#{obj_id}:{status}"
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Mark center point
            cv2.circle(annotated_frame, (cx, cy), 3, (0, 0, 255), -1)
            
            # Check if object crosses the line
            if self.line[0] - 10 < cx < self.line[0] + 10:
                # Highlight the line
                cv2.line(annotated_frame, (self.line[0], self.line[1]), 
                        (self.line[2], self.line[3]), (0, 0, 255), 5)
                
                # Count if not already counted
                if status == "PASS" and obj_id not in self.passed_bottles and obj_id not in self.failed_bottles:
                    self.passed_bottles.append(obj_id)
                elif status == "FAIL" and obj_id not in self.passed_bottles and obj_id not in self.failed_bottles:
                    self.failed_bottles.append(obj_id)
        
        # Draw UI elements
        annotated_frame = self.draw_ui(annotated_frame, stats)
        
        return annotated_frame
    
    def run(self):
        """Run the detection and tracking loop"""
        if not self.open_video():
            return
        
        print("Starting detection. Press 'q' to quit, 'p' to pause/resume.")
        
        paused = False
        skip_frames_counter = 0
        
        while self.cap.isOpened():
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                self.frame_count += 1
                
                # Skip frames if specified
                if self.args.skip_frames > 0:
                    skip_frames_counter += 1
                    if skip_frames_counter % (self.args.skip_frames + 1) != 0:
                        continue
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Show frame
                cv2.imshow('Bottle Quality Control System', annotated_frame)
                
                # Save frame if requested
                if self.out is not None:
                    self.out.write(annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Video " + ("paused" if paused else "resumed"))
            elif key == ord(',') and paused:  # Step backward when paused
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 2))
                self.frame_count = max(0, self.frame_count - 2)
            elif key == ord('.') and paused:  # Step forward when paused
                pass  # The next iteration will naturally move forward one frame
        
        # Clean up
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        
        # Print final report
        self.print_summary()
    
    def print_summary(self):
        """Print final detection summary"""
        total = len(self.passed_bottles) + len(self.failed_bottles)
        if total == 0:
            pass_rate = fail_rate = 0
        else:
            pass_rate = len(self.passed_bottles) / total * 100
            fail_rate = len(self.failed_bottles) / total * 100
            
        print("\n===== BOTTLE QUALITY CONTROL SUMMARY =====")
        print(f"Total bottles detected: {total}")
        print(f"Passed bottles: {len(self.passed_bottles)} ({pass_rate:.1f}%)")
        print(f"Failed bottles: {len(self.failed_bottles)} ({fail_rate:.1f}%)")
        print("==========================================")


def parse_args():
    parser = argparse.ArgumentParser(description='Bottle Detection and Quality Control System')
    parser.add_argument('--video', type=str, default='closeshots_final.mp4', help='Path to the video file')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--line-position', type=float, default=0.5, help='Position of counting line (0-1 as fraction of width)')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--skip-frames', type=int, default=0, help='Skip frames for faster processing (0 = no skipping)')
    return parser.parse_args()


def main():
    args = parse_args()
    system = BottleQCSystem(args)
    system.run()


if __name__ == "__main__":
    main()