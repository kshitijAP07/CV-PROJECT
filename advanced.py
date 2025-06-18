import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from time import time
import os
from sort import Sort
import datetime

class BottleQCSystem:
    def __init__(self, args):
        # Configuration
        self.args = args
        
        # Define a professional color palette
        self.theme = {
            'primary': (65, 105, 225),    # Royal Blue
            'success': (40, 167, 69),     # Green
            'danger': (220, 53, 69),      # Red
            'warning': (255, 193, 7),     # Yellow
            'info': (23, 162, 184),       # Cyan
            'light': (248, 249, 250),     # Light Gray
            'dark': (52, 58, 64),         # Dark Gray
            'white': (255, 255, 255),     # White
            'black': (0, 0, 0),           # Black
            'transparent_bg': (0, 0, 0, 180)  # Semi-transparent black
        }
        
        # Class colors with improved visibility
        self.class_colors = {
            'bottle': self.theme['primary'],     # Royal Blue
            'cap': self.theme['info'],           # Cyan
            'label': self.theme['success'],      # Green
            'missing_label': self.theme['danger'],  # Red
            'damaged_label': self.theme['warning']  # Yellow
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
        
        # UI configuration
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_large = 0.8
        self.font_scale_small = 0.65
        self.font_thickness = 2
        self.thin_line = 1
        self.thick_line = 2
        self.ui_margin = 20
        self.ui_padding = 10
        
        # Session data
        self.session_start_time = time()
        self.log_data = []
    
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'{output_dir}/bottle_qc_{timestamp}.mp4'
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
        has_cap = False
        quality_issues = []
        
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
        
        # Check for cap
        for cap_det in all_detections['cap']:
            if has_overlap(cap_det):
                has_cap = True
                break
        
        # Check for quality issues
        if not has_label:
            quality_issues.append("No Label")
        
        if not has_cap:
            quality_issues.append("No Cap")
        
        # Check for damaged or missing label specifically detected
        for issue_type in ['missing_label', 'damaged_label']:
            for issue_det in all_detections[issue_type]:
                if has_overlap(issue_det):
                    nice_name = "Damaged Label" if issue_type == "damaged_label" else "Missing Label"
                    quality_issues.append(nice_name)
                    break
        
        # Determine bottle status
        if quality_issues:
            return "FAIL", self.theme['danger'], quality_issues
        else:
            return "PASS", self.theme['success'], []
    
    def draw_semi_transparent_rect(self, frame, x1, y1, x2, y2, color, alpha=0.7):
        """Draw a semi-transparent rectangle overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def draw_ui_panel(self, frame):
        """Draw main control panel UI"""
        # Panel dimensions
        panel_width = 280
        panel_height = 200
        margin = self.ui_margin
        
        # Draw semi-transparent panel background
        self.draw_semi_transparent_rect(frame, margin, margin, 
                                       margin + panel_width, margin + panel_height, 
                                       self.theme['black'], 0.7)
        
        # Add panel border
        cv2.rectangle(frame, (margin, margin), (margin + panel_width, margin + panel_height), 
                     self.theme['light'], 1)
        
        # Add title bar
        title_height = 40
        cv2.rectangle(frame, (margin, margin), 
                     (margin + panel_width, margin + title_height), 
                     self.theme['primary'], -1)
        
        # Title text
        cv2.putText(frame, "BOTTLE QC SYSTEM", 
                   (margin + 10, margin + 28), 
                   self.font, self.font_scale_large, self.theme['white'], self.font_thickness)
        
        # Stats content
        y_offset = margin + title_height + 20
        
        # FPS indicator with color coding
        fps_color = self.theme['success'] if self.processing_fps > 15 else \
                   (self.theme['warning'] if self.processing_fps > 5 else self.theme['danger'])
                   
        cv2.putText(frame, f"FPS: {self.processing_fps:.1f}", 
                   (margin + 10, y_offset), 
                   self.font, self.font_scale_small, fps_color, self.font_thickness)
        y_offset += 30
        
        # Display counts
        total_bottles = len(self.passed_bottles) + len(self.failed_bottles)
        cv2.putText(frame, f"Total Bottles: {total_bottles}", 
                   (margin + 10, y_offset), 
                   self.font, self.font_scale_small, self.theme['light'], self.font_thickness)
        y_offset += 30
        
        # Passed bottles counter
        cv2.putText(frame, f"PASSED: {len(self.passed_bottles)}", 
                   (margin + 10, y_offset), 
                   self.font, self.font_scale_small, self.theme['success'], self.font_thickness)
        y_offset += 30
        
        # Failed bottles counter
        cv2.putText(frame, f"FAILED: {len(self.failed_bottles)}", 
                   (margin + 10, y_offset), 
                   self.font, self.font_scale_small, self.theme['danger'], self.font_thickness)
        y_offset += 30
        
        # Current frame info
        frame_text = f"Frame: {self.frame_count}/{self.total_frames}"
        cv2.putText(frame, frame_text, 
                   (margin + 10, y_offset), 
                   self.font, self.font_scale_small, self.theme['light'], self.font_thickness)
    
    def draw_help_text(self, frame):
        """Draw keyboard controls help"""
        help_text = "Press: [Q] Quit  [P] Pause  [.] Next Frame  [,] Prev Frame"
        text_size = cv2.getTextSize(help_text, self.font, self.font_scale_small, self.font_thickness)[0]
        
        # Draw semi-transparent background
        bg_margin = 10
        bg_x = (self.width - text_size[0]) // 2 - bg_margin
        bg_y = self.height - 40
        bg_width = text_size[0] + bg_margin * 2
        bg_height = text_size[1] + bg_margin * 2
        
        self.draw_semi_transparent_rect(frame, bg_x, bg_y, 
                                      bg_x + bg_width, bg_y + bg_height, 
                                      self.theme['black'], 0.7)
        
        # Draw text
        text_x = (self.width - text_size[0]) // 2
        text_y = self.height - 20
        cv2.putText(frame, help_text, (text_x, text_y), 
                   self.font, self.font_scale_small, self.theme['light'], self.font_thickness)
    
    def draw_progress_bar(self, frame):
        """Draw video progress bar"""
        bar_height = 8
        bar_y = self.height - bar_height - 5
        
        # Draw background bar
        cv2.rectangle(frame, (5, bar_y), (self.width - 5, bar_y + bar_height), 
                     self.theme['dark'], -1)
        
        # Draw progress
        if self.total_frames > 0:
            progress_width = int((self.width - 10) * (self.frame_count / self.total_frames))
            cv2.rectangle(frame, (5, bar_y), (5 + progress_width, bar_y + bar_height), 
                         self.theme['primary'], -1)
    
    def draw_counting_line(self, frame, highlight=False):
        """Draw the bottle counting line"""
        color = self.theme['danger'] if highlight else self.theme['warning']
        thickness = 3 if highlight else 2
        
        # Draw dashed line
        dash_length = 20
        gap_length = 10
        y = 0
        
        while y < self.height:
            y_end = min(y + dash_length, self.height)
            cv2.line(frame, (self.line[0], y), (self.line[0], y_end), color, thickness)
            y = y_end + gap_length
        
        # Draw line label
        label = "COUNTING LINE"
        text_size = cv2.getTextSize(label, self.font, self.font_scale_small, self.font_thickness)[0]
        
        # Background for label
        bg_x = self.line[0] - text_size[0] // 2 - 5
        bg_y = 10
        bg_width = text_size[0] + 10
        bg_height = text_size[1] + 10
        
        self.draw_semi_transparent_rect(frame, bg_x, bg_y, 
                                      bg_x + bg_width, bg_y + bg_height, 
                                      self.theme['black'], 0.7)
        
        # Draw text
        cv2.putText(frame, label, 
                   (self.line[0] - text_size[0] // 2, 30), 
                   self.font, self.font_scale_small, color, self.font_thickness)
    
    def draw_bottle_info(self, frame, x1, y1, x2, y2, obj_id, status, status_color, quality_issues):
        """Draw professional information overlay for tracked bottles"""
        # Draw bounding box with varying thickness based on status
        thickness = 3 if status == "FAIL" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, thickness)
        
        # Calculate center point
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        
        # Draw center crosshair
        crosshair_size = 5
        cv2.line(frame, (cx - crosshair_size, cy), (cx + crosshair_size, cy), self.theme['warning'], 1)
        cv2.line(frame, (cx, cy - crosshair_size), (cx, cy + crosshair_size), self.theme['warning'], 1)
        
        # Prepare label text
        label_text = f"#{obj_id}: {status}"
        
        # Draw label background
        text_size = cv2.getTextSize(label_text, self.font, self.font_scale_small, self.font_thickness)[0]
        
        # Position label above bottle
        label_x = x1
        label_y = y1 - text_size[1] - 8
        
        # Ensure label is within frame
        if label_y < 5:
            # If not enough space above, put it below
            label_y = y2 + text_size[1] + 5
        
        # Draw semi-transparent background for the label
        self.draw_semi_transparent_rect(frame, 
                                      label_x - 2, 
                                      label_y - text_size[1] - 2, 
                                      label_x + text_size[0] + 10, 
                                      label_y + 5, 
                                      self.theme['black'], 0.7)
        
        # Draw status text
        cv2.putText(frame, label_text, 
                   (label_x + 3, label_y), 
                   self.font, self.font_scale_small, status_color, self.font_thickness)
        
        # If there are quality issues, display them
        if quality_issues:
            issue_text = ", ".join(quality_issues)
            issue_size = cv2.getTextSize(issue_text, self.font, 0.5, 1)[0]
            
            # Position issues text below bottle
            issue_x = max(5, x1 - (issue_size[0] - (x2 - x1)) // 2)
            issue_y = y2 + 15
            
            # Keep within frame bounds
            if issue_x + issue_size[0] > self.width - 5:
                issue_x = self.width - issue_size[0] - 5
            
            # Draw semi-transparent background
            self.draw_semi_transparent_rect(frame, 
                                          issue_x - 2, 
                                          issue_y - issue_size[1] - 2, 
                                          issue_x + issue_size[0] + 2, 
                                          issue_y + 2, 
                                          self.theme['black'], 0.7)
            
            # Draw issues text
            cv2.putText(frame, issue_text, 
                       (issue_x, issue_y), 
                       self.font, 0.5, self.theme['light'], 1)
    
    def draw_ui(self, frame, stats):
        """Draw all UI elements on the frame"""
        # Add a timestamp
        current_time = time() - self.session_start_time
        timestamp = f"Time: {int(current_time // 60):02d}:{int(current_time % 60):02d}"
        
        cv2.putText(frame, timestamp, 
                   (self.width - 150, 30), 
                   self.font, self.font_scale_small, self.theme['light'], self.font_thickness)
        
        # Draw main UI panel
        self.draw_ui_panel(frame)
        
        # Draw counting line
        self.draw_counting_line(frame)
        
        # Draw progress bar
        self.draw_progress_bar(frame)
        
        # Draw help text
        self.draw_help_text(frame)
        
        # Add watermark
        watermark = "Bottle QC v2.0"
        cv2.putText(frame, watermark, 
                   (self.width - 120, self.height - 10), 
                   self.font, 0.5, self.theme['light'], 1)
        
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
        
        # Draw object detections (label, cap, etc)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                conf = float(box.conf[0])
                
                # Skip bottle class as we handle it separately with tracker
                if class_name == 'bottle':
                    continue
                
                # Draw detection with appropriate color
                color = self.class_colors.get(class_name, self.theme['light'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                
                # Add small label for non-bottle objects
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Stats for this frame
        stats = {
            'bottles_in_frame': 0,
            'passed_in_frame': 0,
            'failed_in_frame': 0
        }
        
        # Flag to highlight counting line if crossed
        highlight_line = False
        
        # Process tracked bottles
        for tracked_obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, tracked_obj)
            obj_id_str = str(obj_id)
            
            # Count bottles
            stats['bottles_in_frame'] += 1
            
            # Calculate bottle center
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            
            # Check quality
            status, status_color, quality_issues = self.check_quality([x1, y1, x2, y2], all_detections)
            
            # Update bottle status history
            if obj_id_str not in self.bottle_status:
                self.bottle_status[obj_id_str] = status
            
            # Update frame stats
            if status == "PASS":
                stats['passed_in_frame'] += 1
            else:
                stats['failed_in_frame'] += 1
            
            # Draw detailed bottle info
            self.draw_bottle_info(annotated_frame, x1, y1, x2, y2, obj_id, status, status_color, quality_issues)
            
            # Check if object crosses the line
            if self.line[0] - 10 < cx < self.line[0] + 10:
                highlight_line = True
                
                # Count if not already counted
                if status == "PASS" and obj_id not in self.passed_bottles and obj_id not in self.failed_bottles:
                    self.passed_bottles.append(obj_id)
                    self.log_data.append({
                        'frame': self.frame_count,
                        'bottle_id': obj_id,
                        'status': 'PASS',
                        'time': time() - self.session_start_time
                    })
                    
                elif status == "FAIL" and obj_id not in self.passed_bottles and obj_id not in self.failed_bottles:
                    self.failed_bottles.append(obj_id)
                    self.log_data.append({
                        'frame': self.frame_count,
                        'bottle_id': obj_id,
                        'status': 'FAIL',
                        'issues': quality_issues,
                        'time': time() - self.session_start_time
                    })
        
        # Draw counting line (highlight if crossed)
        self.draw_counting_line(annotated_frame, highlight_line)
        
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
        
        # Create window with name
        window_name = 'Bottle Quality Control System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size
        if self.width > 1280 or self.height > 720:
            # Scale down larger videos
            window_width = min(self.width, 1280)
            window_height = min(self.height, 720)
            cv2.resizeWindow(window_name, window_width, window_height)
        
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
                cv2.imshow(window_name, annotated_frame)
                
                # Save frame if requested
                if self.out is not None:
                    self.out.write(annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                status_text = "PAUSED" if paused else "RUNNING"
                print(f"Video {status_text}")
                
                # Show pause indicator on frame
                if paused:
                    pause_overlay = annotated_frame.copy()
                    # Draw semi-transparent overlay
                    self.draw_semi_transparent_rect(pause_overlay, 0, 0, self.width, self.height, 
                                                  self.theme['black'], 0.3)
                    
                    # Draw pause text
                    text = "PAUSED"
                    text_size = cv2.getTextSize(text, self.font, 2.0, 3)[0]
                    text_x = (self.width - text_size[0]) // 2
                    text_y = (self.height + text_size[1]) // 2
                    
                    cv2.putText(pause_overlay, text, (text_x, text_y), 
                               self.font, 2.0, self.theme['white'], 3)
                    
                    cv2.imshow(window_name, pause_overlay)
                
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
        """Print final detection summary with enhanced formatting"""
        total = len(self.passed_bottles) + len(self.failed_bottles)
        if total == 0:
            pass_rate = fail_rate = 0
        else:
            pass_rate = len(self.passed_bottles) / total * 100
            fail_rate = len(self.failed_bottles) / total * 100
        
        elapsed_time = time() - self.session_start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
            
        print("\n" + "="*50)
        print(f"{'BOTTLE QUALITY CONTROL SUMMARY':^50}")
        print("="*50)
        print(f"{'Total bottles processed:':<30} {total:>20}")
        print(f"{'Passed bottles:':<30} {len(self.passed_bottles):>20} ({pass_rate:.1f}%)")
        print(f"{'Failed bottles:':<30} {len(self.failed_bottles):>20} ({fail_rate:.1f}%)")
        print(f"{'Processing time:':<30} {minutes:>17} min {seconds:>2} sec")
        print(f"{'Frames processed:':<30} {self.frame_count:>20}")
        print("="*50)
        
        # Display top issues if any failures
        if self.log_data and len(self.failed_bottles) > 0:
            issues_count = {}
            for entry in self.log_data:
                if 'issues' in entry:
                    for issue in entry['issues']:
                        if issue in issues_count:
                            issues_count[issue] += 1
                        else:
                            issues_count[issue] = 1
            
            if issues_count:
                print("\nCommon Quality Issues:")
                for issue, count in sorted(issues_count.items(), key=lambda x: x[1], reverse=True):
                    percent = (count / len(self.failed_bottles)) * 100
                    print(f"  {issue:<20}: {count:>3} occurrences ({percent:.1f}%)")
        
        print("="*50 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Bottle Detection and Quality Control System')
    parser.add_argument('--video', type=str, default='closeshots_final.mp4', help='Path to the video file')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--line-position', type=float, default=0.5, help='Position of counting line (0-1 as fraction of width)')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--skip-frames', type=int, default=0, help='Skip frames for faster processing (0 = no skipping)')
    return parser.parse_args()


def main():
    # Display startup banner
    print("\n" + "="*60)
    print(f"{'BOTTLE QUALITY CONTROL SYSTEM':^60}")
    print(f"{'v2.0 Professional Edition':^60}")
    print("="*60)
    
    # Parse arguments
    args = parse_args()
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Video source: {args.video}")
    print(f"  Model: {args.model}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Counting line position: {args.line_position * 100:.0f}% of width")
    print(f"  Frame skipping: {args.skip_frames}")
    print(f"  Save output: {'Yes' if args.save else 'No'}")
    print("\n" + "-"*60)
    
    try:
        # Initialize and run the system
        system = BottleQCSystem(args)
        system.run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\nBottle QC System terminated")


class BottleQCLogger:
    """Optional class for logging QC results to file"""
    def __init__(self, log_file=None):
        self.log_file = log_file or f"bottle_qc_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("timestamp,bottle_id,status,issues\n")
    
    def log_bottle(self, bottle_id, status, issues=None):
        """Log a bottle inspection result"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        issues_str = "|".join(issues) if issues else ""
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{bottle_id},{status},{issues_str}\n")
    
    def log_system_event(self, event_type, message):
        """Log system events"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},SYSTEM,{event_type},{message}\n")


if __name__ == "__main__":
    main()