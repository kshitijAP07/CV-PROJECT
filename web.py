import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from time import time
import os
from sort import Sort
import streamlit as st
from PIL import Image
import tempfile
import io
import pandas as pd

class BottleQCSystem:
    def __init__(self, config):
        # Configuration
        self.config = config
        self.class_colors = {
            'bottle': (0, 255, 0),     # Green
            'cap': (255, 0, 0),        # Blue
            'label': (0, 255, 255),    # Yellow
            'missing_label': (0, 0, 255),  # Red
            'damaged_label': (255, 0, 255)  # Magenta
        }
        
        # Initialize model
        self.model = YOLO(config['model_path'])
        
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
        self.total_frames = 0
        
        # Video properties
        self.width = 0
        self.height = 0
        self.fps = 0
        self.line_x = 0
        self.line = [0, 0, 0, 0]
        
    def setup_video(self, video_file):
        """Set up video capture from file"""
        temp_file = None
        if video_file is not None:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(video_file.read())
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Open the video file
            self.cap = cv2.VideoCapture(temp_file_path)
            
            if not self.cap.isOpened():
                st.error(f"Error: Could not open video file")
                return False
            
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Define counting line
            self.line_x = int(self.width * self.config['line_position'])
            self.line = [self.line_x, 0, self.line_x, self.height]
            
            st.sidebar.text(f"Video info: {self.width}x{self.height}")
            st.sidebar.text(f"FPS: {self.fps}")
            st.sidebar.text(f"Total frames: {self.total_frames}")
            
            return True
        return False
    
    def detect_objects(self, frame):
        """Detect objects using YOLOv8"""
        results = self.model(frame, conf=self.config['conf_threshold'])
        
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
        
        return annotated_frame, stats
    
    def get_summary(self):
        """Get detection summary"""
        total = len(self.passed_bottles) + len(self.failed_bottles)
        if total == 0:
            pass_rate = fail_rate = 0
        else:
            pass_rate = len(self.passed_bottles) / total * 100
            fail_rate = len(self.failed_bottles) / total * 100
            
        return {
            "total": total,
            "passed": len(self.passed_bottles),
            "failed": len(self.failed_bottles),
            "pass_rate": pass_rate,
            "fail_rate": fail_rate
        }
    
    def run_analysis(self, video_file, progress_bar, frame_placeholder, stats_container):
        """Run analysis on video file with UI updates"""
        if not self.setup_video(video_file):
            return
        
        # Reset counters
        self.passed_bottles = []
        self.failed_bottles = []
        self.bottle_status = {}
        self.frame_count = 0
        
        # Set up columns for stats
        stats_cols = stats_container.columns(5)
        total_metric = stats_cols[0].empty()
        passed_metric = stats_cols[1].empty()
        failed_metric = stats_cols[2].empty()
        pass_rate_metric = stats_cols[3].empty()
        fail_rate_metric = stats_cols[4].empty()
        
        # Streamlit components for displaying results
        frame_skip = self.config.get('skip_frames', 0)
        skip_frames_counter = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Update progress bar
            progress_bar.progress(min(1.0, self.frame_count / self.total_frames))
            
            # Skip frames if specified
            skip_frames_counter += 1
            if frame_skip > 0 and skip_frames_counter % (frame_skip + 1) != 0:
                continue
            
            # Process frame
            annotated_frame, stats = self.process_frame(frame)
            
            # Convert to RGB for Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update frame display
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
            
            # Update stats
            summary = self.get_summary()
            total_metric.metric("Total Bottles", summary["total"])
            passed_metric.metric("Passed", summary["passed"])
            failed_metric.metric("Failed", summary["failed"])
            pass_rate_metric.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")
            fail_rate_metric.metric("Fail Rate", f"{summary['fail_rate']:.1f}%")
        
        # Clean up
        self.cap.release()
        
        # Return the final summary
        return self.get_summary()


def main():
    st.set_page_config(
        page_title="Bottle Quality Control System",
        page_icon="🍾",
        layout="wide",
    )
    
    st.title("Bottle Quality Control System")
    st.write("Upload a video file for bottle quality inspection")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    model_path = st.sidebar.text_input("Model Path", value="best.pt")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    line_position = st.sidebar.slider("Counting Line Position", 0.0, 1.0, 0.5, 0.05)
    skip_frames = st.sidebar.slider("Skip Frames", 0, 10, 0, 1)
    
    # Configuration dictionary
    config = {
        "model_path": model_path,
        "conf_threshold": conf_threshold,
        "line_position": line_position,
        "skip_frames": skip_frames
    }
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Initialize system
        system = BottleQCSystem(config)
        
        # Set up layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Video Analysis")
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        with col2:
            st.subheader("Real-time Statistics")
            stats_container = st.container()
            
            # Add run button
            if st.button("Run Analysis"):
                with st.spinner("Processing video..."):
                    summary = system.run_analysis(uploaded_file, progress_bar, frame_placeholder, stats_container)
                
                # Final summary
                st.subheader("Analysis Complete")
                st.success(f"Analysis complete! {summary['total']} bottles processed.")
                
                # Show detailed results
                st.write("### Results Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Bottles", summary["total"])
                    st.metric("Passed Bottles", summary["passed"])
                    st.metric("Failed Bottles", summary["failed"])
                
                with col2:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")
                    st.metric("Fail Rate", f"{summary['fail_rate']:.1f}%")
                
                # Create visualization of results
                if summary["total"] > 0:
                    # Create dataframe for chart
                    chart_data = pd.DataFrame({
                        "Status": ["Passed", "Failed"],
                        "Count": [summary["passed"], summary["failed"]]
                    })
                    st.subheader("Results Visualization")
                    st.bar_chart(chart_data.set_index("Status"))


if __name__ == "__main__":
    main()