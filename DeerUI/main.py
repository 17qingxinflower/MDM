import tkinter as TT
from tkinter import filedialog, simpledialog, messagebox
from tkinter.ttk import Frame, Label
import cv2
import threading
import time
from queue import Queue, Empty
from collections import deque
from PIL import Image, ImageTk
from ultralytics import YOLO
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from datetime import datetime, timedelta
import matplotlib
import subprocess

# Global Configuration
max_width, max_height = 800, 700
action_stats_label = None
model_path = ""
video_source = 0
is_detecting = False
OUTPUT_DIR = os.path.join(os.getcwd(), "UI_result")  # Default to relative path
VIDEO_FILE_PATH = "" # Placeholder for video path

# Multi-video analysis variables
multi_video_windows = {}  # Dictionary to store multiple video windows
multi_processor_instances = {}  # Dictionary to store processor instances

# Performance optimization parameters
FRAME_QUEUE_SIZE = 2
PROCESSING_SIZE = (320, 240)
DISPLAY_FPS = 30
ACTION_HISTORY = 30
ACTION_THRESHOLD = 0.7  # Confidence threshold

# Sample Data (Static for demo)
temperature = "22°C"
humidity = "50%"
health_info = "Good"
deer_id = "001"

# Raw Class Mapping (Matches the Output IDs of your YOLO Model)
# Note: Keys must match exactly what your trained model outputs
class_mapping = {
    0: "hunt", 1: "egestion", 2: "Standing_feeding",
    3: "Stand_or_walk", 4: "Lie_down_and_rest",
    5: "Licking_the_pussy", 6: "Comb_and_lick"
}

# Display Class Mapping (Maps Model Output -> Professional English Display)
display_class_mapping = {
    "hunt": "Foraging",
    "egestion": "Excretion",
    "Standing_feeding": "Standing Feeding",
    "Stand_or_walk": "Walking/Standing",
    "Lie_down_and_rest": "Resting",
    "Licking_the_pussy": "Anogenital Licking", # Corrected to scientific term
    "Comb_and_lick": "Grooming"
}

def show_plot_in_main_thread(self, fig):
    """
    Safely display plots in the main thread
    """
    def show_plot():
        # Temporarily switch to an interactive backend to show the plot
        from matplotlib import get_backend
        current_backend = get_backend()
        
        if current_backend == 'Agg':
            import matplotlib.pyplot as plt
            plt.switch_backend('TkAgg') 
            fig.show()
            plt.switch_backend(current_backend) 
        else:
            fig.show()
    
    self.root.after(0, show_plot)

class VideoProcessor:
    def __init__(self, root, video_id="default", is_multi_video=False):
        self.root = root
        self.video_id = video_id
        self.is_multi_video = is_multi_video
        self.model = None
        self.cap = None
        self.class_mapping = class_mapping
        self.original_size = (640, 480)
        self.aspect_ratio = 640 / 480
        self.action_history = deque(maxlen=ACTION_HISTORY)
        self.current_action = None
        self.processing_size = PROCESSING_SIZE
        self.action_start_time = None
        self.action_end_time = None
        self.action_durations = []
        self.frame_count = 0
        self.last_results = None
        self.last_detection_time = 0
        self.excel_file_path = None
        self.action_durations_total = {}
        self.min_action_duration = 5  # Default min duration (seconds)
        self.max_transition_rate = 10  # Default max transition rate (times/10 min)
        
        # Threshold Settings
        self.min_duration_threshold_val = 5.0
        self.max_transition_rate_val = 10.0
        self.action_thresholds = {}  # Specific thresholds per action
        
        # Threading
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.display_queue = Queue(maxsize=1)
        self.is_detecting = False
        self.video_path = None
        self.video_source = 0
        
        # GUI Component References
        self.video_label = None
        self.action_label = None
        self.stats_label = None
        self.time_label = None
        self.fps_label = None
        self.start_stop_button = None

    def init_camera(self, source=0):
        """Initialize camera or video file"""
        self.video_source = source
        if self.cap is not None:
            self.cap.release()
        
        if isinstance(source, int):
            # Camera
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise IOError("Cannot open camera")
        else:
            # Video File
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise IOError("Cannot open video file")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.original_size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        if self.original_size[1] == 0:
            self.original_size = (640, 480)
        self.aspect_ratio = self.original_size[0] / max(self.original_size[1], 1)
        return True

    def load_model(self):
        global model_path
        if model_path:
            self.model = YOLO(model_path)
            self.model.fuse()
            self.model.eval()
            return True
        return False

    def update_action_list(self, display_action):
        if self.action_label:
            if display_action:
                self.action_label.config(text=f"Current Action:\n{display_action}", state="normal")
            else:
                self.action_label.config(text="", state="hidden")
        self.update_action_stats()

    def update_action_stats(self):
        if not self.stats_label:
            return
            
        action_durations_total = self.action_durations_total.copy()
        
        # Add current running action to stats
        if self.current_action and self.action_start_time:
            current_duration = time.time() - self.action_start_time
            display_action = display_class_mapping.get(self.current_action, "Unknown")
            if display_action in action_durations_total:
                action_durations_total[display_action] += current_duration
            else:
                action_durations_total[display_action] = current_duration
        
        total_duration = sum(action_durations_total.values())
        
        if total_duration > 0:
            stats_text = f"Video {self.video_id} Stats:\n"
            for action, duration in sorted(action_durations_total.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_duration) * 100
                stats_text += f"\n{action}: {duration:.1f}s ({percentage:.1f}%)\n"
        else:
            stats_text = f"Video {self.video_id} Stats:\nNo Data"
            
        self.stats_label.config(text=stats_text)

    def video_capture_thread_func(self):
        """Video Capture Thread"""
        frame_count = 0
        start_time = time.time()
        
        while self.is_detecting:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()

                if ret:
                    try:
                        self.frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    except:
                        continue
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 1:
                        fps = frame_count / elapsed_time
                        frame_count = 0
                        start_time = time.time()
                        if self.fps_label:
                            self.fps_label.config(text=f"FPS: {fps:.2f}")
                    try:
                        cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        if cap_fps > 0:
                            time.sleep(1 / cap_fps)
                        else:
                            time.sleep(1 / 30)
                    except Exception as e:
                        print(f"FPS Error: {str(e)}")
                        time.sleep(1 / 30)
                else:
                    # End of file
                    if isinstance(self.video_source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
                    else:
                        break
            else:
                time.sleep(0.1)
    
    def processing_thread_func(self):
        """Inference Processing Thread"""
        while self.is_detecting:
            try:
                frame = self.frame_queue.get(timeout=1)
                processed_frame = self.process_frame(frame)
                if self.display_queue.empty():
                    self.display_queue.put(processed_frame)
            except Empty:
                continue
    
    def process_frame(self, frame, is_video_file=False):
        if frame is None:
            return frame.copy()
        
        current_time = time.time()
        frame_out = frame.copy()
        
        # Detect every 5 frames
        if self.is_detecting and self.model and self.frame_count % 5 == 0:
            results = self.model(frame, verbose=False)
            self.last_results = results
            self.last_detection_time = current_time
            
            if results[0].boxes:
                highest_conf_box = max(results[0].boxes, key=lambda box: box.conf[0].item())
                cls = int(highest_conf_box.cls[0])
                conf = highest_conf_box.conf[0].item()
                
                highest_conf_action = self.class_mapping.get(cls, "Unknown")
                display_action = display_class_mapping.get(highest_conf_action, "Unknown")
                
                if self.current_action != highest_conf_action:
                    if self.current_action:
                        self.action_end_time = time.time()
                        duration = self.action_end_time - self.action_start_time
                        self.action_durations.append({
                            "action": self.current_action,
                            "start_time": self.action_start_time,
                            "end_time": self.action_end_time,
                            "duration": duration
                        })
                        # Update stats
                        display_prev_action = display_class_mapping.get(self.current_action, "Unknown")
                        if display_prev_action in self.action_durations_total:
                            self.action_durations_total[display_prev_action] += duration
                        else:
                            self.action_durations_total[display_prev_action] = duration
                    
                    self.current_action = highest_conf_action
                    self.action_start_time = current_time
                    self.update_action_list(display_action)
                    self.update_excel()
                else:
                    self.update_action_list(display_action)
                
                if conf >= ACTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, highest_conf_box.xyxy[0].tolist())
                    color = (0, 255, 0)
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                    # Display Label on Box (Using the raw model class or English display)
                    label = f"{display_action} {conf:.2f}" 
                    cv2.putText(frame_out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                self.current_action = None
                self.update_action_list(None)
        else:
            # Reuse last result between detections
            if self.last_results is not None and self.last_results[0].boxes:
                highest_conf_box = max(self.last_results[0].boxes, key=lambda box: box.conf[0].item())
                cls = int(highest_conf_box.cls[0])
                conf = highest_conf_box.conf[0].item()
                
                highest_conf_action = self.class_mapping.get(cls, "Unknown")
                display_action = display_class_mapping.get(highest_conf_action, "Unknown")
                
                if self.current_action != highest_conf_action:
                    if self.current_action:
                        self.action_end_time = time.time()
                        duration = self.action_end_time - self.action_start_time
                        self.action_durations.append({
                            "action": self.current_action,
                            "start_time": self.action_start_time,
                            "end_time": self.action_end_time,
                            "duration": duration
                        })
                        display_prev_action = display_class_mapping.get(self.current_action, "Unknown")
                        if display_prev_action in self.action_durations_total:
                            self.action_durations_total[display_prev_action] += duration
                        else:
                            self.action_durations_total[display_prev_action] = duration
                    
                    self.current_action = highest_conf_action
                    self.action_start_time = current_time
                    self.update_action_list(display_action)
                    self.update_excel()
                else:
                    self.update_action_list(display_action)
                
                if conf >= ACTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, highest_conf_box.xyxy[0].tolist())
                    color = (0, 255, 0)
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                    label = f"{display_action} {conf:.2f}"
                    cv2.putText(frame_out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                self.current_action = None
                self.update_action_list(None)

        self.frame_count += 1
        return frame_out
    
    def update_display(self):
        """Update GUI Video Label"""
        try:
            frame = self.display_queue.get_nowait()
            if self.video_label:
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()
                if label_width > 0 and label_height > 0:
                    ratio = min(label_width / self.original_size[0], 
                               label_height / self.original_size[1])
                    scaled_size = (int(self.original_size[0] * ratio), 
                                  int(self.original_size[1] * ratio))
                    img = Image.fromarray(frame).resize(scaled_size, Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
        except Empty:
            pass
        
        if self.time_label:
            current_time = time.strftime("%H:%M:%S")
            self.time_label.config(text=f"Time: {current_time}")
        
        if self.is_detecting and self.video_label:
            self.video_label.after(int(1000 / DISPLAY_FPS), self.update_display)
    
    def toggle_detection(self):
        """Start/Stop Detection"""
        if not self.is_detecting:
            # Start
            if not model_path:
                messagebox.showerror("Error", "Please select a model file first!")
                return
            
            try:
                if not self.load_model():
                    messagebox.showerror("Error", "Failed to load model!")
                    return
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
                return
            
            if self.cap is None or not self.cap.isOpened():
                try:
                    self.init_camera(0)
                except Exception as e:
                    messagebox.showerror("Camera Error", f"Cannot open camera, please select a video source: {str(e)}")
                    return
            
            self.is_detecting = True
            self.action_start_time = time.time()
            self.action_durations = []
            self.action_durations_total = {}
            self.update_excel()
            
            if self.start_stop_button:
                self.start_stop_button.config(text="Stop Detection")
            
            threading.Thread(target=self.video_capture_thread_func, daemon=True).start()
            threading.Thread(target=self.processing_thread_func, daemon=True).start()
            self.update_display()
            
            if self.stats_label:
                self.stats_label.config(text=f"Video {self.video_id} Stats:\nDetecting...")
        else:
            # Stop
            self.is_detecting = False
            self.last_results = None
            self.update_action_list(None)
            
            self.auto_save_and_filter()
            
            if self.start_stop_button:
                self.start_stop_button.config(text="Start Detection")
            
            if self.cap:
                self.cap.release()
                self.cap = None
            self.frame_queue.queue.clear()
            self.display_queue.queue.clear()
    
    def auto_save_and_filter(self):
        """Auto save and filter results based on thresholds"""
        original_excel_path = self.save_original_results()
        
        if original_excel_path:
            self.generate_action_timeline()
            self.auto_filter_results()
    
    def save_original_results(self):
        """Save raw analysis results"""
        global OUTPUT_DIR
        today = time.strftime("%Y%m%d")
        today_dir = os.path.join(OUTPUT_DIR, today)
        if not os.path.exists(today_dir):
            os.makedirs(today_dir)
        
        try:
            start_time_str = time.strftime("%H%M", time.localtime(self.action_start_time)) if self.action_start_time else "unknown"
            excel_file_path = os.path.join(today_dir, f"{deer_id}_{start_time_str}_action_durations.xlsx")
            
            action_durations_with_str_time = []
            for action in self.action_durations:
                if isinstance(action["start_time"], (int, float)) and action["start_time"] < 86400:
                    start_time_str = str(timedelta(seconds=action["start_time"]))
                    end_time_str = str(timedelta(seconds=action["end_time"]))
                else:
                    start_time_str = time.strftime("%H:%M:%S", time.localtime(action["start_time"]))
                    end_time_str = time.strftime("%H:%M:%S", time.localtime(action["end_time"]))
                    
                action_durations_with_str_time.append({
                    "Action Type": display_class_mapping.get(action["action"], action["action"]),
                    "Raw Action": action["action"],
                    "Start Time": start_time_str,
                    "End Time": end_time_str,
                    "Duration (s)": action["duration"]
                })
            
            df = pd.DataFrame(action_durations_with_str_time)
            df.to_excel(excel_file_path, index=False)
            
            self.root.after(0, lambda: messagebox.showinfo("Saved", 
                f"Raw results saved to:\n{excel_file_path}"))
            
            return excel_file_path
            
        except PermissionError:
            self.root.after(0, lambda: messagebox.showerror("Permission Error", 
                f"Cannot write to file. Please close the file if it is open."))
            return None
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"Error saving results: {str(e)}"))
            return None

    def auto_filter_results(self):
        """Auto filter results"""
        min_duration_threshold = getattr(self, 'min_duration_threshold_val', 5.0)
        max_transition_rate = getattr(self, 'max_transition_rate_val', 10.0)
        
        filtered_segments = self.analyze_and_filter_results(
            min_duration_threshold=min_duration_threshold,
            max_transition_rate=max_transition_rate
        )
        
        if not filtered_segments:
            self.root.after(0, lambda: messagebox.showinfo("Filter Results", 
                "No segments matched the filter criteria."))
            return
        
        self.save_filtered_results_to_excel(filtered_segments, min_duration_threshold, max_transition_rate)
    
    def save_filtered_results_to_excel(self, filtered_segments, min_duration_threshold, max_transition_rate):
        """Save filtered results to Excel"""
        global OUTPUT_DIR, deer_id
        
        today = time.strftime("%Y%m%d")
        today_dir = os.path.join(OUTPUT_DIR, today)
        if not os.path.exists(today_dir):
            os.makedirs(today_dir)
        
        try:
            start_time_str = time.strftime("%H%M", time.localtime(self.action_start_time)) if self.action_start_time else "filtered"
            excel_file_path = os.path.join(today_dir, f"{deer_id}_{start_time_str}_filtered_segments.xlsx")
            
            filtered_data = []
            for segment in filtered_segments:
                if segment["type"] == "long_duration":
                    start_time_str = str(timedelta(seconds=segment["start_time"])) if isinstance(segment["start_time"], (int, float)) and segment["start_time"] < 86400 else time.strftime("%H:%M:%S", time.localtime(segment["start_time"]))
                    end_time_str = str(timedelta(seconds=segment["end_time"])) if isinstance(segment["end_time"], (int, float)) and segment["end_time"] < 86400 else time.strftime("%H:%M:%S", time.localtime(segment["end_time"]))
                    
                    filtered_data.append({
                        "Filter Type": "Long Duration",
                        "Action Type": segment["display_action"],
                        "Start Time": start_time_str,
                        "End Time": end_time_str,
                        "Duration (s)": segment["duration"],
                        "Threshold (s)": segment.get("threshold", min_duration_threshold),
                        "Note": f"Exceeded {segment.get('threshold', min_duration_threshold)} sec"
                    })
                elif segment["type"] == "high_transition":
                    start_time_str = str(timedelta(seconds=segment["start_time"])) if isinstance(segment["start_time"], (int, float)) and segment["start_time"] < 86400 else time.strftime("%H:%M:%S", time.localtime(segment["start_time"]))
                    end_time_str = str(timedelta(seconds=segment["end_time"])) if isinstance(segment["end_time"], (int, float)) and segment["end_time"] < 86400 else time.strftime("%H:%M:%S", time.localtime(segment["end_time"]))
                    
                    filtered_data.append({
                        "Filter Type": "High Frequency Transition",
                        "Action Type": "Behavioral Pattern",
                        "Start Time": start_time_str,
                        "End Time": end_time_str,
                        "Duration (s)": segment["end_time"] - segment["start_time"],
                        "Rate (times/min)": f"{segment['transition_rate']:.2f}",
                        "Note": f"Exceeded {max_transition_rate} times/min"
                    })
            
            df = pd.DataFrame(filtered_data)
            df.to_excel(excel_file_path, index=False)
            
            def show_and_open():
                messagebox.showinfo("Filtering Complete", 
                    f"Filtered results saved to:\n{excel_file_path}\n"
                    f"Found {len(filtered_segments)} matching segments.\n"
                    f"Opening Excel file...")
                open_excel_file(excel_file_path)
            
            self.root.after(0, show_and_open)
            
        except PermissionError:
            self.root.after(0, lambda: messagebox.showerror("Permission Error", 
                f"Cannot write to file. Please close it if open."))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"Error saving filtered results: {str(e)}"))

    def show_threshold_settings(self):
        """Show Threshold Settings Window"""
        threshold_window = TT.Toplevel(self.root)
        threshold_window.title("Threshold Settings")
        threshold_window.geometry("500x600")
        threshold_window.transient(self.root)
        threshold_window.grab_set()
        
        threshold_window.update_idletasks()
        x = (threshold_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (threshold_window.winfo_screenheight() // 2) - (600 // 2)
        threshold_window.geometry(f"500x600+{x}+{y}")
        
        self.min_duration_threshold = TT.DoubleVar(value=self.min_duration_threshold_val)
        self.max_transition_rate = TT.DoubleVar(value=self.max_transition_rate_val)
        
        if not hasattr(self, 'action_duration_thresholds'):
            self.action_duration_thresholds = {}
            for eng_action in self.class_mapping.values():
                display_action = display_class_mapping.get(eng_action, eng_action)
                default_value = self.action_thresholds.get(display_action, 5.0)
                self.action_duration_thresholds[display_action] = TT.DoubleVar(value=default_value)
        
        TT.Label(threshold_window, text="Behavior Filter Thresholds", font=('Arial', 16, 'bold')).pack(pady=10)
        
        global_frame = TT.LabelFrame(threshold_window, text="Global Settings", font=('Arial', 12, 'bold'))
        global_frame.pack(fill=TT.X, padx=20, pady=10)
        
        duration_frame = TT.Frame(global_frame)
        duration_frame.pack(fill=TT.X, padx=10, pady=10)
        
        TT.Label(duration_frame, text="Min Action Duration (sec):", font=('Arial', 12)).pack(anchor=TT.W)
        duration_entry = TT.Entry(duration_frame, textvariable=self.min_duration_threshold, font=('Arial', 12))
        duration_entry.pack(fill=TT.X, pady=5)
        
        transition_frame = TT.Frame(global_frame)
        transition_frame.pack(fill=TT.X, padx=10, pady=10)
        
        TT.Label(transition_frame, text="Max Transition Rate (times/min):", font=('Arial', 12)).pack(anchor=TT.W)
        transition_entry = TT.Entry(transition_frame, textvariable=self.max_transition_rate, font=('Arial', 12))
        transition_entry.pack(fill=TT.X, pady=5)
        
        action_frame = TT.LabelFrame(threshold_window, text="Per-Action Duration Thresholds", font=('Arial', 12, 'bold'))
        action_frame.pack(fill=TT.BOTH, expand=True, padx=20, pady=10)
        
        canvas = TT.Canvas(action_frame)
        scrollbar = TT.Scrollbar(action_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = TT.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for display_action in self.action_duration_thresholds.keys():
            action_threshold_frame = TT.Frame(scrollable_frame)
            action_threshold_frame.pack(fill=TT.X, padx=10, pady=5)
            
            TT.Label(action_threshold_frame, text=f"{display_action}:", font=('Arial', 11), width=20, anchor=TT.W).pack(side=TT.LEFT)
            threshold_entry = TT.Entry(action_threshold_frame, 
                                    textvariable=self.action_duration_thresholds[display_action], 
                                    font=('Arial', 12), width=10)
            threshold_entry.pack(side=TT.LEFT, padx=10)
            TT.Label(action_threshold_frame, text="s", font=('Arial', 12)).pack(side=TT.LEFT)
        
        canvas.pack(side=TT.LEFT, fill=TT.BOTH, expand=True)
        scrollbar.pack(side=TT.RIGHT, fill=TT.Y)
        
        button_frame = TT.Frame(threshold_window)
        button_frame.pack(fill=TT.X, padx=20, pady=20)
        
        def save_settings():
            try:
                self.min_duration_threshold_val = self.min_duration_threshold.get()
                self.max_transition_rate_val = self.max_transition_rate.get()
                self.action_thresholds = {action: var.get() for action, var in self.action_duration_thresholds.items()}
                threshold_window.destroy()
                messagebox.showinfo("Success", "Settings Saved")
            except TT.TclError:
                messagebox.showerror("Input Error", "Please enter valid numbers.")
        
        TT.Button(button_frame, text="Save", command=save_settings, font=('Arial', 12)).pack(side=TT.LEFT, padx=10)
        TT.Button(button_frame, text="Cancel", command=threshold_window.destroy, font=('Arial', 12)).pack(side=TT.RIGHT, padx=10)
        
        if hasattr(self, 'min_duration_threshold_val'):
            self.min_duration_threshold.set(self.min_duration_threshold_val)
        if hasattr(self, 'max_transition_rate_val'):
            self.max_transition_rate.set(self.max_transition_rate_val)
        if hasattr(self, 'action_thresholds'):
            for action, threshold in self.action_thresholds.items():
                if action in self.action_duration_thresholds:
                    self.action_duration_thresholds[action].set(threshold)

    def analyze_and_filter_results(self, min_duration_threshold=5, max_transition_rate=10):
        if not self.action_durations:
            print("No action data to analyze.")
            return []
        
        long_duration_segments = []
        for action in self.action_durations:
            action_name = display_class_mapping.get(action["action"], action["action"])
            action_threshold = self.action_thresholds.get(action_name, min_duration_threshold) if hasattr(self, 'action_thresholds') else min_duration_threshold
            
            if action["duration"] >= action_threshold:
                long_duration_segments.append({
                    "type": "long_duration",
                    "action": action["action"],
                    "start_time": action["start_time"],
                    "end_time": action["end_time"],
                    "duration": action["duration"],
                    "display_action": action_name,
                    "threshold": action_threshold
                })
        
        if len(self.action_durations) > 1:
            total_duration = self.action_durations[-1]["end_time"] - self.action_durations[0]["start_time"]
            transition_count = len(self.action_durations) - 1
            
            if total_duration > 0:
                transitions_per_minute = (transition_count / total_duration) * 60
            else:
                transitions_per_minute = 0
                
            high_transition_segments = []
            if transitions_per_minute > max_transition_rate:
                high_transition_segments.append({
                    "type": "high_transition",
                    "transition_rate": transitions_per_minute,
                    "total_transitions": transition_count,
                    "total_duration": total_duration,
                    "start_time": self.action_durations[0]["start_time"],
                    "end_time": self.action_durations[-1]["end_time"]
                })
        else:
            high_transition_segments = []
        
        filtered_segments = long_duration_segments + high_transition_segments
        filtered_segments.sort(key=lambda x: x["start_time"])
        
        return filtered_segments

    def hide_action_list(self):
        if self.action_label:
            self.action_label.config(text="", state="hidden")

    def update_excel(self):
        global OUTPUT_DIR
        today = time.strftime("%Y%m%d")
        today_dir = os.path.join(OUTPUT_DIR, today)
        if not os.path.exists(today_dir):
            os.makedirs(today_dir)
        try:
            start_time_str = time.strftime("%H%M", time.localtime(self.action_start_time)) if self.action_start_time else "unknown"
            excel_file_path = os.path.join(today_dir, f"{deer_id}_{start_time_str}_action_durations.xlsx")
            
            action_durations_with_str_time = []
            for action in self.action_durations:
                start_time_str = time.strftime("%H:%M:%S", time.localtime(action["start_time"]))
                end_time_str = time.strftime("%H:%M:%S", time.localtime(action["end_time"]))
                action_durations_with_str_time.append({
                    "Action": action["action"],
                    "Start Time": start_time_str,
                    "End Time": end_time_str,
                    "Duration": action["duration"]
                })
            
            df = pd.DataFrame(action_durations_with_str_time)
            df.to_excel(excel_file_path, index=False)
        except PermissionError:
            messagebox.showerror("Permission Error", f"Cannot write to file.")
        except Exception as e:
            messagebox.showerror("Error", f"Error updating Excel: {str(e)}")

    def generate_action_timeline(self):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            # English doesn't need special font handling for standard characters
            # plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong'] 
            # plt.rcParams['axes.unicode_minus'] = False 
        except ImportError:
            messagebox.showerror("Dependency Missing", "Please install matplotlib: pip install matplotlib")
            return

        if not self.action_durations:
            print("No action data for timeline.")
            return

        timeline_data = []
        action_counts = {}
        action_durations_total = {}

        for action in self.action_durations:
            start_dt = time.strftime("%H:%M:%S", time.localtime(action["start_time"]))
            end_dt = time.strftime("%H:%M:%S", time.localtime(action["end_time"]))
            duration = action["duration"]
            action_name = display_class_mapping.get(action["action"], action["action"])

            timeline_data.append({
                'start': start_dt,
                'end': end_dt,
                'action': action_name,
                'duration': duration,
                'start_time_raw': action["start_time"],
                'end_time_raw': action["end_time"]
            })

            if action_name in action_counts:
                action_counts[action_name] += 1
            else:
                action_counts[action_name] = 1
                
            if action_name in action_durations_total:
                action_durations_total[action_name] += duration
            else:
                action_durations_total[action_name] = duration

        if not timeline_data:
            return

        global OUTPUT_DIR, deer_id
        today = time.strftime("%Y%m%d")
        today_dir = os.path.join(OUTPUT_DIR, today)

        unique_actions = list(set([item['action'] for item in timeline_data]))
        colors = plt.cm.Set3(range(len(unique_actions)))
        action_color_map = dict(zip(unique_actions, colors))

        # Plot 1: Timeline
        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 8))
        fig1.suptitle(f'Musk Deer Behavior Analysis Timeline (ID: {deer_id})', fontsize=24)

        actions = [item['action'] for item in timeline_data]
        durations = [item['duration'] for item in timeline_data]

        y_positions = [1] * len(timeline_data)
        colors_for_bars = [action_color_map[action] for action in actions]
        
        start_positions = [0]
        for i in range(1, len(durations)):
            start_positions.append(start_positions[i-1] + durations[i-1])
        
        bars = ax1.barh(y_positions, durations, left=start_positions, 
                        color=colors_for_bars, alpha=0.7, height=0.5)
        
        ax1.set_ylim(0.5, 1.5)
        ax1.set_xlim(0, sum(durations))
        ax1.set_yticks([])
        ax1.set_xlabel('Time (seconds)', fontsize=20)
        ax1.set_title('Action Duration Horizontal Timeline', fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', labelsize=18)

        for i, (bar, duration) in enumerate(zip(bars, durations)):
            center_x = bar.get_x() + bar.get_width() / 2
            ax1.text(center_x, 1, f'{duration:.1f}s', 
                    ha='center', va='center', fontsize=16, rotation=90)

        annot = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", edgecolor="black", linewidth=1.5),
                            arrowprops=dict(arrowstyle="->", linewidth=1.5),
                            fontsize=14)
        annot.set_visible(False)

        def on_hover(event):
            annot.set_visible(False)
            if event.inaxes == ax1:
                for i, bar in enumerate(bars):
                    bbox = bar.get_bbox()
                    if (bbox.x0 <= event.xdata <= bbox.x1 and 
                        bbox.y0 <= event.ydata <= bbox.y1):
                        item = timeline_data[i]
                        text = f"Action: {item['action']}\n" \
                            f"Duration: {item['duration']:.1f}s\n" \
                            f"Start: {item['start']}\n" \
                            f"End: {item['end']}"
                        annot.xy = (event.xdata, event.ydata)
                        annot.set_text(text)
                        annot.set_visible(True)
                        fig1.canvas.draw()
                        return
                fig1.canvas.draw()

        fig1.canvas.mpl_connect("motion_notify_event", on_hover)
        plt.tight_layout()

        # Plot 2: Frequency
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
        fig2.suptitle(f'Behavior Frequency Statistics (ID: {deer_id})', fontsize=24)

        action_names = list(action_counts.keys())
        counts = list(action_counts.values())

        bars2 = ax2.bar(range(len(action_names)), counts, 
                    color=[action_color_map[name] for name in action_names])
        ax2.set_xlabel('Action Type', fontsize=20)
        ax2.set_ylabel('Frequency (Count)', fontsize=20)
        ax2.set_title('Frequency by Action', fontsize=20)
        ax2.set_xticks(range(len(action_names)))
        ax2.set_xticklabels(action_names, rotation=45, ha='right', fontsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.tick_params(axis='x', labelsize=18)

        for i, (bar, count) in enumerate(zip(bars2, counts)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=16)

        plt.tight_layout()

        # Plot 3: Proportion Pie Chart
        fig3, ax3 = plt.subplots(1, 1, figsize=(14, 10))
        fig3.suptitle(f'Behavior Duration Proportion (ID: {deer_id})', fontsize=24)

        total_duration = sum(action_durations_total.values())
        
        action_names_for_pie = list(action_durations_total.keys())
        durations_for_pie = list(action_durations_total.values())
        
        colors_for_pie = [action_color_map[name] for name in action_names_for_pie]
        
        wedges, texts, autotexts = ax3.pie(durations_for_pie, labels=action_names_for_pie, 
                                        colors=colors_for_pie, autopct='%1.1f%%', startangle=90, 
                                        textprops={'fontsize': 16})
        ax3.set_title('Time Allocation by Action', fontsize=20)
        
        for text in texts:
            text.set_fontsize(18)
        for autotext in autotexts:
            autotext.set_fontsize(16)
            
        legend_labels = [f'{name}: {duration:.1f}s' for name, duration in zip(action_names_for_pie, durations_for_pie)]
        ax3.legend(wedges, legend_labels, title="Action & Duration", loc="center left", 
                bbox_to_anchor=(1, 0, 0.5, 1), fontsize=16, title_fontsize=18)

        plt.tight_layout()

        if self.action_start_time:
            start_time_str = time.strftime("%H%M", time.localtime(self.action_start_time))
            
            timeline_image_path = os.path.join(today_dir, f"{deer_id}_{start_time_str}_action_timeline.png")
            fig1.savefig(timeline_image_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"Timeline saved to: {timeline_image_path}")
            
            frequency_image_path = os.path.join(today_dir, f"{deer_id}_{start_time_str}_action_frequency.png")
            fig2.savefig(frequency_image_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"Frequency chart saved to: {frequency_image_path}")
            
            proportion_image_path = os.path.join(today_dir, f"{deer_id}_{start_time_str}_action_proportion.png")
            fig3.savefig(proportion_image_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"Proportion chart saved to: {proportion_image_path}")

        plt.show()

def open_excel_file(file_path):
    """Open Excel File Cross-Platform"""
    try:
        os.startfile(file_path) # Windows
    except AttributeError:
        try:
            subprocess.call(["open", file_path])  # macOS
        except:
            try:
                subprocess.call(["xdg-open", file_path])  # Linux
            except:
                print(f"Cannot open file automatically. Please open manually: {file_path}")

def create_multi_video_window(video_id="New Video"):
    """Create Multi-Video Analysis Window"""
    global multi_video_windows, multi_processor_instances, root
    
    if video_id in multi_video_windows:
        multi_video_windows[video_id].lift()
        return
    
    video_window = TT.Toplevel(root)
    video_window.title(f"Behavior Analysis - {video_id}")
    video_window.geometry("800x600")
    
    processor = VideoProcessor(video_window, video_id=video_id, is_multi_video=True)
    multi_processor_instances[video_id] = processor
    
    main_frame = Frame(video_window)
    main_frame.pack(fill=TT.BOTH, expand=True, padx=10, pady=10)
    main_frame.columnconfigure(0, weight=1, minsize=200)
    main_frame.columnconfigure(1, weight=4)
    main_frame.rowconfigure(0, weight=1)
    
    data_frame = Frame(main_frame)
    data_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    data_frame.columnconfigure(0, weight=1)
    
    control_frame = Frame(data_frame)
    control_frame.pack(fill=TT.X, pady=5)
    
    def select_video_for_multi():
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if video_path:
            try:
                processor.init_camera(video_path)
                messagebox.showinfo("Success", f"Loaded: {os.path.basename(video_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
    
    TT.Button(control_frame, text="Select Video", command=select_video_for_multi, font=('Arial', 12)).pack(side=TT.LEFT, padx=5)
    
    start_stop_btn = TT.Button(control_frame, text="Start", command=processor.toggle_detection, 
                               font=('Arial', 12), bg="lightgreen")
    start_stop_btn.pack(side=TT.LEFT, padx=5)
    processor.start_stop_button = start_stop_btn
    
    TT.Button(control_frame, text="Thresholds", command=processor.show_threshold_settings, font=('Arial', 12)).pack(side=TT.LEFT, padx=5)
    
    info_labels = [
        Label(data_frame, text=f"Temp: {temperature}", font=('Arial', 12)),
        Label(data_frame, text=f"Humidity: {humidity}", font=('Arial', 12)),
        Label(data_frame, text=f"Health: {health_info}", font=('Arial', 12)),
        Label(data_frame, text=f"ID: {deer_id}", font=('Arial', 12)),
    ]
    
    for label in info_labels:
        label.pack(fill=TT.X, pady=5)
    
    action_label = Label(data_frame, text="", font=('Arial', 20), state="hidden", foreground="red", wraplength=200)
    action_label.pack(fill=TT.X, pady=5)
    processor.action_label = action_label
    
    stats_label = Label(data_frame, text=f"Video {video_id} Stats:\nNo Data", font=('Arial', 12), justify=TT.LEFT)
    stats_label.pack(fill=TT.X, pady=10)
    processor.stats_label = stats_label
    
    time_label = Label(data_frame, text="", font=('Arial', 12))
    time_label.pack(fill=TT.X, pady=5)
    processor.time_label = time_label
    
    fps_label = Label(data_frame, text="FPS: 0.00", font=('Arial', 12))
    fps_label.pack(fill=TT.X, pady=5)
    processor.fps_label = fps_label
    
    video_label = TT.Label(main_frame, bg='black')
    video_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    video_label.configure(width=640, height=480)
    processor.video_label = video_label
    
    def on_closing():
        if processor.is_detecting:
            processor.toggle_detection()
        
        if video_id in multi_video_windows:
            del multi_video_windows[video_id]
        if video_id in multi_processor_instances:
            del multi_processor_instances[video_id]
        
        video_window.destroy()
    
    video_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    multi_video_windows[video_id] = video_window
    
    return video_window

def start_multi_video_analysis():
    """Start Multi-Video Analysis"""
    try:
        num_videos = simpledialog.askinteger("Multi-Video Analysis", 
            "Enter number of videos (1-4):", parent=root, minvalue=1, maxvalue=4)
        
        if num_videos and num_videos > 0:
            for i in range(num_videos):
                video_id = f"Cam_{i+1}"
                create_multi_video_window(video_id)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create windows: {str(e)}")

def quick_video_analysis():
    """Quick Video Analysis"""
    global VIDEO_FILE_PATH, model_path, processor, root
    
    video_path = filedialog.askopenfilename(
        title="Select Video for Quick Analysis",
        filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
    )
    
    if not video_path:
        return
        
    VIDEO_FILE_PATH = video_path
    
    if not model_path:
        select_model_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO Model", "*.pt")]
        )
        if not select_model_path:
            return
        model_path = select_model_path
    
    if not hasattr(processor, 'min_duration_threshold_val') or not hasattr(processor, 'max_transition_rate_val'):
        response = messagebox.askyesno("Thresholds", "Thresholds not set. Configure now?")
        if response:
            processor.show_threshold_settings()
        else:
            processor.min_duration_threshold_val = 5.0
            processor.max_transition_rate_val = 10.0
    
    messagebox.showinfo("Analyzing", "Starting quick analysis. Filtering will run automatically upon completion.")
    threading.Thread(target=perform_quick_analysis_with_filtering, daemon=True).start()

def perform_quick_analysis_with_filtering():
    """Run quick analysis and filtering"""
    global processor, VIDEO_FILE_PATH, model_path, root
    
    temp_processor = VideoProcessor(root, video_id="QuickAnalysis")
    
    if hasattr(processor, 'min_duration_threshold_val'):
        temp_processor.min_duration_threshold_val = processor.min_duration_threshold_val
        temp_processor.max_transition_rate_val = processor.max_transition_rate_val
        temp_processor.action_thresholds = processor.action_thresholds.copy()
    
    try:
        temp_processor.load_model()
    except Exception as e:
        messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
        return
    
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    progress_window = TT.Toplevel()
    progress_window.title("Analysis Progress")
    progress_window.geometry("300x150")
    progress_window.transient(root)
    progress_window.grab_set()
    
    progress_label = TT.Label(progress_window, text="Analyzing Video...")
    progress_label.pack(pady=10)
    
    progress_var = TT.DoubleVar()
    progress_bar = TT.ttk.Progressbar(
        progress_window, 
        variable=progress_var, 
        maximum=100,
        length=250
    )
    progress_bar.pack(pady=10)
    
    info_label = TT.Label(progress_window, text="")
    info_label.pack(pady=5)
    
    temp_processor.action_durations = []
    temp_processor.action_durations_total = {}
    temp_processor.current_action = None
    start_time = 0.0
    temp_processor.frame_count = 0
    
    frame_idx = 0
    processed_frames = 0
    last_results = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx % max(1, int(total_frames / 100)) == 0:
            progress_percent = (frame_idx / total_frames) * 100
            progress_var.set(progress_percent)
            info_label.config(text=f"Analyzing... {frame_idx}/{total_frames} frames")
            progress_window.update()
        
        current_time = frame_idx / fps
        
        # Detect every 5 frames
        if frame_idx % 5 == 0:
            try:
                results = temp_processor.model(frame, verbose=False)
                last_results = results
                
                if results[0].boxes:
                    highest_conf_box = max(results[0].boxes, key=lambda box: box.conf[0].item())
                    cls = int(highest_conf_box.cls[0])
                    conf = highest_conf_box.conf[0].item()
                    
                    if conf >= ACTION_THRESHOLD:
                        action = temp_processor.class_mapping.get(cls, "Unknown")
                        
                        if temp_processor.current_action != action:
                            if temp_processor.current_action:
                                action_end_time = current_time
                                duration = action_end_time - start_time
                                
                                if duration > 0:
                                    temp_processor.action_durations.append({
                                        "action": temp_processor.current_action,
                                        "start_time": start_time,
                                        "end_time": action_end_time,
                                        "duration": duration
                                    })
                                    
                                    display_prev_action = display_class_mapping.get(temp_processor.current_action, "Unknown")
                                    if display_prev_action in temp_processor.action_durations_total:
                                        temp_processor.action_durations_total[display_prev_action] += duration
                                    else:
                                        temp_processor.action_durations_total[display_prev_action] = duration
                            
                            temp_processor.current_action = action
                            start_time = current_time
                
                processed_frames += 1
                
            except Exception as e:
                print(f"Frame Error: {e}")
                continue
    
    if temp_processor.current_action:
        action_end_time = frame_idx / fps
        duration = action_end_time - start_time
        
        if duration > 0:
            temp_processor.action_durations.append({
                "action": temp_processor.current_action,
                "start_time": start_time,
                "end_time": action_end_time,
                "duration": duration
            })
            
            display_prev_action = display_class_mapping.get(temp_processor.current_action, "Unknown")
            if display_prev_action in temp_processor.action_durations_total:
                temp_processor.action_durations_total[display_prev_action] += duration
            else:
                temp_processor.action_durations_total[display_prev_action] = duration
    
    cap.release()
    progress_window.destroy()
    
    temp_processor.auto_save_and_filter()

def select_output_dir():
    global OUTPUT_DIR
    new_output_dir = filedialog.askdirectory()
    if new_output_dir:
        OUTPUT_DIR = new_output_dir
        file_menu.entryconfig(5, label=f"Current Output Dir: {OUTPUT_DIR.split('/')[-1]}")

def create_gui():
    global root, processor, video_label, action_label, file_menu, action_menu, time_label, fps_label, action_stats_label
    
    root = TT.Tk()
    root.title("Musk Deer Behavior Monitor - Multi-Video Edition")
    root.geometry(f"{max_width}x{max_height}")
    root.minsize(800, 600)
    
    messagebox.showinfo("Welcome", "Please select a camera source and load a model file.")
    
    menubar = TT.Menu(root)
    
    file_menu = TT.Menu(menubar, tearoff=0)
    
    def select_model():
        global model_path
        model_path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if model_path:
            file_menu.entryconfig(1, label=f"Current Model: {model_path.split('/')[-1]}")
    
    def select_video_file():
        global VIDEO_FILE_PATH
        VIDEO_FILE_PATH = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if VIDEO_FILE_PATH:
            file_menu.entryconfig(3, label=f"Current Video: {VIDEO_FILE_PATH.split('/')[-1]}")
    
    def set_cam_source():
        global video_source
        try:
            new_source = simpledialog.askinteger("Camera Settings", 
                "Enter Camera Index (0=Default, 1=External):", parent=root)
            if new_source is not None:
                video_source = int(new_source)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    file_menu.add_command(label="Select Model File", command=select_model)
    file_menu.add_command(label="Model: None", state='disabled')
    file_menu.add_command(label="Select Video File", command=select_video_file)
    file_menu.add_command(label="Video: None", state='disabled')
    file_menu.add_command(label="Select Output Dir", command=select_output_dir)
    file_menu.add_command(label="Output: Default", state='disabled')
    file_menu.add_command(label="Camera Settings", command=set_cam_source)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    action_menu = TT.Menu(menubar, tearoff=0)
    
    action_menu.add_command(label="Start Single Detection", command=lambda: processor.toggle_detection())
    action_menu.add_command(label="Start Multi-Video Analysis", command=start_multi_video_analysis)
    action_menu.add_command(label="Quick Video Analysis", command=quick_video_analysis)
    action_menu.add_separator()
    action_menu.add_command(label="Set Filter Thresholds", command=lambda: processor.show_threshold_settings())
    
    menubar.add_cascade(label="Settings", menu=file_menu)
    menubar.add_cascade(label="Operations", menu=action_menu)
    root.config(menu=menubar)
    
    main_frame = Frame(root)
    main_frame.pack(fill=TT.BOTH, expand=True, padx=10, pady=10)
    main_frame.columnconfigure(0, weight=1, minsize=200)
    main_frame.columnconfigure(1, weight=4)
    main_frame.rowconfigure(0, weight=1)
    
    data_frame = Frame(main_frame)
    data_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    data_frame.columnconfigure(0, weight=1)
    
    control_frame = Frame(data_frame)
    control_frame.pack(fill=TT.X, pady=10)
    
    def select_video_source():
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if video_path:
            try:
                processor.init_camera(video_path)
                messagebox.showinfo("Success", f"Loaded Video: {os.path.basename(video_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
        else:
            try:
                processor.init_camera(0)
                messagebox.showinfo("Info", "Switched to Default Camera")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open camera: {str(e)}")
    
    TT.Button(control_frame, text="Select Camera/Video", 
              command=select_video_source, font=('Arial', 12)).pack(fill=TT.X, pady=5)
    
    start_single_btn = TT.Button(control_frame, text="Start Single Detection", 
              command=lambda: processor.toggle_detection(), font=('Arial', 12), bg="lightblue")
    start_single_btn.pack(fill=TT.X, pady=5)
    
    TT.Button(control_frame, text="Start Multi-Video Analysis", 
              command=start_multi_video_analysis, font=('Arial', 12), bg="lightgreen").pack(fill=TT.X, pady=5)
    
    TT.Button(control_frame, text="Quick Video Analysis", 
              command=quick_video_analysis, font=('Arial', 12), bg="lightyellow").pack(fill=TT.X, pady=5)
    
    TT.Button(control_frame, text="Set Thresholds", 
              command=lambda: processor.show_threshold_settings(), font=('Arial', 12), bg="lightcoral").pack(fill=TT.X, pady=5)
    
    info_labels = [
        Label(data_frame, text=f"Temp: {temperature}", font=('Arial', 12)),
        Label(data_frame, text=f"Humidity: {humidity}", font=('Arial', 12)),
        Label(data_frame, text=f"Health: {health_info}", font=('Arial', 12)),
        Label(data_frame, text=f"ID: {deer_id}", font=('Arial', 12)),
    ]
    
    for label in info_labels:
        label.pack(fill=TT.X, pady=5)
    
    action_label = Label(data_frame, text="", font=('Arial', 20), state="hidden", foreground="red", wraplength=200)
    action_label.pack(fill=TT.X, pady=5)
    
    action_stats_label = Label(data_frame, text="Main Video Stats:\nNo Data", font=('Arial', 12), justify=TT.LEFT)
    action_stats_label.pack(fill=TT.X, pady=10)
    
    time_label = Label(data_frame, text="", font=('Arial', 12))
    time_label.pack(fill=TT.X, pady=5)
    
    fps_label = Label(data_frame, text="FPS: 0.00", font=('Arial', 12))
    fps_label.pack(fill=TT.X, pady=5)
    
    threshold_status_label = Label(data_frame, text="Filter Thresholds: Not Set", font=('Arial', 12), foreground="blue")
    threshold_status_label.pack(fill=TT.X, pady=5)
    
    multi_status_label = Label(data_frame, text="Multi-Video: Inactive", font=('Arial', 12), foreground="blue")
    multi_status_label.pack(fill=TT.X, pady=10)
    
    def update_threshold_status():
        if hasattr(processor, 'min_duration_threshold_val'):
            status = f"Thresholds: Min {processor.min_duration_threshold_val}s, Max {processor.max_transition_rate_val} rate"
            threshold_status_label.config(text=status, foreground="green")
        else:
            threshold_status_label.config(text="Thresholds: Not Set", foreground="red")
        root.after(2000, update_threshold_status)
    
    def update_multi_status():
        if multi_video_windows:
            status = f"Analyzing {len(multi_video_windows)} videos"
            multi_status_label.config(text=f"Multi-Video: {status}", foreground="green")
        else:
            multi_status_label.config(text="Multi-Video: Inactive", foreground="blue")
        root.after(1000, update_multi_status)
    
    video_label = TT.Label(main_frame, bg='black')
    video_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    video_label.configure(width=640, height=480)
    
    processor = VideoProcessor(root, video_id="Main Video")
    processor.video_label = video_label
    processor.action_label = action_label
    processor.stats_label = action_stats_label
    processor.time_label = time_label
    processor.fps_label = fps_label
    processor.start_stop_button = start_single_btn
    
    update_threshold_status()
    update_multi_status()
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()
