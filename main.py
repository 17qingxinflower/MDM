import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import threading
import time
from queue import Queue, Empty, Full
from collections import deque
from PIL import Image, ImageTk
from ultralytics import YOLO
import pandas as pd
import os
import math
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import timedelta
import base64
import re

# Global configuration and Matplotlib font settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# --- Dark ecology color palette ---
COLOR_BG_MAIN = "#0A0F0D"      
COLOR_BG_SIDEBAR = "#0D1411"   
COLOR_PANEL = "#111814"        
COLOR_BORDER = "#1B261F"       
COLOR_PRIMARY = "#10B981"      
COLOR_PRIMARY_HOVER = "#059669"
COLOR_DANGER = "#EF4444"       
COLOR_TEXT_MAIN = "#F3F4F6"    
COLOR_TEXT_SUB = "#6B7280"     

ctk.set_appearance_mode("dark")

# Global database concurrency lock
db_lock = threading.Lock()

TRANSPARENT_ICON = b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='

def set_window_icon(window):
    try:
        if os.path.exists("deer.ico"):
            window.after(200, lambda: window.iconbitmap("deer.ico"))
        elif os.path.exists("deer.png"):
            img = tk.PhotoImage(file="deer.png")
            window.after(200, lambda: window.iconphoto(False, img))
        else:
            img = tk.PhotoImage(data=TRANSPARENT_ICON)
            window.after(200, lambda: window.iconphoto(False, img))
    except: pass

def force_focus(window):
    window.attributes('-topmost', True)
    window.update()
    window.attributes('-topmost', False)
    window.focus_force()

max_width, max_height = 1400, 900
model_path = ""
video_source = 0
OUTPUT_DIR = os.path.join(os.getcwd(), "UI_result")
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_FILE_PATH = ""

multi_processor_instances = {}
FRAME_QUEUE_SIZE = 2
DISPLAY_FPS = 30
ACTION_THRESHOLD = 0.45

temperature = "18.5"
humidity = "62"
health_info = "Normal"
deer_id = "2.1.6" 

# === 1. Database initialization ===
def init_db():
    with db_lock:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS analysis_records
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      task_id TEXT, deer_id TEXT, session_start TEXT, session_end TEXT,
                      has_alert INTEGER, excel_raw TEXT, excel_alert TEXT,
                      img_timeline TEXT, img_freq TEXT, img_pie TEXT)''')
                      
        c.execute('''CREATE TABLE IF NOT EXISTS deer_info
                     (deer_id TEXT PRIMARY KEY, added_date TEXT)''')
                     
        c.execute('''CREATE TABLE IF NOT EXISTS daily_summary
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TEXT, 
                      deer_id TEXT,
                      action_name TEXT, 
                      total_duration REAL, 
                      frequency INTEGER,
                      percentage REAL,
                      UNIQUE(date, deer_id, action_name))''')

        c.execute('''CREATE TABLE IF NOT EXISTS action_stream_log
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TEXT, deer_id TEXT, action_name TEXT, 
                      start_time TEXT, end_time TEXT, duration REAL)''')

        c.execute("INSERT OR IGNORE INTO deer_info (deer_id, added_date) VALUES ('2.1.6', date('now'))")
        conn.commit()
        conn.close()

init_db()

class_mapping = {0: "hunt", 1: "egestion", 2: "Standing_feeding", 3: "Stand_or_walk", 4: "Lie_down_and_rest", 5: "Licking_the_pussy", 6: "Comb_and_lick"}
display_class_mapping = {"hunt": "Foraging", "egestion": "Excretion", "Standing_feeding": "Standing Feeding", "Stand_or_walk": "Walking", "Lie_down_and_rest": "Resting", "Licking_the_pussy": "Anogenital Licking", "Comb_and_lick": "Grooming"}
english_display_mapping = {"hunt": "Foraging", "egestion": "Excretion", "Standing_feeding": "Standing Feeding", "Stand_or_walk": "Walking", "Lie_down_and_rest": "Resting", "Licking_the_pussy": "Anogenital Licking", "Comb_and_lick": "Grooming"}

# === 2. Core video analysis engine ===
class VideoProcessor:
    def __init__(self, root, video_id="default", is_multi_video=False):
        self.root = root
        self.video_id = video_id
        self.is_multi_video = is_multi_video
        self.deer_id = "Unassigned"
        
        self.model = None
        self.cap = None
        self.class_mapping = class_mapping
        self.original_size = (640, 480)
        
        self.current_action = None
        self.action_start_time = None
        self.session_start_time = None 
        self.action_durations = []
        
        self.action_durations_total = {}
        self.action_frequency_total = {}
        self.last_saved_durations = {}
        self.last_saved_frequencies = {}
        
        self.frame_count = 0
        self.current_fps = 0.0
        
        self.pending_action = None
        self.pending_start_time = None
        self.action_tolerance = 0.5 
        
        self.current_display_action = None 
        self.stat_cards = [] 
        
        self.min_duration_threshold_val = 5.0
        self.max_transition_rate_val = 10.0
        self.action_thresholds = {}
        
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.display_queue = Queue(maxsize=1)
        self.is_detecting = False
        
        self.is_video_file = False
        self.total_frames = 0
        self.playback_speed = 1.0
        self.seek_requested = -1
        self.has_skipped = False
        self.is_dragging_slider = False
        
        self.video_label = None
        self.action_label = None
        self.stats_container = None
        self.fps_label = None
        self.start_stop_button = None
        
        self.control_panel = None
        self.progress_slider = None
        self.speed_var = None

    def init_camera(self, source=0):
        self.video_source = source
        if self.cap is not None: self.cap.release()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened(): raise IOError("Unable to connect to the monitoring stream.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.original_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.original_size[1] == 0: self.original_size = (640, 480)
        
        self.is_video_file = isinstance(source, str)
        if self.is_video_file:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_frames = 0
        return True

    def get_current_time(self):
        """Timing core: video files use virtual timestamps, so playback speed does not alter the behavior time scale."""
        if self.is_video_file and self.cap:
            msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            return msec / 1000.0 if msec >= 0 else 0.0
        return time.time()

    def load_model(self):
        global model_path
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.model.fuse()
                self.model.eval()
                return True
            except Exception as e:
                messagebox.showerror("Engine Load Failed", f"Unable to load model:\n{str(e)}")
                self.model = None
                return False
        return False

    def update_action_stats(self):
        if not self.stats_container: return
        action_durations_total = self.action_durations_total.copy()
        
        if self.current_action and self.action_start_time is not None:
            current_duration = self.get_current_time() - self.action_start_time
            if current_duration >= 0.1:
                display_action = display_class_mapping.get(self.current_action, "Unknown")
                action_durations_total[display_action] = action_durations_total.get(display_action, 0) + current_duration
        
        if not action_durations_total:
            for card_dict in self.stat_cards: card_dict["card"].pack_forget()
            if not hasattr(self, "empty_stats_label"):
                self.empty_stats_label = ctk.CTkLabel(self.stats_container, text="No behavior data yet", text_color=COLOR_TEXT_SUB)
                self.empty_stats_label.pack(pady=30)
            else: self.empty_stats_label.pack(pady=30)
            return
            
        if hasattr(self, "empty_stats_label") and self.empty_stats_label:
            self.empty_stats_label.pack_forget()
            
        total_duration = sum(action_durations_total.values())
        sorted_items = sorted(action_durations_total.items(), key=lambda x: x[1], reverse=True)
        
        while len(self.stat_cards) < len(sorted_items):
            card = ctk.CTkFrame(self.stats_container, fg_color="transparent", border_width=1, border_color=COLOR_BORDER, corner_radius=8)
            card.pack(fill="x", pady=4, padx=2)
            left_info = ctk.CTkFrame(card, fg_color="transparent")
            left_info.pack(side="left", padx=12, pady=8)
            ctk.CTkLabel(left_info, text="Record", font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(0, 2))
            lbl_act = ctk.CTkLabel(left_info, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_MAIN)
            lbl_act.pack(anchor="w")
            
            right_info = ctk.CTkFrame(card, fg_color="transparent")
            right_info.pack(side="right", padx=15, pady=8)
            lbl_dur = ctk.CTkLabel(right_info, text="", font=ctk.CTkFont(size=15, weight="bold"), text_color=COLOR_PRIMARY)
            lbl_dur.pack(anchor="e")
            lbl_pct = ctk.CTkLabel(right_info, text="", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_SUB)
            lbl_pct.pack(anchor="e")
            self.stat_cards.append({"card": card, "lbl_act": lbl_act, "lbl_dur": lbl_dur, "lbl_pct": lbl_pct})
            
        for i, (action, duration) in enumerate(sorted_items):
            percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
            self.stat_cards[i]["lbl_act"].configure(text=action)
            self.stat_cards[i]["lbl_dur"].configure(text=f"{duration:.1f} s")
            self.stat_cards[i]["lbl_pct"].configure(text=f"Share: {percentage:.1f}%")
            self.stat_cards[i]["card"].pack(fill="x", pady=4, padx=2)
            
        for i in range(len(sorted_items), len(self.stat_cards)):
            self.stat_cards[i]["card"].pack_forget()

    def _close_current_action(self, end_time):
        if self.current_action and self.action_start_time is not None:
            dur = end_time - self.action_start_time
            if dur >= 0.1:
                self.action_durations.append({
                    "action": self.current_action, "start_time": self.action_start_time,
                    "end_time": end_time, "duration": dur
                })
                prev_disp = display_class_mapping.get(self.current_action, "Unknown")
                self.action_durations_total[prev_disp] = self.action_durations_total.get(prev_disp, 0) + dur
                self.action_frequency_total[prev_disp] = self.action_frequency_total.get(prev_disp, 0) + 1
                
                # Data guard: replay mode does not write live stream records to the database
                if not self.is_video_file:
                    with db_lock:
                        try:
                            conn = sqlite3.connect('history.db')
                            c = conn.cursor()
                            st_str = time.strftime("%H:%M:%S", time.localtime(self.action_start_time)) if self.action_start_time > 86400 else str(timedelta(seconds=int(self.action_start_time)))
                            et_str = time.strftime("%H:%M:%S", time.localtime(end_time)) if end_time > 86400 else str(timedelta(seconds=int(end_time)))
                            task_date = time.strftime('%Y-%m-%d', time.localtime(self.session_start_time)) if self.session_start_time and self.session_start_time > 86400 else time.strftime('%Y-%m-%d')
                            
                            c.execute("INSERT INTO action_stream_log (date, deer_id, action_name, start_time, end_time, duration) VALUES (?,?,?,?,?,?)",
                                      (task_date, self.deer_id, prev_disp, st_str, et_str, round(dur, 2)))
                            conn.commit()
                            conn.close()
                        except: pass
                
        self.current_action = None
        self.action_start_time = None

    def auto_checkpoint_worker(self):
        while self.is_detecting:
            for _ in range(300):
                if not self.is_detecting: return
                time.sleep(1)
            self._silent_checkpoint()

    def _silent_checkpoint(self):
        if self.is_video_file or not self.action_durations_total: return
        
        total_valid_seconds = sum(self.action_durations_total.values())
        if total_valid_seconds < 1800: return  
            
        task_date = time.strftime('%Y-%m-%d', time.localtime(self.session_start_time)) if self.session_start_time and self.session_start_time > 86400 else time.strftime('%Y-%m-%d')
        
        with db_lock:
            conn = sqlite3.connect('history.db')
            c = conn.cursor()
            
            for act_name, current_dur in self.action_durations_total.items():
                current_freq = self.action_frequency_total.get(act_name, 1)
                delta_dur = current_dur - self.last_saved_durations.get(act_name, 0)
                delta_freq = current_freq - self.last_saved_frequencies.get(act_name, 0)
                
                if delta_dur > 0 or delta_freq > 0:
                    c.execute("SELECT total_duration, frequency FROM daily_summary WHERE date=? AND deer_id=? AND action_name=?", (task_date, self.deer_id, act_name))
                    row = c.fetchone()
                    if row:
                        c.execute("UPDATE daily_summary SET total_duration=?, frequency=? WHERE date=? AND deer_id=? AND action_name=?", 
                                  (row[0] + delta_dur, row[1] + delta_freq, task_date, self.deer_id, act_name))
                    else:
                        c.execute("INSERT INTO daily_summary (date, deer_id, action_name, total_duration, frequency, percentage) VALUES (?, ?, ?, ?, ?, 0)", 
                                  (task_date, self.deer_id, act_name, delta_dur, delta_freq))
                    
                    self.last_saved_durations[act_name] = current_dur
                    self.last_saved_frequencies[act_name] = current_freq
            
            c.execute("SELECT SUM(total_duration) FROM daily_summary WHERE date=? AND deer_id=?", (task_date, self.deer_id))
            total_day = c.fetchone()[0] or 1
            c.execute("UPDATE daily_summary SET percentage = round((total_duration / ?) * 100, 2) WHERE date=? AND deer_id=?", (total_day, task_date, self.deer_id))
            
            conn.commit()
            conn.close()

    def _on_slider_drag_start(self, event):
        self.is_dragging_slider = True

    def _on_slider_release(self, event):
        if self.is_video_file and self.total_frames > 0 and self.progress_slider:
            val = self.progress_slider.get()
            self.seek_requested = int(val * self.total_frames)
            self.has_skipped = True
        self.is_dragging_slider = False

    def _on_speed_change(self, *args):
        if self.speed_var:
            try:
                self.playback_speed = float(self.speed_var.get().replace("x", ""))
            except:
                self.playback_speed = 1.0

    def video_capture_thread_func(self):
        frame_count, start_time = 0, time.time()
        while self.is_detecting:
            if self.seek_requested >= 0 and self.cap:
                self._close_current_action(self.get_current_time())
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_requested)
                self.current_display_action = None
                self.pending_action = None
                with self.frame_queue.mutex: self.frame_queue.queue.clear()
                self.seek_requested = -1

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    try: self.frame_queue.put_nowait(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    except Full: pass
                    except: continue
                    frame_count += 1
                    
                    if time.time() - start_time >= 1:
                        self.current_fps = frame_count / (time.time() - start_time)
                        frame_count, start_time = 0, time.time()
                    
                    if self.is_video_file and self.progress_slider and not self.is_dragging_slider:
                        curr_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        if curr_frame % 15 == 0:
                            self.root.after(0, self.progress_slider.set, curr_frame / self.total_frames)

                    base_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    sleep_time = (1.0 / base_fps) / self.playback_speed if base_fps > 0 else (1.0/30)/self.playback_speed
                    try: time.sleep(sleep_time)
                    except: time.sleep(1/30)
                else:
                    if self.is_video_file: break
            else: time.sleep(0.1)
    
    def processing_thread_func(self):
        while self.is_detecting:
            try:
                frame = self.frame_queue.get(timeout=1)
                processed_frame = self.process_frame(frame)
                if self.display_queue.empty(): self.display_queue.put(processed_frame)
            except Empty: continue
    
    def process_frame(self, frame, is_video_file=False):
        if frame is None: return frame.copy()
        current_time = self.get_current_time()
        frame_out = frame.copy()
        
        detected_action = None 
        
        if self.is_detecting and self.model and self.frame_count % 5 == 0:
            results = self.model(frame, verbose=False)
            self.last_results = results
            
        if hasattr(self, 'last_results') and self.last_results is not None and self.last_results[0].boxes:
            box = max(self.last_results[0].boxes, key=lambda b: b.conf[0].item())
            cls, conf = int(box.cls[0]), box.conf[0].item()
            if conf >= ACTION_THRESHOLD:
                detected_action = self.class_mapping.get(cls, "Unknown")
                eng_action = english_display_mapping.get(detected_action, "Unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (16, 185, 129), 2)
                cv2.putText(frame_out, f"{eng_action} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)

        if detected_action == self.current_action:
            self.pending_action = None 
            self.pending_start_time = None
        else:
            if self.pending_action != detected_action or self.pending_start_time is None:
                self.pending_action = detected_action
                self.pending_start_time = current_time
            else:
                if current_time - self.pending_start_time >= self.action_tolerance:
                    self._close_current_action(self.pending_start_time) 
                    self.current_action = detected_action
                    self.action_start_time = self.pending_start_time 
                    self.current_display_action = display_class_mapping.get(detected_action, "Unknown") if detected_action else None
                    self.pending_action = None
                    self.pending_start_time = None

        self.frame_count += 1
        return frame_out
    
    def update_display(self):
        try:
            frame = self.display_queue.get_nowait()
            if self.video_label:
                lw, lh = self.video_label.winfo_width(), self.video_label.winfo_height()
                if lw > 0 and lh > 0:
                    ratio = min(lw / self.original_size[0], lh / self.original_size[1])
                    sz = (int(self.original_size[0] * ratio), int(self.original_size[1] * ratio))
                    
                    resized_frame = cv2.resize(frame, sz, interpolation=cv2.INTER_LINEAR)
                    img = Image.fromarray(resized_frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk, text="")
                    
                    if self.fps_label: self.fps_label.configure(text=f"FPS: {int(self.current_fps)}")
                    if self.action_label:
                        if self.current_display_action: 
                            self.action_label.configure(text=f"● {self.current_display_action}", text_color=COLOR_PRIMARY)
                        else: 
                            self.action_label.configure(text="Waiting", text_color=COLOR_TEXT_SUB)
                    self.update_action_stats()
        except Empty: pass
        if self.is_detecting and self.video_label: self.video_label.after(int(1000 / DISPLAY_FPS), self.update_display)
    
    def toggle_detection(self):
        if not self.is_detecting:
            if not model_path: return messagebox.showerror("Notice", "Please load the AI model first.")
            if not self.load_model(): return
            if self.cap is None or not self.cap.isOpened(): return messagebox.showerror("Notice", "No video source is connected. Please check the source.")
            if self.deer_id == "Unassigned": return messagebox.showerror("Notice", "This analysis stream is not assigned to a deer ID. Please connect the source again.")
            
            self.is_detecting = True
            self.session_start_time = time.time()
            self.action_start_time = self.get_current_time() 
            self.seek_requested = -1
            self.has_skipped = False
            self.playback_speed = 1.0
            
            # Load progress controls according to single-view or multi-view mode
            if self.is_video_file and self.control_panel:
                if self.is_multi_video:
                    self.control_panel.pack(fill="x", pady=5, after=self.start_stop_button)
                else:
                    self.control_panel.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
                
                if self.progress_slider:
                    self.progress_slider.set(0)
                if self.speed_var:
                    self.speed_var.set("1.0x")
            elif self.control_panel:
                if self.is_multi_video:
                    self.control_panel.pack_forget()
                else:
                    self.control_panel.grid_forget()
            
            self.action_durations.clear()
            self.action_durations_total.clear()
            self.action_frequency_total.clear()
            self.last_saved_durations.clear()
            self.last_saved_frequencies.clear()
            
            self.current_display_action = None
            self.pending_action = None
            self.pending_start_time = None
            
            if self.fps_label: self.fps_label.grid(row=0, column=1, sticky="ne", padx=15, pady=15)
            if self.is_multi_video and self.start_stop_button:
                self.start_stop_button.configure(text="Stop Analysis", fg_color=COLOR_DANGER, hover_color="#B91C1C")
            
            threading.Thread(target=self.video_capture_thread_func, daemon=True).start()
            threading.Thread(target=self.processing_thread_func, daemon=True).start()
            threading.Thread(target=self.auto_checkpoint_worker, daemon=True).start() 
            self.update_display()
        else:
            self.is_detecting = False
            self._close_current_action(self.get_current_time())
            self.current_display_action = None
            self.pending_action = None
            self.pending_start_time = None
            
            if self.action_label: self.action_label.configure(text="Waiting", text_color=COLOR_TEXT_SUB)
            if self.cap: self.cap.release(); self.cap = None
            self.frame_queue.queue.clear(); self.display_queue.queue.clear()
            
            if self.control_panel:
                if self.is_multi_video: self.control_panel.pack_forget()
                else: self.control_panel.grid_forget()
            
            if hasattr(self.root, "show_placeholder_fn") and not self.is_multi_video: 
                self.root.show_placeholder_fn()
            if self.fps_label: self.fps_label.grid_forget()
            if self.is_multi_video and self.start_stop_button:
                self.start_stop_button.configure(text="Start Analysis", fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER)
            
            threading.Thread(target=self.save_all_data, daemon=True).start()

    def analyze_and_filter_results(self, min_dur, max_trans):
        if not self.action_durations: return []
        filtered = []
        for a in self.action_durations:
            act_name = display_class_mapping.get(a["action"], a["action"])
            if a["duration"] >= self.action_thresholds.get(act_name, min_dur):
                filtered.append({"type": "long_duration", "action": a["action"], "start_time": a["start_time"], "end_time": a["end_time"], "duration": a["duration"], "display_action": act_name})
        
        if len(self.action_durations) > 1:
            td = self.action_durations[-1]["end_time"] - self.action_durations[0]["start_time"]
            tc = len(self.action_durations) - 1
            if td > 0 and (tc / td) * 60 > max_trans:
                filtered.append({"type": "high_transition", "transition_rate": (tc / td) * 60, "start_time": self.action_durations[0]["start_time"], "end_time": self.action_durations[-1]["end_time"]})
        filtered.sort(key=lambda x: x["start_time"])
        return filtered

    def save_all_data(self):
        if not self.action_durations: return
        
        if self.has_skipped:
            print(f"[Skip guard] {self.video_id}  thread detected a progress seek; this replay will not be saved.")
            def post_skip_ui():
                if not self.is_multi_video:
                    messagebox.showinfo("Sandbox Mode Triggered", "This playback included a seek action and will only be shown in the live panel.\nThe data will not be saved to history or daily summaries.")
            self.root.after(0, post_skip_ui)
            return

        today_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d"))
        os.makedirs(today_dir, exist_ok=True)
        ts = time.strftime("%H%M%S", time.localtime(self.session_start_time)) if self.session_start_time else time.strftime("%H%M%S")
        
        suffix = f"_{self.video_id}" if self.is_multi_video or self.video_id != "Main Console" else ""
        task_id = f"#JOB-{ts}{suffix}"
        
        data = []
        for a in self.action_durations:
            st = str(timedelta(seconds=int(a["start_time"]))) if a["start_time"] < 86400 else time.strftime("%H:%M:%S", time.localtime(a["start_time"]))
            et = str(timedelta(seconds=int(a["end_time"]))) if a["end_time"] < 86400 else time.strftime("%H:%M:%S", time.localtime(a["end_time"]))
            data.append({"Behavior": display_class_mapping.get(a["action"], a["action"]), "Start Time": st, "End Time": et, "Duration (s)": round(a["duration"], 2)})
        raw_path = os.path.join(today_dir, f"{task_id}_behavior_stream.xlsx")
        pd.DataFrame(data).to_excel(raw_path, index=False)
        
        filtered = self.analyze_and_filter_results(self.min_duration_threshold_val, self.max_transition_rate_val)
        alert_path = ""
        if filtered:
            alert_path = os.path.join(today_dir, f"{task_id}_behavior_alerts.xlsx")
            ad = []
            for seg in filtered:
                st = str(timedelta(seconds=int(seg["start_time"]))) if seg["start_time"]<86400 else time.strftime("%H:%M:%S", time.localtime(seg["start_time"]))
                et = str(timedelta(seconds=int(seg["end_time"]))) if seg["end_time"]<86400 else time.strftime("%H:%M:%S", time.localtime(seg["end_time"]))
                if seg["type"] == "long_duration": ad.append({"Alert Type": "Long behavior duration", "Behavior": seg["display_action"], "Time Range": f"{st} - {et}", "Recommended Action": "Check for illness or stereotyped behavior"})
                else: ad.append({"Alert Type": "Frequent behavior switching", "Behavior": "Rapid mixed-behavior changes", "Time Range": f"{st} - {et}", "Recommended Action": f"Rate {seg['transition_rate']:.1f}/min; check estrus or courtship behavior"})
            pd.DataFrame(ad).to_excel(alert_path, index=False)

        real_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.session_start_time)) if self.session_start_time and self.session_start_time > 86400 else f"{time.strftime('%Y-%m-%d')} Replay"
        real_end = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with db_lock:
            conn = sqlite3.connect('history.db')
            c = conn.cursor()
            
            c.execute("INSERT INTO analysis_records (task_id, deer_id, session_start, session_end, has_alert, excel_raw, excel_alert, img_timeline, img_freq, img_pie) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (task_id, self.deer_id, real_start, real_end, 1 if filtered else 0, raw_path, alert_path, "", "", ""))
                      
            if self.action_durations_total:
                total_valid_seconds = sum(self.action_durations_total.values())
                if total_valid_seconds >= 1800:
                    task_date = real_start.split(" ")[0] if "Replay" not in real_start else time.strftime('%Y-%m-%d')
                    for act_name, duration in self.action_durations_total.items():
                        freq = self.action_frequency_total.get(act_name, 1)
                        c.execute("SELECT total_duration, frequency FROM daily_summary WHERE date=? AND deer_id=? AND action_name=?", (task_date, self.deer_id, act_name))
                        row = c.fetchone()
                        if row:
                            new_dur = row[0] + duration
                            new_freq = row[1] + freq
                            c.execute("UPDATE daily_summary SET total_duration=?, frequency=? WHERE date=? AND deer_id=? AND action_name=?", (new_dur, new_freq, task_date, self.deer_id, act_name))
                        else:
                            c.execute("INSERT INTO daily_summary (date, deer_id, action_name, total_duration, frequency, percentage) VALUES (?, ?, ?, ?, ?, 0)", (task_date, self.deer_id, act_name, duration, freq))
                    
                    c.execute("SELECT SUM(total_duration) FROM daily_summary WHERE date=? AND deer_id=?", (task_date, self.deer_id))
                    total_day_dur = c.fetchone()[0] or 1
                    c.execute("UPDATE daily_summary SET percentage = round((total_duration / ?) * 100, 2) WHERE date=? AND deer_id=?", (total_day_dur, task_date, self.deer_id))
                else:
                    print(f"[Data guard] Task {task_id} ran for only {total_valid_seconds:.1f} s and was excluded from the baseline summary.")
            
            conn.commit()
            conn.close()
            
        if hasattr(self, '_silent_checkpoint'):
            self._silent_checkpoint()
            
        def post_analysis_ui():
            if hasattr(self.root, "refresh_history_fn"): self.root.refresh_history_fn()
            if not self.is_multi_video:
                if hasattr(self.root, "view_charts_fn"): self.root.view_charts_fn(raw_path, task_id)
                if filtered: messagebox.showwarning("Musk Deer Alert System", f"The system detected {len(filtered)} alert segment(s).\nAn alert sheet has been generated for review.")

        self.root.after(0, post_analysis_ui)

    def show_threshold_settings(self):
        win = ctk.CTkToplevel(self.root)
        win.title("Alert Settings")
        win.geometry("500x600")
        win.configure(fg_color=COLOR_BG_MAIN)
        set_window_icon(win)
        force_focus(win)
        win.transient(self.root); win.grab_set()
        
        self.min_dur = tk.DoubleVar(value=self.min_duration_threshold_val)
        self.max_trans = tk.DoubleVar(value=self.max_transition_rate_val)
        
        if not hasattr(self, 'action_duration_thresholds'):
            self.action_duration_thresholds = {display_class_mapping.get(v, v): tk.DoubleVar(value=self.action_thresholds.get(display_class_mapping.get(v, v), 5.0)) for v in self.class_mapping.values()}
            
        ctk.CTkLabel(win, text="Alert Threshold Settings", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=25, pady=(25,10))
        gf = ctk.CTkFrame(win, fg_color=COLOR_PANEL, corner_radius=8)
        gf.pack(fill="x", padx=25, pady=10)
        
        f1 = ctk.CTkFrame(gf, fg_color="transparent"); f1.pack(fill="x", padx=20, pady=15)
        ctk.CTkLabel(f1, text="Long-duration alert (s)", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=13)).pack(side="left")
        ctk.CTkEntry(f1, textvariable=self.min_dur, width=70, border_color=COLOR_BORDER, fg_color=COLOR_BG_MAIN).pack(side="right")
        
        f2 = ctk.CTkFrame(gf, fg_color="transparent"); f2.pack(fill="x", padx=20, pady=(0, 15))
        ctk.CTkLabel(f2, text="Rapid-switch alert (events/min)", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=13)).pack(side="left")
        ctk.CTkEntry(f2, textvariable=self.max_trans, width=70, border_color=COLOR_BORDER, fg_color=COLOR_BG_MAIN).pack(side="right")

        ctk.CTkLabel(win, text="Per-behavior duration limits (s)", font=ctk.CTkFont(weight="bold", size=14), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=25, pady=(15,5))
        sf = ctk.CTkScrollableFrame(win, fg_color=COLOR_PANEL, corner_radius=8)
        sf.pack(fill="both", expand=True, padx=25, pady=5)
        
        for k in self.action_duration_thresholds.keys():
            frm = ctk.CTkFrame(sf, fg_color="transparent"); frm.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(frm, text=f"{k}", width=120, anchor="w", text_color=COLOR_TEXT_SUB).pack(side="left")
            ctk.CTkEntry(frm, textvariable=self.action_duration_thresholds[k], width=60, border_color=COLOR_BORDER, fg_color=COLOR_BG_MAIN).pack(side="right", padx=10)

        def save():
            try:
                self.min_duration_threshold_val, self.max_transition_rate_val = self.min_dur.get(), self.max_trans.get()
                self.action_thresholds = {k: v.get() for k, v in self.action_duration_thresholds.items()}
                win.destroy(); messagebox.showinfo("Success", "Thresholds have been applied.")
            except: messagebox.showerror("Error", "Please enter numeric values.")
            
        bf = ctk.CTkFrame(win, fg_color="transparent")
        bf.pack(fill="x", padx=25, pady=20)
        ctk.CTkButton(bf, text="Save Settings", command=save, fg_color=COLOR_PRIMARY, text_color="white", font=ctk.CTkFont(weight="bold", size=14), height=40).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(bf, text="Cancel", command=win.destroy, fg_color="transparent", border_width=1, border_color=COLOR_TEXT_SUB, text_color=COLOR_TEXT_SUB, height=40).pack(side="right", expand=True, padx=5)

# === 3. Special mode: multi-view identity assignment ===
def start_multi_video_analysis():
    global root, multi_processor_instances
    num = simpledialog.askinteger("Multi-View Matrix", "How many monitoring panes should be created? (for example, 4)", parent=root, minvalue=1, maxvalue=8)
    if not num: return
    gw = ctk.CTkToplevel(root)
    gw.title("Eco-Intelligent Monitoring Matrix")
    gw.geometry("1450x900")
    gw.configure(fg_color=COLOR_BG_MAIN)
    set_window_icon(gw)
    force_focus(gw) 

    cols = 2 if num > 1 else 1; rows = math.ceil(num / cols)
    for i in range(rows): gw.grid_rowconfigure(i, weight=1, uniform="rg")
    for j in range(cols): gw.grid_columnconfigure(j, weight=1, uniform="cg")
    
    for idx in range(num):
        r, c = divmod(idx, cols); cid = f"Camera_{idx+1}"
        cf = ctk.CTkFrame(gw, corner_radius=15, fg_color=COLOR_PANEL, border_color=COLOR_BORDER, border_width=1)
        cf.grid(row=r, column=c, sticky="nsew", padx=10, pady=10)
        cf.grid_columnconfigure(0, weight=7, uniform="cin"); cf.grid_columnconfigure(1, weight=3, uniform="cin"); cf.grid_rowconfigure(0, weight=1)
        
        vl = ctk.CTkLabel(cf, text=f"{cid} Standby", bg_color="black", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=15))
        vl.grid(row=0, column=0, sticky="nsew", padx=(10,0), pady=10)
        
        sf = ctk.CTkFrame(cf, fg_color="transparent")
        sf.grid(row=0, column=1, sticky="nsew", padx=15, pady=10)
        
        p = VideoProcessor(gw, video_id=cid, is_multi_video=True)
        multi_processor_instances[cid] = p; p.video_label = vl
        
        ctk.CTkLabel(sf, text=f"{cid} Controls", font=ctk.CTkFont(weight="bold", size=15), text_color="white").pack(pady=(5,5))
        
        bind_lbl = ctk.CTkLabel(sf, text="No deer assigned", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_SUB)
        bind_lbl.pack(pady=(0, 10))
        
        def load_v(vp=p, d_lbl=bind_lbl):
            with db_lock:
                conn = sqlite3.connect('history.db')
                c = conn.cursor()
                c.execute("SELECT deer_id FROM deer_info ORDER BY deer_id")
                deer_ids = [row[0] for row in c.fetchall()]
                conn.close()

            if not deer_ids:
                messagebox.showwarning("No Profiles", "No deer profiles are available.\nPlease create a deer profile from the Deer Profiles module in the sidebar.")
                return

            bind_win = ctk.CTkToplevel(gw)
            bind_win.title("Pane Identity Assignment")
            bind_win.geometry("350x250")
            bind_win.configure(fg_color=COLOR_BG_MAIN)
            set_window_icon(bind_win)
            force_focus(bind_win)
            bind_win.transient(gw); bind_win.grab_set()

            ctk.CTkLabel(bind_win, text=f"Please select {vp.video_id}'s target deer ID", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(25, 15))
            
            id_var = tk.StringVar(value=deer_ids[0])
            combo = ctk.CTkComboBox(bind_win, variable=id_var, values=deer_ids, state="readonly", fg_color=COLOR_PANEL, border_color=COLOR_BORDER)
            combo.pack(pady=10)

            def confirm_binding():
                selected_id = id_var.get()
                vp.deer_id = selected_id
                d_lbl.configure(text=f"Assigned: {selected_id}", text_color=COLOR_PRIMARY)
                bind_win.destroy()
                
                ans = messagebox.askquestion("Video Source", f"Assigned deer ID {selected_id}.\nLoad a local video file? Choose No to use the live camera.")
                fp = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv")]) if ans == 'yes' else 0
                if fp or ans == 'no':
                    vp.init_camera(fp)

            ctk.CTkButton(bind_win, text="Confirm and Connect", command=confirm_binding, fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER).pack(pady=20)
            
        ctk.CTkButton(sf, text="Connect Source", command=load_v, height=35, fg_color="#1E2923", border_color=COLOR_BORDER, border_width=1).pack(fill="x", pady=5)
        
        btn = ctk.CTkButton(sf, text="Start Analysis", fg_color=COLOR_PRIMARY, text_color="white", font=ctk.CTkFont(weight="bold"), command=p.toggle_detection, height=35)
        btn.pack(fill="x", pady=5)
        p.start_stop_button = btn 
        
        # --- Multi-thread progress and speed controls ---
        cp = ctk.CTkFrame(sf, fg_color="transparent")
        
        pvar = tk.DoubleVar(value=0)
        pslider = ctk.CTkSlider(cp, variable=pvar, progress_color=COLOR_PRIMARY, button_color="white", height=14)
        pslider.pack(side="left", fill="x", expand=True, padx=(0, 5))
        # UI event bindings
        pslider.bind("<Button-1>", p._on_slider_drag_start)
        pslider.bind("<ButtonRelease-1>", p._on_slider_release)
        
        svar = ctk.StringVar(value="1.0x")
        scb = ctk.CTkComboBox(cp, variable=svar, values=["0.5x", "1.0x", "1.5x", "2.0x", "3.0x"], width=65, height=24)
        scb.pack(side="right")
        svar.trace_add("write", p._on_speed_change)
        
        p.control_panel = cp
        p.progress_slider = pslider
        p.speed_var = svar
        # ----------------------------------

        al = ctk.CTkLabel(sf, text="Waiting", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_TEXT_SUB)
        al.pack(pady=(10,5)); p.action_label = al
        
        ctk.CTkLabel(sf, text="Live Behavior Summary", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(5,0))
        slf = ctk.CTkScrollableFrame(sf, fg_color="transparent")
        slf.pack(fill="both", expand=True)
        p.stats_container = slf
        
    gw.protocol("WM_DELETE_WINDOW", lambda: [p.toggle_detection() for p in multi_processor_instances.values() if p.is_detecting] or multi_processor_instances.clear() or gw.destroy())

# === 4. Special mode: batch identity assignment ===
def quick_video_analysis():
    global VIDEO_FILE_PATH, model_path, root
    if not (path := filedialog.askopenfilename(title="Select a video for batch analysis", filetypes=[("Video", "*.mp4 *.avi *.mkv")])): return
    VIDEO_FILE_PATH = path
    if not model_path:
        if not (model_path := filedialog.askopenfilename(title="Load YOLO model", filetypes=[("YOLO Model", "*.pt")])): return

    with db_lock:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT deer_id FROM deer_info ORDER BY deer_id")
        deer_ids = [row[0] for row in c.fetchall()]
        conn.close()

    if not deer_ids:
        messagebox.showwarning("No Profiles", "No deer profiles are available.\nPlease create a deer profile from the Deer Profiles module in the sidebar.")
        return

    bind_win = ctk.CTkToplevel(root)
    bind_win.title("Batch Identity Assignment")
    bind_win.geometry("350x250")
    bind_win.configure(fg_color=COLOR_BG_MAIN)
    set_window_icon(bind_win)
    force_focus(bind_win)
    bind_win.transient(root); bind_win.grab_set()

    ctk.CTkLabel(bind_win, text="Select the deer ID for this video", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(25, 15))
    
    id_var = tk.StringVar(value=deer_ids[0])
    combo = ctk.CTkComboBox(bind_win, variable=id_var, values=deer_ids, state="readonly", fg_color=COLOR_PANEL, border_color=COLOR_BORDER)
    combo.pack(pady=10)

    def confirm_binding():
        selected_id = id_var.get()
        bind_win.destroy()
        messagebox.showinfo("Started", f"Assigned deer ID {selected_id}.\nBatch analysis has started. Results will be saved to the History Library.")
        threading.Thread(target=perform_quick_analysis_with_filtering, args=(selected_id,), daemon=True).start()

    ctk.CTkButton(bind_win, text="Confirm and Start", command=confirm_binding, fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER).pack(pady=20)

def perform_quick_analysis_with_filtering(target_deer_id):
    global VIDEO_FILE_PATH, model_path, root, processor
    tp = VideoProcessor(root, video_id="Offline Batch")
    tp.deer_id = target_deer_id  
    
    tp.session_start_time = time.time()
    if hasattr(processor, 'min_duration_threshold_val'):
        tp.min_duration_threshold_val = processor.min_duration_threshold_val
        tp.max_transition_rate_val = processor.max_transition_rate_val
        tp.action_thresholds = processor.action_thresholds.copy()
    tp.load_model()
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    tf, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30
    
    pw = ctk.CTkToplevel()
    pw.title("Fast Batch Analysis")
    pw.geometry("450x180"); pw.configure(fg_color=COLOR_BG_MAIN)
    set_window_icon(pw)
    force_focus(pw) 

    ctk.CTkLabel(pw, text="Analyzing the monitoring video...", font=ctk.CTkFont(weight="bold", size=16), text_color=COLOR_PRIMARY).pack(pady=(25,15))
    pvar = tk.DoubleVar(); ctk.CTkProgressBar(pw, variable=pvar, width=350, progress_color=COLOR_PRIMARY).pack()
    il = ctk.CTkLabel(pw, text="Preparing frames...", text_color=COLOR_TEXT_SUB)
    il.pack(pady=15)
    
    st, fi = 0.0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fi += 1
        if fi % max(1, int(tf / 100)) == 0:
            pvar.set(fi / tf); il.configure(text=f"Progress: frame {fi} / {tf}"); pw.update()
        ct = fi / fps
        if fi % 5 == 0:
            res = tp.model(frame, verbose=False)
            detected_act = None
            if res[0].boxes:
                box = max(res[0].boxes, key=lambda b: b.conf[0].item())
                if box.conf[0].item() >= ACTION_THRESHOLD:
                    detected_act = tp.class_mapping.get(int(box.cls[0]), "Unknown")
            
            if detected_act == tp.current_action:
                tp.pending_action = None
                tp.pending_start_time = None
            else:
                if tp.pending_action != detected_act or tp.pending_start_time is None:
                    tp.pending_action = detected_act
                    tp.pending_start_time = ct
                else:
                    if ct - tp.pending_start_time >= tp.action_tolerance:
                        tp._close_current_action(tp.pending_start_time)
                        tp.current_action = detected_act
                        tp.action_start_time = tp.pending_start_time
                        tp.pending_action = None
                        tp.pending_start_time = None
                        
    tp._close_current_action(fi / fps)
    cap.release(); pw.destroy()
    tp.save_all_data()

# === 5. GUI assembly ===
def create_gui():
    global root, processor, deer_id
    root = ctk.CTk()
    root.title("MuskDeer Monitor | Intelligent Behavior Monitoring")
    root.geometry("1400x900")
    root.minsize(1200, 780)
    root.configure(fg_color=COLOR_BG_MAIN)
    
    set_window_icon(root) 

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    
    selected_records = {}

    sidebar = ctk.CTkFrame(root, width=260, corner_radius=0, fg_color=COLOR_BG_SIDEBAR)
    sidebar.grid(row=0, column=0, sticky="nsew")
    sidebar.grid_rowconfigure(8, weight=1)
    
    ctk.CTkLabel(sidebar, text="MuskDeer\nMonitor", font=ctk.CTkFont(family="Arial Black", size=26, weight="bold"), text_color=COLOR_PRIMARY, justify="left").pack(anchor="w", padx=25, pady=(40, 5))
    ctk.CTkLabel(sidebar, text="Intelligent Husbandry Assistant", font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_SUB).pack(anchor="w", padx=25, pady=(0,35))
    
    def switch_tab(tab_name):
        history_view.grid_forget()
        realtime_view.grid_forget()
        nav_realtime.configure(fg_color="transparent", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=14))
        nav_history.configure(fg_color="transparent", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=14))
        
        if tab_name == "realtime":
            realtime_view.grid(row=0, column=0, sticky="nsew")
            nav_realtime.configure(fg_color=COLOR_PANEL, text_color=COLOR_PRIMARY, font=ctk.CTkFont(size=14, weight="bold"))
        elif tab_name == "history":
            history_view.grid(row=0, column=0, sticky="nsew")
            nav_history.configure(fg_color=COLOR_PANEL, text_color=COLOR_PRIMARY, font=ctk.CTkFont(size=14, weight="bold"))
            search_id_var.set("")
            search_status_var.set("Alert status (all)")
            load_history_data()

    nav_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
    nav_frame.pack(fill="x", padx=15, pady=5)
    
    nav_realtime = ctk.CTkButton(nav_frame, text=" 👁  Realtime Monitor", command=lambda: switch_tab("realtime"), anchor="w", fg_color=COLOR_PANEL, text_color=COLOR_PRIMARY, font=ctk.CTkFont(size=14, weight="bold"), height=45)
    nav_realtime.pack(fill="x", pady=2)
    ctk.CTkButton(nav_frame, text=" 🎛  Multi-View Matrix", command=start_multi_video_analysis, anchor="w", fg_color="transparent", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=14), hover_color=COLOR_PANEL, height=45).pack(fill="x", pady=2)
    ctk.CTkButton(nav_frame, text=" ⚡  Video Batch", command=quick_video_analysis, anchor="w", fg_color="transparent", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=14), hover_color=COLOR_PANEL, height=45).pack(fill="x", pady=2)
    nav_history = ctk.CTkButton(nav_frame, text=" 📊  History Library", command=lambda: switch_tab("history"), anchor="w", fg_color="transparent", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=14), hover_color=COLOR_PANEL, height=45)
    nav_history.pack(fill="x", pady=2)

    # ================= Deer profile management module =================
    def manage_deer_profiles():
        win = ctk.CTkToplevel(root)
        win.title("Deer Profile Management")
        win.geometry("400x500")
        win.configure(fg_color=COLOR_BG_MAIN)
        set_window_icon(win)
        force_focus(win)
        win.transient(root); win.grab_set()

        ctk.CTkLabel(win, text="Register Deer Profile", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(20, 10))

        add_frame = ctk.CTkFrame(win, fg_color="transparent")
        add_frame.pack(fill="x", padx=20, pady=10)
        
        new_id_var = tk.StringVar()
        entry = ctk.CTkEntry(add_frame, textvariable=new_id_var, placeholder_text="Enter ID (e.g., 2.1.6)", width=200, height=35, fg_color=COLOR_PANEL, border_color=COLOR_BORDER)
        entry.pack(side="left", padx=(0, 10))

        def add_id():
            new_id = new_id_var.get().strip()
            if not new_id: return
            
            if not re.match(r"^\d+\.\d+\.\d+$", new_id):
                messagebox.showerror("Invalid ID Format", "Deer IDs must use the standard format, such as 2.1.6.\nPlease check and try again.")
                return
                
            with db_lock:
                conn = sqlite3.connect('history.db')
                c = conn.cursor()
                try:
                    c.execute("INSERT INTO deer_info (deer_id, added_date) VALUES (?, date('now'))", (new_id,))
                    conn.commit()
                    new_id_var.set("")
                    messagebox.showinfo("Profile Created", f"Deer ID {new_id} has been created.")
                except sqlite3.IntegrityError:
                    messagebox.showerror("Duplicate Profile", "This deer ID already exists in the profile library.")
                finally:
                    conn.close()
            refresh_list()

        ctk.CTkButton(add_frame, text="Register", command=add_id, width=80, height=35, fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER).pack(side="left")

        list_frame = ctk.CTkScrollableFrame(win, fg_color=COLOR_PANEL, corner_radius=8)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        def delete_id(did):
            if messagebox.askyesno("Delete Profile?", f"Delete deer profile {did}?\nExisting history records will not be deleted."):
                with db_lock:
                    conn = sqlite3.connect('history.db')
                    c = conn.cursor()
                    c.execute("DELETE FROM deer_info WHERE deer_id=?", (did,))
                    conn.commit()
                    conn.close()
                refresh_list()

        def refresh_list():
            for w in list_frame.winfo_children(): w.destroy()
            with db_lock:
                conn = sqlite3.connect('history.db')
                c = conn.cursor()
                c.execute("SELECT deer_id FROM deer_info ORDER BY deer_id")
                rows = c.fetchall()
                conn.close()
            for row in rows:
                did = row[0]
                row_f = ctk.CTkFrame(list_frame, fg_color="transparent")
                row_f.pack(fill="x", pady=5)
                ctk.CTkLabel(row_f, text=f"🦌 {did}", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(side="left", padx=10)
                ctk.CTkButton(row_f, text="Delete", width=50, fg_color="transparent", border_width=1, border_color=COLOR_DANGER, text_color=COLOR_DANGER, command=lambda d=did: delete_id(d)).pack(side="right", padx=10)

        refresh_list()

    nav_profile = ctk.CTkButton(nav_frame, text=" 🦌  Deer Profiles", command=manage_deer_profiles, anchor="w", fg_color="transparent", text_color=COLOR_TEXT_SUB, font=ctk.CTkFont(size=14), hover_color=COLOR_PANEL, height=45)
    nav_profile.pack(fill="x", pady=2)
    # ========================================================

    ctk.CTkFrame(sidebar, height=1, fg_color=COLOR_BORDER).pack(fill="x", padx=25, pady=30)
    
    ctk.CTkLabel(sidebar, text="AI Controller", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_SUB, anchor="w").pack(fill="x", padx=25, pady=(0, 10))
    
    def load_m():
        global model_path
        if model_path := filedialog.askopenfilename(filetypes=[("AI Weights", "*.pt")]): messagebox.showinfo("Success", f"Model loaded.")
        
    def load_s():
        with db_lock:
            conn = sqlite3.connect('history.db')
            c = conn.cursor()
            c.execute("SELECT deer_id FROM deer_info ORDER BY deer_id")
            deer_ids = [row[0] for row in c.fetchall()]
            conn.close()

        if not deer_ids:
            messagebox.showwarning("No Profiles", "No deer profiles are available.\nPlease create a deer profile from the Deer Profiles module in the sidebar.")
            return

        bind_win = ctk.CTkToplevel(root)
        bind_win.title("Main Console Identity Assignment")
        bind_win.geometry("350x250")
        bind_win.configure(fg_color=COLOR_BG_MAIN)
        set_window_icon(bind_win)
        force_focus(bind_win)
        bind_win.transient(root); bind_win.grab_set()

        ctk.CTkLabel(bind_win, text="Select target deer ID", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(25, 15))
        
        id_var = tk.StringVar(value=deer_ids[0])
        combo = ctk.CTkComboBox(bind_win, variable=id_var, values=deer_ids, state="readonly", fg_color=COLOR_PANEL, border_color=COLOR_BORDER)
        combo.pack(pady=10)

        def confirm_binding():
            global deer_id
            deer_id = id_var.get()
            processor.deer_id = deer_id  
            deer_id_label.configure(text=deer_id) 
            
            bind_win.destroy()
            
            ans = messagebox.askquestion("Video Source", f"Assigned deer ID {deer_id}.\nLoad a local video file? Choose No to use the live camera.")
            fp = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv")]) if ans == 'yes' else 0
            if fp or ans == 'no':
                processor.init_camera(fp)

        ctk.CTkButton(bind_win, text="Confirm and Connect Monitor", command=confirm_binding, fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER).pack(pady=20)

    btn_style = {"fg_color": "transparent", "border_width": 1, "border_color": COLOR_BORDER, "text_color": COLOR_TEXT_MAIN, "hover_color": COLOR_PANEL, "height": 40}
    ctk.CTkButton(sidebar, text="Load AI Model          >", command=load_m, font=ctk.CTkFont(size=13), **btn_style).pack(fill="x", padx=25, pady=6)
    ctk.CTkButton(sidebar, text="Connect Video Source   >", command=load_s, font=ctk.CTkFont(size=13), **btn_style).pack(fill="x", padx=25, pady=6)
    ctk.CTkButton(sidebar, text="Behavior Alert Settings", command=lambda: processor.show_threshold_settings(), font=ctk.CTkFont(size=13), **btn_style).pack(fill="x", padx=25, pady=6)
    
    bottom_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
    bottom_frame.pack(side="bottom", fill="x", padx=25, pady=30)
    
    start_btn = ctk.CTkButton(bottom_frame, text="Start Detection", command=lambda: processor.toggle_detection(), font=ctk.CTkFont(size=15, weight="bold"), fg_color=COLOR_PRIMARY, text_color="white", hover_color=COLOR_PRIMARY_HOVER, height=45)
    start_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))
    
    def on_stop_click():
        if processor.is_detecting:
            processor.toggle_detection()
            start_btn.configure(fg_color=COLOR_PRIMARY, text="Start Detection", state="normal")
            
    stop_btn = ctk.CTkButton(bottom_frame, text="⏹", command=on_stop_click, width=45, height=45, fg_color=COLOR_DANGER, hover_color="#B91C1C", text_color="white", font=ctk.CTkFont(size=18))
    stop_btn.pack(side="right")

    # --------------------- Right-side container ---------------------
    right_container = ctk.CTkFrame(root, fg_color="transparent")
    right_container.grid(row=0, column=1, sticky="nsew", padx=30, pady=30)
    right_container.grid_rowconfigure(0, weight=1); right_container.grid_columnconfigure(0, weight=1)

    realtime_view = ctk.CTkFrame(right_container, fg_color="transparent")
    
    cards_frame = ctk.CTkFrame(realtime_view, fg_color="transparent", height=80)
    cards_frame.pack(fill="x", pady=(0, 20))
    cards_frame.grid_columnconfigure((0,1,2,3), weight=1)
    
    def mk_card(col, title, val, hl=False):
        c = ctk.CTkFrame(cards_frame, corner_radius=12, fg_color=COLOR_PANEL, border_width=1, border_color=COLOR_PRIMARY if hl else COLOR_BORDER)
        c.grid(row=0, column=col, sticky="nsew", padx=8)
        ctk.CTkLabel(c, text=title, font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_SUB).pack(anchor="w", padx=20, pady=(15, 2))
        lbl = ctk.CTkLabel(c, text=val, font=ctk.CTkFont(size=24, weight="bold"), text_color=COLOR_PRIMARY if hl else "white")
        lbl.pack(anchor="w", padx=20, pady=(0, 15))
        return lbl
    
    deer_id_label = mk_card(0, "Target Deer ID", deer_id)
    mk_card(1, "Temperature", f"{temperature} °C")
    mk_card(2, "Relative Humidity", f"{humidity} %")
    mk_card(3, "AI Health Status", health_info, hl=True)

    content_frame = ctk.CTkFrame(realtime_view, fg_color="transparent")
    content_frame.pack(fill="both", expand=True)
    content_frame.grid_columnconfigure(0, weight=7); content_frame.grid_columnconfigure(1, weight=3)
    content_frame.grid_rowconfigure(0, weight=1)
    
    vc = ctk.CTkFrame(content_frame, corner_radius=15, fg_color=COLOR_PANEL, border_width=1, border_color=COLOR_BORDER)
    vc.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
    vc.grid_rowconfigure(0, weight=1); vc.grid_columnconfigure(0, weight=1)
    # Reserve row 1 for video playback controls
    vc.grid_rowconfigure(1, weight=0)
    
    placeholder_frame = ctk.CTkFrame(vc, fg_color="transparent")
    placeholder_frame.grid(row=0, column=0, sticky="nsew")
    
    pf_inner = ctk.CTkFrame(placeholder_frame, fg_color="transparent")
    pf_inner.pack(expand=True)
    ctk.CTkLabel(pf_inner, text="Waiting for video source...", font=ctk.CTkFont(size=30, weight="bold"), text_color=COLOR_TEXT_SUB).pack(pady=(10, 10))
    ctk.CTkLabel(pf_inner, text="Engine ready | Load a model and connect a video source from the left panel", font=ctk.CTkFont(size=14), text_color="#4B5563").pack()

    video_label = ctk.CTkLabel(vc, text="")
    tag_label = ctk.CTkLabel(vc, text=" REC  CAM-01 BREEDING ROOM ", font=ctk.CTkFont(size=10, weight="bold"), text_color="black", fg_color="white", corner_radius=4)
    tag_label.grid(row=0, column=0, sticky="nw", padx=20, pady=20)
    
    fps_label = ctk.CTkLabel(vc, text="FPS: 0", font=ctk.CTkFont(size=10, weight="bold"), text_color="white", fg_color=COLOR_BG_SIDEBAR, corner_radius=4)
    
    def show_placeholder():
        video_label.grid_forget()
        tag_label.grid_forget()
        fps_label.grid_forget()
        placeholder_frame.grid(row=0, column=0, sticky="nsew")
        
    def hide_placeholder():
        placeholder_frame.grid_forget()
        video_label.grid(row=0, column=0, sticky="nsew")
        tag_label.grid(row=0, column=0, sticky="nw", padx=20, pady=20)
        fps_label.grid(row=0, column=0, sticky="ne", padx=20, pady=20)
        
    root.show_placeholder_fn = show_placeholder

    processor = VideoProcessor(root, video_id="Main Console")
    processor.deer_id = deer_id 
    processor.video_label = video_label
    processor.fps_label = fps_label

    # ====== Main-console video playback and progress controls ======
    control_panel = ctk.CTkFrame(vc, fg_color="transparent", height=45)
    
    progress_var = tk.DoubleVar(value=0)
    progress_slider = ctk.CTkSlider(control_panel, variable=progress_var, progress_color=COLOR_PRIMARY, button_color="white", button_hover_color=COLOR_TEXT_MAIN)
    progress_slider.pack(side="left", fill="x", expand=True, padx=(20, 10))
    # UI event bindings
    progress_slider.bind("<Button-1>", processor._on_slider_drag_start)
    progress_slider.bind("<ButtonRelease-1>", processor._on_slider_release)
    
    speed_var = ctk.StringVar(value="1.0x")
    speed_cb = ctk.CTkComboBox(control_panel, variable=speed_var, values=["0.5x", "1.0x", "1.5x", "2.0x", "3.0x"], width=80, fg_color=COLOR_BG_MAIN, border_color=COLOR_BORDER)
    speed_cb.pack(side="right", padx=(0, 20))
    speed_var.trace_add("write", processor._on_speed_change)

    processor.control_panel = control_panel
    processor.progress_slider = progress_slider
    processor.speed_var = speed_var
    # ====================================

    log_panel = ctk.CTkFrame(content_frame, fg_color="transparent")
    log_panel.grid(row=0, column=1, sticky="nsew")
    
    action_box = ctk.CTkFrame(log_panel, fg_color=COLOR_PANEL, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
    action_box.pack(fill="x", pady=(0, 15))
    ctk.CTkLabel(action_box, text="Current Behavior", font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_SUB).pack(anchor="w", padx=20, pady=(15, 0))
    action_label = ctk.CTkLabel(action_box, text="Waiting", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_TEXT_SUB)
    action_label.pack(anchor="w", padx=20, pady=(5, 20))
    processor.action_label = action_label
    
    log_list_box = ctk.CTkFrame(log_panel, fg_color=COLOR_PANEL, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
    log_list_box.pack(fill="both", expand=True)
    
    lh = ctk.CTkFrame(log_list_box, fg_color="transparent")
    lh.pack(fill="x", padx=20, pady=15)
    ctk.CTkLabel(lh, text="Live Behavior Summary", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_PRIMARY).pack(side="left")
    
    sf_main = ctk.CTkScrollableFrame(log_list_box, fg_color="transparent")
    sf_main.pack(fill="both", expand=True, padx=10, pady=(0,10))
    processor.stats_container = sf_main

    def custom_start_toggle():
        if not processor.is_detecting:
            processor.toggle_detection()
            if processor.is_detecting: 
                hide_placeholder()
                start_btn.configure(fg_color="#064E3B", text="Running inference...", state="disabled")
    
    start_btn.configure(command=custom_start_toggle)

    history_view = ctk.CTkFrame(right_container, fg_color="transparent")
    
    top_bar = ctk.CTkFrame(history_view, fg_color="transparent")
    top_bar.pack(fill="x", pady=(0, 20))
    
    title_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
    title_frame.pack(side="left")
    ctk.CTkLabel(title_frame, text="History Library", font=ctk.CTkFont(size=22, weight="bold"), text_color="white").pack(anchor="w")
    ctk.CTkLabel(title_frame, text="Search and manage all recorded musk deer behavior analysis tasks", font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_SUB).pack(anchor="w")
    
    def batch_delete_selected():
        to_delete_ids = []
        to_delete_paths = []
        
        for r_id, data in selected_records.items():
            if data["var"].get():
                to_delete_ids.append(r_id)
                to_delete_paths.extend(data["paths"])
                
        if not to_delete_ids:
            return messagebox.showinfo("Notice", "Please select records to delete first.")
            
        if not messagebox.askyesno("Destructive Action", f"Permanently delete the selected {len(to_delete_ids)} analysis record(s)?\nRelated Excel files will be permanently removed from disk."):
            return
            
        with db_lock:
            conn = sqlite3.connect('history.db')
            c = conn.cursor()
            placeholders = ','.join('?' for _ in to_delete_ids)
            c.execute(f"DELETE FROM analysis_records WHERE id IN ({placeholders})", tuple(to_delete_ids))
            conn.commit()
            conn.close()
        
        for p in to_delete_paths:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
                
        load_history_data()

    select_all_var = tk.BooleanVar(value=False)
    def toggle_select_all():
        state = select_all_var.get()
        for data in selected_records.values():
            data["var"].set(state)

    search_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
    search_frame.pack(side="right")
    
    batch_action_bar = ctk.CTkFrame(search_frame, fg_color="transparent")
    batch_action_bar.pack(side="left", padx=(0, 20))
    ctk.CTkCheckBox(batch_action_bar, text="Select All", variable=select_all_var, command=toggle_select_all, width=60, font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_MAIN).pack(side="left", padx=5)
    ctk.CTkButton(batch_action_bar, text="Batch Delete", command=batch_delete_selected, width=70, height=32, fg_color="transparent", border_width=1, border_color=COLOR_DANGER, text_color=COLOR_DANGER, hover_color="#3F1616", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)

    search_status_var = tk.StringVar(value="Alert status (all)")
    status_combo = ctk.CTkComboBox(search_frame, variable=search_status_var, values=["Alert status (all)", "✔ Normal", "⚠ Alert"], width=130, height=36, fg_color=COLOR_PANEL, border_color=COLOR_BORDER, font=ctk.CTkFont(size=12))
    status_combo.pack(side="left", padx=5)

    search_id_var = tk.StringVar()
    search_entry = ctk.CTkEntry(search_frame, textvariable=search_id_var, placeholder_text="Search deer ID or job ID...", width=200, height=36, fg_color=COLOR_PANEL, border_color=COLOR_BORDER, font=ctk.CTkFont(size=12))
    search_entry.pack(side="left", padx=(5, 10))

    ctk.CTkButton(search_frame, text="Search", command=lambda: load_history_data(), width=60, height=36, fg_color="#1E2923", hover_color=COLOR_BORDER).pack(side="left")

    table_container = ctk.CTkFrame(history_view, fg_color=COLOR_PANEL, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
    table_container.pack(fill="both", expand=True)

    th = ctk.CTkFrame(table_container, fg_color="transparent", height=45)
    th.pack(fill="x", pady=(5, 0))
    th.pack_propagate(False)
    h_font = ctk.CTkFont(size=12, weight="bold")
    
    ctk.CTkLabel(th, text="Select", width=40, anchor="center", text_color=COLOR_TEXT_SUB, font=h_font).pack(side="left", padx=(10, 0))
    ctk.CTkLabel(th, text="Job ID", width=150, anchor="center", text_color=COLOR_TEXT_SUB, font=h_font).pack(side="left", padx=10)
    ctk.CTkLabel(th, text="Deer ID", width=100, anchor="center", text_color=COLOR_TEXT_SUB, font=h_font).pack(side="left", padx=10)
    ctk.CTkLabel(th, text="Date", width=120, anchor="center", text_color=COLOR_TEXT_SUB, font=h_font).pack(side="left", padx=10)
    ctk.CTkLabel(th, text="Time Window", width=160, anchor="center", text_color=COLOR_TEXT_SUB, font=h_font).pack(side="left", padx=10)
    ctk.CTkLabel(th, text="Alert Status", width=100, anchor="center", text_color=COLOR_TEXT_SUB, font=h_font).pack(side="left", padx=10)
    ctk.CTkLabel(th, text="Actions", anchor="center", text_color=COLOR_TEXT_SUB, font=h_font).pack(side="right", padx=60)

    ctk.CTkFrame(table_container, height=1, fg_color=COLOR_BORDER).pack(fill="x", padx=10)

    hist_scroll = ctk.CTkScrollableFrame(table_container, fg_color="transparent", corner_radius=0)
    hist_scroll.pack(fill="both", expand=True, padx=5, pady=5)

    def view_interactive_charts(excel_path, task_id):
        if not excel_path or not os.path.exists(excel_path): return messagebox.showerror("Error", "Raw data file was not found.")
        df = pd.read_excel(excel_path)
        if df.empty: return messagebox.showinfo("Notice", "No behavior data is available.")
        
        win = ctk.CTkToplevel(root)
        win.title("Behavior Analytics Dashboard")
        win.geometry("1100x750")
        win.configure(fg_color=COLOR_BG_MAIN)
        set_window_icon(win)
        force_focus(win)

        hdr = ctk.CTkFrame(win, fg_color="transparent", height=60)
        hdr.pack(fill="x", padx=30, pady=(20, 10))
        ctk.CTkLabel(hdr, text="Behavior Analytics Dashboard", font=ctk.CTkFont(size=24, weight="bold"), text_color="white").pack(anchor="w")
        ctk.CTkLabel(hdr, text=f"Job ID: {task_id} | Analysis type: AI-assisted review", font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_SUB).pack(anchor="w")
        
        scr = ctk.CTkScrollableFrame(win, fg_color="transparent")
        scr.pack(fill="both", expand=True, padx=20, pady=10)
        
        actions, durations, start_times, end_times = df['Behavior'].tolist(), df['Duration (s)'].tolist(), df['Start Time'].tolist(), df['End Time'].tolist()
        unique_acts = list(set(actions))
        colors = ['#10B981', '#F59E0B', '#3B82F6', '#8B5CF6', '#EC4899', '#F43F5E', '#64748B']
        cmap = dict(zip(unique_acts, [colors[i % len(colors)] for i in range(len(unique_acts))]))
        
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = COLOR_PANEL
        plt.rcParams['axes.facecolor'] = COLOR_PANEL

        fig1, ax1 = plt.subplots(figsize=(12, 2.5))
        
        def time_str_to_sec(t_str):
            h, m, s = map(float, str(t_str).split(':'))
            return h*3600 + m*60 + s
            
        real_starts = [time_str_to_sec(t) for t in start_times]
        
        bars = ax1.barh([1]*len(durations), durations, left=real_starts, color=[cmap[a] for a in actions], alpha=0.9)
        ax1.set_yticks([]); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False); ax1.spines['bottom'].set_visible(False); ax1.spines['left'].set_visible(False)
        ax1.tick_params(axis='x', colors=COLOR_TEXT_SUB)
        
        annot = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points", bbox=dict(boxstyle="round", fc=COLOR_TEXT_MAIN, ec=COLOR_BG_MAIN), color=COLOR_BG_MAIN, fontsize=10, fontweight='bold')
        annot.set_visible(False)

        def on_hover(event):
            annot.set_visible(False)
            if event.inaxes == ax1:
                for i, bar in enumerate(bars):
                    if bar.contains(event)[0]:
                        annot.xy = (event.xdata, event.ydata)
                        annot.set_text(f"Behavior: {actions[i]}\nDuration: {durations[i]:.1f} s\nStart: {start_times[i]}\nEnd: {end_times[i]}")
                        annot.set_visible(True)
                        fig1.canvas.draw_idle()
                        return
            fig1.canvas.draw_idle()

        fig1.canvas.mpl_connect("motion_notify_event", on_hover)
        
        c1_frame = ctk.CTkFrame(scr, fg_color=COLOR_PANEL, corner_radius=12)
        c1_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(c1_frame, text="Behavior Timeline", font=ctk.CTkFont(weight="bold"), text_color="white").pack(anchor="w", padx=20, pady=(15,0))
        canvas1 = FigureCanvasTkAgg(fig1, master=c1_frame); canvas1.draw(); canvas1.get_tk_widget().pack(pady=(0,15), fill="x", padx=15)

        bottom_charts = ctk.CTkFrame(scr, fg_color="transparent")
        bottom_charts.pack(fill="x", pady=10)
        bottom_charts.grid_columnconfigure((0,1), weight=1)
        
        c2_frame = ctk.CTkFrame(bottom_charts, fg_color=COLOR_PANEL, corner_radius=12)
        c2_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ctk.CTkLabel(c2_frame, text="Behavior Frequency", font=ctk.CTkFont(weight="bold"), text_color="white").pack(anchor="w", padx=20, pady=(15,0))
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        action_counts = df['Behavior'].value_counts()
        ax2.bar(action_counts.index, action_counts.values, color=[cmap[n] for n in action_counts.index], width=0.4)
        ax2.tick_params(axis='x', rotation=0, colors=COLOR_TEXT_SUB); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        canvas2 = FigureCanvasTkAgg(fig2, master=c2_frame); canvas2.draw(); canvas2.get_tk_widget().pack(pady=(0,15), fill="x", padx=15)
        
        c3_frame = ctk.CTkFrame(bottom_charts, fg_color=COLOR_PANEL, corner_radius=12)
        c3_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        ctk.CTkLabel(c3_frame, text="Time Share", font=ctk.CTkFont(weight="bold"), text_color="white").pack(anchor="w", padx=20, pady=(15,0))
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        time_totals = df.groupby('Behavior')['Duration (s)'].sum()
        ax3.pie(time_totals.values, labels=time_totals.index, colors=[cmap[n] for n in time_totals.index], autopct='%1.1f%%', textprops={'fontsize':9, 'color':'white'}, wedgeprops={'width':0.4, 'edgecolor':COLOR_PANEL})
        canvas3 = FigureCanvasTkAgg(fig3, master=c3_frame); canvas3.draw(); canvas3.get_tk_widget().pack(pady=(0,15), fill="x", padx=15)

    root.view_charts_fn = view_interactive_charts

    def open_excel(path):
        if path and os.path.exists(path):
            try: os.startfile(path)
            except: messagebox.showerror("Error", "Unable to open the file.")

    def delete_record(r_id, paths):
        if not messagebox.askyesno("Destructive Action", "Permanently delete this analysis record? Files will be removed from disk."): return
        with db_lock:
            conn = sqlite3.connect('history.db'); c = conn.cursor()
            c.execute("DELETE FROM analysis_records WHERE id=?", (r_id,))
            conn.commit(); conn.close()
        for p in paths:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
        load_history_data()

    def load_history_data():
        for widget in hist_scroll.winfo_children(): widget.destroy()
        
        selected_records.clear()
        select_all_var.set(False)
        
        s_status = search_status_var.get()
        s_id = search_id_var.get()
        
        query = "SELECT * FROM analysis_records WHERE 1=1"
        params = []
        if s_id: 
            query += " AND (task_id LIKE ? OR deer_id LIKE ?)"
            params.extend([f"%{s_id}%", f"%{s_id}%"])
        if s_status == "⚠ Alert": query += " AND has_alert = 1"
        elif s_status == "✔ Normal": query += " AND has_alert = 0"
        query += " ORDER BY id DESC"
        
        with db_lock:
            conn = sqlite3.connect('history.db'); c = conn.cursor()
            c.execute(query, params); records = c.fetchall(); conn.close()
        
        if not records:
            ctk.CTkLabel(hist_scroll, text="No matching records found", text_color=COLOR_TEXT_SUB).pack(pady=50)
            return

        for i, r in enumerate(records):
            r_id, task_id, deer_id_rec, st_t, end_t, has_alert, ex_raw, ex_al, _, _, _ = r
            
            if st_t.endswith("Replay"): date_str, time_str = st_t.split(" ")[0], "Offline analysis/replay"
            else:
                try: date_str, time_str = st_t.split(" ")[0], f"{st_t.split(' ')[1][:5]} - {end_t.split(' ')[1][:5]}"
                except: date_str, time_str = st_t, end_t

            row_bg = "transparent" if i % 2 == 0 else "#141C18"
            row = ctk.CTkFrame(hist_scroll, fg_color=row_bg, height=45, corner_radius=0)
            row.pack(fill="x")
            row.pack_propagate(False) 
            
            chk_var = tk.BooleanVar(value=False)
            selected_records[r_id] = {"var": chk_var, "paths": [ex_raw, ex_al]}
            ctk.CTkCheckBox(row, variable=chk_var, text="", width=40).pack(side="left", padx=(10, 0))
            
            f_font = ctk.CTkFont(size=12, weight="bold")
            ctk.CTkLabel(row, text=task_id, width=150, font=f_font, text_color=COLOR_TEXT_MAIN).pack(side="left", padx=10)
            
            tag_frame = ctk.CTkFrame(row, fg_color="#064E3B", corner_radius=4, width=100, height=24)
            tag_frame.pack(side="left", padx=10); tag_frame.pack_propagate(False)
            ctk.CTkLabel(tag_frame, text=deer_id_rec, font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_PRIMARY).pack(expand=True)
            
            ctk.CTkLabel(row, text=date_str, width=120, font=f_font, text_color=COLOR_TEXT_MAIN).pack(side="left", padx=10)
            ctk.CTkLabel(row, text=time_str, width=160, font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_SUB).pack(side="left", padx=10)
            
            status_color = COLOR_DANGER if has_alert else COLOR_PRIMARY
            status_text = "● Alert" if has_alert else "● Normal"
            ctk.CTkLabel(row, text=status_text, width=100, text_color=status_color, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)
            
            f_action = ctk.CTkFrame(row, fg_color="transparent")
            f_action.pack(side="right", padx=10)
            
            ctk.CTkButton(f_action, text="Dashboard", width=50, height=28, fg_color="#1E2923", hover_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN, font=ctk.CTkFont(size=12), command=lambda e=ex_raw, t=task_id: view_interactive_charts(e, t)).pack(side="left", padx=3)
            ctk.CTkButton(f_action, text="Records", width=65, height=28, fg_color="transparent", border_width=1, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN, font=ctk.CTkFont(size=12), command=lambda e=ex_raw: open_excel(e)).pack(side="left", padx=3)
            
            if has_alert and ex_al:
                ctk.CTkButton(f_action, text="Alert Sheet", width=55, height=28, fg_color="transparent", border_width=1, border_color=COLOR_BORDER, text_color=COLOR_DANGER, font=ctk.CTkFont(size=12), command=lambda e=ex_al: open_excel(e)).pack(side="left", padx=3)
            
            ctk.CTkButton(f_action, text="Delete", width=40, height=28, fg_color="transparent", border_width=1, border_color=COLOR_DANGER, text_color=COLOR_DANGER, font=ctk.CTkFont(size=12), command=lambda i=r_id, paths=[ex_raw, ex_al]: delete_record(i, paths)).pack(side="left", padx=3)

    root.refresh_history_fn = load_history_data
    switch_tab("realtime")
    root.mainloop()

if __name__ == "__main__":
    create_gui()