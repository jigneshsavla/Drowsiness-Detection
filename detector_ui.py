import cv2
import dlib
import numpy as np
import time
import winsound
import threading
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.spatial import distance as dist
from imutils import face_utils
from collections import deque
import customtkinter as ctk  # pip install customtkinter
from PIL import Image, ImageTk

# Set appearance mode and color theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

# ==============================
# Step 1: EAR & MAR Calculation
# ==============================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # Vertical
    B = dist.euclidean(mouth[4], mouth[8])   # Vertical
    C = dist.euclidean(mouth[0], mouth[6])   # Horizontal
    mar = (A + B) / (2.0 * C)
    return mar

# ==============================
# Step 2: Activity Logging
# ==============================

class ActivityLogger:
    def __init__(self):
        self.log_dir = "drowsiness_logs"
        self.ensure_log_directory()
        self.log_file = self.create_log_file()
        self.write_header()
        
        # Statistics tracking
        self.drowsiness_count = 0
        self.yawn_count = 0
        self.total_drowsy_time = 0
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.blink_times = []
        self.blink_rate_history = deque(maxlen=60)  # Store 60 data points for blink rate
        self.drowsy_events = []
        self.yawn_events = []
        self.session_start_time = time.time()
        
    def ensure_log_directory(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def create_log_file(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return open(f"{self.log_dir}/drowsiness_log_{timestamp}.csv", "w")
    
    def write_header(self):
        self.log_file.write("timestamp,event_type,duration,ear,mar\n")
        self.log_file.flush()
        
    def log_event(self, event_type, duration=0, ear=0, mar=0):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"{timestamp},{event_type},{duration:.2f},{ear:.3f},{mar:.3f}\n")
        self.log_file.flush()
        
        # Update statistics
        if event_type == "drowsiness":
            self.drowsiness_count += 1
            self.total_drowsy_time += duration
            self.drowsy_events.append((timestamp, duration))
        elif event_type == "yawn":
            self.yawn_count += 1
            self.yawn_events.append(timestamp)
        elif event_type == "blink":
            self.blink_count += 1
            current_time = time.time()
            time_since_last = current_time - self.last_blink_time
            self.last_blink_time = current_time
            
            if 0.1 < time_since_last < 4.0:  # Valid blink interval
                self.blink_times.append(time_since_last)
                # Calculate blink rate (blinks per minute)
                if len(self.blink_times) >= 3:
                    recent_blinks = self.blink_times[-10:] if len(self.blink_times) >= 10 else self.blink_times
                    avg_time_between = sum(recent_blinks) / len(recent_blinks)
                    blink_rate = 60.0 / avg_time_between if avg_time_between > 0 else 0
                    self.blink_rate_history.append(blink_rate)
    
    def get_blink_rate(self):
        if not self.blink_rate_history:
            return 0
        return self.blink_rate_history[-1]
    
    def get_average_blink_rate(self):
        if not self.blink_rate_history:
            return 0
        return sum(self.blink_rate_history) / len(self.blink_rate_history)
    
    def get_session_duration(self):
        return time.time() - self.session_start_time
        
    def close(self):
        self.log_file.close()

# ==============================
# Step 3: Modern UI Application
# ==============================

class DrowsinessDetectionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Alertness Monitor")
        self.root.geometry("1280x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup variables
        self.running = False
        self.cap = None
        self.logger = ActivityLogger()
        self.detector = dlib.get_frontal_face_detector()
        self.landmarks_path = r"F:\drowsiness_detector\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.landmarks_path)
        self.frame_skip = 2
        self.frame_count = 0
        
        # Facial indices
        (self.left_start, self.left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.right_start, self.right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mouth_start, self.mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        
        # Detection thresholds
        self.EAR_THRESHOLD = 0.25
        self.BLINK_THRESHOLD = 0.3
        self.MAR_THRESHOLD = 0.6
        self.DROWSINESS_DURATION = 2.0
        
        # Alert state variables
        self.beep_active = False
        self.drowsy_start_time = None
        self.blinking = False
        self.drowsy_duration = 0
        self.text_overlay = {
            "drowsiness_alert": False,
            "eyes_status": "Eyes Open",
            "yawning": False,
        }
        
        # Create UI
        self.create_ui()
        self.update_ui_once()
    def on_closing(self):
            """Handle window closing"""
            if self.running:
                self.toggle_monitoring()  # Stop monitoring if running
            self.logger.close()  # Close log file
            self.root.destroy()
        
    def create_ui(self):
        # Create a frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create left panel for video feed
        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Video feed placeholder
        self.video_frame = ctk.CTkFrame(self.left_panel, width=640, height=480)
        self.video_frame.pack(padx=10, pady=10)
        
        # Canvas for video display
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.canvas.pack()
        
        # Controls under video feed
        self.controls_frame = ctk.CTkFrame(self.left_panel)
        self.controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Start/Stop button
        self.start_button = ctk.CTkButton(
            self.controls_frame, 
            text="Start Monitoring", 
            command=self.toggle_monitoring,
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.start_button.pack(side="left", padx=10, pady=10)
        
        # Settings button
        self.settings_button = ctk.CTkButton(
            self.controls_frame, 
            text="Settings", 
            command=self.open_settings
        )
        self.settings_button.pack(side="left", padx=10, pady=10)
        
        # Right panel for statistics and alerts
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Status indicators
        self.status_frame = ctk.CTkFrame(self.right_panel)
        self.status_frame.pack(fill="x", padx=10, pady=10)
        
        # Alert status
        self.alert_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", font=("Helvetica", 16, "bold"))
        self.alert_label.pack(pady=5)
        
        # Session info
        self.session_frame = ctk.CTkFrame(self.right_panel)
        self.session_frame.pack(fill="x", padx=10, pady=10)
        
        self.session_label = ctk.CTkLabel(self.session_frame, text="Session Statistics", font=("Helvetica", 16, "bold"))
        self.session_label.pack(pady=5)
        
        # Session duration
        self.duration_label = ctk.CTkLabel(self.session_frame, text="Duration: 00:00:00")
        self.duration_label.pack(anchor="w", padx=10, pady=2)
        
        # Drowsiness events
        self.drowsy_count_label = ctk.CTkLabel(self.session_frame, text="Drowsiness Events: 0")
        self.drowsy_count_label.pack(anchor="w", padx=10, pady=2)
        
        # Total drowsy time
        self.drowsy_time_label = ctk.CTkLabel(self.session_frame, text="Total Drowsy Time: 0.0s")
        self.drowsy_time_label.pack(anchor="w", padx=10, pady=2)
        
        # Yawn events
        self.yawn_count_label = ctk.CTkLabel(self.session_frame, text="Yawn Events: 0")
        self.yawn_count_label.pack(anchor="w", padx=10, pady=2)
        
        # Blink count
        self.blink_count_label = ctk.CTkLabel(self.session_frame, text="Blink Count: 0")
        self.blink_count_label.pack(anchor="w", padx=10, pady=2)
        
        # Current blink rate
        self.blink_rate_label = ctk.CTkLabel(self.session_frame, text="Current Blink Rate: 0.0 bpm")
        self.blink_rate_label.pack(anchor="w", padx=10, pady=2)
        
        # Eye and mouth metrics
        self.metrics_frame = ctk.CTkFrame(self.right_panel)
        self.metrics_frame.pack(fill="x", padx=10, pady=10)
        
        self.metrics_label = ctk.CTkLabel(self.metrics_frame, text="Real-time Metrics", font=("Helvetica", 16, "bold"))
        self.metrics_label.pack(pady=5)
        
        # EAR value and progress bar
        self.ear_label = ctk.CTkLabel(self.metrics_frame, text="Eye Aspect Ratio (EAR): 0.00")
        self.ear_label.pack(anchor="w", padx=10, pady=2)
        
        self.ear_progress = ttk.Progressbar(self.metrics_frame, length=200, mode="determinate")
        self.ear_progress.pack(anchor="w", padx=10, pady=2)
        
        # MAR value and progress bar
        self.mar_label = ctk.CTkLabel(self.metrics_frame, text="Mouth Aspect Ratio (MAR): 0.00")
        self.mar_label.pack(anchor="w", padx=10, pady=2)
        
        self.mar_progress = ttk.Progressbar(self.metrics_frame, length=200, mode="determinate")
        self.mar_progress.pack(anchor="w", padx=10, pady=2)
        
        # Charts frame
        self.charts_frame = ctk.CTkFrame(self.right_panel)
        self.charts_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize matplotlib figure
        self.fig = plt.figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Blink Rate Over Time")
        self.ax.set_ylabel("Blinks per Minute")
        self.ax.set_xlabel("Time (s)")
        self.line, = self.ax.plot([], [], 'b-')
        
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
        self.canvas_chart.get_tk_widget().pack(fill="both", expand=True)
        
        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=2)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Start update thread for UI
        self.update_thread = threading.Thread(target=self.update_ui_once, daemon=True)
        self.update_thread.start()
    
    def toggle_monitoring(self):
        if not self.running:
            # Start monitoring
            self.running = True
            self.cap = cv2.VideoCapture(0)

             # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust as needed
            self.cap.set(cv2.CAP_PROP_CONTRAST, 150)    # Adjust as needed
            self.cap.set(cv2.CAP_PROP_SATURATION, 150)  # Adjust as needed


            
            self.start_button.configure(text="Stop Monitoring", fg_color="#dc3545", hover_color="#c82333")
            self.alert_label.configure(text="Status: Monitoring")
            
            # Start video processing thread
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
            
            # Start video update from main thread
            self.update_video()
        else:
            # Stop monitoring
            self.running = False
            if self.cap:
                self.cap.release()
            self.start_button.configure(text="Start Monitoring", fg_color="#28a745", hover_color="#218838")
            self.alert_label.configure(text="Status: Stopped")
            # Stop any active beeping
            self.beep_active = False

    def update_video(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            # Create an PIL Image from the frame
            image = Image.fromarray(self.current_frame)
            
            # Convert PIL Image to tkinter PhotoImage
            img = ImageTk.PhotoImage(image=image)
            self.canvas.img = img  # Keep a reference
        
            if hasattr(self, 'image_on_canvas'):
                self.canvas.itemconfig(self.image_on_canvas, image=img)
            else:
                self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=img)
        
        # Schedule next update
        if self.running:
            self.root.after(30, self.update_video)
    
    def process_video(self):
        self.drowsy_start_time = None
        self.blinking = False
        self.frame_count = 0
        self.ear_value = 0
        self.mar_value = 0

        
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Adjust brightness and contrast
            alpha = 1.1  # Contrast control (1.0 means no change)
            beta = 5     # Brightness control (0 means no change)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            # Optional: Apply slight color correction
            # This can make the colors more natural
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(frame_lab)
            # Apply CLAHE to L-channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            # Merge channels back
            merged = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
            # Create a copy of the frame for display
            display_frame = frame.copy()    
                
            
            # Frame skipping logic for processing
            self.frame_count += 1
            process_this_frame = self.frame_count % (self.frame_skip + 1) == 0
            
            # Always convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) == 0:
                cv2.putText(display_frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.text_overlay["drowsiness_alert"] = False
                self.text_overlay["eyes_status"] = "No Face"
                self.text_overlay["yawning"] = False
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                self.current_frame = rgb_frame
                continue
                
            # Process detection on specified frames
            if process_this_frame:
                for face in faces:
                    # Get facial landmarks
                    shape = self.predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    
                    # Extract eye coordinates
                    leftEye = shape[self.left_start:self.left_end]
                    rightEye = shape[self.right_start:self.right_end]
                    mouth = shape[self.mouth_start:self.mouth_end]
                    
                    # Calculate EAR for both eyes
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    self.ear_value = ear
                    
                    # Calculate MAR
                    mar = mouth_aspect_ratio(mouth)
                    self.mar_value = mar
                    
                    # Draw facial landmarks
                    for (x, y) in shape:
                        cv2.circle(display_frame, (x, y), 1, (0, 255, 0), -1)
                    
                    # Draw eye contours
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    mouthHull = cv2.convexHull(mouth)
                    
                    cv2.drawContours(display_frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(display_frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(display_frame, [mouthHull], -1, (0, 255, 0), 1)
                    
                    # Add EAR and MAR values to display
                    cv2.putText(display_frame, f"EAR: {ear:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"MAR: {mar:.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check for blinks
                    if not self.blinking and ear < self.BLINK_THRESHOLD:
                        self.blinking = True
                        self.logger.log_event("blink", 0, ear, mar)
                    elif self.blinking and ear > self.BLINK_THRESHOLD:
                        self.blinking = False
                    
                    # Check for drowsiness
                    if ear < self.EAR_THRESHOLD:
                        if self.drowsy_start_time is None:
                            self.drowsy_start_time = time.time()
                        
                        # Calculate current drowsy duration
                        self.drowsy_duration = time.time() - self.drowsy_start_time
                        
                        # Add alert text
                        if self.drowsy_duration >= self.DROWSINESS_DURATION:
                            # Display alert on screen
                            cv2.putText(display_frame, "DROWSINESS ALERT!", (150, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            self.text_overlay["drowsiness_alert"] = True
                            self.text_overlay["eyes_status"] = "Eyes Closed"
                            
                            # Trigger alert if not already active
                            if not self.beep_active:
                                self.beep_active = True
                                self.logger.log_event("drowsiness", self.drowsy_duration, ear, mar)
                                beep_thread = threading.Thread(target=self.beep_continuous, daemon=True)
                                beep_thread.start()
                    else:
                        # Reset drowsiness counter if eyes are open
                        if self.drowsy_start_time is not None:
                            if self.text_overlay["drowsiness_alert"]:
                                # Log the full drowsy episode when it ends
                                drowsy_episode_duration = time.time() - self.drowsy_start_time
                                self.logger.log_event("drowsiness_end", drowsy_episode_duration, ear, mar)
                            
                            self.drowsy_start_time = None
                            self.drowsy_duration = 0
                            self.beep_active = False
                            self.text_overlay["drowsiness_alert"] = False
                            self.text_overlay["eyes_status"] = "Eyes Open"
                    
                    # Check for yawning
                    if mar > self.MAR_THRESHOLD:
                        cv2.putText(display_frame, "Yawning", (150, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if not self.text_overlay["yawning"]:
                            self.text_overlay["yawning"] = True
                            self.logger.log_event("yawn", 0, ear, mar)
                            yawn_thread = threading.Thread(target=self.beep_once, daemon=True)
                            yawn_thread.start()
                    else:
                        self.text_overlay["yawning"] = False
                    
            # Display status on frame
            status_y = display_frame.shape[0] - 10
            cv2.putText(display_frame, f"Status: {'DROWSY' if self.text_overlay['drowsiness_alert'] else 'Alert'}", 
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 0, 255) if self.text_overlay["drowsiness_alert"] else (0, 255, 0), 2)
            
            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.current_frame = rgb_frame
            
            # Sleep briefly to prevent high CPU usage
            time.sleep(0.01)        
    def update_ui_once(self):
        if hasattr(self, 'logger'):
            # Format session duration
            duration = self.logger.get_session_duration()
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Update labels with latest values
            self.duration_label.configure(text=f"Duration: {duration_str}")
            self.drowsy_count_label.configure(text=f"Drowsiness Events: {self.logger.drowsiness_count}")
            self.drowsy_time_label.configure(text=f"Total Drowsy Time: {self.logger.total_drowsy_time:.1f}s")
            self.yawn_count_label.configure(text=f"Yawn Events: {self.logger.yawn_count}")
            self.blink_count_label.configure(text=f"Blink Count: {self.logger.blink_count}")
            self.blink_rate_label.configure(text=f"Current Blink Rate: {self.logger.get_blink_rate():.1f} bpm")
            
            # Update alert status based on current state
            if hasattr(self, 'text_overlay'):
                if self.text_overlay["drowsiness_alert"]:
                    self.alert_label.configure(text="Status: ⚠️ DROWSY ⚠️", text_color="#dc3545")
                elif self.text_overlay["yawning"]:
                    self.alert_label.configure(text="Status: Yawning", text_color="#fd7e14")
                elif self.running:
                    self.alert_label.configure(text="Status: Alert", text_color="#28a745")
            
            # Update metrics if available
            if hasattr(self, 'ear_value') and hasattr(self, 'mar_value'):
                self.ear_label.configure(text=f"Eye Aspect Ratio (EAR): {self.ear_value:.2f}")
                self.ear_progress['value'] = min(self.ear_value * 100, 100)
                
                self.mar_label.configure(text=f"Mouth Aspect Ratio (MAR): {self.mar_value:.2f}")
                self.mar_progress['value'] = min(self.mar_value * 50, 100)  # Scaled for better visibility
            
            # Update chart
            if hasattr(self, 'line') and hasattr(self, 'ax') and hasattr(self, 'fig'):
                blink_data = list(self.logger.blink_rate_history)
                x_data = list(range(len(blink_data)))
                
                if blink_data:
                    self.line.set_data(x_data, blink_data)
                    self.ax.set_xlim(0, max(len(blink_data), 30))
                    self.ax.set_ylim(0, max(max(blink_data, default=15) * 1.2, 15))
                    self.fig.canvas.draw_idle()
        
        # Schedule next update
        if hasattr(self, 'root'):
            self.root.after(500, self.update_ui_once)    
    def beep_continuous(self):
        """Progressive beep for drowsiness alert that intensifies over time"""
    
        start_time = time.time()
        base_frequency = 1000  # Starting frequency in Hz
        base_duration = 500    # Starting duration in ms
        
        while self.beep_active and self.running:
            # Calculate how long eyes have been closed
            elapsed_time = time.time() - start_time
            
            # Increase intensity based on elapsed time
            if elapsed_time < 3:
                # First stage - normal beeps
                frequency = base_frequency
                duration = base_duration
                interval = 0.5  # Time between beeps
            elif elapsed_time < 6:
                # Second stage - faster, higher pitch
                frequency = base_frequency * 1.2
                duration = base_duration * 0.8
                interval = 0.3
            else:
                # Final stage - urgent, even higher pitch
                frequency = base_frequency * 1.5
                duration = base_duration * 0.7
                interval = 0.2
            
            # Generate beep with current intensity
            winsound.Beep(int(frequency), int(duration))
            time.sleep(interval)
    
    def beep_once(self):
        """Single beep for yawn alert"""
        winsound.Beep(1500, 500)  # 1500 Hz for 500ms
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Detection Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)  # Make it modal
        
        # Create settings UI
        settings_frame = ctk.CTkFrame(settings_window)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # EAR threshold setting
        ear_label = ctk.CTkLabel(settings_frame, text="Eye Aspect Ratio Threshold:")
        ear_label.pack(anchor="w", pady=(10, 0))
        
        ear_slider = ctk.CTkSlider(settings_frame, from_=0.1, to=0.5, 
                                   number_of_steps=40)
        ear_slider.set(self.EAR_THRESHOLD)
        ear_slider.pack(fill="x", padx=10, pady=5)
        
        ear_value_label = ctk.CTkLabel(settings_frame, text=f"{self.EAR_THRESHOLD:.2f}")
        ear_value_label.pack()
        
        # MAR threshold setting
        mar_label = ctk.CTkLabel(settings_frame, text="Mouth Aspect Ratio Threshold:")
        mar_label.pack(anchor="w", pady=(10, 0))
        
        mar_slider = ctk.CTkSlider(settings_frame, from_=0.3, to=1.0, 
                                   number_of_steps=70)
        mar_slider.set(self.MAR_THRESHOLD)
        mar_slider.pack(fill="x", padx=10, pady=5)
        
        mar_value_label = ctk.CTkLabel(settings_frame, text=f"{self.MAR_THRESHOLD:.2f}")
        mar_value_label.pack()
        
        # Drowsiness duration setting
        duration_label = ctk.CTkLabel(settings_frame, text="Drowsiness Duration (seconds):")
        duration_label.pack(anchor="w", pady=(10, 0))
        
        duration_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=5.0, 
                                        number_of_steps=45)
        duration_slider.set(self.DROWSINESS_DURATION)
        duration_slider.pack(fill="x", padx=10, pady=5)
        
        duration_value_label = ctk.CTkLabel(settings_frame, text=f"{self.DROWSINESS_DURATION:.1f}")
        duration_value_label.pack()
        
        # Update labels on slider change
        def update_ear_label(value):
            ear_value_label.configure(text=f"{float(value):.2f}")
                
        def update_mar_label(value):
            mar_value_label.configure(text=f"{float(value):.2f}")
                
        def update_duration_label(value):
            duration_value_label.configure(text=f"{float(value):.1f}")
                
        ear_slider.configure(command=update_ear_label)
        mar_slider.configure(command=update_mar_label)
        duration_slider.configure(command=update_duration_label)
        
        # Save button
        def save_settings():
            self.EAR_THRESHOLD = float(ear_slider.get())
            self.MAR_THRESHOLD = float(mar_slider.get())
            self.DROWSINESS_DURATION = float(duration_slider.get())
            settings_window.destroy()
            messagebox.showinfo("Settings", "Settings updated successfully!")
                
        save_button = ctk.CTkButton(settings_frame, text="Save Settings", command=save_settings)
        save_button.pack(pady=20)

        

# ==============================
# Step 4: Run Application
# ==============================

if __name__ == "__main__":
    root = ctk.CTk()
    app = DrowsinessDetectionApp(root)
    root.mainloop()
