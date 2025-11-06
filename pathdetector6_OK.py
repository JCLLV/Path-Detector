import cv2
import numpy as np
from tkinter import filedialog, Tk, StringVar, ttk
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional
import math

@dataclass
class PathMetrics:
    steering_angle: float
    path_width: float
    confidence: float
    center_offset: float
    curvature: float

class PathDetector:
    def __init__(self, config_file: str = 'config.json'):
        self.load_config(config_file)
        self.setup_logging()
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.setup_kalman()
        self.history = []
        self.frame_count = 0
    
    def load_config(self, config_file: str):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = self.get_default_config()
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
        self.config = config
    
    def get_default_config(self) -> dict:
        return {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 50,
            'min_line_length': 100,
            'max_line_gap': 50,
            'roi_height_factor': 0.5,
            'confidence_threshold': 0.7,
            'smoothing_window': 5
        }
    
    def setup_logging(self):
        logging.basicConfig(
            filename=f'path_detection_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def setup_kalman(self):
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], np.float32)
        self.kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]], np.float32) * 0.03
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        denoised = cv2.fastNlMeansDenoisingColored(enhanced)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blur, enhanced
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        mean = np.mean(image)
        sigma = np.std(image)
        lower = int(max(0, (1.0 - sigma) * mean))
        upper = int(min(255, (1.0 + sigma) * mean))
        edges = cv2.Canny(image, lower, upper)
        return edges
    
    def get_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        height, width = shape
        roi_height = int(height * self.config['roi_height_factor'])
        vertices = np.array([[(0, height),
                            (width//3, roi_height),
                            (2*width//3, roi_height),
                            (width, height)]], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask
    
    def detect_lanes(self, edges: np.ndarray, frame: np.ndarray) -> Tuple[List, List]:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                               threshold=self.config['hough_threshold'],
                               minLineLength=self.config['min_line_length'],
                               maxLineGap=self.config['max_line_gap'])
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length > self.config['min_line_length']:
                    if -0.8 < slope < -0.1:
                        left_lines.append(line[0])
                    elif 0.1 < slope < 0.8:
                        right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def calculate_path_metrics(self, left_lines: List, right_lines: List, 
                             frame_shape: Tuple[int, int]) -> PathMetrics:
        height, width = frame_shape
        center_x = width // 2
        
        if not (left_lines and right_lines):
            return PathMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        left_avg = np.mean(left_lines, axis=0, dtype=np.int32)
        right_avg = np.mean(right_lines, axis=0, dtype=np.int32)
        
        path_center = (left_avg[0] + right_avg[0]) // 2
        deviation = center_x - path_center
        steering_angle = np.arctan2(deviation, height) * 180 / np.pi
        
        measurement = np.array([[path_center], [steering_angle]], np.float32)
        self.kalman_filter.correct(measurement)
        prediction = self.kalman_filter.predict()
        
        smoothed_center = prediction[0]
        smoothed_angle = prediction[1]
        path_width = abs(right_avg[0] - left_avg[0])
        
        left_slope = (left_avg[3] - left_avg[1]) / (left_avg[2] - left_avg[0])
        right_slope = (right_avg[3] - right_avg[1]) / (right_avg[2] - right_avg[0])
        curvature = abs(left_slope - right_slope)
        
        line_confidence = min(len(left_lines), len(right_lines)) / 10.0
        width_confidence = 1.0 if 0.3 * width < path_width < 0.7 * width else 0.5
        angle_confidence = 1.0 - abs(smoothed_angle) / 45.0
        confidence = (line_confidence + width_confidence + angle_confidence) / 3.0
        
        return PathMetrics(
            steering_angle=float(smoothed_angle),
            path_width=float(path_width),
            confidence=float(confidence),
            center_offset=float(deviation),
            curvature=float(curvature)
        )
    
    def draw_visualization(self, frame: np.ndarray, metrics: PathMetrics, 
                        left_lines: List, right_lines: List) -> np.ndarray:
       """Dibuja visualización avanzada"""
       result = frame.copy()
       height, width = frame.shape[:2]
       center_x = width // 2
       path_center = center_x
       
       if left_lines and right_lines:
           left_avg = np.mean(left_lines, axis=0, dtype=np.int32)
           right_avg = np.mean(right_lines, axis=0, dtype=np.int32)
           
           cv2.line(result, (left_avg[0], left_avg[1]),
                   (left_avg[2], left_avg[3]), (0, 255, 0), 2)
           cv2.line(result, (right_avg[0], right_avg[1]),
                   (right_avg[2], right_avg[3]), (0, 255, 0), 2)
           
           path_center = (left_avg[0] + right_avg[0]) // 2
           cv2.line(result, (path_center, height),
                   (path_center, height//2), (0, 0, 255), 2)
           cv2.line(result, (center_x, height),
                   (center_x, height//2), (255, 0, 0), 2)
       
       info_color = (0, 255, 0) if metrics.confidence > 0.7 else (0, 165, 255)
       deviation = center_x - path_center
       status = "Centrado"
       if abs(metrics.steering_angle) > 5:
           """ status = "Girar Izquierda" if deviation < 0 else "Girar Derecha" """
           status = "Girar Derecha" if deviation < 0 else "Girar Izquierda"

       cv2.putText(result, f"Estado: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
       cv2.putText(result, f"Angulo: {metrics.steering_angle:.1f} grados", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
       cv2.putText(result, f"Confianza: {metrics.confidence:.2f}", (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
       cv2.putText(result, f"Curvatura: {metrics.curvature:.2f}", (width - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
       
       arrow_length = 100
       arrow_angle = math.radians(-metrics.steering_angle)
       end_point = (
           int(center_x + arrow_length * math.sin(arrow_angle)),
           int(height - arrow_length * math.cos(arrow_angle))
       )
       cv2.arrowedLine(result, (center_x, height), end_point, (0, 0, 255), 3)
       
       return result
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, PathMetrics]:
        processed, enhanced = self.preprocess_frame(frame)
        edges = self.detect_edges(processed)
        mask = self.get_roi_mask(edges.shape)
        roi = cv2.bitwise_and(edges, mask)
        left_lines, right_lines = self.detect_lanes(roi, frame)
        metrics = self.calculate_path_metrics(left_lines, right_lines, frame.shape[:2])
        
        self.history.append(metrics)
        if len(self.history) > self.config['smoothing_window']:
            self.history.pop(0)
        
        result = self.draw_visualization(enhanced, metrics, left_lines, right_lines)
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.log_metrics(metrics)
        
        return result, metrics
    
    def log_metrics(self, metrics: PathMetrics):
        logging.info(
            f"Frame {self.frame_count}: "
            f"Steering={metrics.steering_angle:.1f}°, "
            f"Confidence={metrics.confidence:.2f}, "
            f"Curvature={metrics.curvature:.2f}"
        )

class VideoProcessor:
    def __init__(self):
        self.detector = PathDetector()
    
    def process_video(self):
        root = Tk()
        root.title("Procesador de Video")
        
        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky='nsew')
        
        ttk.Label(frame, text="Seleccione un video para procesar:").grid(column=0, row=0)
        
        status_var = StringVar()
        status_var.set("Esperando selección...")
        ttk.Label(frame, textvariable=status_var).grid(column=0, row=2)
        
        def select_and_process():
            video_path = filedialog.askopenfilename(
                title="Seleccionar video",
                filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
            )
            
            if not video_path:
                return
                
            status_var.set("Procesando video...")
            root.update()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                status_var.set("Error al abrir el video")
                return
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            out = cv2.VideoWriter(
                f"processed_{timestamp}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
            csv_file = open(f"metrics_{timestamp}.csv", 'w')
            csv_file.write("frame,steering_angle,confidence,curvature,center_offset\n")
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, metrics = self.detector.process_frame(frame)
                out.write(processed_frame)
                
                csv_file.write(f"{frame_count},{metrics.steering_angle},"
                             f"{metrics.confidence},{metrics.curvature},"
                             f"{metrics.center_offset}\n")
                
                progress = (frame_count / total_frames) * 100
                status_var.set(f"Procesando: {progress:.1f}%")
                root.update()
                
                cv2.imshow('Path Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            csv_file.close()
            
            status_var.set("Procesamiento completado")
        
        ttk.Button(frame, text="Seleccionar Video", 
                  command=select_and_process).grid(column=0, row=1)
        
        root.mainloop()

def main():
    processor = VideoProcessor()
    processor.process_video()

if __name__ == "__main__":
    main()