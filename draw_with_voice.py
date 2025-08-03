import pygame
import numpy as np
import sys
import threading
import queue
from voice_recognition.predict import VoiceCommandRecognizer, record_audio
import random

class VoiceDrawingApp:
    def __init__(self, model_path, width=800, height=600):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Voice-Controlled Drawing")
        
        self.drawing = False
        self.last_pos = (width // 2, height // 2) 
        self.color = (0, 0, 0) 
        self.brush_size = 10
        self.command_queue = queue.Queue()
        self.move_distance = 20  
        
        self.recognizer = VoiceCommandRecognizer(model_path)
        
        self.command_actions = {
            'up': self.move_up,
            'down': self.move_down,
            'left': self.move_left,
            'right': self.move_right,
            'stop': self.clear_screen,
            'go': self.change_color,
            'yes': self.increase_brush,
            'no': self.decrease_brush
        }
        
        self.colors = [
            (233, 197, 106),    # Red
            (51, 119, 119),    # Teal
            (171, 182, 200),    # Light Blue
            (93, 117, 153),  # Sky Blue
            (229, 152, 155),  # Pink
            (181, 131, 141),  # Brown
            (109, 104, 117)       # Purple
        ]
        self.current_color_index = 2  
        
        self.draw_cursor()
        pygame.display.flip()
        
        self.running = True
        self.recording_thread = threading.Thread(target=self.record_and_predict)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def draw_cursor(self):
        """Draw the cursor at current position"""
        pygame.draw.circle(self.screen, self.color, self.last_pos, self.brush_size//2)
    
    def draw_line_to(self, new_pos):
        """Draw a line from last position to new position"""
        pygame.draw.line(self.screen, self.color, self.last_pos, new_pos, self.brush_size)
        self.last_pos = new_pos
        pygame.display.flip()
    
    def change_color(self):
        """Cycle to the next color"""
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.color = self.colors[self.current_color_index]
    
    def increase_brush(self):
        """Increase brush size"""
        self.brush_size = min(30, self.brush_size + 2)
    
    def decrease_brush(self):
        """Decrease brush size"""
        self.brush_size = max(1, self.brush_size - 2)
    
    def move_up(self):
        """Move cursor up and draw"""
        if self.last_pos:
            x, y = self.last_pos
            new_y = max(0, y - self.move_distance)
            self.draw_line_to((x, new_y))
    
    def move_down(self):
        """Move cursor down and draw"""
        if self.last_pos:
            x, y = self.last_pos
            new_y = min(self.height, y + self.move_distance)
            self.draw_line_to((x, new_y))
    
    def move_left(self):
        """Move cursor left and draw"""
        if self.last_pos:
            x, y = self.last_pos
            new_x = max(0, x - self.move_distance)
            self.draw_line_to((new_x, y))
    
    def move_right(self):
        """Move cursor right and draw"""
        if self.last_pos:
            x, y = self.last_pos
            new_x = min(self.width, x + self.move_distance)
            self.draw_line_to((new_x, y))
    
    def clear_screen(self):
        """Clear the drawing area"""
        self.screen.fill((255, 255, 255))
        pygame.display.flip()
        self.last_pos = (self.width // 2, self.height // 2)
        self.draw_cursor()
    
    def record_and_predict(self):
        """Thread function to record audio and predict commands"""
        while self.running:
            try:
                audio_data = record_audio(duration=3, sample_rate=16000) 
                if audio_data is not None:
                    predictions = self.recognizer.predict_raw_audio(audio_data, top_k=1)
                    if predictions:
                        command = predictions[0]['class']
                        confidence = predictions[0]['probability']
                        if confidence > 0.3: 
                            print(f"Detected command: {command} (confidence: {confidence:.2f})")
                            self.command_queue.put(command)
            except Exception as e:
                print(f"Error in recording thread: {e}")
    
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        self.clear_screen()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: 
                        self.drawing = True
                        self.last_pos = event.pos
                        self.draw_cursor()
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: 
                        self.drawing = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        current_pos = event.pos
                        self.draw_line_to(current_pos)
            
            try:
                command = self.command_queue.get_nowait()
                action = self.command_actions.get(command)
                if action:
                    action()
            except queue.Empty:
                pass
            
            clock.tick(30)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Voice-controlled drawing application')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the trained voice command recognition model')
    args = parser.parse_args()
    
    pygame.init()
    app = VoiceDrawingApp(args.model)
    app.run()