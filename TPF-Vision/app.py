import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch.nn as nn
import os
import math
import random

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=12, dropout=0.3):
        super(SimpleCNN, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(dropout)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 32)
            out = self.body(dummy)
            self.flattened_size = out.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.flattened_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

clases = ['manzana', 'mariposa', 'nube', 'ojo', 'flor', 'helado', 'luna', 'arcoiris', 'carita feliz', 'estrella', 'sol', 'árbol']
articulos = {
    'manzana': 'una',
    'mariposa': 'una',
    'nube': 'una',
    'ojo': 'un',
    'flor': 'una',
    'helado': 'un',
    'luna': 'una',
    'arcoiris': 'un',
    'carita feliz': 'una',
    'estrella': 'una',
    'sol': 'un',
    'árbol': 'un',
}


script_dir = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(script_dir, "modelo_cnn_final.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = SimpleCNN(num_classes=len(clases)).to(device)
modelo.load_state_dict(torch.load(modelo_path, map_location=device))
modelo.eval()

def procesar_imagen_pil(img, final_size=32, umbral=90):
    img = img.convert("L")
    img_np = np.array(img)
    bin_img = (img_np < umbral).astype(np.uint8) * 255
    coords = np.argwhere(bin_img > 0)
    if coords.size == 0:
        return None  
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = bin_img[y0:y1, x0:x1]

    h, w = cropped.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = cropped

    img_final = Image.fromarray(padded).resize((final_size, final_size), Image.Resampling.LANCZOS)
    return img_final

class DrawingApp:
    def __init__(self, root):
        self.canvas_size = 800

        self.BACKGROUND_COLOR = 'white'
        self.STROKE_COLOR = 'black'
        self.IMAGE_BG_COLOR = 255  
        self.IMAGE_STROKE_COLOR = 0  

        self.MIN_BRUSH_RADIUS = 1
        self.MAX_BRUSH_RADIUS = 5
        self.DEFAULT_BRUSH_RADIUS = 3
        self.current_brush_radius = self.DEFAULT_BRUSH_RADIUS

        self.root = root
        self.root.title("Predictor de Dibujos")
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg=self.BACKGROUND_COLOR)
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=self.IMAGE_BG_COLOR)
        self.draw = ImageDraw.Draw(self.image)

        self.strokes = []  
        self.current_stroke = []  

        self.prediction_text_id = None
        self.prediction_text_opacity = 0

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_stroke)
        self.root.bind_all("<KeyPress>", self.key_pressed)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        self.brush_size_label = tk.Label(control_frame, text=f"Brush Size: {self.current_brush_radius}")
        self.brush_size_label.pack(side=tk.LEFT, padx=5)

        size_frame = tk.Frame(control_frame)
        size_frame.pack(side=tk.LEFT, padx=10)
        tk.Button(size_frame, text="+", command=self.increase_brush_size).pack(side=tk.LEFT)
        tk.Button(size_frame, text="-", command=self.decrease_brush_size).pack(side=tk.LEFT)

        tk.Button(control_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Reset", command=self.reset_canvas).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)

        self.anim_icons = {}
        recursos_path = os.path.join(script_dir, "recursos")
        for clase in clases:
            nombre = clase.replace(" ", "_").lower()
            path = os.path.join(recursos_path, f"{nombre}.png")
            if os.path.exists(path):
                self.anim_icons[clase] = ImageTk.PhotoImage(Image.open(path).resize((48, 48)))

    def key_pressed(self, event):
        key = event.keysym.lower()

        if key == 'escape': 
            self.terminate()  
        ctrl_pressed = (event.state & 0x0C)
        if ctrl_pressed:
            if key == 'z':
                self.undo()
            elif key == 'plus':
                self.increase_brush_size()
            elif key == 'minus':
                self.decrease_brush_size()

    def terminate(self):
        if messagebox.askyesno("Quit Application", "¿Seguro que querés salir?", icon='warning'):
            self.root.destroy()  

    def paint(self, event):
        x, y = event.x, event.y
        self.current_stroke.append((x, y))
        self.redraw_image()

    def reset_stroke(self, event):
        if self.current_stroke:
            self.strokes.append((self.current_stroke.copy(), self.current_brush_radius))
            self.current_stroke = []

    def increase_brush_size(self, event=None):
        self.current_brush_radius = min(self.MAX_BRUSH_RADIUS, self.current_brush_radius + 1)
        self.brush_size_label.config(text=f"Brush Size: {self.current_brush_radius}")

    def decrease_brush_size(self, event=None):
        self.current_brush_radius = max(self.MIN_BRUSH_RADIUS, self.current_brush_radius - 1)
        self.brush_size_label.config(text=f"Brush Size: {self.current_brush_radius}")

    def undo(self, event=None):
        if self.strokes:
            self.strokes.pop()
            self.redraw_image()

    def reset_canvas(self):
        self.strokes = []
        self.current_stroke = []
        self.redraw_image()

    def cancel_current_stroke(self):
        if self.current_stroke:
            self.current_stroke = []
            self.redraw_image()

    def get_stroke(self, points, brush_radius):
        stroke = []
        if not points:
            return stroke
        
        x0, y0 = points[0]
        stroke.append((round(x0), round(y0), brush_radius))

        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]

            distance = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            num_interpolated = max(2, int(distance / 2))

            for j in range(1, num_interpolated + 1):
                t = j / num_interpolated
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                radius = brush_radius * (0.9 + 0.2 * math.sin(t * math.pi))
                stroke.append((round(x), round(y), round(radius)))

        return stroke

    def redraw_image(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=self.IMAGE_BG_COLOR)
        self.draw = ImageDraw.Draw(self.image)

        for points, brush_radius in self.strokes:
            self.draw_stroke(points, brush_radius)

        if self.current_stroke:
            self.draw_stroke(self.current_stroke, self.current_brush_radius)

    def draw_stroke(self, points, brush_radius):
        smoothed_stroke = self.get_stroke(points, brush_radius)

        if not smoothed_stroke:
            return
        for i in range(1, len(smoothed_stroke)):
            x0, y0, r0 = smoothed_stroke[i - 1]
            x1, y1, r1 = smoothed_stroke[i]
            self.canvas.create_line(x0, y0, x1, y1, fill=self.STROKE_COLOR,
                                    width=max(r0, r1) * 2,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self.draw.line((x0, y0, x1, y1), fill=self.IMAGE_STROKE_COLOR, width=max(1, max(r0, r1) * 2))

    def predict(self):
        img_final = procesar_imagen_pil(self.image)
        if img_final is None:
            self.show_prediction_text("No se detectó dibujo.")
            return

        img_tensor = torch.tensor(np.array(img_final)).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            salida = modelo(img_tensor)
            pred_idx = salida.argmax(dim=1).item()
            pred_clase = clases[pred_idx]
            self.animate_icons(pred_clase)
            articulo = articulos.get(pred_clase, "un/una")
            self.show_prediction_text(f"¡Parece {articulo} {pred_clase}!")

    def animate_icons(self, clase):
        if clase not in self.anim_icons:
            return
        icon = self.anim_icons[clase]
        icon_ids = []
        for _ in range(20):
            x = random.randint(0, self.canvas_size - 50)
            y = random.randint(0, self.canvas_size - 50)
            icon_id = self.canvas.create_image(x, y, image=icon)
            icon_ids.append((icon_id, random.choice([-2, 2]), random.choice([-2, 2])))

        def move(step=0):
            if step >= 60:
                for icon_id, _, _ in icon_ids:
                    self.canvas.delete(icon_id)
                return
            for i in range(len(icon_ids)):
                icon_id, dx, dy = icon_ids[i]
                self.canvas.move(icon_id, dx, dy)
            self.root.after(50, lambda: move(step + 1))
        move()

    def show_prediction_text(self, text):
        if self.prediction_text_id:
            self.canvas.delete(self.prediction_text_id)
            self.canvas.delete("text_shadow")

        x = self.canvas_size // 2
        y = self.canvas_size // 2

        self.canvas.create_text(
            x + 2, y + 2,
            text=text,
            fill="grey",
            font=("Helvetica", 44, "bold"),
            tags="text_shadow"
        )

        self.prediction_text_id = self.canvas.create_text(
            x, y,
            text=text,
            fill="#f4c20d",
            font=("Helvetica", 44, "bold")
        )

        self.root.after(3000, self._clear_prediction_text)

    def _clear_prediction_text(self):
        if self.prediction_text_id:
            self.canvas.delete(self.prediction_text_id)
            self.prediction_text_id = None
        self.canvas.delete("text_shadow") 

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    app.run()
