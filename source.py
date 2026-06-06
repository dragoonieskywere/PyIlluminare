import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import rawpy
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
import glob
import sys

try:
    import scipy.interpolate as interp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    HAS_EXIFREAD = False

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False

# Ativa a aceleração por hardware (OpenCL / GPU) nativa do OpenCV
cv2.ocl.setUseOpenCL(True)

# Dicionário de Traduções Expandido
LANGUAGES = {
    'pt': {
        'title': 'PyIlluminare',
        'open_raw': 'Abrir Arquivo',
        'print': 'Imprimir',
        'compare': 'Segure para Antes/Depois',
        'tab_basic': 'Básico',
        'tab_adv': 'Avançado',
        'tab_lens': 'Lente',
        'tab_curve': 'Curva',
        'tab_brush': 'Pincel',
        'tab_about': 'Sobre',
        'temp': 'Temperatura de Cor',
        'exp': 'Exposição',
        'cont': 'Contraste',
        'hl': 'Realces',
        'shad': 'Sombras',
        'sat': 'Saturação',
        'noise_lum': 'Redução de Ruído RAW',
        'noise_color': 'Ruído de Cor',
        'sharpness': 'Nitidez (Sharpness)',
        'texture': 'Textura',
        'distortion': 'Distorção (Barril/Almofada)',
        'curve_hint': 'Clique p/ adicionar. Botão direito p/ apagar.',
        'brush_paint': 'Pintar (+)',
        'brush_erase': 'Apagar (-)',
        'brush_size': 'Tamanho do Pincel',
        'brush_feather': 'Suavidade (Feather)',
        'show_mask': 'Mostrar Máscara (Vermelho)',
        'clear_mask': 'Limpar Máscara',
        'local_adjust': 'Ajustes Aplicados na Máscara:',
        'loc_exp': 'Exposição Local',
        'loc_cont': 'Contraste Local',
        'loc_sat': 'Saturação Local',
        'save_jpg': 'Salvar JPEG',
        'save_tiff': 'Salvar TIFF (16-bit)',
        'no_img': "Nenhuma imagem carregada.\nSelecione um Arquivo."
    },
    'en': {
        'title': 'PyIlluminare',
        'open_raw': 'Open File',
        'print': 'Print',
        'compare': 'Hold for Before/After',
        'tab_basic': 'Basic',
        'tab_adv': 'Advanced',
        'tab_lens': 'Lens',
        'tab_curve': 'Curve',
        'tab_brush': 'Brush',
        'tab_about': 'About',
        'temp': 'Color Temperature',
        'exp': 'Exposure',
        'cont': 'Contrast',
        'hl': 'Highlights',
        'shad': 'Shadows',
        'sat': 'Saturation',
        'noise_lum': 'RAW Noise Reduction',
        'noise_color': 'Color Noise',
        'sharpness': 'Sharpness',
        'texture': 'Texture',
        'distortion': 'Distortion (Barrel/Pincushion)',
        'curve_hint': 'Click to add point. Right-click to delete.',
        'brush_paint': 'Paint (+)',
        'brush_erase': 'Erase (-)',
        'brush_size': 'Brush Size',
        'brush_feather': 'Feather',
        'show_mask': 'Show Mask (Red)',
        'clear_mask': 'Clear Mask',
        'local_adjust': 'Adjustments inside Mask:',
        'loc_exp': 'Local Exposure',
        'loc_cont': 'Local Contrast',
        'loc_sat': 'Local Saturation',
        'save_jpg': 'Save JPEG',
        'save_tiff': 'Save TIFF (16-bit)',
        'no_img': "No image loaded.\nSelect a File."
    },
    'es': {
        'title': 'PyIlluminare',
        'open_raw': 'Abrir Archivo',
        'print': 'Imprimir',
        'compare': 'Mantener para Antes/Después',
        'tab_basic': 'Básico',
        'tab_adv': 'Avanzado',
        'tab_lens': 'Lente',
        'tab_curve': 'Curva',
        'tab_brush': 'Pincel',
        'tab_about': 'Sobre',
        'temp': 'Temperatura de Color',
        'exp': 'Exposición',
        'cont': 'Contraste',
        'hl': 'Altas Luces',
        'shad': 'Sombras',
        'sat': 'Saturación',
        'noise_lum': 'Reducción de Ruido RAW',
        'noise_color': 'Ruido de Color',
        'sharpness': 'Nitidez',
        'texture': 'Textura',
        'distortion': 'Distorsión',
        'curve_hint': 'Clic para añadir. Clic derecho para borrar.',
        'brush_paint': 'Pintar (+)',
        'brush_erase': 'Borrar (-)',
        'brush_size': 'Tamaño del Pincel',
        'brush_feather': 'Suavizado (Feather)',
        'show_mask': 'Mostrar Máscara (Rojo)',
        'clear_mask': 'Limpiar Máscara',
        'local_adjust': 'Ajustes en la Máscara:',
        'loc_exp': 'Exposición Local',
        'loc_cont': 'Contraste Local',
        'loc_sat': 'Saturación Local',
        'save_jpg': 'Guardar JPEG',
        'save_tiff': 'Guardar TIFF (16-bit)',
        'no_img': "No hay imagen cargada.\nSeleccione un Archivo."
    }
}

class ProLightroomClone:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1400x900")
        
        # Variável de Otimização (Evita lag nos sliders)
        self.update_timer = None
        
        # Variáveis de Estado
        self.current_lang = 'pt'
        self.raw_path = None
        self.current_dir = None
        self.raw_image = None
        self.preview_base = None 
        self.zoom_factor = 1.0
        self.raw_tags = {} 
        
        # Pan/Mover a imagem
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Rolo de filme
        self.thumbnails = []
        
        # Variáveis Pincel
        self.mask = None
        self.is_brushing = False
        self.tk_image = None

        # Variáveis Curva de Tons
        self.curve_points = [[0.0, 0.0], [0.25, 0.25], [0.75, 0.75], [1.0, 1.0]]
        self.curve_drag_idx = None
        
        self.ui_elements = {} 
        self.sliders = {}

        self.setup_ui()
        self.update_texts()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- BARRA SUPERIOR ---
        top_bar = tk.Frame(main_frame, bg="#333", height=40)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        
        tk.Label(top_bar, text="🌐", bg="#333", fg="white").pack(side=tk.LEFT, padx=5, pady=5)
        self.lang_cb = ttk.Combobox(top_bar, values=["Português", "English", "Español"], state="readonly", width=12)
        self.lang_cb.current(0)
        self.lang_cb.pack(side=tk.LEFT, pady=5)
        self.lang_cb.bind("<<ComboboxSelected>>", self.change_language)
        
        btn_load = tk.Button(top_bar, command=self.load_file, bg="#007acc", fg="white", relief=tk.FLAT)
        btn_load.pack(side=tk.LEFT, padx=15, pady=5)
        self.ui_elements['open_raw'] = btn_load
        
        btn_print = tk.Button(top_bar, command=self.print_image, bg="#6c757d", fg="white", relief=tk.FLAT)
        btn_print.pack(side=tk.LEFT, padx=5, pady=5)
        self.ui_elements['print'] = btn_print

        self.btn_compare = tk.Button(top_bar, bg="#dc3545", fg="white", relief=tk.FLAT)
        self.btn_compare.pack(side=tk.RIGHT, padx=10, pady=5)
        self.btn_compare.bind("<ButtonPress-1>", self.show_original)
        self.btn_compare.bind("<ButtonRelease-1>", self.show_edited)
        self.ui_elements['compare'] = self.btn_compare

        # --- ROLO DE FILME (INFERIOR) ---
        self.filmstrip_frame = tk.Frame(main_frame, bg="#252526", height=120)
        self.filmstrip_frame.pack(side=tk.BOTTOM, fill=tk.X)

        scroll_x = ttk.Scrollbar(self.filmstrip_frame, orient="horizontal")
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.filmstrip_canvas = tk.Canvas(self.filmstrip_frame, bg="#252526", height=100, highlightthickness=0, xscrollcommand=scroll_x.set)
        self.filmstrip_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        scroll_x.config(command=self.filmstrip_canvas.xview)
        
        self.filmstrip_canvas.bind("<Enter>", lambda e: self.filmstrip_canvas.bind_all("<MouseWheel>", lambda ev: self.filmstrip_canvas.xview_scroll(int(-1*(ev.delta/120)), "units")))
        self.filmstrip_canvas.bind("<Leave>", lambda e: self.filmstrip_canvas.unbind_all("<MouseWheel>"))

        # --- ÁREA CENTRAL (IMAGEM + ABAS) ---
        center_frame = tk.Frame(main_frame, bg="#1e1e1e")
        center_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(center_frame, bg="#1e1e1e")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#1e1e1e", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.CENTER)
        self.canvas.bind("<Configure>", lambda e: self.update_canvas_position())
        
        self.img_placeholder = tk.Label(self.canvas_frame, bg="#1e1e1e", fg="white", font=("Arial", 14))
        self.img_placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.ui_elements['no_img'] = self.img_placeholder

        self.exif_label = tk.Label(self.canvas_frame, bg="#1e1e1e", fg="#999", font=("Courier", 10, "bold"))

        # Eventos Pincel (Botão Esquerdo)
        self.canvas.bind("<ButtonPress-1>", self.start_brush)
        self.canvas.bind("<B1-Motion>", self.draw_brush)
        self.canvas.bind("<ButtonRelease-1>", self.end_brush)
        
        # Eventos Pan / Mover Imagem (Botão Direito)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        
        # Zoom (Windows, Mac e suporte Linux nums 4 e 5)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)  
        self.canvas.bind("<MouseWheel>", self.on_zoom)  
        self.canvas.bind("<Control-Button-4>", self.on_zoom)    
        self.canvas.bind("<Control-Button-5>", self.on_zoom)    
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)

        # --- ABAS DE CONTROLE (DIREITA) ---
        control_container = tk.Frame(center_frame, width=380, bg="#2d2d2d")
        control_container.pack(side=tk.RIGHT, fill=tk.Y)
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook.Tab', background="#333", foreground="white", padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', '#007acc')])
        
        self.notebook = ttk.Notebook(control_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_basic = self.create_scrollable_tab(self.notebook)
        self.tab_adv = self.create_scrollable_tab(self.notebook)
        self.tab_lens = self.create_scrollable_tab(self.notebook)
        self.tab_curve = self.create_scrollable_tab(self.notebook)
        self.tab_brush = self.create_scrollable_tab(self.notebook)
        self.tab_about = self.create_scrollable_tab(self.notebook) # NOVA ABA
        
        self.notebook.add(self.tab_basic.master.master, text="Básico")
        self.notebook.add(self.tab_adv.master.master, text="Avançado")
        self.notebook.add(self.tab_lens.master.master, text="Lente")
        self.notebook.add(self.tab_curve.master.master, text="Curva")
        self.notebook.add(self.tab_brush.master.master, text="Pincel")
        self.notebook.add(self.tab_about.master.master, text="Sobre")

        self.populate_tabs()

    # --- SISTEMA DE OTIMIZAÇÃO (DEBOUNCER) ---
    def schedule_update(self, event=None):
        """ Evita que o programa trave recalculando a imagem 60 vezes por segundo """
        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(50, self.update_image)

    def update_canvas_position(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.coords(self.canvas_image_id, (w // 2) + self.pan_x, (h // 2) + self.pan_y)

    def start_pan(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def do_pan(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_canvas_position()

    def on_zoom(self, event):
        # Suporte para Linux (num 4 ou 5) ou Windows/Mac (delta)
        if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0:
            self.zoom_factor *= 1.15  
        elif getattr(event, 'num', 0) == 5 or getattr(event, 'delta', 0) < 0:
            self.zoom_factor /= 1.15  
            
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
        self.update_image()

    def create_scrollable_tab(self, parent):
        frame = tk.Frame(parent, bg="#2d2d2d")
        canvas_scroll = tk.Canvas(frame, bg="#2d2d2d", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = tk.Frame(canvas_scroll, bg="#2d2d2d", padx=15, pady=15)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw", width=340)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return scrollable_frame

    def add_slider(self, parent, name, key, from_, to, default):
        lbl = tk.Label(parent, bg="#2d2d2d", fg="white", font=("Arial", 9, "bold"))
        lbl.pack(anchor="w", pady=(10, 0))
        self.ui_elements[key] = lbl
        
        var = tk.DoubleVar(value=default)
        slider = ttk.Scale(parent, from_=from_, to=to, orient="horizontal", variable=var, command=self.schedule_update)
        slider.pack(fill=tk.X)
        self.sliders[name] = var

        # Duplo clique no slider ou no label para resetar valor
        def reset_value(e):
            var.set(default)
            self.update_image()
            
        slider.bind("<Double-Button-1>", reset_value)
        lbl.bind("<Double-Button-1>", reset_value)

    def populate_tabs(self):
        # --- TAB: BÁSICO ---
        self.add_slider(self.tab_basic, 'temp', 'temp', -0.5, 0.5, 0.0)
        self.add_slider(self.tab_basic, 'exposure', 'exp', -100, 100, 0)
        self.add_slider(self.tab_basic, 'contrast', 'cont', 0.1, 3.0, 1.0)
        self.add_slider(self.tab_basic, 'highlights', 'hl', -1.0, 1.0, 0.0)
        self.add_slider(self.tab_basic, 'shadows', 'shad', -1.0, 1.0, 0.0)
        self.add_slider(self.tab_basic, 'saturation', 'sat', 0.0, 3.0, 1.0)
        
        btn_jpg = tk.Button(self.tab_basic, command=lambda: self.save_image("jpg"), bg="#28a745", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT)
        btn_jpg.pack(fill=tk.X, pady=(30, 5))
        btn_tiff = tk.Button(self.tab_basic, command=lambda: self.save_image("tiff"), bg="#17a2b8", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT)
        btn_tiff.pack(fill=tk.X, pady=(0, 20))
        self.ui_elements['save_jpg'] = btn_jpg
        self.ui_elements['save_tiff'] = btn_tiff

        # --- TAB: AVANÇADO ---
        self.add_slider(self.tab_adv, 'noise_lum', 'noise_lum', 0.0, 100.0, 0.0)
        self.add_slider(self.tab_adv, 'noise_color', 'noise_color', 0.0, 100.0, 0.0)
        self.add_slider(self.tab_adv, 'sharpness', 'sharpness', 0.0, 3.0, 0.0)
        self.add_slider(self.tab_adv, 'texture', 'texture', -1.0, 1.0, 0.0)

        # --- TAB: LENTE ---
        self.add_slider(self.tab_lens, 'distortion', 'distortion', -0.5, 0.5, 0.0)

        # --- TAB: CURVA ---
        l_curve_hint = tk.Label(self.tab_curve, fg="#888", bg="#2d2d2d", font=("Arial", 8))
        l_curve_hint.pack(pady=(0,10))
        self.ui_elements['curve_hint'] = l_curve_hint

        self.curve_canvas = tk.Canvas(self.tab_curve, width=250, height=250, bg="#111", highlightthickness=1, highlightbackground="#555", cursor="crosshair")
        self.curve_canvas.pack(pady=5)
        self.curve_canvas.bind("<ButtonPress-1>", self.on_curve_press)
        self.curve_canvas.bind("<B1-Motion>", self.on_curve_drag)
        self.curve_canvas.bind("<ButtonRelease-1>", self.on_curve_release)
        self.curve_canvas.bind("<ButtonPress-3>", self.on_curve_rclick)
        self.draw_curve()

        if not HAS_SCIPY:
            tk.Label(self.tab_curve, text="⚠️ Instale 'scipy' para habilitar a curva.", bg="#2d2d2d", fg="yellow", font=("Arial", 8)).pack()

        # --- TAB: PINCEL ---
        self.brush_mode = tk.StringVar(value="Pintar")
        frame_modos = tk.Frame(self.tab_brush, bg="#2d2d2d")
        frame_modos.pack(fill=tk.X, pady=10)
        rb1 = tk.Radiobutton(frame_modos, variable=self.brush_mode, value="Pintar", bg="#2d2d2d", fg="white", selectcolor="#444")
        rb2 = tk.Radiobutton(frame_modos, variable=self.brush_mode, value="Apagar", bg="#2d2d2d", fg="white", selectcolor="#444")
        rb1.pack(side=tk.LEFT, padx=10)
        rb2.pack(side=tk.LEFT)
        self.ui_elements['brush_paint'] = rb1
        self.ui_elements['brush_erase'] = rb2

        self.add_slider(self.tab_brush, 'brush_size', 'brush_size', 5, 300, 50)
        self.add_slider(self.tab_brush, 'brush_feather', 'brush_feather', 0, 100, 50)
        
        self.show_mask = tk.BooleanVar(value=False)
        cb_mask = tk.Checkbutton(self.tab_brush, variable=self.show_mask, command=self.schedule_update, bg="#2d2d2d", fg="white", selectcolor="#444")
        cb_mask.pack(anchor="w", pady=(15,5))
        self.ui_elements['show_mask'] = cb_mask
        
        btn_clear = tk.Button(self.tab_brush, command=self.clear_mask, bg="#dc3545", fg="white", relief=tk.FLAT)
        btn_clear.pack(fill=tk.X, pady=(5, 15))
        self.ui_elements['clear_mask'] = btn_clear

        l_adj = tk.Label(self.tab_brush, bg="#2d2d2d", fg="#ffcc00", font=("Arial", 9, "bold"))
        l_adj.pack(anchor="w")
        self.ui_elements['local_adjust'] = l_adj
        
        self.add_slider(self.tab_brush, 'local_exposure', 'loc_exp', -100, 100, 0)
        self.add_slider(self.tab_brush, 'local_contrast', 'loc_cont', 0.1, 3.0, 1.0)
        self.add_slider(self.tab_brush, 'local_saturation', 'loc_sat', 0.0, 3.0, 1.0)

        # --- TAB: SOBRE ---
        lbl_title = tk.Label(self.tab_about, text="PyIlluminare", bg="#2d2d2d", fg="#007acc", font=("Arial", 18, "bold"))
        lbl_title.pack(pady=(30, 5))

        lbl_version = tk.Label(self.tab_about, text="Versão 1.0.0", bg="#2d2d2d", fg="white", font=("Arial", 10))
        lbl_version.pack(pady=(0, 20))

        lbl_dev_title = tk.Label(self.tab_about, text="Desenvolvido por:", bg="#2d2d2d", fg="#aaa", font=("Arial", 10))
        lbl_dev_title.pack()

        lbl_author = tk.Label(self.tab_about, text="Gregório Severiano\n(Dragoonie)", bg="#2d2d2d", fg="#ffcc00", font=("Arial", 12, "bold"))
        lbl_author.pack(pady=(5, 20))
        
        lbl_tech = tk.Label(self.tab_about, text="Motor de Processamento:\nOpenCV & Python", bg="#2d2d2d", fg="#888", font=("Arial", 9))
        lbl_tech.pack(side=tk.BOTTOM, pady=20)


    # --- SISTEMA DE TRADUÇÃO ---
    def change_language(self, event=None):
        selection = self.lang_cb.get()
        if selection == "Português": self.current_lang = 'pt'
        elif selection == "English": self.current_lang = 'en'
        elif selection == "Español": self.current_lang = 'es'
        self.update_texts()

    def update_texts(self):
        t = LANGUAGES[self.current_lang]
        self.root.title(t['title'])
        for key, widget in self.ui_elements.items():
            widget.config(text=t.get(key, key))
            
        tabs = [self.tab_basic, self.tab_adv, self.tab_lens, self.tab_curve, self.tab_brush, self.tab_about]
        keys = ['tab_basic', 'tab_adv', 'tab_lens', 'tab_curve', 'tab_brush', 'tab_about']
        for tab, key in zip(tabs, keys):
            self.notebook.tab(tab.master.master, text=t.get(key, key))

    # --- LER METADADOS (EXIF) ---
    def extract_exif(self, file_path):
        self.raw_tags = {}
        if not HAS_EXIFREAD: return
        
        def safe_ratio(val):
            v = str(val)
            if '/' in v:
                try:
                    n, d = map(float, v.split('/'))
                    if d == 0: return ""
                    return str(int(n/d)) if n/d == int(n/d) else f"{n/d:.1f}"
                except: pass
            return v

        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                self.raw_tags = tags 
            
            cam = tags.get('Image Model', tags.get('Image Make', ''))
            focal = safe_ratio(tags.get('EXIF FocalLength', ''))
            fnum = safe_ratio(tags.get('EXIF FNumber', ''))
            expo = tags.get('EXIF ExposureTime', '')
            iso = tags.get('EXIF ISOSpeedRatings', '')

            parts = []
            if cam: parts.append(str(cam))
            if focal: parts.append(f"{focal}mm")
            if fnum: parts.append(f"f/{fnum}")
            if expo: parts.append(f"{expo}s")
            if iso: parts.append(f"ISO {iso}")

            info = " | ".join(parts)
            if info:
                self.exif_label.config(text=info)
                self.exif_label.place(x=20, y=20) 
            else:
                self.exif_label.place_forget()
        except:
            self.exif_label.place_forget()

    # --- GERADOR DE METADADOS PARA EXPORTAÇÃO ---
    def generate_exif_bytes(self):
        """ Cria um pacote EXIF seguro para a aba de Detalhes do Windows """
        if not HAS_PIEXIF: return b""
        
        # 1. Se o arquivo original for JPEG, tentamos copiar direto de forma segura
        if self.raw_path.lower().endswith(('.jpg', '.jpeg')):
            try:
                exif_dict = piexif.load(self.raw_path)
                if "thumbnail" in exif_dict: del exif_dict["thumbnail"]
                return piexif.dump(exif_dict)
            except Exception as e:
                print(f"Aviso EXIF JPEG: {e}")
                pass 
                
        # 2. Fallback Manual Seguro para arquivos RAW
        if not self.raw_tags: return b""
        
        exif_dict = {"0th": {}, "Exif": {}, "1st": {}, "thumbnail": None, "GPS": {}}
        
        def safe_rational(val_str):
            try:
                val_str = str(val_str).replace("f/", "").strip()
                if '/' in val_str:
                    n, d = val_str.split('/')
                    return (int(float(n)), int(float(d)))
                else:
                    return (int(float(val_str) * 1000), 1000)
            except:
                return None

        exif_dict["0th"][piexif.ImageIFD.Software] = b"PyIlluminare"
        
        if 'Image Make' in self.raw_tags:
            exif_dict["0th"][piexif.ImageIFD.Make] = str(self.raw_tags['Image Make'])[:30].encode('utf-8')
        if 'Image Model' in self.raw_tags:
            exif_dict["0th"][piexif.ImageIFD.Model] = str(self.raw_tags['Image Model'])[:30].encode('utf-8')
        if 'Image DateTime' in self.raw_tags:
            exif_dict["0th"][piexif.ImageIFD.DateTime] = str(self.raw_tags['Image DateTime'])[:19].encode('utf-8')

        if 'EXIF ISOSpeedRatings' in self.raw_tags:
            try:
                iso_str = str(self.raw_tags['EXIF ISOSpeedRatings']).replace('[','').replace(']','')
                iso_val = int(iso_str.split(',')[0])
                exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = min(iso_val, 65535)
            except: pass

        if 'EXIF FNumber' in self.raw_tags:
            rat = safe_rational(self.raw_tags['EXIF FNumber'])
            if rat: exif_dict["Exif"][piexif.ExifIFD.FNumber] = rat

        if 'EXIF ExposureTime' in self.raw_tags:
            rat = safe_rational(self.raw_tags['EXIF ExposureTime'])
            if rat: exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = rat

        if 'EXIF FocalLength' in self.raw_tags:
            rat = safe_rational(self.raw_tags['EXIF FocalLength'])
            if rat: exif_dict["Exif"][piexif.ExifIFD.FocalLength] = rat

        try:
            return piexif.dump(exif_dict)
        except Exception as e:
            print(f"Aviso Exif Fallback de Segurança: {e}")
            return b""

    # --- INTERAÇÃO DA CURVA DE TONS ---
    def draw_curve(self):
        self.curve_canvas.delete("all")
        w, h = 250, 250
        for i in range(1, 4):
            pos = i * (w/4)
            self.curve_canvas.create_line(pos, 0, pos, h, fill="#333", dash=(2, 2))
            self.curve_canvas.create_line(0, pos, w, pos, fill="#333", dash=(2, 2))
            
        if not HAS_SCIPY: return

        try:
            x_pts = [p[0] for p in self.curve_points]
            y_pts = [p[1] for p in self.curve_points]
            interpolator = interp.PchipInterpolator(x_pts, y_pts)
            plot_x = np.linspace(0, 1, 100)
            plot_y = np.clip(interpolator(plot_x), 0, 1)
            for i in range(len(plot_x)-1):
                x1, y1 = plot_x[i]*w, (1 - plot_y[i])*h
                x2, y2 = plot_x[i+1]*w, (1 - plot_y[i+1])*h
                self.curve_canvas.create_line(x1, y1, x2, y2, fill="white", width=2)
        except Exception:
            pass
            
        for i, p in enumerate(self.curve_points):
            px, py = p[0]*w, (1 - p[1])*h
            color = "#ffcc00" if i == self.curve_drag_idx else "white"
            self.curve_canvas.create_oval(px-4, py-4, px+4, py+4, fill=color, outline="black")

    def on_curve_press(self, event):
        w, h = 250, 250
        ex, ey = event.x / w, 1 - (event.y / h)
        for i, p in enumerate(self.curve_points):
            if abs(p[0]-ex) < 0.05 and abs(p[1]-ey) < 0.05:
                self.curve_drag_idx = i
                self.draw_curve()
                return
        if 0.05 < ex < 0.95:
            self.curve_points.append([ex, ey])
            self.curve_points.sort(key=lambda p: p[0])
            self.curve_drag_idx = self.curve_points.index([ex, ey])
            self.draw_curve()
            self.schedule_update()

    def on_curve_drag(self, event):
        if self.curve_drag_idx is not None:
            idx = self.curve_drag_idx
            w, h = 250, 250
            ex = np.clip(event.x / w, 0.0, 1.0)
            ey = np.clip(1 - (event.y / h), 0.0, 1.0)
            if idx == 0: ex = 0.0
            elif idx == len(self.curve_points) - 1: ex = 1.0
            else:
                min_x = self.curve_points[idx-1][0] + 0.01
                max_x = self.curve_points[idx+1][0] - 0.01
                ex = np.clip(ex, min_x, max_x)
            self.curve_points[idx] = [ex, ey]
            self.draw_curve()

    def on_curve_release(self, event):
        self.curve_drag_idx = None
        self.draw_curve()
        self.schedule_update()

    def on_curve_rclick(self, event):
        w, h = 250, 250
        ex, ey = event.x / w, 1 - (event.y / h)
        for i in range(1, len(self.curve_points)-1):
            p = self.curve_points[i]
            if abs(p[0]-ex) < 0.05 and abs(p[1]-ey) < 0.05:
                self.curve_points.pop(i)
                self.draw_curve()
                self.schedule_update()
                break

    # --- ROLO DE FILME ---
    def load_filmstrip(self, folder_path):
        self.filmstrip_canvas.delete("all")
        self.thumbnails.clear()
        
        types = ('*.cr2', '*.nef', '*.arw', '*.dng', '*.jpg', '*.png', '*.tif')
        files = []
        for ext in types: files.extend(glob.glob(os.path.join(folder_path, ext)))
        for ext in types: files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        x_offset = 10
        for f in files[:30]:
            try:
                img = Image.open(f)
                img.thumbnail((100, 100))
                tk_thumb = ImageTk.PhotoImage(img)
                self.thumbnails.append(tk_thumb)
                
                img_id = self.filmstrip_canvas.create_image(x_offset, 10, anchor=tk.NW, image=tk_thumb)
                self.filmstrip_canvas.tag_bind(img_id, "<Button-1>", lambda e, path=f: self.process_file(path))
                x_offset += 110
            except: pass
            
        self.root.update_idletasks()
        self.filmstrip_canvas.config(scrollregion=self.filmstrip_canvas.bbox(tk.ALL))

    # --- MOTOR DE IMAGEM ---
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.cr2 *.nef *.arw *.dng *.jpg *.png *.tiff"), ("All", "*.*")])
        if not file_path: return
        self.current_dir = os.path.dirname(file_path)
        self.process_file(file_path)
        self.load_filmstrip(self.current_dir)

    def process_file(self, file_path):
        self.raw_path = file_path
        self.img_placeholder.place_forget()
        self.root.update()
        
        self.extract_exif(file_path)

        try:
            if file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.raf')):
                with rawpy.imread(file_path) as raw:
                    rgb_image = raw.postprocess(use_camera_wb=True, half_size=True)
                self.raw_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            else:
                self.raw_image = cv2.imread(file_path)

            max_dim = 1200
            h, w = self.raw_image.shape[:2]
            scale = max_dim / max(h, w)
            if scale < 1.0: self.preview_base = cv2.resize(self.raw_image, (int(w * scale), int(h * scale)))
            else: self.preview_base = self.raw_image.copy()

            self.mask = np.zeros(self.preview_base.shape[:2], dtype=np.float32)
            
            self.zoom_factor = 1.0  
            self.pan_x = 0
            self.pan_y = 0
            
            self.update_image()
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar:\n{e}")

    def clear_mask(self):
        if self.mask is not None:
            self.mask.fill(0.0)
            self.update_image()

    def start_brush(self, event):
        self.is_brushing = True
        self.draw_brush(event)

    def draw_brush(self, event):
        if not self.is_brushing or self.mask is None: return
        if self.notebook.index("current") != 4: return 
        
        w_canvas = self.canvas.winfo_width()
        h_canvas = self.canvas.winfo_height()
        h_img, w_img = self.mask.shape
        
        disp_w = int(w_img * self.zoom_factor)
        disp_h = int(h_img * self.zoom_factor)
        
        img_center_x = (w_canvas // 2) + self.pan_x
        img_center_y = (h_canvas // 2) + self.pan_y
        
        img_tl_x = img_center_x - (disp_w // 2)
        img_tl_y = img_center_y - (disp_h // 2)
        
        x = int((event.x - img_tl_x) / self.zoom_factor)
        y = int((event.y - img_tl_y) / self.zoom_factor)
        
        if 0 <= x < w_img and 0 <= y < h_img:
            radius = int(self.sliders['brush_size'].get() / self.zoom_factor)
            radius = max(1, radius)
            color = 1.0 if self.brush_mode.get() == "Pintar" else 0.0
            cv2.circle(self.mask, (x, y), radius, color, -1)
            self.update_image()

    def end_brush(self, event):
        self.is_brushing = False

    def apply_edits(self, img_array, is_export=False, target_mask=None):
        img_float = img_array.astype(np.float32) / (65535.0 if is_export else 255.0)

        v_exp = self.sliders['exposure'].get() / 100.0
        v_contrast = self.sliders['contrast'].get()
        v_sat = self.sliders['saturation'].get()

        img_global = np.clip(img_float * v_contrast + v_exp, 0, 1)

        v_temp = self.sliders['temp'].get()
        if v_temp != 0:
            b, g, r = cv2.split(img_global)
            r = np.clip(r + (v_temp * 0.5), 0, 1) 
            b = np.clip(b - (v_temp * 0.5), 0, 1) 
            img_global = cv2.merge((b, g, r))

        v_hl = self.sliders['highlights'].get()
        v_shad = self.sliders['shadows'].get()
        if v_hl != 0 or v_shad != 0:
            lum = cv2.cvtColor(img_global, cv2.COLOR_BGR2GRAY)
            if v_hl != 0:
                hl_mask = np.clip((lum - 0.5) * 2.0, 0, 1)
                hl_mask = np.repeat(hl_mask[:, :, np.newaxis], 3, axis=2)
                img_global = img_global + (img_global * (v_hl * 0.5) * hl_mask)
            if v_shad != 0:
                shad_mask = np.clip((0.5 - lum) * 2.0, 0, 1)
                shad_mask = np.repeat(shad_mask[:, :, np.newaxis], 3, axis=2)
                img_global = img_global + (img_global * (v_shad * 0.5) * shad_mask)

            img_global = np.clip(img_global, 0, 1)

        gray = cv2.cvtColor(img_global, cv2.COLOR_BGR2GRAY)
        if v_sat != 1.0:
            gray_3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img_global = cv2.addWeighted(img_global, v_sat, gray_3d, 1.0 - v_sat, 0)
            
        img_global = np.clip(img_global, 0.0001, 1.0)

        if HAS_SCIPY:
            x_pts = [p[0] for p in self.curve_points]
            y_pts = [p[1] for p in self.curve_points]
            interpolator = interp.PchipInterpolator(x_pts, y_pts)
            lut_x = np.linspace(0, 1, 1024)
            lut_y = np.clip(interpolator(lut_x), 0, 1)
            img_global = np.interp(img_global, lut_x, lut_y).astype(np.float32)

        umat_img = cv2.UMat(img_global)
        
        v_ncolor = self.sliders['noise_color'].get()
        if v_ncolor > 0:
            ycrcb = cv2.cvtColor(umat_img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            blur_amount = (v_ncolor / 20.0)
            ksize = int(blur_amount) * 2 + 1
            cr = cv2.GaussianBlur(cr, (ksize, ksize), blur_amount)
            cb = cv2.GaussianBlur(cb, (ksize, ksize), blur_amount)
            ycrcb = cv2.merge([y, cr, cb])
            umat_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        v_nlum = self.sliders['noise_lum'].get()
        if v_nlum > 0:
            sigma_color = v_nlum / 500.0 
            sigma_space = v_nlum / 5.0
            umat_img = cv2.bilateralFilter(umat_img, 5, sigma_color, sigma_space)

        v_sharp = self.sliders['sharpness'].get()
        if v_sharp > 0:
            blurred = cv2.GaussianBlur(umat_img, (0, 0), 3.0)
            umat_img = cv2.addWeighted(umat_img, 1.0 + v_sharp, blurred, -v_sharp, 0)
            
        v_tex = self.sliders['texture'].get()
        if v_tex != 0:
            smoothed = cv2.bilateralFilter(umat_img, 9, 0.1, 0.1)
            if v_tex > 0: umat_img = cv2.addWeighted(umat_img, 1.0 + v_tex, smoothed, -v_tex, 0)
            else: umat_img = cv2.addWeighted(umat_img, 1.0 + v_tex, smoothed, abs(v_tex), 0)

        v_dist = self.sliders['distortion'].get()
        if abs(v_dist) > 0.001:
            h_img, w_img = img_array.shape[:2]
            focal_length = max(h_img, w_img) 
            K = np.array([[focal_length, 0, w_img/2], [0, focal_length, h_img/2], [0, 0, 1]], dtype=np.float32)
            D = np.array([v_dist, 0, 0, 0], dtype=np.float32)
            
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w_img, h_img), 1, (w_img, h_img))
            umat_img = cv2.undistort(umat_img, K, D, None, new_K)

        img_global = umat_img.get()

        mask_to_use = self.mask if target_mask is None else target_mask
        
        if mask_to_use is not None and np.max(mask_to_use) > 0:
            v_local_exp = self.sliders['local_exposure'].get() / 100.0
            v_local_contrast = self.sliders['local_contrast'].get()
            v_local_sat = self.sliders['local_saturation'].get()
            v_feather = self.sliders['brush_feather'].get()

            smooth_mask = mask_to_use.copy()
            if v_feather > 0:
                ksize = int(v_feather) * 2 + 1
                if is_export: ksize *= 5 
                smooth_mask = cv2.GaussianBlur(smooth_mask, (ksize, ksize), 0)

            mask_3d = np.repeat(smooth_mask[:, :, np.newaxis], 3, axis=2)
            img_local = np.clip(img_global * v_local_contrast + v_local_exp, 0, 1)
            
            if v_local_sat != 1.0:
                gray_local = cv2.cvtColor(img_local, cv2.COLOR_BGR2GRAY)
                gray_local_3d = cv2.cvtColor(gray_local, cv2.COLOR_GRAY2BGR)
                img_local = cv2.addWeighted(img_local, v_local_sat, gray_local_3d, 1.0 - v_local_sat, 0)

            final_float = img_global * (1.0 - mask_3d) + img_local * mask_3d

            if not is_export and self.show_mask.get():
                red_overlay = np.zeros_like(final_float)
                red_overlay[:,:,2] = 1.0 
                final_float = final_float * (1.0 - (mask_3d * 0.5)) + red_overlay * (mask_3d * 0.5)
        else:
            final_float = img_global

        return final_float

    def update_image(self):
        if self.preview_base is None: return
        img_float = self.apply_edits(self.preview_base, is_export=False)
        edited_preview = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        
        edited_rgb = cv2.cvtColor(edited_preview, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(edited_rgb)

        if self.zoom_factor != 1.0:
            new_w = int(img_pil.width * self.zoom_factor)
            new_h = int(img_pil.height * self.zoom_factor)
            if new_w > 0 and new_h > 0:
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

        self.tk_image = ImageTk.PhotoImage(image=img_pil)
        
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)
        self.update_canvas_position()

    # --- COMPARAÇÃO ANTES/DEPOIS ---
    def show_original(self, event):
        if self.preview_base is not None:
            img_pil = Image.fromarray(cv2.cvtColor(self.preview_base, cv2.COLOR_BGR2RGB))
            
            if self.zoom_factor != 1.0:
                new_w = int(img_pil.width * self.zoom_factor)
                new_h = int(img_pil.height * self.zoom_factor)
                if new_w > 0 and new_h > 0:
                    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
                    
            self.tk_image_orig = ImageTk.PhotoImage(image=img_pil)
            self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image_orig)

    def show_edited(self, event):
        self.update_image()

    # --- FUNÇÃO DE IMPRESSÃO ---
    def print_image(self):
        if self.preview_base is None: return
        try:
            temp_path = os.path.join(os.getcwd(), "temp_print.jpg")
            img_float = self.apply_edits(self.preview_base)
            cv2.imwrite(temp_path, (np.clip(img_float, 0, 1) * 255).astype(np.uint8))
            
            if sys.platform == "win32": os.startfile(temp_path, "print")
            elif sys.platform == "darwin": os.system(f"lpr {temp_path}")
            else: os.system(f"lp {temp_path}")
        except Exception as e:
            messagebox.showerror("Erro de Impressão", f"{e}")

    # --- EXPORTAÇÃO ---
    def save_image(self, fmt):
        if self.raw_image is None or self.raw_path is None: return

        ext = ".jpg" if fmt == "jpg" else ".tiff"
        save_path = filedialog.asksaveasfilename(defaultextension=ext)
        if not save_path: return

        try:
            if self.raw_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.raf')):
                with rawpy.imread(self.raw_path) as raw:
                    rgb_full = raw.postprocess(use_camera_wb=True, output_bps=16)
                bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)
            else:
                bgr_full = cv2.imread(self.raw_path)

            full_h, full_w = bgr_full.shape[:2]
            full_mask = cv2.resize(self.mask, (full_w, full_h), interpolation=cv2.INTER_LINEAR)

            img_float = self.apply_edits(bgr_full, is_export=True, target_mask=full_mask)
            final_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)

            if fmt == "jpg":
                final_uint8 = (np.clip(final_rgb, 0, 1) * 255).astype(np.uint8)
                img_pil = Image.fromarray(final_uint8)
                
                exif_data = self.generate_exif_bytes()
                if exif_data:
                    img_pil.save(save_path, "JPEG", quality=95, exif=exif_data)
                else:
                    img_pil.save(save_path, "JPEG", quality=95)
                    
            elif fmt == "tiff":
                # Usamos a imagem original img_float (que está em BGR) e não a final_rgb
                # O OpenCV espera nativamente o formato BGR para salvar
                final_uint16_bgr = (np.clip(img_float, 0, 1) * 65535).astype(np.uint16)
                cv2.imwrite(save_path, final_uint16_bgr)

            messagebox.showinfo("Sucesso", f"Salvo em:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Erro", f"{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProLightroomClone(root)
    root.mainloop()