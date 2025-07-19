from gui import NormalizationApp
import tkinter as tk
from tkinter import ttk

if __name__ == "__main__":
    root = tk.Tk()
    
    # Estilo moderno
    style = ttk.Style(root)
    style.theme_use('clam')
    
    app = NormalizationApp(root)
    root.mainloop()