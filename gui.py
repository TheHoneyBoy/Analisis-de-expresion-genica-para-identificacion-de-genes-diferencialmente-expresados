import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from data_loader import load_microarray_data, normalize_data

class NormalizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Normalización de Microarray")
        self.root.geometry("500x300")
        
        self.file_path = "./data/GSE7904_series_matrix.txt"
        self.original_data = None
        self.current_data = None
        self.temp_normalized = None
        
        self.create_main_interface()
    
    def create_main_interface(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Botón para cargar datos
        ttk.Button(
            self.main_frame,
            text="Cargar Datos",
            command=self.load_data
        ).pack(pady=10, fill=tk.X)
        
        # Etiqueta de estado
        self.status_label = ttk.Label(
            self.main_frame,
            text="Presione 'Cargar Datos' para comenzar",
            wraplength=400
        )
        self.status_label.pack(pady=10)
        
        # Frame para botones de normalización
        self.norm_frame = ttk.Frame(self.main_frame)
        self.norm_frame.pack(pady=10)
        
        # Botones de normalización (inicialmente deshabilitados)
        ttk.Button(
            self.norm_frame,
            text="Normalización Log2",
            command=lambda: self.show_normalization('log2'),
            state=tk.DISABLED
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Button(
            self.norm_frame,
            text="Normalización Z-score",
            command=lambda: self.show_normalization('zscore'),
            state=tk.DISABLED
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Button(
            self.norm_frame,
            text="Normalización Robust",
            command=lambda: self.show_normalization('robust'),
            state=tk.DISABLED
        ).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # Configurar columnas para que se expandan
        self.norm_frame.columnconfigure(0, weight=1)
        self.norm_frame.columnconfigure(1, weight=1)
        self.norm_frame.columnconfigure(2, weight=1)
    
    def load_data(self):
        try:
            self.original_data = load_microarray_data(self.file_path)
            self.current_data = self.original_data.copy()
            
            self.status_label.config(
                text=f"Datos cargados correctamente\n"
                     f"Genes: {self.original_data.shape[0]}, Muestras: {self.original_data.shape[1]}"
            )
            
            # Habilitar botones de normalización
            for btn in self.norm_frame.winfo_children():
                btn.config(state=tk.NORMAL)
                
        except Exception as e:
            self.status_label.config(text=f"Error al cargar datos: {str(e)}")
    
    def show_normalization(self, method):
        try:
            # Aplicar normalización temporal
            self.temp_normalized = normalize_data(self.original_data, method)
            
            # Crear ventana de visualización de gráficos
            plot_window = tk.Toplevel(self.root)
            plot_window.title(f"Normalización: {method}")
            plot_window.geometry("800x600")
            
            # Crear figura con subplots
            fig = plt.figure(figsize=(10, 6))
            
            # Gráfico de densidad
            ax1 = fig.add_subplot(2, 1, 1)
            genes_to_plot = self.temp_normalized.columns[:20]
            for gene in genes_to_plot:
                sns.kdeplot(self.temp_normalized[gene], ax=ax1, label=gene, alpha=0.7)
            ax1.set_title(f"Distribución ({method})", fontsize=12)
            ax1.set_xlabel("Valores normalizados")
            ax1.set_ylabel("Densidad")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Boxplot
            ax2 = fig.add_subplot(2, 1, 2)
            samples = self.temp_normalized[genes_to_plot]
            sns.boxplot(data=samples, ax=ax2, palette="viridis")
            ax2.set_title(f"Boxplot ({method})", fontsize=12)
            ax2.set_xlabel("Muestras")
            ax2.set_ylabel("Valores normalizados")
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Canvas para matplotlib
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            
            # Crear ventana separada para los botones
            button_window = tk.Toplevel(self.root)
            button_window.title("Confirmación")
            button_window.geometry("300x100")
            
            # Frame para botones
            button_frame = ttk.Frame(button_window, padding="10")
            button_frame.pack(expand=True, fill=tk.BOTH)
            
            # Botón para aceptar normalización
            ttk.Button(
                button_frame,
                text="Elegir esta normalización",
                command=lambda: self.accept_normalization(plot_window, button_window),
                style='Accent.TButton'
            ).pack(side=tk.LEFT, expand=True, padx=5)
            
            # Botón para cancelar
            ttk.Button(
                button_frame,
                text="Cancelar",
                command=lambda: self.cancel_normalization(plot_window, button_window),
                style='Cancel.TButton'
            ).pack(side=tk.LEFT, expand=True, padx=5)
            
            # Configurar estilos para los botones
            style = ttk.Style()
            style.configure('Accent.TButton', foreground='white', background='#4CAF50')
            style.configure('Cancel.TButton', foreground='white', background='#F44336')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar {method}:\n{str(e)}")
    
    def accept_normalization(self, plot_window, button_window):
        """Acepta la normalización y cierra las ventanas"""
        if self.temp_normalized is not None:
            self.current_data = self.temp_normalized
            self.status_label.config(
                text=f"Normalización aplicada correctamente\n"
                     f"Datos listos para análisis adicional"
            )
            
            # Guardar los datos normalizados
            try:
                self.current_data.to_csv("datos_normalizados.csv")
                messagebox.showinfo("Éxito", "Datos guardados en 'datos_normalizados.csv'")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudieron guardar los datos:\n{str(e)}")
        
        plot_window.destroy()
        button_window.destroy()
    
    def cancel_normalization(self, plot_window, button_window):
        """Cancela la normalización y cierra las ventanas"""
        plot_window.destroy()
        button_window.destroy()