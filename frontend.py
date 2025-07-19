import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Configuración
GSE_ID = "GSE9476"
URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE9nnn/{GSE_ID}/suppl/{GSE_ID}_RAW.tar"
DEST_DIR = "./microarray_data"

# Descargar y extraer microarray
def download_microarray():
    os.makedirs(DEST_DIR, exist_ok=True)
    tar_path = os.path.join(DEST_DIR, f"{GSE_ID}_RAW.tar")
    
    if not os.path.exists(tar_path):
        print("Descargando archivo TAR...")
        urllib.request.urlretrieve(URL, tar_path)
        print("Descarga completa.")
    else:
        print("Archivo ya descargado.")

    print("Extrayendo archivos...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=DEST_DIR)
    print("Extracción completa.")

# Simulación de datos para visualizar
def leer_datos_simulados():
    genes = [f"Gene{i}" for i in range(100)]
    muestras = [f"Muestra{j}" for j in range(5)]
    datos = np.random.rand(100, 5)
    return genes, muestras, datos

# Gráfico de dispersión
def graficar_dispersion(genes, muestras, datos):
    plt.figure(figsize=(10, 6))
    for i in range(len(muestras)):
        plt.scatter(range(len(genes)), datos[:, i], label=muestras[i])
    plt.title("Gráfico de dispersión de genes")
    plt.xlabel("Genes")
    plt.ylabel("Expresión")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Heatmap
def graficar_heatmap(genes, muestras, datos):
    plt.figure(figsize=(10, 8))
    sns.heatmap(datos, xticklabels=muestras, yticklabels=genes, cmap="viridis")
    plt.title("Heatmap de expresión génica")
    plt.xlabel("Muestras")
    plt.ylabel("Genes")
    plt.tight_layout()
    plt.show()

# Interfaz gráfica con Tkinter
def main_app():
    def on_cargar():
        download_microarray()
        genes, muestras, datos = leer_datos_simulados()
        messagebox.showinfo("Listo", "Datos simulados cargados. Ahora puedes graficar.")
        app.genes = genes
        app.muestras = muestras
        app.datos = datos

    def on_graficar_dispersion():
        graficar_dispersion(app.genes, app.muestras, app.datos)

    def on_graficar_heatmap():
        graficar_heatmap(app.genes, app.muestras, app.datos)

    app = tk.Tk()
    app.title("Microarray Visualizador")
    app.geometry("400x200")

    ttk.Button(app, text="Descargar y cargar microarray", command=on_cargar).pack(pady=10)
    ttk.Button(app, text="Graficar dispersión", command=on_graficar_dispersion).pack(pady=10)
    ttk.Button(app, text="Graficar heatmap", command=on_graficar_heatmap).pack(pady=10)

    app.mainloop()

# Ejecutar
if __name__ == "__main__":
    main_app()
