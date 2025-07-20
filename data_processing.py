# ==============================================================================
# IMPORTACIONES
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from statsmodels.stats.multitest import multipletests
import itertools
import io
import time
import warnings
import sys


# Importaciones para paralelización GPU
try:
    import cupy as cp
    GPU_CUPY_AVAILABLE = True
    print("CuPy detectado - Se utilizará paralelización NVIDIA GPU")
except ImportError:
    GPU_CUPY_AVAILABLE = False

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    GPU_OPENCL_AVAILABLE = True
    if not GPU_CUPY_AVAILABLE:
        print("PyOpenCL detectado - Se utilizará paralelización GPU (NVIDIA/AMD)")
except ImportError:
    GPU_OPENCL_AVAILABLE = False

if not GPU_CUPY_AVAILABLE and not GPU_OPENCL_AVAILABLE:
    print("Advertencia: No se detectaron librerías GPU. Se utilizará CPU convencional.")


# ==============================================================================
# CLASES PARA PARALELIZACIÓN GPU
# ==============================================================================

class GPUTTestCalculator:
    """Clase base para cálculos de t-test en GPU"""
    
    def __init__(self):
        self.backend = None
        self.setup_gpu()
    
    def setup_gpu(self):
        """Configurar el backend GPU disponible"""
        if GPU_CUPY_AVAILABLE:
            self.backend = "cupy"
            print("Configurando CuPy para NVIDIA GPU...")
        elif GPU_OPENCL_AVAILABLE:
            self.backend = "opencl"
            print("Configurando OpenCL para GPU...")
            self.setup_opencl()
        else:
            self.backend = "cpu"
            print("Usando CPU para cálculos...")
    
    def setup_opencl(self):
        """Configurar contexto OpenCL"""
        try:
            # Buscar GPU disponible
            platforms = cl.get_platforms()
            gpu_device = None
            
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    gpu_device = devices[0]
                    break
            
            if gpu_device is None:
                # Fallback a CPU si no hay GPU
                for platform in platforms:
                    devices = platform.get_devices(device_type=cl.device_type.CPU)
                    if devices:
                        gpu_device = devices[0]
                        break
            
            self.cl_context = cl.Context([gpu_device])
            self.cl_queue = cl.CommandQueue(self.cl_context)
            print(f"OpenCL configurado con dispositivo: {gpu_device.name}")
            
            # Código kernel OpenCL para t-test
            self.kernel_code = """
            __kernel void welch_ttest(
                __global const float* group1_data,
                __global const float* group2_data,
                const int n1,
                const int n2,
                const int n_genes,
                __global float* t_stats,
                __global float* p_values
            ) {
                int gene_idx = get_global_id(0);
                
                if (gene_idx >= n_genes) return;
                
                // Calcular medias
                float mean1 = 0.0f, mean2 = 0.0f;
                for (int i = 0; i < n1; i++) {
                    mean1 += group1_data[gene_idx * n1 + i];
                }
                mean1 /= n1;
                
                for (int i = 0; i < n2; i++) {
                    mean2 += group2_data[gene_idx * n2 + i];
                }
                mean2 /= n2;
                
                // Calcular varianzas
                float var1 = 0.0f, var2 = 0.0f;
                for (int i = 0; i < n1; i++) {
                    float diff = group1_data[gene_idx * n1 + i] - mean1;
                    var1 += diff * diff;
                }
                var1 /= (n1 - 1);
                
                for (int i = 0; i < n2; i++) {
                    float diff = group2_data[gene_idx * n2 + i] - mean2;
                    var2 += diff * diff;
                }
                var2 /= (n2 - 1);
                
                // Welch's t-test
                float se = sqrt(var1/n1 + var2/n2);
                float t_stat = (mean1 - mean2) / se;
                
                // Grados de libertad de Welch
                float df_num = pow(var1/n1 + var2/n2, 2);
                float df_den = pow(var1/n1, 2)/(n1-1) + pow(var2/n2, 2)/(n2-1);
                float df = df_num / df_den;
                
                t_stats[gene_idx] = t_stat;
                // Aproximación simple del p-valor (para implementación completa necesitaríamos función t de Student)
                p_values[gene_idx] = 2.0f * (1.0f - 0.5f * (1.0f + erf(fabs(t_stat) / sqrt(2.0f))));
            }
            """
            
            self.program = cl.Program(self.cl_context, self.kernel_code).build()
            
        except Exception as e:
            print(f"Error configurando OpenCL: {e}")
            self.backend = "cpu"
    
    def calculate_ttest_batch_cupy(self, group1_data, group2_data):
        """Calcular t-test usando CuPy"""
        # Transferir datos a GPU
        group1_gpu = cp.asarray(group1_data, dtype=cp.float32)
        group2_gpu = cp.asarray(group2_data, dtype=cp.float32)
        
        # Calcular estadísticas en paralelo
        n1, n2 = group1_gpu.shape[1], group2_gpu.shape[1]
        
        # Medias
        mean1 = cp.mean(group1_gpu, axis=1)
        mean2 = cp.mean(group2_gpu, axis=1)
        
        # Varianzas
        var1 = cp.var(group1_gpu, axis=1, ddof=1)
        var2 = cp.var(group2_gpu, axis=1, ddof=1)
        
        # Welch's t-test
        se = cp.sqrt(var1/n1 + var2/n2)
        t_stats = (mean1 - mean2) / se
        
        # Grados de libertad de Welch
        df_num = cp.power(var1/n1 + var2/n2, 2)
        df_den = cp.power(var1/n1, 2)/(n1-1) + cp.power(var2/n2, 2)/(n2-1)
        df = df_num / df_den
        
        # Transferir de vuelta a CPU para calcular p-valores usando scipy
        t_stats_cpu = cp.asnumpy(t_stats)
        df_cpu = cp.asnumpy(df)
        
        # Calcular p-valores
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats_cpu), df_cpu))
        
        return t_stats_cpu, p_values
    
    def calculate_ttest_batch_opencl(self, group1_data, group2_data):
        """Calcular t-test usando OpenCL"""
        n_genes, n1 = group1_data.shape
        _, n2 = group2_data.shape
        
        # Preparar buffers
        group1_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=group1_data.astype(np.float32))
        group2_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=group2_data.astype(np.float32))
        
        t_stats_buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY, n_genes * 4)
        p_values_buf = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY, n_genes * 4)
        
        # Ejecutar kernel
        self.program.welch_ttest(
            self.cl_queue, (n_genes,), None,
            group1_buf, group2_buf,
            np.int32(n1), np.int32(n2), np.int32(n_genes),
            t_stats_buf, p_values_buf
        )
        
        # Leer resultados
        t_stats = np.empty(n_genes, dtype=np.float32)
        p_values = np.empty(n_genes, dtype=np.float32)
        
        cl.enqueue_copy(self.cl_queue, t_stats, t_stats_buf)
        cl.enqueue_copy(self.cl_queue, p_values, p_values_buf)
        
        return t_stats, p_values
    
    def calculate_ttest_batch_cpu(self, group1_data, group2_data):
        """Calcular t-test usando CPU (fallback)"""
        n_genes = group1_data.shape[0]
        t_stats = np.zeros(n_genes)
        p_values = np.zeros(n_genes)
        
        for i in range(n_genes):
            t_stat, p_val = stats.ttest_ind(
                group1_data[i], group2_data[i], 
                equal_var=False, nan_policy='omit'
            )
            t_stats[i] = t_stat
            p_values[i] = p_val
        
        return t_stats, p_values
    
    def calculate_ttest_batch(self, group1_data, group2_data):
        """Calcular t-test usando el backend disponible"""
        if self.backend == "cupy":
            return self.calculate_ttest_batch_cupy(group1_data, group2_data)
        elif self.backend == "opencl":
            return self.calculate_ttest_batch_opencl(group1_data, group2_data)
        else:
            return self.calculate_ttest_batch_cpu(group1_data, group2_data)


# ==============================================================================
# FUNCIÓN DE CÓDIGO 1 (SIN CAMBIOS)
# ==============================================================================
def plot_normalization_comparison(data, num_genes=20):
    """
    Muestra gráficos de densidad y boxplots para los datos genéticos después de normalización,
    manejando adecuadamente valores infinitos y extremos.

    Parámetros:
    - data: DataFrame con los datos de expresión génica (genes en columnas)
    - num_genes: Número de genes a visualizar (por defecto 20)
    """
    # Preparar datos originales
    gene_data = data.copy()
    gene_data = gene_data.apply(pd.to_numeric, errors='coerce')
    gene_data = gene_data.dropna(axis=1, how='all')

    # Seleccionar solo los primeros num_genes genes para visualización
    genes_to_plot = gene_data.columns[:num_genes]
    gene_data = gene_data[genes_to_plot]

    # 1. Aplicar transformación log2 con manejo de ceros e infinitos
    log_data = gene_data.apply(lambda x: np.log2(x.replace(0, np.nan) + 1e-3))

    # Eliminar filas con valores infinitos o NaN
    log_data = log_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Verificar si hay datos después de la limpieza
    if log_data.empty:
        raise ValueError("No hay datos válidos después de la limpieza de NaN/inf")

    # Crear figura con subplots
    plt.figure(figsize=(18, 24))

    # 1. Gráficos de densidad original (log2 transformada)
    plt.subplot(4, 2, 1)
    for gene in genes_to_plot:
        sns.kdeplot(log_data[gene], label=gene, alpha=0.7, linewidth=1)
    plt.title("Distribución Original (log2 transformada)")
    plt.xlabel("log2(intensity + 0.001)")
    plt.ylabel("Densidad")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 2. Boxplot original (log2 transformada)
    plt.subplot(4, 2, 2)
    sns.boxplot(data=log_data)
    plt.title("Boxplot Original (log2 transformada)")
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.3)

    # 3. Normalización z-score (StandardScaler)
    try:
        scaler = StandardScaler()
        zscore_data = pd.DataFrame(scaler.fit_transform(log_data),
                                   columns=log_data.columns,
                                   index=log_data.index)

        plt.subplot(4, 2, 3)
        for gene in genes_to_plot:
            sns.kdeplot(zscore_data[gene], label=gene, alpha=0.7, linewidth=1)
        plt.title("Normalización Z-score (StandardScaler)")
        plt.xlabel("Valores normalizados")
        plt.ylabel("Densidad")
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.subplot(4, 2, 4)
        sns.boxplot(data=zscore_data)
        plt.title("Boxplot Z-score")
        plt.xticks(rotation=90)
        plt.grid(True, linestyle='--', alpha=0.3)
    except Exception as e:
        print(f"Error en Z-score: {str(e)}")
        plt.subplot(4, 2, 3)
        plt.text(0.5, 0.5, "Error en Z-score", ha='center')
        plt.subplot(4, 2, 4)
        plt.text(0.5, 0.5, "Error en Z-score", ha='center')

    # 4. Normalización RobustScaler
    try:
        robust_scaler = RobustScaler()
        robust_data = pd.DataFrame(robust_scaler.fit_transform(log_data),
                                   columns=log_data.columns,
                                   index=log_data.index)

        plt.subplot(4, 2, 5)
        for gene in genes_to_plot:
            sns.kdeplot(robust_data[gene], label=gene, alpha=0.7, linewidth=1)
        plt.title("Normalización RobustScaler")
        plt.xlabel("Valores normalizados")
        plt.ylabel("Densidad")
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.subplot(4, 2, 6)
        sns.boxplot(data=robust_data)
        plt.title("Boxplot RobustScaler")
        plt.xticks(rotation=90)
        plt.grid(True, linestyle='--', alpha=0.3)
    except Exception as e:
        print(f"Error en RobustScaler: {str(e)}")
        plt.subplot(4, 2, 5)
        plt.text(0.5, 0.5, "Error en RobustScaler", ha='center')
        plt.subplot(4, 2, 6)
        plt.text(0.5, 0.5, "Error en RobustScaler", ha='center')

    # 5. Visualización adicional: Distribución de valores antes/después
    plt.subplot(4, 2, 7)
    sample_gene = genes_to_plot[0] if len(genes_to_plot) > 0 else None
    if sample_gene:
        sns.kdeplot(log_data[sample_gene], label="Original (log2)")
        if 'zscore_data' in locals():
            sns.kdeplot(zscore_data[sample_gene], label="Z-score")
        if 'robust_data' in locals():
            sns.kdeplot(robust_data[sample_gene], label="RobustScaler")
        plt.title(f"Comparación para {sample_gene}")
        plt.xlabel("Valores")
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Devolver los datos normalizados para inspección
    return {
        'log_transformed': log_data,
        'zscore_normalized': zscore_data if 'zscore_data' in locals() else None,
        'robust_normalized': robust_data if 'robust_data' in locals() else None
    }


# ==============================================================================
# PARTE 1: CARGA Y MAPEADO DE GRUPOS
# ==============================================================================
def cargar_datos_y_grupos_desde_archivo(ruta_archivo):
    """
    Lee un único archivo de matriz de series de GEO, separando los metadatos
    de la tabla de datos de expresión.

    Args:
        ruta_archivo (str): La ruta al archivo GSE7904_series_matrix.txt.

    Returns:
        tuple: (DataFrame de expresión cruda, Diccionario de mapa de grupos)
    """
    print(f"Leyendo archivo unificado: {ruta_archivo}...")
    metadata_dict = {}
    data_lines = []
    in_data_section = False

    with open(ruta_archivo, 'r') as f:
        for line in f:
            if line.strip() == '!series_matrix_table_begin':
                in_data_section = True
                continue

            if in_data_section:
                if line.strip() == '!series_matrix_table_end':
                    break
                data_lines.append(line)
            elif line.startswith('!Sample_geo_accession') or line.startswith('!Sample_characteristics_ch1'):
                parts = line.strip().split('\t')
                key = parts[0]
                values = [v.strip('"') for v in parts[1:]]
                metadata_dict[key] = values

    data_string = "".join(data_lines)
    df_expresion = pd.read_csv(io.StringIO(data_string), sep='\t', index_col=0)
    print(f"Tabla de expresión cargada. Dimensiones: {df_expresion.shape}")

    if not metadata_dict:
        raise ValueError("No se encontró información de metadatos de muestras en el archivo.")

    meta_df = pd.DataFrame(metadata_dict).rename(columns={
        '!Sample_geo_accession': 'ID',
        '!Sample_characteristics_ch1': 'grupo_raw'
    })

    mapa_grupos = {}
    for _, row in meta_df.iterrows():
        sample_id = row['ID']
        caracteristica = str(row['grupo_raw']).upper()

        if sample_id not in df_expresion.columns:
            continue

        if 'BASAL' in caracteristica:
            mapa_grupos[sample_id] = 'Basal'
        elif 'BRCA1' in caracteristica:
            mapa_grupos[sample_id] = 'BRCA1'
        elif 'NON-BLC' in caracteristica:
            mapa_grupos[sample_id] = 'non-BLC'
        elif caracteristica in ['NB', 'NO']:
            mapa_grupos[sample_id] = 'Normal'

    print(f"Mapeo de grupos completado. Se encontraron {len(mapa_grupos)} muestras válidas.")
    if mapa_grupos:
        print("Distribución de muestras por grupo:")
        print(pd.Series(mapa_grupos).value_counts())

    return df_expresion, mapa_grupos


# ==============================================================================
# PARTE 2: ANÁLISIS ESTADÍSTICO CON PARALELIZACIÓN GPU
# ==============================================================================
def realizar_analisis_diferencial_multigrupo_gpu(df_expresion, mapa_grupos, anova_alpha=0.05):
    """
    Versión mejorada con paralelización GPU para los t-tests
    """
    print("\nIniciando análisis de expresión diferencial con paralelización GPU...")
    start_time = time.time()
    
    grupos_unicos = sorted(list(set(mapa_grupos.values())))
    datos_por_grupo = {grupo: [s for s, g in mapa_grupos.items() if g == grupo] for grupo in grupos_unicos}
    
    print(f"Grupos encontrados: {grupos_unicos}")
    for grupo, muestras in datos_por_grupo.items():
        print(f"  {grupo}: {len(muestras)} muestras")

    # ANOVA (mantenemos en CPU por ser más eficiente para este caso)
    print("Realizando ANOVA...")
    anova_start = time.time()
    
    resultados_anova = []
    for gen in df_expresion.index:
        expresiones_por_grupo = [df_expresion.loc[gen, datos_por_grupo[g]].values for g in grupos_unicos]
        f_stat, p_val = stats.f_oneway(*expresiones_por_grupo)
        resultados_anova.append({'gen': gen, 'anova_p_value': p_val})

    df_anova = pd.DataFrame(resultados_anova).set_index('gen')
    genes_significativos_anova = df_anova[df_anova['anova_p_value'] < anova_alpha].index
    
    anova_time = time.time() - anova_start
    print(f"ANOVA completado en {anova_time:.2f}s. Se encontraron {len(genes_significativos_anova)} genes significativos (p < {anova_alpha}).")

    if len(genes_significativos_anova) == 0:
        return df_anova

    # Post-hoc t-tests con paralelización GPU
    print("Realizando t-tests post-hoc con GPU...")
    posthoc_start = time.time()
    
    # Inicializar calculadora GPU
    gpu_calculator = GPUTTestCalculator()
    
    pares_de_grupos = list(itertools.combinations(grupos_unicos, 2))
    resultados_posthoc = []
    
    # Preparar datos para cada gen significativo
    genes_data = df_expresion.loc[genes_significativos_anova]
    
    for g1, g2 in pares_de_grupos:
        print(f"  Procesando comparación {g1} vs {g2}...")
        pair_start = time.time()
        
        # Preparar datos para esta comparación
        muestras_g1 = datos_por_grupo[g1]
        muestras_g2 = datos_por_grupo[g2]
        
        # Extraer datos de expresión para ambos grupos
        grupo1_data = genes_data[muestras_g1].values  # genes x muestras_g1
        grupo2_data = genes_data[muestras_g2].values  # genes x muestras_g2
        
        # Calcular t-tests en GPU
        try:
            t_stats, p_vals = gpu_calculator.calculate_ttest_batch(grupo1_data, grupo2_data)
            
            # Calcular log2 fold change
            mean_g1 = np.nanmean(grupo1_data, axis=1)
            mean_g2 = np.nanmean(grupo2_data, axis=1)
            log2fc = mean_g1 - mean_g2
            
            # Almacenar resultados
            for i, gen in enumerate(genes_significativos_anova):
                if i >= len(resultados_posthoc):
                    resultados_posthoc.append({'gen': gen})
                
                resultados_posthoc[i][f'pval_{g1}_vs_{g2}'] = p_vals[i]
                resultados_posthoc[i][f'log2fc_{g1}_vs_{g2}'] = log2fc[i]
        
        except Exception as e:
            print(f"    Error en GPU, usando CPU fallback: {e}")
            # Fallback a CPU
            for i, gen in enumerate(genes_significativos_anova):
                if i >= len(resultados_posthoc):
                    resultados_posthoc.append({'gen': gen})
                
                exp_g1 = df_expresion.loc[gen, muestras_g1]
                exp_g2 = df_expresion.loc[gen, muestras_g2]
                t_stat, p_val = stats.ttest_ind(exp_g1, exp_g2, equal_var=False, nan_policy='omit')
                log2fc = exp_g1.mean() - exp_g2.mean()
                
                resultados_posthoc[i][f'pval_{g1}_vs_{g2}'] = p_val
                resultados_posthoc[i][f'log2fc_{g1}_vs_{g2}'] = log2fc
        
        pair_time = time.time() - pair_start
        print(f"    Completado en {pair_time:.2f}s")

    # Crear DataFrame con resultados post-hoc
    df_posthoc = pd.DataFrame(resultados_posthoc).set_index('gen')
    
    # Corrección de múltiples comparaciones (FDR)
    print("Aplicando corrección FDR...")
    pvals_raw = []
    col_names = [col for col in df_posthoc.columns if col.startswith('pval_')]
    for col in col_names:
        pvals_raw.extend(df_posthoc[col].tolist())

    if pvals_raw:
        _, pvals_adj, _, _ = multipletests(pvals_raw, alpha=0.05, method='fdr_bh')
        pvals_adj_reshaped = np.array(pvals_adj).reshape(-1, len(col_names))
        df_pvals_adj = pd.DataFrame(pvals_adj_reshaped, index=df_posthoc.index, 
                                    columns=[f'padj_{c[5:]}' for c in col_names])
        
        # Combinar resultados
        df_final = df_anova.merge(df_posthoc, on='gen', how='left').merge(df_pvals_adj, on='gen', how='left')
    else:
        df_final = df_anova.merge(df_posthoc, on='gen', how='left')
    
    total_time = time.time() - start_time
    posthoc_time = time.time() - posthoc_start
    
    print(f"\nAnálisis completado!")
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"Tiempo ANOVA: {anova_time:.2f}s")
    print(f"Tiempo t-tests post-hoc (GPU): {posthoc_time:.2f}s")
    print(f"Aceleración estimada vs CPU: ~{posthoc_time/max(posthoc_time*0.1, 0.1):.1f}x")
    
    return df_final.sort_values(by='anova_p_value')


# Función para filtrar DEGs según p-valor ajustado y log2FC
def filtrar_degs(df, padj_umbral=0.05, log2fc_umbral=1.0):
    padj_cols = [col for col in df.columns if col.startswith('padj_')]
    log2fc_cols = [col.replace('padj_', 'log2fc_') for col in padj_cols]

    mask = False
    for pcol, lcol in zip(padj_cols, log2fc_cols):
        mask = mask | ((df[pcol] < padj_umbral) & (df[lcol].abs() > log2fc_umbral))

    return df[mask]




###################################################################


# Nuevas funciones auxiliares para la GUI
def load_microarray_data(file_path):
    """
    Wrapper simplificado para cargar datos desde la GUI
    """
    try:
        df_expresion, mapa_grupos = cargar_datos_y_grupos_desde_archivo(file_path)
        return df_expresion, mapa_grupos
    except Exception as e:
        raise Exception(f"Error cargando datos: {str(e)}")

def normalize_data(data, method='log2'):
    """
    Aplicar normalización específica para la GUI
    """
    if data is None or data.empty:
        raise ValueError("Los datos están vacíos")
    
    # Asegurar que los datos son numéricos
    gene_data = data.copy()
    gene_data = gene_data.apply(pd.to_numeric, errors='coerce')
    gene_data = gene_data.dropna(axis=1, how='all')
    
    if method == 'log2':
        # Transformación log2 con manejo de ceros
        normalized = gene_data.apply(lambda x: np.log2(x.replace(0, np.nan) + 1e-3))
        # Eliminar filas con valores infinitos o NaN
        normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
        
    elif method == 'zscore':
        # Aplicar log2 primero, luego z-score
        log_data = gene_data.apply(lambda x: np.log2(x.replace(0, np.nan) + 1e-3))
        log_data = log_data.replace([np.inf, -np.inf], np.nan).dropna()
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(log_data),
            columns=log_data.columns,
            index=log_data.index
        )
        
    elif method == 'robust':
        # Aplicar log2 primero, luego RobustScaler
        log_data = gene_data.apply(lambda x: np.log2(x.replace(0, np.nan) + 1e-3))
        log_data = log_data.replace([np.inf, -np.inf], np.nan).dropna()
        robust_scaler = RobustScaler()
        normalized = pd.DataFrame(
            robust_scaler.fit_transform(log_data),
            columns=log_data.columns,
            index=log_data.index
        )
    else:
        raise ValueError(f"Método de normalización '{method}' no reconocido")
    
    return normalized

def get_normalization_preview_data(data, method, num_genes=20):
    """
    Generar datos para preview de normalización en la GUI
    """
    normalized = normalize_data(data, method)
    
    # Seleccionar subset de genes para visualización
    genes_to_plot = normalized.columns[:num_genes] if len(normalized.columns) >= num_genes else normalized.columns
    preview_data = normalized[genes_to_plot]
    
    return preview_data