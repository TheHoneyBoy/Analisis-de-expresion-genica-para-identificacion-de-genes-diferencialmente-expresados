
# ==============================================================================
# IMPORTACIONES PRINCIPALES
# ==============================================================================
import streamlit as st
import pandas as pd
import time
import io

# Importar funciones de los módulos del proyecto
# Asegúrate de que estos archivos estén en la misma carpeta
try:
    from data_processing import (
        load_microarray_data,
        normalize_data,
        realizar_analisis_diferencial_multigrupo_gpu,
        filtrar_degs
    )
    from visualization import (
        create_normalization_plots,
        create_results_summary_plot,
        create_comparison_plots,
        graficar_heatmap,
        graficar_boxplots_por_gen
    )
except ImportError as e:
    st.error(f"Error importando módulos. Asegúrate de que 'data_processing.py' y 'visualization.py' estén en la misma carpeta. Detalle: {e}")
    st.stop()


# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTADO DE SESIÓN
# ==============================================================================

# Configurar el layout de la página para que sea ancho y tenga un título
st.set_page_config(layout="wide", page_title="Herramienta de Análisis Génico")

# El estado de sesión (`st.session_state`) se usa para guardar variables
# entre interacciones del usuario, evitando que los datos se pierdan.
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_expresion = None
    st.session_state.mapa_grupos = None
    st.session_state.normalized_data = None
    st.session_state.normalization_method = None
    st.session_state.analysis_run = False
    st.session_state.df_results = None
    st.session_state.degs = None


def convert_df_to_csv(df):
    """Convierte un DataFrame a CSV en memoria para su descarga."""
    return df.to_csv(index=True).encode('utf-8')

# ==============================================================================
# INTERFAZ DE USUARIO (UI)
# ==============================================================================

# --- Título Principal ---
st.title("🔬 Pipeline de Análisis de Expresión Génica")
st.markdown("---")

# --- Barra Lateral de Controles ---
st.sidebar.title("Panel de Control")

# --- PASO 1: Carga de Datos ---
st.sidebar.header("Paso 1: Cargar Datos")
uploaded_file = st.sidebar.file_uploader(
    "Carga tu archivo `_series_matrix.txt`",
    type=['txt']
)

if uploaded_file is not None and not st.session_state.data_loaded:
    with st.spinner('Cargando y procesando archivo...'):
        try:
            # Para que la función lea el archivo subido, necesita un path.
            # Lo guardamos temporalmente.
            with open("temp_file.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            df_exp, mapa_g = load_microarray_data("temp_file.txt")
            
            st.session_state.df_expresion = df_exp
            st.session_state.mapa_grupos = mapa_g
            st.session_state.data_loaded = True
            
            st.sidebar.success("¡Datos cargados con éxito!")

        except Exception as e:
            st.sidebar.error(f"Error al cargar: {e}")


# Si los datos están cargados, mostrar los siguientes pasos
if st.session_state.data_loaded:
    
    # --- PASO 2: Normalización ---
    st.sidebar.header("Paso 2: Normalizar Datos")
    
    # El usuario elige el método de normalización
    norm_method = st.sidebar.selectbox(
        "Elige un método de normalización:",
        ('log2', 'zscore', 'robust'),
        key='norm_choice'
    )

    # Botón para aplicar la normalización
    if st.sidebar.button("Aplicar Normalización", key='normalize_button'):
        with st.spinner(f"Aplicando normalización '{norm_method}'..."):
            try:
                # La normalización se aplica sobre los datos crudos
                normalized = normalize_data(st.session_state.df_expresion, method=norm_method)
                st.session_state.normalized_data = normalized
                st.session_state.normalization_method = norm_method
            except Exception as e:
                st.error(f"Error durante la normalización: {e}")

    # --- PASO 3: Análisis Estadístico ---
    if st.session_state.normalized_data is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("Paso 3: Ejecutar Análisis")
        
        # Parámetros para el análisis
        padj_umbral = st.sidebar.number_input('Umbral de P-valor ajustado (padj):', 0.0, 1.0, 0.05, 0.01)
        log2fc_umbral = st.sidebar.number_input('Umbral de Log2 Fold Change:', 0.0, 5.0, 1.0, 0.5)
        
        # Botón para iniciar el análisis
        if st.sidebar.button("🚀 Iniciar Análisis Estadístico", key='run_analysis'):
            with st.spinner('Realizando análisis ANOVA y t-tests (puede tardar)...'):
                try:
                    # Se usa la data normalizada para el análisis
                    resultados = realizar_analisis_diferencial_multigrupo_gpu(
                        st.session_state.normalized_data,
                        st.session_state.mapa_grupos
                    )
                    
                    degs_filtrados = filtrar_degs(resultados, padj_umbral, log2fc_umbral)
                    
                    st.session_state.df_results = resultados
                    st.session_state.degs = degs_filtrados
                    st.session_state.analysis_run = True
                    st.success("¡Análisis completado!")

                except Exception as e:
                    st.error(f"Error en el análisis: {e}")


# ==============================================================================
# PANEL PRINCIPAL DE VISUALIZACIÓN
# ==============================================================================

if not st.session_state.data_loaded:
    st.info("👋 ¡Bienvenido! Por favor, carga tu archivo de datos en la barra lateral para comenzar.")

# --- Visualización de la Normalización ---
if st.session_state.normalized_data is not None:
    st.header(f"📊 Visualización de la Normalización: '{st.session_state.normalization_method.upper()}'")
    
    with st.spinner("Generando gráficos de normalización..."):
        fig_norm = create_normalization_plots(
            st.session_state.normalized_data,
            st.session_state.normalization_method
        )
        st.pyplot(fig_norm)
    st.markdown("---")


# --- Visualización de los Resultados del Análisis ---
if st.session_state.analysis_run:
    st.header("📈 Resultados del Análisis Diferencial")
    
    # Usar pestañas para organizar los distintos gráficos
    tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Volcano Plots", "Heatmap", "Boxplots por Gen"])
    
    with tab1:
        st.subheader("Resumen del Análisis")
        with st.spinner("Generando resumen..."):
            fig_summary = create_results_summary_plot(
                st.session_state.df_results,
                len(st.session_state.degs),
                len(st.session_state.df_expresion)
            )
            st.pyplot(fig_summary)
            
            st.subheader("Genes Expresados Diferencialmente (DEGs)")
            st.write(f"Se encontraron **{len(st.session_state.degs)}** genes que cumplen con los criterios.")
            st.dataframe(st.session_state.degs)
            
            st.download_button(
               "⬇️ Descargar DEGs - Genes Diferencialmente Expresados (CSV)",
               convert_df_to_csv(st.session_state.degs),
               "DEGs_resultados.csv",
               "text/csv",
               key='download-degs'
            )
            st.download_button(
               "⬇️ Descargar todos los resultados - Genes Potencialmente Interesantes (CSV)",
               convert_df_to_csv(st.session_state.df_results),
               "Todos_los_resultados.csv",
               "text/csv",
               key='download-all'
            )

    with tab2:
        st.subheader("Volcano Plots por Comparación")
        with st.spinner("Generando Volcano Plots..."):
            # NOTA: La función `create_comparison_plots` debe devolver una lista de figuras.
            volcano_plots = create_comparison_plots(
                st.session_state.df_results, st.session_state.mapa_grupos
            )
            if not volcano_plots:
                st.warning("No se pudieron generar los Volcano Plots. Revisa los datos de resultados.")
            else:
                for plot in volcano_plots:
                    st.write(f"**Comparación: {plot['title']}**")
                    st.pyplot(plot['figure'])

    with tab3:
        st.subheader("Heatmap de Genes Significativos")
        with st.spinner("Generando Heatmap..."):
            if not st.session_state.degs.empty:
                # NOTA: La función `graficar_heatmap` debe devolver un objeto de figura.
                fig_heatmap = graficar_heatmap(
                    st.session_state.normalized_data, 
                    st.session_state.degs, 
                    st.session_state.mapa_grupos, 
                    num_genes=50
                )
                st.pyplot(fig_heatmap)
            else:
                st.warning("No hay DEGs para mostrar en el heatmap.")
                
    with tab4:
        st.subheader("Boxplots de DEGs Individuales")
        with st.spinner("Generando Boxplots..."):
            if not st.session_state.degs.empty:
                 # NOTA: La función `graficar_boxplots_por_gen` debe devolver un objeto de figura.
                fig_boxplots = graficar_boxplots_por_gen(
                    st.session_state.normalized_data, 
                    st.session_state.degs, 
                    st.session_state.mapa_grupos, 
                    num_genes=6
                )
                st.pyplot(fig_boxplots)
            else:
                st.warning("No hay DEGs para mostrar en los boxplots.")