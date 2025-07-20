
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


# ==============================================================================
# PARTE 3: VISUALIZACIÓN Y RESULTADOS (SIN CAMBIOS)
# ==============================================================================
def graficar_volcano_plot(df_resultados, grupo_a, grupo_b, log2fc_umbral=1.0, padj_umbral=0.05):
    print(f"\nGenerando Volcano Plot para la comparación: {grupo_a} vs {grupo_b}...")
    logfc_col, padj_col = f'log2fc_{grupo_a}_vs_{grupo_b}', f'padj_{grupo_a}_vs_{grupo_b}'

    if logfc_col not in df_resultados.columns:
        logfc_col_inv = f'log2fc_{grupo_b}_vs_{grupo_a}'
        padj_col_inv = f'padj_{grupo_b}_vs_{grupo_a}'
        if logfc_col_inv not in df_resultados.columns:
            print(f"Advertencia: No se encontraron columnas para la comparación {grupo_a} vs {grupo_b}.")
            return
        df_resultados[logfc_col] = -df_resultados[logfc_col_inv]
        df_resultados[padj_col] = df_resultados[padj_col_inv]

    df_plot = df_resultados[[logfc_col, padj_col]].dropna().copy()
    df_plot['-log10_padj'] = -np.log10(df_plot[padj_col].replace(0, np.finfo(float).eps))
    df_plot['condicion'] = 'No significativo'
    df_plot.loc[(df_plot[logfc_col] > log2fc_umbral) & (df_plot[padj_col] < padj_umbral), 'condicion'] = f'Sobre-expresado en {grupo_a}'
    df_plot.loc[(df_plot[logfc_col] < -log2fc_umbral) & (df_plot[padj_col] < padj_umbral), 'condicion'] = f'Sub-expresado en {grupo_a}'

    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df_plot, x=logfc_col, y='-log10_padj', hue='condicion', palette={'No significativo': 'grey', f'Sobre-expresado en {grupo_a}': 'red', f'Sub-expresado en {grupo_a}': 'blue'}, alpha=0.6)
    plt.axvline(x=log2fc_umbral, linestyle='--', color='black', lw=1)
    plt.axvline(x=-log2fc_umbral, linestyle='--', color='black', lw=1)
    plt.axhline(y=-np.log10(padj_umbral), linestyle='--', color='black', lw=1)
    plt.title(f'Volcano Plot: {grupo_a} vs {grupo_b}')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10(P-valor ajustado)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title='Condición')
    plt.show()


def graficar_heatmap(df_expresion, degs, mapa_grupos, num_genes=50):
    """
    Genera un heatmap de los genes diferencialmente expresados.
    """
    print(f"\nGenerando heatmap para los {min(num_genes, len(degs))} DEGs principales...")

    # Asegurarse que degs está en formato lista
    if isinstance(degs, pd.DataFrame):
        genes = degs.index.tolist()
    else:
        genes = degs

    # Seleccionar solo los genes DEGs y las muestras disponibles
    genes = genes[:num_genes]  # limitar a los más importantes
    df_subset = df_expresion.loc[genes]

    # Ordenar columnas (muestras) por grupo
    muestras_ordenadas = sorted(df_subset.columns, key=lambda x: mapa_grupos.get(x, "Z"))
    df_subset = df_subset[muestras_ordenadas]

    # Crear mapa de colores por grupo
    grupos = [mapa_grupos[m] for m in muestras_ordenadas]
    grupo_palette = sns.color_palette("Set2", len(set(grupos)))
    grupo_dict = {g: grupo_palette[i] for i, g in enumerate(sorted(set(grupos)))}
    col_colors = [grupo_dict[g] for g in grupos]

    # Plot
    cluster_map = sns.clustermap(df_subset, col_colors=col_colors, cmap='vlag', figsize=(14, 10), z_score=0)
    plt.title('Heatmap de DEGs (z-score por gen)')
    #plt.show()
    return cluster_map.fig

def graficar_boxplots_por_gen(df_expresion, degs, mapa_grupos, num_genes=6):
    """
    Genera boxplots para los DEGs seleccionados.
    """
    print(f"\nGenerando boxplots para {min(num_genes, len(degs))} DEGs...")

    # Seleccionar genes y preparar el DataFrame largo para seaborn
    genes = degs.index.tolist()[:num_genes]
    df_subset = df_expresion.loc[genes]
    df_largo = df_subset.T.reset_index().melt(id_vars='index', var_name='Gen', value_name='Expresión')
    df_largo = df_largo.rename(columns={'index': 'Muestra'})
    df_largo['Grupo'] = df_largo['Muestra'].map(mapa_grupos)

    # Crear gráfico
    plt.figure(figsize=(18, 6))
    sns.boxplot(x='Gen', y='Expresión', hue='Grupo', data=df_largo, palette='Set2')
    plt.title("Boxplots de expresión para genes diferencialmente expresados")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    #plt.show()
    return plt.gcf()

#########################################################################################################

# Nuevas funciones para la GUI
def create_normalization_plots(data, method_name, num_genes=20):
    """
    Crear gráficos de normalización para la GUI
    """
    if data is None or data.empty:
        return None
        
    # Seleccionar genes para visualización
    genes_to_plot = data.columns[:num_genes] if len(data.columns) >= num_genes else data.columns
    plot_data = data[genes_to_plot]
    
    # Crear figura
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico de densidad
    ax1 = axes[0]
    for gene in genes_to_plot:
        if gene in plot_data.columns:
            sns.kdeplot(data=plot_data[gene], ax=ax1, label=gene, alpha=0.7, linewidth=1.5)
    
    ax1.set_title(f'Distribución de Expresión - {method_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Valores de Expresión')
    ax1.set_ylabel('Densidad')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Boxplot
    ax2 = axes[1]
    sns.boxplot(data=plot_data, ax=ax2, palette='viridis')
    ax2.set_title(f'Boxplot de Expresión - {method_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Genes')
    ax2.set_ylabel('Valores de Expresión')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_results_summary_plot(resultados, degs_count, total_genes):
    """
    Crear gráfico resumen de resultados del análisis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribución de p-valores ANOVA
    ax1 = axes[0, 0]
    if 'anova_p_value' in resultados.columns:
        ax1.hist(resultados['anova_p_value'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax1.set_xlabel('P-valor ANOVA')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de P-valores (ANOVA)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Genes significativos vs no significativos
    ax2 = axes[0, 1]
    labels = ['DEGs', 'No significativos']
    sizes = [degs_count, total_genes - degs_count]
    colors = ['#ff6b6b', '#4ecdc4']
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                       startangle=90, explode=(0.05, 0))
    ax2.set_title('Proporción de Genes Significativos', fontweight='bold')
    
    # 3. Heatmap de correlación entre comparaciones (si hay múltiples comparaciones)
    ax3 = axes[1, 0]
    pval_cols = [col for col in resultados.columns if col.startswith('padj_')]
    if len(pval_cols) > 1:
        corr_data = resultados[pval_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Correlación entre Comparaciones\n(P-valores ajustados)', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Correlación no disponible\n(Una sola comparación)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Correlación entre Comparaciones', fontweight='bold')
    
    # 4. Top 10 genes más significativos
    ax4 = axes[1, 1]
    if not resultados.empty:
        top_genes = resultados.nsmallest(10, 'anova_p_value')
        y_pos = np.arange(len(top_genes))
        ax4.barh(y_pos, -np.log10(top_genes['anova_p_value']), color='lightcoral')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_genes.index, fontsize=8)
        ax4.set_xlabel('-log10(P-valor)')
        ax4.set_title('Top 10 Genes Más Significativos', fontweight='bold')
        ax4.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_comparison_plots(resultados, mapa_grupos):
    """
    Crear todos los volcano plots para las comparaciones por pares
    """
    grupos_unicos = sorted(list(set(mapa_grupos.values())))
    pares_de_grupos = list(itertools.combinations(grupos_unicos, 2))
    
    plots = []
    for grupo_a, grupo_b in pares_de_grupos:
        try:
            fig = create_single_volcano_plot(resultados, grupo_a, grupo_b)
            if fig is not None:
                plots.append({
                    'figure': fig,
                    'title': f'{grupo_a} vs {grupo_b}',
                    'comparison': (grupo_a, grupo_b)
                })
        except Exception as e:
            print(f"Error creando volcano plot para {grupo_a} vs {grupo_b}: {e}")
    
    return plots

def create_single_volcano_plot(df_resultados, grupo_a, grupo_b, log2fc_umbral=1.0, padj_umbral=0.05):
    """
    Crear un volcano plot individual para dos grupos
    """
    logfc_col = f'log2fc_{grupo_a}_vs_{grupo_b}'
    padj_col = f'padj_{grupo_a}_vs_{grupo_b}'
    
    # Verificar si las columnas existen, si no, intentar la comparación inversa
    if logfc_col not in df_resultados.columns:
        logfc_col_inv = f'log2fc_{grupo_b}_vs_{grupo_a}'
        padj_col_inv = f'padj_{grupo_b}_vs_{grupo_a}'
        if logfc_col_inv not in df_resultados.columns:
            return None
        
        # Usar comparación inversa con fold change negativo
        df_plot = df_resultados[[logfc_col_inv, padj_col_inv]].dropna().copy()
        df_plot[logfc_col] = -df_plot[logfc_col_inv]  # Invertir el fold change
        df_plot[padj_col] = df_plot[padj_col_inv]
    else:
        df_plot = df_resultados[[logfc_col, padj_col]].dropna().copy()
    
    if df_plot.empty:
        return None
    
    # Preparar datos para el plot
    df_plot['-log10_padj'] = -np.log10(df_plot[padj_col].replace(0, np.finfo(float).eps))
    
    # Clasificar genes
    df_plot['categoria'] = 'No significativo'
    up_mask = (df_plot[logfc_col] > log2fc_umbral) & (df_plot[padj_col] < padj_umbral)
    down_mask = (df_plot[logfc_col] < -log2fc_umbral) & (df_plot[padj_col] < padj_umbral)
    
    df_plot.loc[up_mask, 'categoria'] = f'↑ en {grupo_a}'
    df_plot.loc[down_mask, 'categoria'] = f'↓ en {grupo_a}'
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colores para cada categoría
    colors = {
        'No significativo': '#808080',
        f'↑ en {grupo_a}': '#d62728',
        f'↓ en {grupo_a}': '#2ca02c'
    }
    
    # Scatter plot
    for categoria in df_plot['categoria'].unique():
        mask = df_plot['categoria'] == categoria
        ax.scatter(df_plot.loc[mask, logfc_col], 
                  df_plot.loc[mask, '-log10_padj'],
                  c=colors.get(categoria, '#808080'),
                  label=categoria,
                  alpha=0.6,
                  s=20)
    
    # Líneas de referencia
    ax.axvline(x=log2fc_umbral, linestyle='--', color='black', alpha=0.5)
    ax.axvline(x=-log2fc_umbral, linestyle='--', color='black', alpha=0.5)
    ax.axhline(y=-np.log10(padj_umbral), linestyle='--', color='black', alpha=0.5)
    
    # Etiquetas y título
    ax.set_xlabel('Log2 Fold Change', fontsize=12)
    ax.set_ylabel('-Log10(P-valor ajustado)', fontsize=12)
    ax.set_title(f'Volcano Plot: {grupo_a} vs {grupo_b}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig