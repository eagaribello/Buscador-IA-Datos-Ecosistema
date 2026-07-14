import matplotlib.pyplot as plt
import pandas as pd
import textwrap 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = dataset.copy().reset_index(drop=True)

# --- CAMBIO DE COLOR: Activar modo oscuro base ---
plt.style.use('dark_background')

columnas_buscar = ['nme_producto_pd', 'nme_categoria_pd']

dataset['texto_ia_combinado'] = dataset[columnas_buscar].fillna('').astype(str).agg(' '.join, axis=1)

# 1. Obtener la última búsqueda del usuario usando el Índice real
if 'TextoBuscado' in df.columns and not df['TextoBuscado'].isna().all():
    termino_busqueda = str(df['TextoBuscado'].iloc[-1]).strip().lower()
else:
    termino_busqueda = ""

# 2. Motor de IA Rápido (NLP)
if termino_busqueda and termino_busqueda != "nan":
    textos = [termino_busqueda] + dataset['texto_ia_combinado'].str.lower().tolist()
    
    vectorizer = TfidfVectorizer()
    matriz_vectores = vectorizer.fit_transform(textos)
    
    similitudes = cosine_similarity(matriz_vectores[0:1], matriz_vectores[1:]).flatten()
    
    dataset['Score_IA'] = similitudes
    dataset_filtrado = dataset[dataset['Score_IA'] > 0.1]
else:
    dataset_filtrado = dataset

# 3. Preparar los datos
datos_torta = dataset_filtrado['nme_categoria_pd'].value_counts()

# 4. Dibujar la gráfica
fig, ax = plt.subplots(figsize=(16, 8))

# --- CAMBIO DE COLOR: Aplicar fondo azul marino oscuro ---
color_fondo = '#0A0F1C'
fig.patch.set_facecolor(color_fondo)
ax.set_facecolor(color_fondo)

if not datos_torta.empty:
    top_categorias = datos_torta.head(5)
    
    if len(datos_torta) > 5:
        otros = pd.Series([datos_torta.iloc[5:].sum()], index=['Otros'])
        top_categorias = pd.concat([top_categorias, otros])
        
    # --- CAMBIO DE COLOR: Paleta extraída de la imagen para la torta ---
    colores = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#06B6D4', '#EC4899']
    
    # Dibujamos la torta principal
    wedges, texts, autotexts = ax.pie(
           top_categorias.values, 
           autopct='%1.1f%%', 
           startangle=140, 
           colors=colores, 
           # Cambiamos el edgecolor de 'white' al color de fondo para mantener la estética oscura
           wedgeprops={'edgecolor': color_fondo, 'linewidth': 2},
           textprops={'fontsize': 24, 'fontweight': 'bold', 'color': 'white'} 
    )
    
    # Aumentamos el límite de caracteres a 45 porque ahora la leyenda tendrá mucho más espacio
    etiquetas_ajustadas = [textwrap.fill(str(texto), width=45) for texto in top_categorias.index]
    
    # Creamos la leyenda empujándola hacia el espacio vacío
    leyenda = ax.legend(wedges, etiquetas_ajustadas,
              title="Categorías",
              loc="center left",
              bbox_to_anchor=(1.1, 0.5), # El '1.1' saca la leyenda de la caja de la torta hacia la derecha
              fontsize=22, 
              title_fontsize=22,
              frameon=False)
              
    # --- CAMBIO DE COLOR: Forzar textos de la leyenda a blanco ---
    plt.setp(leyenda.get_title(), color='white')
    for text in leyenda.get_texts():
        text.set_color('#E2E8F0') # Gris muy claro para contraste elegante
           
    titulo = f'Categorías de grupos relacionados' if termino_busqueda else 'Distribución General de Categorías'
    #titulo = termino_busqueda
    
    # --- CAMBIO DE COLOR: Título a blanco ---
    ax.set_title(titulo, fontsize=34, fontweight='bold', pad=20, color='white')
    
    # --- LA MAGIA DE ALINEACIÓN ---
    # Le decimos a la torta que solo ocupe la mitad izquierda (del 5% al 50% del área total)
    # Esto deja el 50% restante (toda la derecha) completamente libre para la leyenda.
    plt.subplots_adjust(left=0.05, right=0.5)

else:
    ax.text(0.5, 0.5, f'La IA no encontró resultados para:\n"{termino_busqueda}"', 
            horizontalalignment='center', verticalalignment='center', fontsize=22, color='#E2E8F0')
    ax.axis('off')

# ELIMINAMOS plt.tight_layout() para que no interfiera con nuestra alineación manual
plt.show()