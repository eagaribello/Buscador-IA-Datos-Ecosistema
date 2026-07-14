import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CAMBIO DE COLOR: Activar modo oscuro base ---
plt.style.use('dark_background')

columnas_buscar = ['razon_social', 'description.ciiu', 'camara_comercio', 'ultimo_ano_renovado']

dataset['texto_ia_combinado'] = dataset[columnas_buscar].fillna('').astype(str).agg(' '.join, axis=1)

# Nos aseguramos de que los años sean numéricos y eliminamos nulos
dataset['año_inicial'] = pd.to_numeric(dataset['año_inicial'], errors='coerce')
dataset['ultimo_ano_renovado'] = pd.to_numeric(dataset['ultimo_ano_renovado'], errors='coerce')

# 1. Obtener la última búsqueda del usuario
if 'TextoBuscado' in dataset.columns and not dataset['TextoBuscado'].isna().all():
    termino_busqueda = str(dataset['TextoBuscado'].iloc[0]).strip().lower()
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

dataset_limpio = dataset_filtrado.dropna(subset=['año_inicial', 'ultimo_ano_renovado']).copy()

# Convertimos a enteros
dataset_limpio['año_inicial'] = dataset_limpio['año_inicial'].astype(int)
dataset_limpio['ultimo_ano_renovado'] = dataset_limpio['ultimo_ano_renovado'].astype(int)

# --- CAMBIO 1: Obtener el año máximo dinámicamente ---
año_maximo = dataset_limpio['ultimo_ano_renovado'].max()

# Crear el rango continuo de años para el eje x
min_year = año_maximo - 30
#years_x = np.arange(min_year, año_maximo)
years_x = np.arange(min_year, año_maximo)
df_years = pd.DataFrame({'año': years_x})

# Preparar datos Línea 1: Cantidad por último año de renovación
renovacion_por_año = dataset_limpio.groupby('ultimo_ano_renovado').size().reset_index(name='cantidad_renovada')
linea1 = pd.merge(df_years, renovacion_por_año, left_on='año', right_on='ultimo_ano_renovado', how='left').fillna(0)

# Preparar datos Línea 2: Abrieron en X año y renovaron en el año_maximo
renovados_maximo = dataset_limpio[dataset_limpio['ultimo_ano_renovado'] == año_maximo]
abiertos_y_renovados = renovados_maximo.groupby('año_inicial').size().reset_index(name='cantidad_abiertos')
linea2 = pd.merge(df_years, abiertos_y_renovados, left_on='año', right_on='año_inicial', how='left').fillna(0)

# Dibujar la gráfica
fig, ax = plt.subplots(figsize=(12, 6))

# --- CAMBIO DE COLOR: Aplicar fondo azul marino oscuro ---
color_fondo = '#0A0F1C'
fig.patch.set_facecolor(color_fondo)
ax.set_facecolor(color_fondo)

# --- CAMBIO DE COLOR: Colores extraídos de la imagen (Morado y Cian) ---
color_linea1 = '#8B5CF6'
color_linea2 = '#06B6D4'

ax.plot(years_x, linea1['cantidad_renovada'], label='Total por Último Año de Renovación', color=color_linea1, marker='o', linestyle='-')
ax.plot(years_x, linea2['cantidad_abiertos'], label=f'Año matrícula, Establecimiento vigente', color=color_linea2, marker='^', linestyle='--')

# Configuración de textos
# --- CAMBIO DE COLOR: Textos a blanco y gris claro ---
ax.set_xlabel('Año', fontsize=22, fontweight='bold', color='#E2E8F0')
ax.set_ylabel('Cantidad de Establecimientos', fontsize=20, fontweight='bold', color='#E2E8F0')
ax.set_title(f'Distribución y Renovación de Establecimientos', fontsize=26, fontweight='bold', pad=15, color='white')

# --- CAMBIO 2: Eje X dinámico y legible ---
# Calculamos un salto para mostrar un máximo de 15 etiquetas en el eje X
salto_x = max(1, len(years_x) // 15) 
ax.set_xticks(years_x[::salto_x]) # Tomamos los años saltando de N en N
ax.set_xticklabels(years_x[::salto_x], rotation=45)

# --- CAMBIO DE COLOR: Ticks (números) a gris claro ---
ax.tick_params(axis='both', labelsize=20, colors='#E2E8F0')

# Detalles visuales
# --- CAMBIO DE COLOR: Leyenda flotante sin fondo blanco ---
leyenda = ax.legend(fontsize=22, frameon=False)
for texto in leyenda.get_texts():
    texto.set_color('#E2E8F0')

# --- CAMBIO DE COLOR: Cuadrícula sutil adaptada a fondo oscuro ---
ax.grid(axis='y', linestyle='-', alpha=0.15, color='#E2E8F0')
ax.grid(axis='x', linestyle=':', alpha=0.15, color='#E2E8F0')

# Ajuste automático de márgenes para que nada se corte
plt.tight_layout()
plt.show()