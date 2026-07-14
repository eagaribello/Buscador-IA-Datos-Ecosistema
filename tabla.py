import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = dataset.copy().reset_index(drop=True)

# --- CAMBIO DE COLOR: Activar modo oscuro base ---
plt.style.use('dark_background')

columnas_buscar = ['nme_grupo_gr', 'nme_categoria_pd']

# Usamos 'df' en lugar de 'dataset' en todo el código
df['texto_ia_combinado'] = df[columnas_buscar].fillna('').astype(str).agg(' '.join, axis=1)

# 2. Obtener el texto buscado
if 'TextoBuscado' in df.columns and not df['TextoBuscado'].isna().all():
    termino_busqueda = str(df['TextoBuscado'].iloc[-1]).strip().lower()
else:
    termino_busqueda = ""

# 3. Motor de Búsqueda IA
if termino_busqueda and termino_busqueda != "nan":
    
    # --- SOLUCIÓN 2: Agregar .tolist() para sumar correctamente ---
    textos = [termino_busqueda] + df['texto_ia_combinado'].tolist()
    
    vectorizer = TfidfVectorizer()
    matriz_vectores = vectorizer.fit_transform(textos)
    
    similitudes = cosine_similarity(matriz_vectores[0:1], matriz_vectores[1:]).flatten()
    df['Score_IA'] = similitudes
    
    # Extraemos el Top 5
    top_5 = df.sort_values(by='Score_IA', ascending=False).head(5).copy()
else:
    top_5 = df.head(5).copy()

# 4. Construcción de la matriz (Sin emojis ni sangrías manuales)
tabla_datos = []
mapa_colores = {} 
es_url = {} 

fila_fisica_idx = 1
for resultado_idx, (index, row) in enumerate(top_5.iterrows()):
    nombre = str(row['nme_grupo_gr']).strip()
    
    # Ancho máximo de caracteres
    nombre_wrap = textwrap.fill(nombre, width=120) 
        
    cod_grupo = str(row.get('cod_grupo_gr', '')).strip()
    id_persona = str(row.get('id_persona_pd', '')).strip()
    
    # Limpieza de código
    if cod_grupo.startswith("COL"):
        cod_grupo = cod_grupo[3:]
        
    # Validamos ambos enlaces
    url_grupo = "No disponible"
    if cod_grupo and cod_grupo.lower() != "nan":
        url_grupo = f"https://scienti.minciencias.gov.co/gruplac/jsp/visualiza/visualizagr.jsp?nro={cod_grupo}"
        
    url_persona = "No disponible"
    if id_persona and id_persona.lower() != "nan":
        url_persona = f"https://scienti.minciencias.gov.co/cvlac/visualizador/generarCurriculoCv.do?cod_rh={id_persona}"
    
    # --- CAMBIO DE COLOR: Efecto Cebra en Modo Oscuro ---
    # -- FILA 1: Nombre del Grupo --
    tabla_datos.append([f"{nombre_wrap}"])
    mapa_colores[fila_fisica_idx] = '#141A29' if resultado_idx % 2 == 0 else '#0A0F1C'
    es_url[fila_fisica_idx] = False
    fila_fisica_idx += 1
    
    # -- FILA 2: URL del Grupo --
    tabla_datos.append([f"Grup: {url_grupo}"])
    mapa_colores[fila_fisica_idx] = '#1E293B' if resultado_idx % 2 == 0 else '#0F172A'
    es_url[fila_fisica_idx] = True
    fila_fisica_idx += 1

    # -- FILA 3: URL del Investigador --
    tabla_datos.append([f"CvLAC: {url_persona}"])
    mapa_colores[fila_fisica_idx] = '#1E293B' if resultado_idx % 2 == 0 else '#0F172A'
    es_url[fila_fisica_idx] = True
    fila_fisica_idx += 1

# Rellenar espacio vacío con el color de fondo oscuro
while len(tabla_datos) < 15: 
    tabla_datos.append(["-"])
    mapa_colores[fila_fisica_idx] = '#0A0F1C'
    es_url[fila_fisica_idx] = False
    fila_fisica_idx += 1

# 5. Dibujar la tabla
fig = plt.figure(figsize=(20, 9)) 

# --- CAMBIO DE COLOR: Fondo principal del lienzo ---
color_fondo = '#0A0F1C'
fig.patch.set_facecolor(color_fondo)

# Área de dibujo forzando márgenes mínimos
ax = fig.add_axes([0.02, 0.02, 0.96, 0.85]) 
ax.axis('off')

columnas = ["Grupos e Investigadores Seleccionados - Enlaces Oficiales ScienTI"]

# Creamos la tabla alineada a la izquierda
tabla_visual = ax.table(
    cellText=tabla_datos, 
    colLabels=columnas, 
    loc='center', 
    cellLoc='left', 
    bbox=[0, 0, 1, 1] 
)

tabla_visual.auto_set_font_size(False)
tabla_visual.set_fontsize(25)

# Aplicar colores, estilos y eliminar la sangría
for (row, col), cell in tabla_visual.get_celld().items():
    
    # Sobrescribimos la sangría gigante que causa el bbox y la forzamos a ser minúscula (0.2%)
    cell.PAD = 0.002
    
    if row == 0:
        cell.set_text_props(weight='bold', color='white', fontsize=25)
        # --- CAMBIO DE COLOR: Encabezado oscuro institucional ---
        cell.set_facecolor('#1E293B')
        cell.set_edgecolor(color_fondo)
    else:
        # Aplicamos los colores oscuros de las filas
        cell.set_facecolor(mapa_colores.get(row, color_fondo))
        cell.set_edgecolor('#1E293B') # Bordes oscuros para no desentonar
        
        if es_url.get(row, False):
            # --- CAMBIO DE COLOR: Cian brillante para los links ---
            cell.set_text_props(fontsize=25, style='italic', color='#06B6D4')
        else:
            # --- CAMBIO DE COLOR: Gris claro para el texto de los nombres ---
            cell.set_text_props(weight='semibold', color='#E2E8F0')

titulo = 'Grupos e investigadores relacionados' if termino_busqueda else 'Grupos e investigadores relacionados'

# --- CAMBIO DE COLOR: Título a color blanco ---
fig.text(0.5, 0.94, titulo, ha='center', va='center', fontsize=30, fontweight='bold', color='white')

plt.show()