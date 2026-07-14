import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CAMBIO DE COLOR: Activar modo oscuro base ---
plt.style.use('dark_background')

columnas_buscar = ['camara_comercio', 'description.ciiu', 'razon_social', 'ultimo_ano_renovado']

dataset['texto_ia'] = dataset[columnas_buscar].fillna('').astype(str).agg(' '.join, axis=1)

# 2. Diccionario para traducir Cámara de Comercio a Departamento
# (Añadimos las principales en minúsculas para evitar errores)
mapeo_deptos = {
    # Distrito Capital
    'bogota': 'Bogotá D.C.',
    
    # Casos especiales de cámaras
    'sur y oriente del tolima': 'Tolima',
    'antioquia para antioquia': 'Antioquia',
    
    # Antioquia
    'medellin': 'Antioquia', 'envigado': 'Antioquia', 'bello': 'Antioquia', 
    'rionegro': 'Antioquia', 'itagui': 'Antioquia', 'apartado': 'Antioquia', 'antioquia': 'Antioquia',
    
    # Valle del Cauca
    'cali': 'Valle del Cauca', 'palmira': 'Valle del Cauca', 'buenaventura': 'Valle del Cauca', 
    'tulua': 'Valle del Cauca', 'cartago': 'Valle del Cauca', 'buga': 'Valle del Cauca',
    
    # Cundinamarca
    'soacha': 'Cundinamarca', 'chia': 'Cundinamarca', 'zipaquira': 'Cundinamarca', 
    'facatativa': 'Cundinamarca', 'girardot': 'Cundinamarca', 'mosquera': 'Cundinamarca',
    
    # Atlántico
    'barranquilla': 'Atlántico', 'soledad': 'Atlántico', 'malambo': 'Atlántico', 
    'puerto colombia': 'Atlántico', 'sabanalarga': 'Atlántico',
    
    # Santander
    'bucaramanga': 'Santander', 'floridablanca': 'Santander', 'barrancabermeja': 'Santander', 
    'piedecuesta': 'Santander', 'giron': 'Santander',
    
    # Bolívar
    'cartagena': 'Bolívar', 'magangue': 'Bolívar', 'turbaco': 'Bolívar', 'arjona': 'Bolívar',
    
    # Boyacá
    'tunja': 'Boyacá', 'duitama': 'Boyacá', 'sogamoso': 'Boyacá', 'chiquinquira': 'Boyacá',
    
    # Caldas
    'manizales': 'Caldas', 'chinchina': 'Caldas', 'la dorada': 'Caldas', 'villamaria': 'Caldas',
    
    # Cesar
    'valledupar': 'Cesar', 'aguachica': 'Cesar', 'bosconia': 'Cesar', 'codazzi': 'Cesar',
    
    # Córdoba
    'monteria': 'Córdoba', 'cerete': 'Córdoba', 'lorica': 'Córdoba', 'sahagun': 'Córdoba',
    
    # Huila
    'neiva': 'Huila', 'pitalito': 'Huila', 'garzon': 'Huila', 'la plata': 'Huila',
    
    # Magdalena
    'santa marta': 'Magdalena', 'cienaga': 'Magdalena', 'fundacion': 'Magdalena', 'el banco': 'Magdalena',
    
    # Meta
    'villavicencio': 'Meta', 'acacias': 'Meta', 'granada': 'Meta', 'puerto lopez': 'Meta',
    
    # Nariño
    'pasto': 'Nariño', 'tumaco': 'Nariño', 'ipiales': 'Nariño', 'tuquerres': 'Nariño',
    
    # Norte de Santander
    'cucuta': 'Norte de Santander', 'ocana': 'Norte de Santander', 'pamplona': 'Norte de Santander', 'villa del rosario': 'Norte de Santander',
    
    # Quindío
    'armenia': 'Quindío', 'calarca': 'Quindío', 'quimbaya': 'Quindío', 'montenegro': 'Quindío',
    
    # Risaralda
    'pereira': 'Risaralda', 'dosquebradas': 'Risaralda', 'santa rosa de cabal': 'Risaralda',
    
    # Sucre
    'sincelejo': 'Sucre', 'corozal': 'Sucre', 'tolu': 'Sucre', 'san marcos': 'Sucre',
    
    # Tolima
    'ibague': 'Tolima', 'el espinal': 'Tolima', 'melgar': 'Tolima', 'honda': 'Tolima',
    
    # Resto de capitales y municipios relevantes
    'leticia': 'Amazonas', 'puerto narino': 'Amazonas',
    'arauca': 'Arauca', 'tame': 'Arauca', 'saravena': 'Arauca',
    'florencia': 'Caquetá', 'san vicente del caguan': 'Caquetá',
    'yopal': 'Casanare', 'aguazul': 'Casanare', 'villanueva': 'Casanare',
    'popayan': 'Cauca', 'santander de quilichao': 'Cauca', 'puerto tejada': 'Cauca',
    'quibdo': 'Chocó', 'istmina': 'Chocó', 'nuqui': 'Chocó',
    'inirida': 'Guainía',
    'san jose del guaviare': 'Guaviare', 'calamar': 'Guaviare',
    'riohacha': 'La Guajira', 'maicao': 'La Guajira', 'uribia': 'La Guajira',
    'mocoa': 'Putumayo', 'puerto asis': 'Putumayo', 'sibundoy': 'Putumayo',
    'san andres': 'San Andrés y Providencia', 'providencia': 'San Andrés y Providencia',
    'mitu': 'Vaupés',
    'puerto carreno': 'Vichada', 'la primavera': 'Vichada'
}

# Función rápida para buscar el departamento
def obtener_depto(camara):
    camara_limpia = str(camara).lower().strip()
    for clave, depto in mapeo_deptos.items():
        if clave in camara_limpia:
            return depto
    return 'Otro / No Asignado'

dataset['Departamento'] = dataset['camara_comercio'].apply(obtener_depto)

# 3. Preparación de la Búsqueda
if 'TextoBuscado' in dataset.columns and not dataset['TextoBuscado'].isna().all():
    termino_busqueda = str(dataset['TextoBuscado'].iloc[0]).strip().lower()
else:
    termino_busqueda = ""

# 4. Motor IA Rápido (NLP)
if termino_busqueda and termino_busqueda != "nan":
    # Unimos Razón Social y Actividad Económica (CIIU) para tener más contexto
    
    textos = [termino_busqueda] + dataset['texto_ia'].str.lower().tolist()
    
    vectorizer = TfidfVectorizer()
    matriz_vectores = vectorizer.fit_transform(textos)
    
    similitudes = cosine_similarity(matriz_vectores[0:1], matriz_vectores[1:]).flatten()
    dataset['Score_IA'] = similitudes
    
    # Ordenamos por mayor puntaje y tomamos el TOP 50
    dataset_top50 = dataset[dataset['Score_IA'] > 0.02].sort_values(by='Score_IA', ascending=False).head(100)
else:
    # Si no hay búsqueda, mostramos un resumen aleatorio de 50
    dataset_top50 = dataset.head(50)

# 5. Dibujar la Gráfica de Densidad
fig, ax = plt.subplots(figsize=(14, 8))

# --- CAMBIO DE COLOR: Aplicar fondo azul marino oscuro ---
color_fondo = '#0A0F1C'
fig.patch.set_facecolor(color_fondo)
ax.set_facecolor(color_fondo)

if not dataset_top50.empty:
    # Contamos cuántos de los Top 50 hay por departamento
    densidad = dataset_top50['Departamento'].value_counts().reset_index()
    densidad.columns = ['Departamento', 'Cantidad']
    
    # --- CAMBIO DE COLOR: Degradado de morados a azules (BuPu_r) e igualar borde al fondo ---
    sns.barplot(data=densidad, x='Cantidad', y='Departamento', 
                palette='BuPu_r', edgecolor=color_fondo, ax=ax)
    
    # Configuraciones visuales para que se vea premium
    titulo = f'Densidad por Departamento' if termino_busqueda else 'Densidad de Establecimientos'
    
    # --- CAMBIO DE COLOR: Textos a blanco y gris claro ---
    ax.set_title(titulo, fontsize=26, fontweight='bold', pad=20, color='white')
    ax.set_xlabel('Número de Establecimientos', fontsize=22, color='#E2E8F0')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=22, colors='#E2E8F0')
    ax.tick_params(axis='x', labelsize=14, colors='#E2E8F0')
    
    # Agregamos los números al final de cada barra
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', 
                    (p.get_width() + 0.3, p.get_y() + p.get_height() / 2.), 
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')
                    
else:
    ax.text(0.5, 0.5, f'La IA no encontró resultados relevantes para:\n"{termino_busqueda}"', 
            horizontalalignment='center', verticalalignment='center', fontsize=20, color='#E2E8F0')
    ax.axis('off')

# Ajuste de márgenes
plt.tight_layout()
plt.show()