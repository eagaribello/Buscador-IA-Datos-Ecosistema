import sys
import subprocess

# 1. Lista de librerías que tu dashboard necesita para la IA y las gráficas
librerias_requeridas = ["pandas", "matplotlib", "seaborn", "sentence-transformers", "sklearn", "textwrap"]

# 2. Bucle inteligente: Solo instala lo que te haga falta
for libreria in librerias_requeridas:
    try:
        # Intenta cargar la librería
        __import__(libreria)
    except ImportError:
        # Si no existe, la instala apuntando al Python activo en Power BI
        subprocess.check_call([sys.executable, "-m", "pip", "install", libreria])