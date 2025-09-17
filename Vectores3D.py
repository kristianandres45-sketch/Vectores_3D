# -*- coding: utf-8 -*-
"""
Transformación y operaciones con vectores en 3D
Created on Wed Sep 12 10:42:11 2025
@author: Cristhian Almache
"""


# Correr en terminal->  streamlit run "C:\Users\ANGIE\Desktop\Proyecto_Curso\Vectores3D.py"

# =========================
# Librerías
# =========================
import os                               # Manejo de rutas y verificación de existencia de archivos
import numpy as np                      # Lo utilizamos para los cálculos numéricos
import pandas as pd                     # No lo utilizamos en el código pero es una librería importante en otras aplicaciones
import matplotlib.pyplot as plt         # Lo utilizaremos para la gráfica estática
import plotly.graph_objects as go       # Lo utilizaremos para la gráfica dinámica
import streamlit as st                  # Framework para publicar nuestra aplicación

# =============================
# Config general
# =============================

st.set_page_config(page_title="VECTORES 3D", layout="wide")       # Título de la pestaña de la aplicación

# =============================
# Parámetros (EDITABLES)
# =============================

LOGO_IZQ = r"C:\Users\ANGIE\Escritorio\Proyecto Curso\ESPE_102_años.png"       # Colocamos la ruta al logo izquierdo (opcional)
LOGO_DER = r"C:\Users\ANGIE\Escritorio\Proyecto Curso\Aguila_ESPE.jpeg"        # Colocamos la ruta al logo derecho (opcional)

# =============================
# UI: Encabezado con logos
# =============================

c1, c2, c3 = st.columns([1, 6, 1])                      # Dividimos el encabezado en 3 columnas para colocar los logos y título
with c1:
    if os.path.exists(LOGO_IZQ):                        # Verifica que el archivo de imagen exista
        st.image(LOGO_IZQ, use_container_width=True)    # Muestra el logo izquierdo
with c2:
    st.title("🔢 PROYECTO CURSO DE PYTHON")            # La segunda columna es para el Título principal en el centro
with c3:
    if os.path.exists(LOGO_DER):                        # Verifica que el archivo de imagen 2 exista
        st.image(LOGO_DER, use_container_width=True)    # Muestra el logo derecho

# =========================
# Funciones
# =========================

def magnitud(v):                             #Definimos la función magnitud
    return np.linalg.norm(v).round(2)        #Hallamos la magnitud o módulo del vector

def rumbo(x, y):                             #Definimos la función rumbo con los parámetros de entrada X e Y
    if x == 0 and y == 0:                    #Utilizamos un condicional para conocer en que cuadrante se encuentra el vector 
        return "Indeterminado"
    tetha = np.degrees(np.arctan2(x, y))     # Calculamos el ángulo theta del rumbo
    tetha = abs(tetha)                       # Convertimos el ángulo a positivo
    if x > 0 and y > 0:                      
        return f"N {tetha:.2f}° E"           # Condicionales para Primer Cuadrante
    elif x > 0 and y < 0:
        return f"S {180-tetha:.2f}° E"       # Condicionales para Cuarto Cuadrante
    elif x < 0 and y < 0:
        return f"S {180-tetha:.2f}° O"       # Condicionales para Tercer Cuadrante
    else:
        return f"N {tetha:.2f}° O"           # Condicionales para Segundo Cuadrante

def inclinacion(x, y, z):                    # Definimos la función inclinación con los parámetros de entrada X Y Z
    r = np.sqrt(x**2 + y**2 + z**2)          # Volvemos hallar el módulo del vector
    if r == 0:
        return "0°"
    ang = np.degrees(np.arcsin(z / r))       # Calculamos el ángulo de inclinación
    return f"{ang:.2f}°"                     # Convertimos número resultante en un texto para añadir los puntos geográficos y el simbolo de grados.

# =========================
#Gráfica Estática
# =========================

def graf_vector(v1, v2=None):                                                 # Definimos la función para la gráfica estática
    fig = plt.figure(figsize=(8,8))                                           # Creamos la figura utilizando la librería matplot con un tamaño de 8x8
    ax = fig.add_subplot(111, projection='3d')                                # Creamos los ejes en 3D dentro de la figura

    # Vector 1
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label="Vector 1")      # Dibujamos los vectores indicando todos sus parámetros

    # Vector 2 (opcional)
    if v2 is not None:                                                        # Con la condicional colocamos al segundo vector como un parámetros opcional 
        ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label="Vector 2")  # Dibujamos el segundo vector indicando todos sus parámetro

# Configurar límites para mostrar los 8 octantes
    max_range = max(np.max(np.abs(v1)), np.max(np.abs(v2)) if v2 is not None else 1, 1)   # Hallamos el valor más alto de las componentes del vector
    ax.set_xlim([-max_range, max_range])                                                  # Con ese valor colocamos los valores máx y min de cada eje
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
# Agregar planos para visualizar octantes 
    # Plano XY (Z=0)
    xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 10), np.linspace(-max_range, max_range, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
    
    # Plano XZ (Y=0)
    xx2, zz2 = np.meshgrid(np.linspace(-max_range, max_range, 10), np.linspace(-max_range, max_range, 10))
    yy2 = np.zeros_like(xx2)
    ax.plot_surface(xx2, yy2, zz2, color='gray', alpha=0.1)
    
    # Plano YZ (X=0)
    yy3, zz3 = np.meshgrid(np.linspace(-max_range, max_range, 10), np.linspace(-max_range, max_range, 10))
    xx3 = np.zeros_like(yy3)
    ax.plot_surface(xx3, yy3, zz3, color='gray', alpha=0.1) 
    
# Configuración de las etiquetas de los vectores 
    ax.set_xlabel("EJE X")
    ax.set_ylabel("EJE Y")
    ax.set_zlabel("EJE Z")
    ax.legend()
    ax.set_title("Gráfica Estática")

    plt.tight_layout()
    return fig

# =========================
#Gráfica Dinámica
# =========================

def graf_vector2(v1, v2=None):                        # Definimos la función para la gráfica dinámica con Plotly 
    fig = go.Figure()                                 # Creamos una figura vacía
    origin = [0, 0, 0]                                # Específicamos el punto de origen del vector
    # Graficar v1
    fig.add_trace(go.Scatter3d(
        x=[origin[0], v1[0]],
        y=[origin[1], v1[1]],
        z=[origin[2], v1[2]],
        mode='lines+markers+text',                    # Indicamos los parametros del trazo
        line=dict(color='red', width=6),              
        marker=dict(size=4),
        text=['Origen', f'v1: {v1}'],
        textposition='middle right',
        name='Vector 1'
    ))
    # Graficar v2
    if v2 is not None:
        fig.add_trace(go.Scatter3d(
            x=[origin[0], v2[0]],
            y=[origin[1], v2[1]],
            z=[origin[2], v2[2]],
            mode='lines+markers+text',
            line=dict(color='blue', width=6),
            marker=dict(size=4),
            text=['Origen', f'v2: {v2}'],
            textposition='middle left',
            name='Vector 2'
        ))
        
# Configurar los ejes para mostrar los 8 octantes
    max_range = max(np.max(np.abs(v1)), np.max(np.abs(v2)) if v2 is not None else 1, 1) + 1
        
# Agregar planos para visualizar octantes
    # Plano XY (Z=0)
    xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 10), np.linspace(-max_range, max_range, 10))
    zz = np.zeros_like(xx)
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, 'rgba(128,128,128,0.1)'], [1, 'rgba(128,128,128,0.1)']], showscale=False, name='Plano XY'))
    
    # Plano XZ (Y=0)
    xx2, zz2 = np.meshgrid(np.linspace(-max_range, max_range, 10), np.linspace(-max_range, max_range, 10))
    yy2 = np.zeros_like(xx2)
    fig.add_trace(go.Surface(x=xx2, y=yy2, z=zz2, colorscale=[[0, 'rgba(128,128,128,0.1)'], [1, 'rgba(128,128,128,0.1)']], showscale=False, name='Plano XZ'))
    
    # Plano YZ (X=0)
    yy3, zz3 = np.meshgrid(np.linspace(-max_range, max_range, 10), np.linspace(-max_range, max_range, 10))
    xx3 = np.zeros_like(yy3)
    fig.add_trace(go.Surface(x=xx3, y=yy3, z=zz3, colorscale=[[0, 'rgba(128,128,128,0.1)'], [1, 'rgba(128,128,128,0.1)']], showscale=False, name='Plano YZ'))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-max_range, max_range]),
            yaxis=dict(title='Y', range=[-max_range, max_range]),
            zaxis=dict(title='Z', range=[-max_range, max_range]),
            aspectmode='cube',  # Mantiene proporciones iguales
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Vista inicial
        ),
        title="Gráfica Dinámica",
        showlegend=True,
        width=800,
        height=600
    )
    
    return fig

# =========================
# Interfaz Streamlit
# =========================
st.title("🔢 Transformación y operaciones con vectores en 3D")
st.write("Aplicación interactiva para el cálculo de operaciones básicas de vectores en 3D, Transformación de Coordenadas Rectangulares a Coordenadas Geográficas y la Gráfica en el espacio")

# Entradas de usuario
st.sidebar.header("📥 Ingresar las componentes rectangulares del Vector")
st.sidebar.subheader("➡️ Coordenadas del Vector 1")
x1 = st.sidebar.number_input("Componente X", value=2.0)
y1 = st.sidebar.number_input("Componente Y", value=-3.0)
z1 = st.sidebar.number_input("Componente Z", value=4.0)
st.sidebar.subheader("➡️ Coordenadas del Vector 2")
x2 = st.sidebar.number_input("Componente X", value=1.0)
y2 = st.sidebar.number_input("Componente Y", value=2.0)
z2 = st.sidebar.number_input("Componente Z", value=3.0)

vector1 = np.array([x1, y1, z1])
vector2 = np.array([x2, y2, z2])

# Resultados
st.subheader("📊 Operaciones básicas")
st.write(f"**Vector 1:** {vector1}")
st.write(f"**Vector 2:** {vector2}")
st.write(f"Suma: {vector1 + vector2}")
st.write(f"Resta: {vector1 - vector2}")
st.write(f"Producto punto: {np.dot(vector1, vector2)}")
st.write(f"Producto cruz: {np.cross(vector1, vector2)}")

st.subheader("📐 Coordenadas Geográficas Vector 1")
st.write(f"Magnitud: {magnitud(vector1)}")
st.write(f"Rumbo: {rumbo(vector1[0], vector1[1])}")
st.write(f"Inclinación: {inclinacion(*vector1)}")
st.write(f"**V1 = ({magnitud(vector1)}, {rumbo(vector1[0], vector1[1])}, {inclinacion(*vector1)})**")

st.subheader("📐 Coordenadas Geográficas del Vector 2")
st.write(f"Magnitud: {magnitud(vector2)}")
st.write(f"Rumbo: {rumbo(vector2[0], vector2[1])}")
st.write(f"Inclinación: {inclinacion(*vector2)}")
st.write(f"**V2 = ({magnitud(vector2)}, {rumbo(vector2[0], vector2[1])}, {inclinacion(*vector2)})**")

# Gráficas
st.subheader("🖼️ Gráficas 3D de los vectores")

# Gráfica estática (Matplotlib)
st.write("**Gráfica Estática**")
fig_static = graf_vector(vector1, vector2)
st.pyplot(fig_static)

# Gráfica interactiva (Plotly)
st.write("**Gráfica Dinámica**")
fig_interactive = graf_vector2(vector1, vector2)
st.plotly_chart(fig_interactive, use_container_width=True)

