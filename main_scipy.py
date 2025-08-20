"""
import numpy as np
from scipy.sparse import csr_matrix

calificaciones_peliculas = np.array([
        [5, 0, 0, 0, 3],
        [0, 4, 0, 2, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 3, 0, 0],
        [0, 2, 0, 0, 0]
    ])

# Convertir a matriz dispersa CSR
matriz_csr = csr_matrix(calificaciones_peliculas)

print("Valores no ceros:", matriz_csr.data)
print("Índices columnas:", matriz_csr.indices)
print("Punteros de filas (indptr):", matriz_csr.indptr)


import numpy as np
from scipy.sparse import csr_matrix

datos_usuarios = [1, 2, 3, 4]
indices_columna_usuarios = [0, 2, 2, 0]
indptr_usuarios = [0, 2, 3, 4]
matriz_usuarios = csr_matrix((datos_usuarios, indices_columna_usuarios, indptr_usuarios), shape=(3, 3) )
datos_productos = [5, 6, 7]
indices_columna_productos = [1, 0, 2]
indptr_productos = [0, 1, 2, 3]
matriz_productos = csr_matrix((datos_productos, indices_columna_productos, indptr_productos), shape=(3, 3) )
sum_matriz = matriz_usuarios + matriz_productos
mult_matriz = matriz_usuarios.dot(matriz_productos)
print("Matriz de usuarios:\n", matriz_usuarios.toarray())
print("Matriz de productos:\n", matriz_productos.toarray())
print("Suma de matrices CSR:")
print(sum_matriz.toarray())
print("Multiplicación de matrices CSR:")
# La celda [i][j] del resultado es el producto punto entre la fila i de matriz_usuarios y la columna j de matriz_productos.
print(mult_matriz.toarray())
# Fila 0 de A: [1, 0, 2]
# Columna 0 de B: [0, 6, 0] → 1x0 + 0x6 + 2x0 = 0
# Fila 0 de A: [1, 0, 2]
# Columna 1 de B: [5, 0, 0] → 1x5 + 0x0 + 2x0 = 5
# Fila 0 de A: [1, 0, 2]
# Columna 2 de B: [0, 0, 7] → 1x0 + 0x0 + 2x7 = 14
print("Transpuesta de la matriz multiplicada:")
print(mult_matriz.transpose().toarray())




import numpy as np
from scipy.sparse import csr_matrix
datos_originales = np.array([
                    [0, 0, 3],
                    [4, 0, 0],
                    [0, 1, 0],
                    [3, 0, 5]
                ])
matriz_csr = csr_matrix(datos_originales)
print("Matriz original:\n", datos_originales)
print ("Matriz CSR:", matriz_csr)
print("Número de elementos no cero:", matriz_csr.count_nonzero())
matriz_csr[1, 1] = 0
print("Matriz CSR después de ingresar un valor 0:", matriz_csr)
print("Número de elementos no ceros:", matriz_csr.nnz)
matriz_csr.eliminate_zeros()
print("Matriz CSR después de eliminar ceros:", matriz_csr)
print("Número de elementos no ceros:", matriz_csr.nnz)



import numpy as np 
from scipy. linalg import det
matrix = np.array([
        [4, 2, 3, 1],
        [2, 3, 1, 4],
        [1, 1, 2, 2],
        [2, 3, 2, 4]
    ])
determinante = det(matrix)
print("El determinante de la matriz es:", determinante)
if np.isclose(determinante, 0): # Tiene una tolerancia.
    print("No tiene una solución única y el sistema podría ser inestable.")
else:
    print("El puente es estable.")


import numpy as np 
from scipy. linalg import eig
rigidez_edificio = np.array ([
                    [6, -2, 1],
                    [-2, 5, -2],
                    [1, -2, 4]
                ])
eigenvalues, eigenvectors = eig(rigidez_edificio)
print("Valores propios:\n", eigenvalues)
print("Vectores propios:\n", eigenvectors)



from scipy.spatial import distance
ubicacion_maria = [2, 3]
ubicacion_ana = [5, 1]
cafe_1 = [3, 2]
cafe_2 = [5, 3]
distancia_maria_cafe1 = distance.euclidean(ubicacion_maria, cafe_1)
print ("Distancia entre María y el Café 1:", distancia_maria_cafe1)
distancia_ana_cafe1 = distance.euclidean(ubicacion_ana, cafe_1)
print("Distancia entre Ana y el Café 1:", distancia_ana_cafe1)
distancia_maria_cafe2 = distance.euclidean(ubicacion_maria, cafe_2)
print("Distancia entre María y el Café 2:", distancia_maria_cafe2)
distancia_ana_cafe2 = distance.euclidean(ubicacion_ana, cafe_2)
print("Distancia entre Ana y el Café 2:", distancia_ana_cafe2)
distancia_total_cafe1 = distancia_maria_cafe1 + distancia_ana_cafe1
distancia_total_cafe2 = distancia_maria_cafe2 + distancia_ana_cafe2

if distancia_total_cafe1 < distancia_total_cafe2:
    print("El Café 1 es el punto de encuentro más cercano para Ana y María.")
elif distancia_total_cafe1 > distancia_total_cafe2:
    print("El Café 2 es el punto de encuentro más cercano para Ana y María.")
else:
    print("Ambos cafés están a la misma distancia total para Ana y María.")




import matplotlib.pyplot as plt
from scipy.spatial import distance

# Coordenadas
ubicacion_maria = [2, 3]
ubicacion_ana = [5, 1]
cafe_1 = [3, 2]
cafe_2 = [5, 3]

# Calcular distancias Manhattan
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

dist_maria_cafe1 = manhattan(ubicacion_maria, cafe_1)
dist_maria_cafe2 = manhattan(ubicacion_maria, cafe_2)
dist_ana_cafe1 = manhattan(ubicacion_ana, cafe_1)
dist_ana_cafe2 = manhattan(ubicacion_ana, cafe_2)

# Mostrar distancias
print("Distancias tipo Manhattan:")
print(f"María -> Café 1: {dist_maria_cafe1} cuadras")
print(f"María -> Café 2: {dist_maria_cafe2} cuadras")
print(f"Ana   -> Café 1: {dist_ana_cafe1} cuadras")
print(f"Ana   -> Café 2: {dist_ana_cafe2} cuadras")

# Crear el gráfico
plt.figure(figsize=(8, 8))
plt.grid(True)

# Dibujar puntos
plt.plot(*ubicacion_maria, 'ro', label="María")
plt.plot(*ubicacion_ana, 'bo', label="Ana")
plt.plot(*cafe_1, 'g^', label="Café 1")
plt.plot(*cafe_2, 'm^', label="Café 2")

# Dibujar rutas tipo Manhattan (en L)
def dibujar_ruta(origen, destino, color):
    x0, y0 = origen
    x1, y1 = destino
    plt.plot([x0, x1], [y0, y0], color, linestyle='--')  # Movimiento en X
    plt.plot([x1, x1], [y0, y1], color, linestyle='--')  # Movimiento en Y

# Rutas de María
dibujar_ruta(ubicacion_maria, cafe_1, 'g')
dibujar_ruta(ubicacion_maria, cafe_2, 'm')

# Rutas de Ana
dibujar_ruta(ubicacion_ana, cafe_1, 'g')
dibujar_ruta(ubicacion_ana, cafe_2, 'm')

# Ajustes visuales
plt.xticks(range(0, 8))
plt.yticks(range(0, 8))
plt.xlabel("X (cuadras)")
plt.ylabel("Y (cuadras)")
plt.title("Rutas de María y Ana a los Cafés (Manhattan)")
plt.legend()
plt.axis('equal')
plt.show()



# Calcular el volumen total de agua en un tanque
# usando la integral definida de la función de velocidad
from scipy.integrate import quad
#f(t) = 2t^2 + 3t + 1
def velocidad(t):
    return 2*t**2 + 3*t + 1

resultado, error = quad(velocidad, 0, 5)
print(f"El volumen total de agua: {resultado} unidades cúbicas")
print(f"Error estimado: {error}")





from scipy.integrate import dblquad
# Definir la función que representa la superficie del terreno
# f(x, y) = x^2 + y^2
def funcion_terreno(x, y):
    return x**2 + y**2

# Calcular la integral doble de la función sobre el área:
# x va de 0 a 2, y va de 0 a 1 para cada x
# dblquad(func, x_min, x_max, y_min_func, y_max_func)
resultado, error = dblquad(funcion_terreno, 0, 2, lambda x: 0, lambda x: 1)

# Mostrar el resultado y el error estimado
print(f"El volumen total de tierra que necesita ser nivelado es: {resultado} unidades cúbicas")
print(f"Error estimado del cálculo: {error}")



import numpy as np
from scipy import stats
grupo_vitamina = [1.1, 2.3, 1.8, 2.5, 2.2, 3.0, 2.9, 3.1, 2.8, 3.3]
grupo_agua = [0.5, 0.6, 0.4, 0.8, 0.5, 0.7, 0.9, 0.6, 0.8, 0.5]
# t_prueba: el valor del estadístico t de la prueba.
# p_valor: la probabilidad de obtener una diferencia igual o mayor por azar si no hubiera diferencia real.
t_prueba, p_valor = stats.ttest_ind(grupo_vitamina, grupo_agua)
print(f"Estadístico t: {t_prueba}")
print(f"P-valor: {p_valor}")
if p_valor < 0.05: # Nivel de significancia del 5%. p < 0.05. Improbable que la diferencia sea por azar
    print("La diferencia entre los grupos es significativa. Por lo tanto, puedes tener confianza razonable en que la vitamina realmente tiene un efecto diferente que el agua (en el contexto de tu experimento).")
else:
    print("No hay una diferencia significativa entre los grupos.")



import numpy as np
from scipy import stats
horas_estudio = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
notas_examen = [50, 55, 60, 65, 70, 75, 78, 82, 85, 88]
pendiente_linea_regresion, interseccion, r_valor, p_valor, std_err = stats.linregress(horas_estudio, notas_examen)
print(f"Pendiente: {pendiente_linea_regresion}")
print(f"Intersección: {interseccion}")
print(f"Valor R-cuadrado: {r_valor**2}")
print(f"p-valor: {p_valor}")
horas_futuras = 15
nota_predicha = pendiente_linea_regresion * horas_futuras + interseccion
print(f"Predicción de nota con {horas_futuras} horas de estudio: {nota_predicha}")



import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage 
import matplotlib.pyplot as plt
usuarios = np.array ([
            [13, 10],
            [14, 12],
            [5, 9],
            [15, 3],
            [14, 4],
            [16, 5],
            [25, 1],
            [24, 2],
            [26, 1]
        ]) # [horas viendo películas, horas viendo series
Z = linkage(usuarios, method="ward")
clustering = fcluster(Z, 3, criterion='maxclust')
print("Clústeres asignados a los usuarios:", clustering)
plt.scatter(usuarios[:,0], usuarios[:,1], c=clustering, cmap='rainbow')
plt.title("Agrupación de Usuarios")
plt.xlabel("Horas viendo películas")
plt.ylabel("Horas viendo series")
plt.show()




import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Definir la tasa de muestreo y el tiempo total de la señal
tasa_muestreo = 1000  # muestras por segundo
tiempo_total = 1.0    # duración en segundos

# Crear el vector de tiempo
t = np.linspace(0.0, tiempo_total, int(tasa_muestreo * tiempo_total), endpoint=False)

# Definir las frecuencias de la guitarra y el bajo
frecuencia_guitarra = 150  # Hz
frecuencia_bajo = 60       # Hz

# Generar la señal sumando dos ondas senoidales (guitarra y bajo)
sonido = np.sin(frecuencia_guitarra * 2.0 * np.pi * t) + 0.5 * np.sin(frecuencia_bajo * 2.0 * np.pi * t)

# Calcular la Transformada Rápida de Fourier (FFT) de la señal
yf = fft(sonido)

# Calcular los valores de frecuencia correspondientes a la FFT
xf = fftfreq(int(tasa_muestreo * tiempo_total), 1/tasa_muestreo)

# Crear la figura para graficar
plt.figure(figsize=(10, 6))

# Graficar la señal en el dominio del tiempo
plt.subplot(2, 1, 1) 
plt.plot(t, sonido)
plt.title("Sonido de la Guitarra y el Bajo en el Tiempo")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Amplitud")

# Graficar la magnitud de la FFT (dominio de la frecuencia)
plt.subplot(2, 1, 2)
plt.plot(xf[:tasa_muestreo//2], np.abs(yf[:tasa_muestreo//2]))
plt.title("Transformada Rápida de Fourier (Frecuencias)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()

# Ajustar el espacio entre los subplots y mostrar el gráfico
plt.subplots_adjust(hspace=0.5)
plt.show()


"""

import numpy as np
from scipy.fftpack import dct, idct 
import matplotlib.pyplot as plt 
from PIL import Image
imagen_real = Image.open("planta.png")
imagen_real = imagen_real.convert("L")
imagen_np = np.array(imagen_real)
print(imagen_np)
dct_imagen = dct(dct(imagen_np.T, norm='ortho').T, norm='ortho')
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Transformada de Coseno Discreta")
plt.imshow(dct_imagen, cmap='gray')
plt.colorbar()

dct_imagen_comprimida = dct_imagen.copy()
dct_imagen_comprimida[50:, 50:] = 0
imagen_comprimida = idct(idct(dct_imagen_comprimida.T, norm='ortho').T, norm='ortho')

plt.subplot (1, 2, 2)
plt.title("Imagen Comprimida" ) 
plt.imshow(imagen_comprimida, cmap='gray')
plt.colorbar()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
imagen_color = Image.open("planta.png")
imagen_np = np.array(imagen_color)
print(imagen_np)
imagen_desenfocada = ndimage.gaussian_filter(imagen_np, sigma=1) # sigma controla qué tanto desenfoque se aplicará
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow (imagen_np)
plt.subplot(1, 2, 2)
plt.title("Imagen Desenfocada")
plt.imshow(imagen_desenfocada)
plt.show()



import numpy as np 
from scipy.spatial import distance 
import matplotlib.pyplot as plt

escuelas = np.array([
    [37.77, -122.42],
    [37.78, -122.43],
    [37.76, -122.45],
    [37.79, -122.40]
])
hospitales = np.array([
    [37.80, -122.41],
    [37.74, -122.44],
])
distancias_escuela_hospital_1 = distance.cdist(escuelas, [hospitales [0]], 'euclidean')
print("Distancias entre las escuelas y el Hospital 1:")
print(distancias_escuela_hospital_1)
plt.figure(figsize=(8, 6))
plt.scatter(escuelas[:,1], escuelas[:,0], color='blue', label='Escuelas', marker='o')
plt. scatter(hospitales[:,1], hospitales[:,0], color='red', label='Hospitales', marker='x')

plt.title("Distribución de Escuelas y Hospitales")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.legend()
plt.show()