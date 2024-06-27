# libreria_final_tomas_hick
libreria_final_tomas_hick


README - LIBRERÍA CDIII 2024 - HICK, TOMÁS GUILLERMO

Librería de Análisis Estadístico en Python
Esta librería en Python proporciona herramientas para realizar análisis estadístico y exploración de datos de diversa índole. Fue desarrollada como parte del cursado de Ciencia de Datos intentando abarcar todos los temas tratados durante el cuatrimestre, tiene como finalidad ser entregada como parte de un examen parcial para su corrección.

Funcionalidades Principales
	•	ResumenNumérico: Clase para calcular estadísticas descriptivas como media, mediana, desviación estándar, y percentiles de un conjunto de datos.
	•	ResumenGrafico: Clase para generar gráficos estadísticos como histogramas, gráficos de densidad, y QQ plots para evaluar la distribución de los datos.
	•	GeneradoraDeDatos: Clase para generar datos aleatorios con distribuciones normales y una en partículas vista en la cátedra (BS), además de calcular funciones de densidad de probabilidad para estas distribuciones, entre otras funciones.
	•	RegresionLinealSimple: Implementación de regresión lineal simple para ajustar un modelo a datos unidimensionales, calcular coeficientes, y visualizar la recta de mejor ajuste junto con diagnósticos de residuos.
	•	RegresionLinealMultiple: Clase para ajustar modelos de regresión lineal múltiple, graficar la dispersión de datos junto con las rectas de mejor ajuste para cada variable, y analizar los residuos.
	•	RegresionLogistica: Implementación de regresión logística para evaluar el modelo, y realizar predicciones binarias. Incluye métodos para calcular la matriz de confusión y el error de clasificación.
	•	TestChiCuadrado: Clase para realizar pruebas de chi cuadrado y calcular valores p, permitiendo evaluar la bondad del ajuste de datos observados con respecto a una distribución esperada.
	•	Anova: Implementación de ANOVA para comparar medias entre grupos utilizando el paquete statsmodels. Permite calcular intervalos de confianza para las diferencias de medias y obtener un resumen detallado del modelo ANOVA ajustado.

Necesario para su uso
Para utilizar esta librería, es necesario importar las siguientes librerías estándar de Python:

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import norm

import statsmodels.api as sm

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm
