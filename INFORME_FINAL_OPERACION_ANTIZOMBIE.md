# 🧟 Reporte Final: Operación Anti-Zombie
**Fecha:** 3 de Diciembre de 2025  
**Autor:** Equipo de Ciencia de Datos - Operación Anti-Zombie

---

## 1. Resumen Ejecutivo

El proyecto **Operación Anti-Zombie** ha completado exitosamente el ciclo de vida CRISP-DM, procesando datos clínicos, epidemiológicos y de redes de contacto para combatir el brote. 

Se han logrado los siguientes hitos críticos:
- **Diagnóstico Automático:** Se desarrolló un modelo de clasificación con una precisión del **98.1%** para determinar el estado actual de un paciente (Sano, Infectado, Zombificado).
- **Predicción de Gravedad:** Un modelo de regresión capaz de estimar el "Nivel de Zombificación" futuro con un error promedio (RMSE) de **5.84** puntos.
- **Contención del Brote:** Identificación precisa de los "Super-spreaders" (pacientes con alto potencial de contagio) mediante análisis de grafos, permitiendo focalizar los esfuerzos de cuarentena.

---

## 2. Metodología

Se utilizó la metodología estándar de la industria **CRISP-DM** (Cross-Industry Standard Process for Data Mining), abarcando las siguientes fases:

1.  **Entendimiento de Datos (EDA):** Análisis univariado, bivariado y multivariado de 4 datasets principales (Pacientes, Evolución, Tratamientos, Red de Contagios).
2.  **Preparación de Datos:** Limpieza de valores nulos, imputación estadística, ingeniería de características (e.g., `Zombification_Velocity`, `Composite_Risk_Score`) y codificación de variables.
3.  **Modelado:** Entrenamiento de algoritmos de Clasificación, Regresión y Clustering (K-Means, Hierarchical).
4.  **Evaluación:** Comparación rigurosa mediante métricas como F1-Score, RMSE, Curvas ROC y Silhouette Score.

---

## 3. Hallazgos Clave

### 3.1 Correlaciones Clínicas
Se descubrieron relaciones fuertes que determinan la irreversibilidad de la infección:
*   **Nivel de Consciencia vs. Zombificación:** Correlación inversa casi perfecta (**-0.95**). A menor consciencia, mayor nivel de transformación.
*   **Signos Vitales:** La caída drástica de la presión arterial y temperatura corporal son los indicadores tempranos más fiables de una zombificación inminente.

### 3.2 Dinámica del Brote
*   El brote sigue un comportamiento exponencial en sus primeras fases.
*   Los tratamientos experimentales muestran una eficacia variable, siendo el **Costo de Producción** un factor limitante para el despliegue masivo.

### 3.3 Análisis de Redes (Contagio)
Se identificó que el contagio no es aleatorio. Un pequeño porcentaje de infectados (Super-spreaders) es responsable de la mayoría de las transmisiones.
*   **Paciente Cero y Nodos Críticos:** Se detectaron nodos con alta centralidad de intermediación que actúan como puentes entre comunidades de sanos e infectados.

---

## 4. Visualizaciones y Resultados

### 4.1 Análisis de Correlaciones
El siguiente mapa de calor muestra las variables más influyentes en el dataset de pacientes.

![Matriz de Correlación](resultados/eda/bivariate/correlacion_matriz_pacientes.png)

### 4.2 Análisis de Redes de Contagio
Visualización de la red de contactos, destacando los nodos más conectados (Super-spreaders).

![Grafo de Red](resultados/eda/network/network_graph.png)

### 4.3 Rendimiento de Modelos de Clasificación
Comparativa de los algoritmos evaluados para predecir el `Estado_Actual`.

![Comparación Modelos Clasificación](resultados/reports/model_comparison_classification.png)

#### Matriz de Confusión (Mejor Modelo: Decision Tree)
El modelo **Decision Tree** obtuvo un F1-Score de **0.9810**. A continuación se muestra su capacidad para distinguir entre clases.

![Matriz Confusión Decision Tree](resultados/classification/visualizations/confusion_matrix_decision_tree_clf.png)

### 4.4 Rendimiento de Modelos de Regresión
Comparativa para la predicción del `Nivel_Zombificacion`. El modelo **Random Forest** fue el ganador con el menor error.

![Comparación Modelos Regresión](resultados/reports/model_comparison_regression.png)

---

## 5. Conclusiones

1.  **Viabilidad del Triaje IA:** Con un 98% de acierto, el sistema puede automatizar el diagnóstico en campo, liberando personal médico para atender casos críticos.
2.  **Ventana de Actuación:** Las variables temporales (`Dias_Desde_Tratamiento`, `Dias_Incubacion`) son críticas. La intervención temprana reduce el `Nivel_Zombificacion` final significativamente.
3.  **Segmentación:** El clustering reveló 4 perfiles claros de pacientes, sugiriendo que no existe una "cura única", sino que se requieren protocolos diferenciados por segmento.

---

## 6. Recomendaciones

### Acciones Inmediatas (Contención)
Basado en el análisis de centralidad de red, se recomienda el **aislamiento inmediato** y vigilancia estricta de los siguientes pacientes (Top 5 Prioridad de Intervención):

1.  **Paciente P0544** (Prioridad: 0.0035)
2.  **Paciente P0591** (Prioridad: 0.0034)
3.  **Paciente P0447** (Prioridad: 0.0032)
4.  **Paciente P0478** (Prioridad: 0.0031)
5.  **Paciente P0439** (Prioridad: 0.0031)

### Estrategia de Tratamiento
*   Desplegar el modelo **Decision Tree** en dispositivos móviles para los equipos de respuesta rápida.
*   Priorizar el uso de recursos en pacientes clasificados como "Infectado Leve" con alta probabilidad de transición a "Grave" según el modelo de regresión, maximizando el ROI de los tratamientos limitados.

---
*Generado automáticamente por el Pipeline Operación Anti-Zombie*
