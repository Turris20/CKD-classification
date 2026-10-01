[README.md](https://github.com/user-attachments/files/32926504/README.md)
# CKD-classification

**Calculadora de enfermedad renal crónica (ERC) con interfaz gráfica que compara tres clasificadores implementados desde cero en GNU Octave: Naive Bayes gaussiano, una red neuronal y un clasificador lineal.**

![Octave](https://img.shields.io/badge/GNU%20Octave-MATLAB-0790C0?logo=octave&logoColor=white)
![ML](https://img.shields.io/badge/machine%20learning-desde%20cero-8A2BE2)
![Dominio](https://img.shields.io/badge/dominio-salud%20%2F%20nefrología-2E8B57)

> **English summary:** Chronic kidney disease (CKD) calculator with a graphical user interface. Three classifiers are implemented **from scratch** in GNU Octave, without machine-learning toolboxes: a Gaussian Naive Bayes, a one-hidden-layer neural network trained with hand-written backpropagation, and a linear classifier on hemoglobin and packed cell volume. The GUI takes 24 clinical variables and shows the three models' predictions side by side.

---

## ¿Qué hace?

El usuario captura 24 variables clínicas de un paciente (edad, presión arterial, creatinina sérica, hemoglobina, etc.) en una interfaz gráfica. Al pulsar **Calculate**, la aplicación consulta tres modelos distintos y muestra si cada uno predice que el paciente tiene o no ERC. Ver varias opiniones a la vez permite comparar cómo razona cada tipo de modelo.

Cada campo de la interfaz tiene un botón de ayuda **ⓘ** que explica la unidad o la codificación esperada (por ejemplo, `normal: 1 / anormal: 0`).

## Los tres modelos

Todos están implementados a mano, sin *toolboxes* de aprendizaje automático, para entender qué ocurre dentro de cada algoritmo.

| Modelo | Script | Entradas | Implementación |
|---|---|---|---|
| **Naive Bayes gaussiano** | `bayesianockd.m` | 24 variables normalizadas | Media y desviación estándar por clase, probabilidades *a priori* y verosimilitud con `normpdf` |
| **Red neuronal** | `redneuronalckd.m` | 24 variables normalizadas | Arquitectura 24 → 5 (tanh) → 2 (softmax), entropía cruzada, *backpropagation* escrito a mano, regularización L2 y reentrenamiento hasta superar 90 % de exactitud |
| **Clasificador lineal** | `svmckd.m` | Hemoglobina y hematocrito | Función de costo logística optimizada con `fminunc`; grafica la frontera de decisión y los pares de puntos más cercanos entre clases |

Los modelos se guardan en archivos `.mat` solo si alcanzan un **F1 > 90 %** en el conjunto de prueba, y la interfaz (`interfazingles.m`) los carga para predecir.

## Datos

Se usa el conjunto **Chronic Kidney Disease** del repositorio UCI Machine Learning, con 24 variables clínicas por paciente. Los datos están limpios y separados por clase en archivos de entrenamiento y prueba:

| Archivo | Contenido |
|---|---|
| `datasetckd_bien_train_limpio.csv` / `datasetckd_mal_train_limpio.csv` | Entrenamiento: pacientes sin ERC / con ERC |
| `datasetckd_bien_test.csv` / `datasetckd_mal_test.csv` | Prueba: pacientes sin ERC / con ERC |

Las variables se normalizan al rango [0.1, 1] con el mínimo y el máximo del conjunto de entrenamiento, y esos mismos valores se reutilizan en prueba y en la interfaz para evitar fuga de información.

## Cómo ejecutarlo

Requisitos: [GNU Octave](https://octave.org/) con los paquetes `statistics`, `optim` e `io`.

```octave
pkg install -forge statistics optim io   % solo la primera vez

% 1. Entrenar los modelos (cada script imprime matriz de confusión, exactitud, precisión, recall y F1)
bayesianockd     % genera modelo_nb.mat
redneuronalckd   % genera el archivo .mat de la red
svmckd           % genera modelo_svm.mat

% 2. Abrir la calculadora
interfazingles
```

## Estructura

```
.
├── bayesianockd.m      # Naive Bayes gaussiano
├── redneuronalckd.m    # Red neuronal con backpropagation manual
├── svmckd.m            # Clasificador lineal sobre hemoglobina y hematocrito
├── interfazingles.m    # Interfaz gráfica (en inglés) que combina los tres modelos
└── datasetckd_*.csv    # Datos de entrenamiento y prueba por clase
```

## Lo que aprendí

- Cómo funcionan por dentro tres familias de clasificadores, al implementar sus ecuaciones sin librerías: probabilístico (Bayes), conexionista (red neuronal) y lineal.
- A derivar y programar el *backpropagation* de una red con tanh y softmax.
- A evaluar modelos con matriz de confusión, precisión, *recall* y F1, y a normalizar los datos de prueba con los parámetros de entrenamiento.
- A integrar varios modelos en una herramienta con interfaz gráfica pensada para un usuario no técnico.

---

 **Contacto:** [GitHub @Turris20](https://github.com/Turris20)
