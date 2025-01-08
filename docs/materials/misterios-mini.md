<div class="content-2columns" markdown>
![](assets/misterios/imgs/gato.jpg){: .rounded-title-img}

# Revelando los misterios de la IA
</div>

Esta es una versión reducida del documento de apoyo del taller *Revelando los misterios de la IA*. El documento completo está disponible en [aquí][completo].

[completo]: misterios.md

### Un poco de contexto histórico

1. ¿Cuándo se construyó el primer computador moderno?
2. ¿Cuándo se acuñó el término inteligencia artificial?
3. ¿Cuándo se vendió la primera videoconsola?
4. ¿Cuándo ganó una máquina al ajedrez por primera vez a un humano?
5. ¿Cuándo se empezó a jugar a videojuegos multijugador en red?
6. ¿Cuándo propuso alguien que las máquinas son una especie con mecanismos de evolución propios?
7. ¿Cuándo se crearon los primeros programas de correo electrónico?
8. ¿Cuándo se diseñó el primer lenguaje de programación?
9. ¿Qué cosas que se veían como futuristas en las películas de ciencia ficción de hace unos años son hoy posibles?
10. ¿Cuáles siguen pareciendo futuristas?
11. ¿Cuándo tendremos una inteligencia artificial de propósito general?
12. ¿Cómo te imaginas que funciona ChatGPT?
13. ¿Desde cuándo puede tener una persona una conversación natural con una máquina?
14. ¿Qué aplicaciones tiene la inteligencia artificial actual?
15. ¿Quién pondrá el primer pie en Ganímedes? ¿Y en alguno de los planetas (por descubrir) de Alfa Centauri?
16. ¿Qué riesgos ves en la inteligencia artificial?
17. ¿Qué beneficios ves en la inteligencia artificial?

### Redes neuronales

- Aunque existen otras aproximaciones, las redes neuronales han permitido avances espectaculares.

```mermaid
graph LR
    input("Entrada (números)") --> nn["Red neuronal<br>(operaciones)"]
    nn --> output("Salida (números)")
```

- Las redes neuronales son funciones matemáticas que transforman vectores de entrada en vectores de salida.
- Probablemente has estudiado $y = f(x)$ en donde $x$ y $y$ son números reales (*escalares*).
- Aquí funciones las funciones son *multivariables*: $y_1, y_2, y_3 = f(x_1, x_2, x_3, x_4, x_5)$.

| Números a la entrada representan...  | Números a la salida representan...    | Aplicación de la red neuronal             |
|--------------------------------------|---------------------------------------|-------------------------------------------|
| Píxeles de una imagen                | Probabilidad entre 0 y 1 de que haya un gato      | Detectar gatos en imágenes                |
| Palabras de una frase                | Píxeles de una imagen                 | Generar imágenes a partir de descripciones|
| Señal de voz                         | Palabras de un texto                             | Transcribir voz a texto                   |
| Palabras en un idioma                | Palabras en otro idioma               | Traducir automáticamente                  |
| Palabras de una reseña de un restaurante| Estimación de opinión buena, mala o regular | Clasificar reseñas                |
| Palabras de un texto                 | Probabilidad de la siguiente palabra  | Continuar textos (**modelo de lengua**)                         |

### Representación de palabras como números a la entrada de una red neuronal

!!! note "Ejercicio"

    Considera el intervalo $[0,1]$ de números reales y la tarea de asignar un número distinto que represente cada una de las palabras de la siguiente lista, de forma que palabras que tengan un significado similar o que puedan aparecer en las mismas frases tengan números cercanos. 
    
    La lista de palabras es: *Júpiter*, *gato*, *minino*, *pequeño*, *sombrero*, *domingo*, *Ganímedes*.

!!! note "Ejercicio"

    Prueba ahora a colocar cada palabra en un cuadrado en el espacio bidimensional $[0,1]\times[0,1]$. Observa que las palabras ahora se representarían con vectores como $[0.2, 0.3]$.

- Las palabras se representan como vectores (*embeddings*) de alta dimensión (1000-10000 dimensiones).
- Los embeddings permiten a las redes neuronales generalizar entre palabras relacionadas.

### Representación de las probabilidades de las palabras a la salida de una red neuronal

- Los vectores de salida representan distribuciones de probabilidad para cada palabra del vocabulario.

!!! note "Ejercicio"

    ¿Qué quiere decir que el vector de salida representa una distribución de probabilidad?


- Podríamos decidir que la primera dimensión representa la probabilidad de que la siguiente palabra sea "atardecer", la segunda dimensión la probabilidad de que sea "desde", la tercera la probabilidad de que sea "viajamos", etc.
- "Hoy hace un día muy...". La red neuronal podría decidir que la probabilidad de que la siguiente palabra sea *caluroso* es de 0.25, la de que sea *frío* es de 0.18, la de que sea *lluvioso* es de 0.05, etc.

!!! note "Ejercicio"

    Considera la siguiente frase: "Hoy hace un día de invierno muy...". ¿Cómo cambiarían las probabilidades anteriores?

!!! note "Ejercicio"

    Considera diferentes frases y qué palabras podrían seguir a ellas con mucha o poca probabilidad.

- Generar probabilidades permite obtener múltiples continuaciones coherentes de un mismo texto.

### Entrenamiento, generalización e inferencia

- Las redes neuronales aprenden ajustando parámetros internos para minimizar el error en los datos de entrenamiento.
- Los datos de entrenamiento están formados por las entradas y las salidas deseadas.
- Nos interesa aproximar "bastante" la salida de la red a la salida deseada, pero no del todo.
- El *sobreentrenamiento* ocurre cuando la red se ajusta demasiado a los ejemplos vistos y no *generaliza* bien.
- Durante la inferencia, los parámetros permanecen congelados.

### Generación de textos con modelos de lengua

- Los modelos de lengua predicen la siguiente palabra basándose en el contexto previo.
- Si hacemos lo anterior iterativamente, podemos generar texto de cualquier longitud.
- Dado "¿Cuál es la capital de Francia?", el modelo puede generar estas probabilidades:

| Siguiente palabra | Probabilidad |
|---------|--------------|
| París | 0.2 |
| La | 0.1 |
| ... | ... |
| Londres | 0.05 |
| Madrid | 0.01 |
| Francia | 0.01 |
| ... | ... |
| Desde | 0.001 |
| ... | ... |

- Si escogemos la palabra con mayor probabilidad, ahora le damos al modelo "¿Cuál es la capital de Francia? La" y nos devuelve:

| Siguiente palabra | Probabilidad |
|---------|--------------|
| París | 0.2 |
| La | 0.1 |
| capital | 0.05 |
| Londres | 0.05 |
| Madrid | 0.01 |
| Francia | 0.01 |

- "¿Cuál es la capital de Francia? La capital". 
- Y así sucesivamente hasta conseguir idealmente algo como "¿Cuál es la capital de Francia? La capital de Francia es París".

### Datos de entrenamiento

- Los datos se extraen de textos grandes como la Wikipedia y se dividen en fragmentos.
- Simplificación: modelo de lengua que predice la siguiente palabra a partir de las 2 anteriores.
- Texto para usar en el entrenamiento: "Hilbert propuso una lista amplia de 23 problemas".
- Datos de entrenamiento:

| Entrada | Salida deseada |
|---------|--------|
| Hilbert propuso | una |
| propuso una | lista |
| una lista | amplia |
| lista amplia | de |
| amplia de | 23 |
| de 23 | problemas |

- Los modelos actuales (como ChatGPT, Gemini o Claude) se han entrenado con textos no repetidos de billones de palabras.
- De ahí, que generen comportamientos coherentes e "inteligentes".

!!! note "Ejercicio"

    ¿A cuántas veces la saga completa de Harry Potter equivale un billón de palabras? ¿Cómo nos referimos en inglés a un billón?

!!! note "Ejercicio"

    ¿Cómo se representa numéricamente la salida deseada de la red neuronal en el ejemplo anterior? ¿Y las palabras de la entrada?

### Un modelo muy simple de red neuronal

- Veamos un modelo simple, pero muy parecido al que se usaba en los teclados predictivos de los teléfonos móviles.

![](assets/misterios/imgs/teclado.jpg){: .rounded-title-img}

- Las redes básicas usan matrices de pesos para transformar las entradas en salidas.
- Ejemplo: Un modelo con 2000 valores de entrada (2 palabras a razón de 1000 componentes por palabra) y 20000 valores de salida (tamaño de nuestro vocabulario).

$$
[y_1, y_2, \ldots, y_{20000}] = [x_1, x_2, \ldots, x_{2000}] \cdot W
$$

donde $W$ es lo que se conoce como una matriz de *parámetros* (también llamados *pesos*):

$$
W = \begin{bmatrix}
w_{1,1} & w_{1,2} & \ldots & w_{1,20000} \\
w_{2,1} & w_{2,2} & \ldots & w_{2,20000} \\
\vdots & \vdots & \ddots & \vdots \\
w_{2000,1} & w_{2000,2} & \ldots & w_{2000,20000} \\
\end{bmatrix}
$$

```mermaid
graph LR
    input("Vector de entrada<br>2000 componentes<br>2 palabras") --> nn["Multiplicación por la matriz W<br>Tamaño de W: 2000 x 20000"]
    nn --> output("Vector de salida<br>20000 probabilidades")
```

- [Multiplicación de matrices][wiki].

[wiki]: https://es.wikipedia.org/wiki/Multiplicaci%C3%B3n_de_matrices

- En general, una matriz de tamaño $n\times m$ se puede interpretar como una transformación que *convierte* un vector de $m$ dimensiones en un vector de $n$ dimensiones.

### El poder de las GPUs

- Las GPUs aceleran operaciones con matrices muy rápido.
- Modelos grandes requieren clusters de GPUs potentes, aunque es posible experimentar en plataformas como Google Colab con GPUs caseras de *gaming*.
- RTX 5080, 16 GB de memoria, 1000€.
- H200, mucho más rápida, 141 GB de memoria, 30000€.

### El papel de los parámetros en una red neuronal

- Durante el entrenamiento, ajustamos la matriz de parámetros $W$ para que la salida de la red sea lo más parecida posible a la salida deseada.
- Ejemplo: Si los embeddings tienen dimensión 1000 y el vocabulario 20000 palabras, los datos de entrenamiento son pares de entrada de 2000 valores y salida de 20000 probabilidades.
 
- Inicio del entrenamiento: $W$ se inicializa con valores aleatorios.
- Predicción y error: Se computa la salida de la red y se mide el error con una *función de pérdida*.
- Ajuste de parámetros: 
  - Se calcula la *derivada* de la función de pérdida respecto a los valores de $W$.
  - Se ajustan los parámetros para minimizar la función de pérdida.
- Este proceso se repite para todos los ejemplos y varias veces para mejorar las predicciones.

### Aprendizaje de los embeddings de palabras

- Los *embeddings* de palabras que se usan a la entrada son parámetros de la red neuronal y se aprenden durante el entrenamiento igual que la matriz $W$.

### Código: entrenar y ejecutar un modelo de lengua

- Vamos a ver en acción una implementación en Python de un modelo de lengua similar al explicado hasta este momento. 
- Necesitarás una cuenta de Google para poder acceder a Google Colab. Si no tienes una y tienes permiso de tus padres o tutores, puedes crearla en [este enlace][google].

[google]: https://accounts.google.com/signup

<a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/assets/misterios/notebooks/simple-lm.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

!!! note "Programa"

    Modifica el programa cambiando el número de pasos de entrenamiento y observa cómo evoluciona el error. Añade nuevas frases al conjunto de entrenamiento y reentrena el modelo. En particular, añade alguna frase que comparta prefijo con otra ya existente (por ejemplo, si estaba "I like apples", añade "I like oranges") y observa las probabilidades de salida tras el entrenamiento (para la palabra siguiente a "I like"). Modifica el tamaño de los embeddings. Lee sobre la tasa de aprendizaje (*learning rate*) y prueba con diferentes valores.

### Mentirijillas

- Los modelos actuales son mucho más complejos que el ejemplo descrito.
- Pueden procesar hasta cientos de miles de palabras en lugar de solo dos.
- Utilizan miles de matrices de parámetros y operaciones avanzadas.
- Requieren superordenadores debido al tamaño masivo de los parámetros y los datos.
- Algunos modelos comerciales superan el billón de parámetros, mejorando aprendizaje y generalización.
- Su uso masivo conlleva retos de escalabilidad, seguridad, privacidad, etc.

### El mecanismo de atención

- Los modelos actuales consideran contextos largos gracias al mecanismo de *atención*.
- Permite a la red *seleccionar* las palabras más relevantes del contexto para predecir la siguiente palabra.
- En el contexto "Rosalind Franklin trabajó incansablemente en la obtención de imágenes de difracción de rayos X del ADN. Estas imágenes resultaron cruciales para que Watson y Crick pudieran desarrollar su conocido modelo de la doble hélice en 1953 cuando investigaban en la Universidad de Cambridge. Aunque su contribución fue fundamental, pocos reconocieron en aquel momento el impacto del trabajo de...", palabras como "Rosalind" son más importantes que "Cambridge" para predecir la siguiente palabra.
- La atención evita que los modelos se *desborden* ante enormes cantidades de embeddings de entrada.
- Se aprende automáticamente durante el ajuste de parámetros para minimizar la función de pérdida.

### Código: observando el mecanismo de atención 

- El siguiente código nos permite evaluar cómo la atención favorece unas palabras de la entrada sobre otras a la hora de generar una posible continuación.

<a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/assets/misterios/notebooks/alti.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

!!! note "Programa"

    Prueba con diferentes frases de contexto y palabras objetivo y sorpresa. Estudia la relevancia que da el modelo a las palabras de entrada.

### Más allá de predecir la siguiente palabra

- Los modelos que predicen la siguiente palabra a veces generan continuaciones sorprendentes, pero otras veces no coinciden con nuestras expectativas.
- Ejemplo: 
  - "María ve caer una piedra y una pluma en el vacío. Como ha estudiado física, María sabe...", "que ambas llegarán al suelo al mismo tiempo."
  - "María ve caer una piedra y una pluma en el vacío. Como María no estudió física, María piensa...", "que la piedra llegará antes al suelo."

!!! note "Ejercicio"

    ¿Se puede decir que un modelo de lengua capaz de generar continuaciones como las anteriores entiende el mundo?"

- También pueden darse continuaciones frustrantes como:
  - "¿De qué color es el cielo?" → "¿De qué color es el mar?"
  - "La traducción de 'I like tomatoes' al español es..." → "diferente de la de 'I like potatoes'."
  - "¿Cuál es la capital de Francia?" → "A: Londres. B: París. C: Madrid. D: Berlín."

- Tras el entrenamiento para predecir la siguiente palabra, se realiza una fase de *alineamiento* para ajustar el comportamiento del modelo a las preferencias del usuario, promoviendo respuestas coherentes, éticas y educadas.
- El modelo puede disculparse por errores o negarse a responder preguntas inapropiadas.
- Etapas del alineamiento:
  1. Aprender a predecir palabras en textos revisados por personas para inducir respuestas apropiadas.
  2. Generar varias respuestas para una pregunta y ordenar las opciones de mejor a peor con ayuda humana.
  3. Usar algoritmos que refuercen respuestas apropiadas (bien puntuadas) y reduzcan las inapropiadas (mal puntuadas).

![](assets/misterios/imgs/rlhf.png)

### Código: comparando modelos base con modelos que siguen instrucciones

- Vas a ejecutar dos tipos de modelos: uno entrenado solo para predecir la siguiente palabra; otro entrenado para seguir *instrucciones*.
- Ventajas de los modelos Llama-3: son *abiertos*: se pueden descargar y usar sin coste; son *pequeños*: pueden ejecutarse en GPUs modestas.
- Inconvenientes: no son tan potentes como los modelos comerciales, que pueden tener hasta mil veces más parámetros.
- Acceso completo: permiten realizar operaciones como acceder a las probabilidades de salida de la red neuronal, algo que no es posible con modelos comerciales.

<a target="_blank" href="https://colab.research.google.com/github/jaspock/me/blob/main/docs/materials/assets/misterios/notebooks/llama.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

!!! note "Programa"

    Observa cómo el contexto influye en la probabilidad de la siguiente palabra. Juega con diferentes frases de contexto y estudia las predicciones de los modelos. ¿Qué diferencias observas entre los modelos base y los que siguen instrucciones?

### Qué aprenden realmente los modelos de lengua

- Los modelos de lengua han sido llamados *loros probabilísticos* por generar textos coherentes sin *entender* el mundo, estando condicionados por los datos de entrenamiento.
- Ejemplo de *alucinación*: "El 20 de julio de 1969, el astronauta Neil Armstrong pisó la luna. La misión fue un éxito y la humanidad celebró el acontecimiento...", "Sin embargo, la llegada a la luna fue un montaje y nunca ocurrió porque la Luna es un holograma."
- Efectos estudiados en modelos que predicen la siguiente palabra:
  - Sensibilidad a la frecuencia de la tarea: 
    - Mejor desempeño en tareas frecuentes que en tareas raras.
    - Ejemplo: GPT-4 tiene un 42% de precisión en traducción al *pig latin* común, pero solo un 23% en variantes raras.
    - Ejemplo: Calcula mejor $9/5 x + 32$ que $7/5 x + 31$. ¿Por qué?
  - Sensibilidad a la probabilidad de la respuesta: 
    - Precisión más alta cuando la respuesta correcta es de alta probabilidad.
    - Ejemplo: GPT-4 invierte secuencias de palabras con 97% de precisión si el resultado es de alta probabilidad, pero solo con 53% si es de baja probabilidad.
  
![](assets/misterios/imgs/griffiths-abreviatura.png)  
![](assets/misterios/imgs/griffiths-article-swapping.png)  
![](assets/misterios/imgs/griffiths-pig-latin.png)  

### Limitaciones de los modelos de lengua

- Los modelos pueden *alucinar*, generando textos sin sentido, incorrectos o alejados de la realidad.
- Se incorporan modos de razonamiento (como las personas) para verificar y contrastar respuestas antes de emitirlas, sometiéndolas a procesos adicionales.
  
![](assets/misterios/imgs/socratic.png)

- La inteligencia artificial general, que resolvería cualquier tarea humana, aún enfrenta problemas difíciles para los modelos actuales. Estas son tareas en las que aún fallan:
- 
  - [Ciertos rompecabezas](https://arcplayground.com/#).
  - [Tareas de la Olimpiada Lingüística](https://arxiv.org/abs/2409.12126).
  
![](assets/misterios/imgs/arcagi.png)  
![](assets/misterios/imgs/linguini.png)

- Se anuncia la posible llegada de la *superinteligencia* (inteligencia artificial que supera a la humana y que lleva a la ciencia y la tecnología a cotas inimaginables), pero la historia demuestra que muchas predicciones futuristas no se cumplen.

### Sesgos

- Los modelos de lengua reflejan sesgos presentes en los datos de entrenamiento, como prejuicios sociales, estereotipos o desigualdades.
- Técnicas para mitigar sesgos:
  - Selección cuidadosa de datos de entrenamiento.
  - Filtración de contenido problemático.
  - Ajustes en etapas de alineamiento.
  - Uso de métricas específicas para monitorear y reducir el impacto de los sesgos.
- Desafío: La eliminación completa de los sesgos sigue siendo difícil.
- 
- Ejemplo: Traducción sesgada por género:
  - "La doctora Elizabeth Rossi ha sido contratada por sus conocimientos sobre la microbiota intestinal y su impacto en la salud digestiva."
  - Traducción incorrecta al inglés: "Dr. Elizabeth Rossi has been hired for *his* knowledge about the gut microbiota and its impact on digestive health."
- Amplificación de sesgos: los modelos refuerzan patrones frecuentes, sobregeneralizando y exacerbando prejuicios más allá de lo que hay en los datos originales.

### Qué estudiar

- Divulgación en Youtube: [DotCSV][dot] o [Xavier Mitjana][xavier].
- Matemáticas: álgebra, cálculo, probabilidad y estadística.
- Programación: Python, Hugging Face, PyTorch
- Grados en Ingeniería Informática o Inteligencia Artificial.
- Cursos de especialización o másteres en IA o ciencia de datos.
- Cualquier otra carrera, pero complementando con aprendizaje autónomo sobre IA y programación.
- [Guía][guía] del profesor sobre procesamiento del lenguaje natural (la rama de la IA centrada en el lenguaje humano).

[guía]: https://www.dlsi.ua.es/~japerez/materials/transformers/intro/
[dot]: https://www.youtube.com/dotcsv
[xavier]: https://www.youtube.com/@XavierMitjana


### Dudas

- Ahora o en el futuro a Juan Antonio Pérez Ortiz de la Universidad de Alicante: japerez@ua.es
