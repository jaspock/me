
<div class="content-2columns" markdown>
![](assets/imgs/engine-words.png){: .rounded-title-img}

# Transformers y modelos de atención: conociendo los modelos de atención y su aplicación en el modelo transformer
</div>

En 2017 las redes neuronales recurrentes basadas en unidades LSTM eran la arquitectura habitual para el procesamiento neuronal de secuencias, en general, y del lenguaje natural, en particular. Algunos investigadores comenzaban a obtener también buenos resultados en esta área con las redes neuronales convolucionales, tradicionalmente empleadas con imágenes. Por otro lado, los mecanismos de atención introducidos unos años antes en las redes recurrentes habían mejorado su capacidad para resolver ciertas tareas y abierto el abanico de posibilidades de estos modelos. Además, el modelo conocido como codificador-descodificador (*encoder-decoder* en inglés) se convertía en la piedra angular de los sistemas que transformaban una secuencia en otra (sistemas conocidos como *seq2seq* como, por ejemplo, los sistemas de traducción automática o de obtención de resúmenes). A mediados de 2017, sin embargo, aparece un artículo ("[Attention Is All You Need](https://arxiv.org/abs/1706.03762)") que propone eliminar la recurrencia del modelo codificador-descodificador y sustituirla por lo que se denomina autoatención (*self-attention*); aunque el artículo se centra en la tarea de la traducción automática, en muy poco tiempo la aplicación de esta arquitectura, bautizada como *transformer*, a muchos otros campos se descubre altamente eficaz hasta el punto de relegar a las arquitecturas recurrentes a un segundo plano. El transformer sería, además, uno de los elementos fundamentales de los modelos preentrenados que estudiaremos más adelante y que comenzarían a aparecer en los meses o años siguientes. No obstante, sigue habiendo investigadores trabajando con redes recurrentes, por lo que no puede descartarse que recobren mayor relevancia en el futuro.

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Fundamentos de los transformers

Vamos a acometer el estudio de los elementos básicos de la arquitectura transformer siguiendo el capítulo [:octicons-book-24:][basictransformer] "[Deep Learning Architectures for Sequence Processing][basictransformer]". Aquí entenderás qué significa una de las ecuaciones más importantes de los últimos años dentro del aprendizaje automático:

$$
\text{Atención}(Q,K,V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) \, V
$$

Salta las secciones 9.2 a 9.6, que se centran en otro modelo alternativo para el procesamiento de secuencias, las redes neuronales recurrentes, que se han venido usando menos en el área del procesamiento del lenguaje natural tras la llegada del transformer.

[basictransformer]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/9.pdf


## Anotaciones al libro

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--recomendable-start-->"
   end="<!--recomendable-end-->"
%}

Básicamente, el transformer es una arquitectura que permite trabajar con secuencias de diferente naturaleza. Cuando estemos usándolo para analizar o generar frases en lenguaje natural, las secuencias serán frases formadas por tokens de palabras. Como ya sabemos, para poder tener "calculadoras de palabras", estos tokens se han de representar mediante números y, como ya hemos visto, las representaciones profundas en forma de vectores de embeddings que capturan ciertas propiedades de las palabras subyacentes son especialmente útiles. El transformer, de hecho, no es más que una máquina para calcular embeddings contextuales de palabras y esta contextualidad los diferencia de algoritmos como el de skip-grams, que daban una representación única para cada palabra. Los transformers comienzan con una representación incontextual de las palabras a su entrada y la van refinando capa tras capa. Desde la salida de la primera capa, las representaciones son contextuales, de forma que la representación de *estación* en "El verano es mi estación favorita" es diferente a la de *estación* en "La estación de tren está a la vuelta de la esquina" e incluso diferente a la de "El invierno es mi estación favorita". A medida que avanzamos por las capas, las representaciones no solo se hacen más contextuales, sino que van adecuándose a la tarea concreta que espera resolverse con la salida del modelo en su última capa. La última capa será normalmente un clasificador formado, por ejemplo, por una capa densa (instrumentada, como hemos visto, a través de una matriz) que generará un vector de logits que serán transformados en probabilidades por una función de activación softmax. Algunas de las operatorias posibles con el transformer son:

- Se generan tantos vectores de probabilidad a la salida como tokens tiene la entrada, de forma que cada uno de ellos corresponde a la probabilidad de que el correspondiente token de la entrada pertenezca a una determinada clase. De esta manera, por ejemplo, podemos usar la salida del modelo para saber si cada token es un nombre propio o no (tarea muy útil, por ejemplo, para evitar que los sistemas de traducción automática intenten traducirlos cuando no procede).
- Podemos combinar todos los embeddings de salida en un único vector (por ejemplo, calculando su media) y usar el embedding resultante como representante de la frase completa. Pasando este embedding por una capa densa y una función de activación softmax, podemos obtener la probabilidad de que la frase pertenezca a una determinada clase. De esta manera, por ejemplo, podemos usar la salida del modelo para saber si una frase habla de forma positiva, negativa o neutra sobre un tema.
- Un enfoque alternativo al anterior para tareas de clasificación a nivel de frase es añadir un token *ficticio* (tradicionalmente se representa como `CLS`) al principio o al final de la frase y usarlo como entrada adicional al transformer. La salida del transformer para este token será la representación de la frase completa que suministraremos al clasificador.
- Podemos entrenar el modelo para que el vector de probabilidad obtenido a la salida para un token concreto nos indique la probabilidad de que cada una de las palabras de nuestro vocabulario sea el siguiente token de la frase. De esta manera, por ejemplo, una vez acabado el entrenamiento podemos usar la salida de la red para ir generando texto a partir de un prefijo dado y así conseguir un modelo de lengua generativo con el que poder mantener un diálogo, contestar preguntas, traducir un texto a otro idioma o resumir un documento.

En su formulación más general, el transformer está compuesto por dos módulos principales: un codificador (*encoder*) y un descodificador (*decoder*). El codificador es el encargado de generar representaciones profundas de los tokens de la frase de entrada que serán usadas por el descodificador para generar la frase de salida. Cuando queremos clasificar el texto de entrada o etiquetar sus tokens puede ser suficiente con usar el codificador. Cuando queremos transformar un texto en otro (traducción automática u obtención de resúmenes), tradicionalmente se han usado ambos. Cuando estamos interesados en un modelo de lengua generativo, se suele usar un descodificador, que va generando de forma *autorregresiva* la frase de salida token a token. Sin embargo, en los últimos tiempos se ha venido produciendo una convergencia hacia el uso de modelos que integran codificador y descodificador en un solo módulo. Mucha gente llama a todas las opciones "transformer", pero otras personas reservan el término "transformer" para el modelo que integra codificador y descodificador, y usan términos como "encoder-like transformer" o "decoder-like transformer" para referirse a los modelos que solo usan uno de los dos módulos.

Una de las grandes ventajas de los transformers frente a otros modelos como las redes neuronales *feedforward* es que pueden procesar secuencias de longitud variable, ya que no aprenden parámetros diferentes para procesar la primera palabra, la segunda, la tercera, etc. En cambio, los transformers aprenden en cada capa una única transformación que se aplica a todos los tokens de la entrada. Como desde la entrada del modelo cada token se representa con un embeddings diferente, la transformación da resultados (consulta, clave y valor) diferentes para cada token.

El apartado 9.7 de este capítulo del libro se centra en el transformer como modelo de lengua generativo y, por tanto, en el descodificador. En realidad, el codificador no funciona de forma muy diferente, como veremos más adelante. 

Apartado 9.7
{: .section}

El apartado comienza introduciendo la idea de atención: el embedding de un token se refinará *mezclándolo* con los embeddings de otros tokens de la entrada. De este modo, una secuencia de $n$ embeddings $\mathbf{x}_1,\ldots,\mathbf{x}_n$ se transforma en una secuencia $\mathbf{y}_1,\ldots,\mathbf{y}_n$, donde cada $\mathbf{y}_i$ es un *cóctel* diferente de los embeddings de entrada. En principio, podríamos pensar que un embedding debería mezclarse más con aquellos que representan palabras con las que está relacionado. Así, en la frase "La mujer comenzó el viaje cansada", el embedding de "cansada" debería mezclarse más con el de "mujer" que con el de "viaje", ya que, a fin de cuentas, la palabra está calificando a la mujer y no al viaje. Desde que estudiamos el tema de embeddings incontextuales, sabemos que una manera de medir la similitud de dos vectores es mediante el producto escalar. Por tanto, en una primera aproximación, podríamos calcular el producto escalar entre el embedding de un token y el de cada uno de los tokens de la entrada, y usar el resultado como argumento de la función softmax, que nos indicará el grado de mezcla de los embeddings. Observa que si permitimos que un embedding pueda mezclarse, entre otros, consigo mismo, la función softmax asignará una gran parte de la mezcla a sí mismo. 

Aunque la forma anterior de combinar embeddings puede ser útil en ciertos contextos, el transformer lo hace de una forma ligeramente más compleja. Cada embedding de la entrada se transforma en tres vectores:

- la consulta (*query*) define *qué busca* el token en otros tokens;
- la clave (*key*) indica *cómo se define* cada token a sí mismo;
- el valor (*value*) define *qué da* cada token a otros tokens.

Un símil habitual para explicar esta forma de atención es el de una aplicación de citas en la que las personas indican qué buscan (*consulta*) y cómo se definen a sí mismas (*clave*). Una vez determinado el grado de afinidad entre cada persona y el resto (mediante el producto escalar), se combinan los genes (*valores*) de cada individuo para obtener un nuevo embedding. Este proceso se repite con cada token, por lo que el número de productos escalares es del orden de $n^2$. Aquí, a diferencia de las aplicaciones de citas, un token se puede emparentar con un cierto grado de afinidad consigo mismo y los nuevos embeddings suelen resultar de combinar los genes de más de dos individuos. Otro símil que ayuda a entender la atención es el basado en el acceso a un diccionario de un lenguaje de programación, que puedes consultar [más abajo][diccionario]. 

[diccionario]: #un-símil-del-mecanismo-de-autoatención

Lo interesante aquí es que la transformación de un vector $\mathbf{x}_i$ en tres vectores de dimensión $d$ se realiza mediante tres transformaciones lineales $\mathbf{W}^Q$, $\mathbf{W}^K$ y $\mathbf{W}^V$, que se aplican a todos los tokens de la entrada. Por tanto, el número de parámetros a aprender es mucho menor que el número de parámetros que tendríamos que aprender si aplicáramos una transformación lineal a cada uno de los tokens de la entrada. Además, como las transformaciones lineales son independientes entre sí, podemos paralelizarlas y, por tanto, acelerar el cálculo. En el libro se ve, además, que los diferentes productos matriciales de los embeddings que representan todos las palabras de la frase se pueden realizar también en paralelo si disponemos los embeddings de entrada *por filas* en una matriz $X$. De hecho, como veremos en la implementación, podemos agrupar los embeddings de todas las frases de un mini-batch en un único tensor (equivalente a un vector de matrices $X$) y paralelizar aún más los cálculos. Las GPUs están optimizadas para realizar eficientemente todo este tipo de operaciones matriciales.

Un elemento que merece un pequeño análisis es la división por $\sqrt{d_k}$ en el denominador de la fórmula de la atención. Esta división se realizar para evitar que haya valores de los productos escalares excesivamente altos que se lleven la mayor parte de la distribución de probabilidad generada por la softmax. Hay un [apartado especial][división] más abajo que analiza este asunto con un poco más de detalle.

[división]: #atención-escalada

Un aspecto reseñable de este apartado es que se centra en el enfoque autorregresivo del transformer, es decir, en su uso como modelo generativo. Durante la etapa de uso (también denominada *inferencia*) el modelo genera para el token actual el vector de probabilidades del siguiente token. Podemos, en este caso, quedarnos con el token con mayor probabilidad y retroalimentarlo a la entrada. Es como si usáramos el sistema para responder a la siguiente solicitud: dado el prefijo que ya has visto, si le añado este nuevo token, dame las probabilidades de los tokens que pueden ir a continuación. Elegir el token con mayor probabilidad es solo una manera de usar el vector de probabilidades, pero hay muchas otras. Más adelante, veremos la técnica conocida como *beam search*, pero una opción más sencilla es muestrear el siguiente token en base a las probabilidades (cuanto más probable es un token, mayor probabilidad hay de elegirlo), de forma que en cada ejecución el modelo puede generar una secuencia diferente. Esto explica por qué en ocasiones los modelos generativos producen secuencias diferentes ante el mismo prefijo (*prompt*).

El uso autorregresivo del transformer implica que durante el entrenamiento el modelo tiene que trabajar de una manera similar a como se va a usar durante el tiempo de inferencia. Para emular este comportamiento, el mecanismo de atención no puede tener en consideración la información de tokens posteriores. De otra manera, el algoritmo de aprendizaje podría fácilmente terminar centrándose en el token siguiente a la hora de generar el vector de probabilidad, lo que tendría efectos muy negativos durante la inferencia, cuando esta información no estará disponible. Esto justifica la máscara de la figura 9.17. 

Veremos más adelante que hay ocasiones en las que nos interesa que todos los tokens de una frase puedan mezclarse con todos los demás y no solo con los anteriores. Por ejemplo, en el caso de la tarea de clasificar la temática de los textos de entrada, tenemos todos los tokens disponibles desde el primer momento, por lo que no es necesario renunciar a la información de los tokens posteriores a la hora de preparar el cóctel de embeddings. La versión del transformer basada en el codificador que veremos más adelante permitirá hacerlo. En este apartado, sin embargo, se estudia la visión del transformer como descodificador, que es la que se usa para generar autorregresivamente secuencias de salida.

Subapartado 9.7.1
{: .section}

Una vez estudiados los fundamentos del mecanismo de autoatención, este subapartado analiza los otros elementos complementarios que están también presentes en cada capa del transformer: la red *feedforward*, las conexiones residuales y la normalización de capa.

La red *feedforward* tiene normalmente una única capa oculta con una función de activación no lineal.

Las conexiones residuales permiten sortear parcialmente el problema del *gradiente evanescente* de las arquitecturas de múltiples capas. Este problema se puede resumir en que las capas inferiores de las redes con muchas capas reciben una señal de retroalimentación muy débil durante el entrenamiento, lo que dificulta la propagación de información del error y la convergencia del modelo. Entenderás mejor los motivos de esto si consideras un transformer como una sucesiva composición de funciones desde la capa de entrada a la de salida (la salida de cada capa se basa en las operaciones realizadas en esa capa sobre las salidas de la capa anterior), por lo que la derivada de la función de error respecto a parámetros de las capas iniciales estará formada por una gran cantidad de productos (por la aplicación de la regla de la cadena) que pueden fácilmente irse a cero o a valores extremadamente grandes, provocando que las actualizaciones aplicadas por el algoritmo de descenso por gradiente estocástico sean muy pequeñas o muy grandes. Con las conexiones residuales, la derivada de la función de error se bifurca (la derivada de una suma de funciones es la suma de las derivadas) en dos caminos: uno que sigue la ruta convencional y otro que sigue la ruta de la conexión residual. Teóricamente, parte del error tiene ahora la capacidad de llegar intacto a cualquier punto de la red, por muy alejado que se encuentre de la capa de salida, saltando módulos enteros a través de las conexiones residuales. 

La normalización de capa sirve también para evitar valores excesivamente grandes o pequeños en las salidas intermedias de la red, que suelen afectar negativamente al entrenamiento (por ejemplo, llevando las funciones de activación a zonas planas sin gradiente o haciendo que la mayor parte de la probabilidad recaiga en uno o unos pocos elementos después de la función softmax). Puedes encontrar en esta guía un análisis más detallado de esta [normalización][normalización].

[normalización]: #normalización-de-capa

Subapartado 9.7.2
{: .section}

Dado que los tokens de un texto se relacionan de diferentes formas entre ellos, el transformer replica el mecanismo de atención en cada capa mediante la aplicación de múltiples cabezales de atención. Así, por ejemplo, algunos cabezales pueden centrarse en los tokens cercanos, mientras que otros pueden centrarse en aquellos con los que se guarda una determinada relación sintáctica o semántica independientemente de su distancia. Conceptualmente, los múltiples cabezales no introducen apenas novedad respecto al mecanismo ya visto para la atención. Como, en cualquier caso, los elementos subsiguientes esperan recibir un vector de un tamaño dado, los resultados de los diferentes cabezales se concatenan para formar un único vector de salida y se pasan por una proyección lineal tanto para uniformar las representaciones de cada cabezal como para obtener un vector del tamaño adecuado, si corresponde.

Observa que ahora cada cabezal tiene sus propias matrices de parámetros $\mathbf{W}^Q_i$, $\mathbf{W}^K_i$ y $\mathbf{W}^V_i$.

Subapartado 9.7.3
{: .section}

Hasta ahora, hemos vendido como algo positivo el hecho de que las matrices que se usan en cada capa sean las mismas para todos los tokens de la entrada (sabiendo, eso sí, que si hay más de un cabezal, estas matrices son diferentes). Esto nos permite reducir el número de parámetros respecto, por ejemplo, una red *feedforward*, que tendría diferentes parámetros para cada token de la entrada, lo que nos obligaría seguramente a introducir en cada paso una pequeña ventana de tokens. Sin embargo, esto plantea un problema, ya que (asegúrate de que entiendes por qué) se computarían exactamente los mismos embeddings para las palabras de oraciones como "Perro muerde hombre" y "Hombre muerde perro". No obstante, es fácil deducir que el rol y las propiedades semánticas de cada una de las palabras es bien diferente en ambas oraciones. Por ello, al embedding incontextual de cada palabra se le suma antes de la primera capa un vector de *embedding posicional* que idealmente cumple varias propiedades:

- no se repite en dos posiciones diferentes;
- es más parecido para tokens que se encuentran en posiciones cercanas.

Se han propuesto muchas formas de dar valor a los embeddings posicionales. El artículo original de la arquitectura transformer usaba una codificación fija de cada posición basada en una serie de funciones sinusoidales con diferentes frecuencias en cada posición del vector, pero pronto se vio que el propio algoritmo de aprendizaje podía encargarse de aprender valores adecuados para ellos. Si durante el entrenamiento las entradas van tener una longitud máxima acotada, pero durante la inferencia las secuencias pueden crecer arbitrariamente, puede usarse una combinación de embeddings aprendidos (para las primeras posiciones) y embeddings fijos (a partir de ellas).

Apartado 9.8
{: .section}

Este apartado muestra un ejemplo de uso del transformer como modelo de lengua. En cada paso, se procesa un token más a la entrada y se genera un vector de probabilidades para el siguiente token. Se puede usar una estrategia voraz, por ejemplo, para administrar el token ganador como siguiente token a la entrada. 

Apartado 9.9
{: .section}

Observa que el ejemplo de modelo de lengua anterior es bastante limitado, ya que se le pide al modelo que genere un texto sin darle ninguna *semilla*. Si, como hemos comentado, muestreamos posibles palabras de la distribución de probabilidad, podemos generar más de una secuencia. Si simplemente elegimos el token de mayor probabilidad, la secuencia generada sería única. 

Este apartado demuestra cómo puede aportarse un prefijo (*prompt*) al modelo y pedirle que lo continúe. Esta es la idea subyacente a los recientes modelos de lengua (GPT, PaLM, LLaMA, etc.) que han demostrado ser capaces de generar textos, mantener diálogos y presentar argumentarios de una calidad sorprendente. Estos modelos han sido entrenados para generar el siguiente token de una secuencia, pero tienen algunas fases de entrenamiento adicionales que básicamente constan de:

- ajuste fino (*fine-tuning*) de los pesos con textos formados por preguntas y respuestas para que así el sistema aprenda a desenvolverse en diálogos;
- ajuste fino en base a las valoraciones emitidas por personas (*human feedback*) sobre la calidad de las respuestas generadas por el modelo; en este caso, los evaluadores ordenan diferentes respuestas para la misma pregunta por orden de calidad (respuestas verídicas y educadas reciben las puntuaciones más altas); dado que en este caso no existe una función de pérdida derivable, se usan técnicas de entrenamiento basadas en aprendizaje por refuerzo (*reinforcement learning*) que ajustan los pesos siguiendo otras políticas.

Observa cómo la arquitectura autorregresiva que hemos estudiado se puede usar también para realizar resúmenes o traducción de textos a otros idiomas. En estos casos, el prefijo está formado por el texto a resumir o a traducir, respectivamente. Un token especial indica al modelo que el prefijo ha acabado y que debe empezar a generar el texto de salida.

Finalmente, las representaciones aprendidas tras el entrenamiento por un transformer en cada una de sus capas para una nueva frase de entrada pueden considerarse como embeddings contextuales de los diferentes tokens de la entrada. Estos embeddings pueden ser muy útiles en diversas tareas de procesamiento del lenguaje natural. En principio, cualquier capa puede ser adecuada para obtener estas representaciones, pero algunos trabajos han demostrado que ciertas capas son más adecuadas que otras para ciertas tareas. Las capas más cercanas a la entrada parecen representar información más relacionada con la morfología, mientras que las capas finales se relacionan más con la semántica.

## Un símil del mecanismo de autoatención

El mecanismo de autoatención se puede introducir con propósitos didácticos basándonos en una hipotética versión de Python en la que se permitiera acceder a los valores de un diccionario usando claves *aproximadas*. Supongamos el siguiente diccionario de Python almacenado en la variable `d`; como cualquier diccionario de Python este contiene también un conjunto de claves (`manzana`, por ejemplo) y sus valores asociados (`8` es el valor asociado a la clave `manzana`, por ejemplo):

```python
d = {"manzana":8, "albaricoque":4, "naranja":3}
```

En Python *convencional* ahora podemos realizar una *consulta* al diccionario con una sintaxis como `d["manzana"]` para obtener el valor `8`. El intérprete de Python ha usado el nombre de nuestra consulta (`manzana`) para buscar entre todas las claves del diccionario una cuyo nombre coincida *exactamente* y devolver su valor (`8` en este caso).

Observa cómo en la discusión anterior hemos usado los términos "consulta" (*query*), "clave" (*key*) y "valor" (*value*) que aparecen también cuando se discute el mecanismo de autoatención del transformer.

Vayamos ahora más allá y consideremos que realizamos una consulta como `d["narancoque"]`. Un intérprete de Python *real* lanzará una excepción ante la consulta anterior, pero un intérprete *imaginario* podría recorrer el diccionario, comparar el término de la consulta con cada clave del diccionario y ponderar los valores en función del parecido encontrado. Consideremos una función `similitud` que recibe dos cadenas y devuelve un número, no necesariamente acotado, que es mayor cuanto más parecidas son las cadenas (los valores concretos no son ahora relevantes):

```
similitud("narancoque","manzana") → 0
similitud("narancoque","albaricoque") → 20
similitud("narancoque","naranja") → 30
```

Estos resultados normalizados para que su suma sea 1 son `0`, `0,4` y `0,6`. Nuestro intérprete de Python imaginario podría ahora devolvernos para la consulta `d["narancoque"]` el valor 0 x 8 + 0,4 x 4 + 0,6 x 3 = 3,4. 

En el caso del transformer, las consultas, las claves y los valores son vectores de una cierta dimensión, y la función de similitud empleada es el producto escalar de la consulta y las diferentes claves. Los grados de similitud se normalizan mediante la función softmax y se utlizan igualmente para ponderar después los distintos valores:

$$
\text{SelfAtt}(Q,K,V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

## Guías visuales

Jay Alammar publicó hace tiempo una serie de artículos muy conocidos que ilustran de forma muy didáctica y visual el funcionamiento de los transformers. Puedes consultarlos para afianzar conceptos:

- "[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)"
- "[Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)"
- "[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)"

## Atención escalada

Un factor que puede parecer arbitrario en la ecuación de la atención es la división por la raíz cuadrada de la dimensión de la clave. Para entender la motivación de esta operación, observa que cuanto mayor es el tamaño de los embeddings, mayor es el resultado de cada producto escalar $q_i k_j$. El problema es que cuando la función softmax se aplica a valores muy altos, su carácter exponencial hace que asigne valores muy pequeños a todos los elementos excepto al que tiene el valor más alto. Es decir, cuando la función softmax se satura, tiende a un vector *one-hot*. Esto provocará que la atención se centre en un único token e ignore el resto, lo que no es un comportamiento deseable.

Consideremos que $Q$ y $K$ tienen tamaño $B \times T \times C$, donde $C$ es el tamaño de las consultas y las claves. Para simplificar, si asumimos que los elementos de las matrices $Q$ y $K$ tienen varianza alrededor de 1, la varianza de los elementos del producto será del orden de $C$. Como se cumple que, dado un escalar $m$, $\mathrm{var}(mX) = m^2 \mathrm{var}(x)$, al multiplicar cada elemento por $1 / \sqrt{C}$, la varianza del producto matricial se reduce en $\left(1 / \sqrt{C}\right)^2 = 1 / C$. Por tanto, si la varianza de los elementos de $Q$ y $K$ es 1, ahora la varianza del producto matricial también estará alrededor de 1. 

El siguiente código permite comprobar los extremos anteriores:

```python
import torch

B, T, C = 10, 10, 5000
m = 1
Q = torch.randn(B, T, C)*m
K = torch.randn(B, T, C)*m

print(f'Variance of Q: {Q.var().item():.2f}')
print(f'Variance of K: {K.var().item():.2f}')
# variances close to m**2

QK = Q.matmul(K.transpose(-2, -1))

print(f'Variance of QK: {QK.var().item():.2f}') 
# very high variance close to C*(m**4)!

s = torch.softmax(QK, dim=-1)
torch.set_printoptions(precision=2)
print(f'Mean value of highest softmax: {s.max(dim=-1)[0].mean().item():.2f}') 
# max value of each channel close to 1

QK = QK / (C ** 0.5)

print(f'Variance of QK after normalization: {QK.var().item():.2f}')
# variance close to m**4

s = torch.softmax(QK, dim=-1)
print(f'Mean value of highest softmax: {s.max(dim=-1)[0].mean().item():.2f}') 
# max value of each channel smaller than 1
```

En general, si la varianza de los elementos de $Q$ y $K$ es $m$, la varianza del producto matricial estará alrededor de $m^4 C$. Si $m=2$, por ejemplo, la normalización no nos deja elementos con varianza de 1, pero sí la reduce para dejarla en el orden de $m^4 = 16$.


## Normalización de capa

Sean $\hat{\mu}$ y $\hat{\sigma}^2$ la media y la varianza, respectivamente, de todas las entradas, que representaremos por $\boldsymbol{x}$, a las neuronas de una capa formada por $H$ neuronas:

$$
\begin{align}
\hat{\mu} &= \frac{1}{H} \sum_{i=1}^H x_i \\
\hat{\sigma}^2 &= \frac{1}{H} \sum_{i=1}^H \left(x_i - \hat{\mu} \right)^2 + \epsilon
\end{align}
$$

donde $\epsilon$ tiene un valor muy pequeño para evitar una división por cero en la siguiente ecuación. La función LN de normalización para cada entrada de la capa se define como la estandarización:

$$
\text{LN}(x_i) = \gamma_i \frac{x_i - \hat{\mu}}{\hat{\sigma}^2} + \beta
$$

La fracción permite que todos las entradas de la capa en un determinado instante tengan media cero y varianza 1. Como estos valores son arbitrarios, en cualquier caso, se añaden dos parámetros aprendibles $\boldsymbol{\gamma}$ y $\boldsymbol{\beta}$ para reescalarlos. Los valores normalizados se convierten en la nueva entrada de cada neurona y a estos se aplica la función de activación que corresponda; en el caso del transformer, no hay ninguna función de activación adicional.

El código para PyTorch de la normalización de capa es bastante sencillo:

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

