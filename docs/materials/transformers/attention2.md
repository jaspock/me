
<div class="content-2columns" markdown>
![](assets/imgs/2engines.png){: .rounded-title-img}

# La arquitectura transformer completa
</div>

En el bloque anterior se estudió el transformer como un sistema que va generando progresivamente una secuencia de tokens, pero el primer uso de la arquitectura fue para un caso diferente: el de transformar una secuencia de tokens en otra para poder realizar traducción automática entre lenguas. Para este uso, se ideó un sistema en dos fases, codificación y descodificación, de las que el transformer del bloque anterior se corresponde con la segunda. En este tema veremos el codificador y su interacción con el descodificador. Existe, además, la posibilidad de usar únicamente el codificador en otro tipo de tareas que también veremos. En las tres opciones oirás que se utiliza la palabra *transformer* para describir la arquitectura subyacente.   

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## El modelo codificador-descodificador

En el bloque anterior, hemos estudiado todos los elementos del transformer, pero en algunas aplicaciones en las que una secuencia de tokens se transforma en otra secuencia de tokens, las capas del transformer se asocian a dos submodelos bien diferenciados y conectados entre ellos mediante mecanismos de atención: el codificador y el descodificador. El capítulo "[Machine translation][mt]" [<i class="fas fa-file"></i>][mt] se centra en la arquitectura completa. En este capítulo solo es necesario que estudies por ahora las secciones 10.2 y 10.6.

[mt]: https://web.archive.org/web/20221104204739/https://web.stanford.edu/~jurafsky/slp3/10.pdf

El capítulo aborda el estudio de la arquitectura completa del transformer desde el punto de vista de la traducción automática (la primera aplicación para la que se usaron los transformers), pero los apartados en los que nos vamos a centrar son bastante generales y no asumen una aplicación concreta dentro de la tarea de transformar una secuencia de entrada en otra secuencia de salida.

Observa que ya vimos en el bloque anterior que tareas como la traducción automática o la obtención de resúmenes pueden también ser abordadas con modelos basados únicamente en el descodificador.

## Anotaciones al libro

Es recomendable que estudies estos comentarios después de una primera lectura del capítulo y antes de la segunda lectura.

Apartado 10.2
{: .section}

Este apartado es muy breve y se limita a presentar la idea de la arquitectura codificador-descodificador, que no solo está presente en los transformers. Observa cómo el codificador genera una representación intermedia de la secuencia de entrada que es usada por el descodificador para guiar la generación de la secuencia de salida.

## Apartado 10.6
{: .section}

Otro apartado breve donde se ve que las únicas novedades que aporta la arquitectura completa a lo que ya sabías son:

- el codificador no es autorregresivo (observa la figura 10.15), por lo que no es necesario aplicar las máscaras que anulan la atención a los tokens futuros; más adelante, veremos las ecuaciones completas del modelo bajo este enfoque, pero la ausencia de máscara es el único cambio reseñable.
- en la *atención cruzada* las claves y los valores se obtienen mediante sendas matrices a partir de los embeddings de la última capa del codificador mientras que las consultas se generan en el descodificador.

## Búsqueda y subpalabras

Hay algunos elementos adicionales a la definición de los diferentes modelos neuronales que son también relevantes cuando estos se aplican en el área del procesamiento del lenguaje natural. 

- Estudia el mecanismo de búsqueda en haz (*beam search*) descrito en la [sección 10.5][mt].
- Lee lo que se dice sobre la obtención de subpalabras en las secciones [10.7.1][mt] y [2.4.3][cap2].

[cap2]: https://web.archive.org/web/20221026071429/https://web.stanford.edu/~jurafsky/slp3/2.pdf

Apartado 10.5
{: .section}

A la hora de generar la secuencia de salida a partir de los sucesivos vectores de probabilidad emitidos por el descodificador, hay muchos enfoques posibles. Uno de los más sencillos consiste en elegir el token con mayor probabilidad en cada paso, pero, como se ve en el libro, esta estrategia suele ser muy ineficiente. Un enfoque ligeramente más elaborado es el de la búsqueda en haz (*beam search*), que consiste en mantener en cada paso un conjunto de posibles secuencias de salida y elegir las mejores para continuar con el proceso.

Apartados 2.4.3 y 10.7.1
{: .section}

La mayoría de los modelos neuronales actuales de procesamiento de textos no consideran cada palabra del texto como una entrada diferente, sino que habitualmente dividen (*tokenizan*) el texto en unidades más pequeñas antes de procesarlo por la red neuronal. Dos de los algoritmos más usados para esta tarea son *BPE* y *SentencePiece*. El primero necesita que el texto esté dividido en palabras, mientras que el segundo trata el texto carácter a carácter sin que los posibles separadores (el espacio en blanco, por ejemplo) jueguen un papel especial. Ambos terminan haciendo que las palabras muy frecuentes se representen con una subpalabra única, mientras que las palabras menos frecuentes se representan con una secuencia de subpalabras. A la hora de determinar la segmentación que se va a usar en nuestro modelo, el desarrollador indica el número máximo de subpalabras a utilizar para obtener el vocabulario del problema.

## Modelos preentrenados

El capítulo "[Transfer Learning with Contextual Embeddings and Pre-trained language models][bert]" [<i class="fas fa-file"></i>][bert] estudia los modelos preentrenados basados en codificador y cómo adaptarlos a nuestras necesidades. Para nuestro estudio son relevantes la introducción y las secciones 11.1 y 11.2. 

[bert]: https://web.archive.org/web/20221026071419/https://web.stanford.edu/~jurafsky/slp3/11.pdf

Introducción 
{: .section}

Brevemente, la introducción presenta el concepto de modelo preentrenado y su ajuste (*fine-tuning*) para una tarea concreta.

Apartado 11.1 
{: .section}

Este apartado repasa todas las ideas que hemos visto sobre el codificador del transformer, lo que te puede ayudar a terminar de entender el mecanismo de atención y la arquitectura del transformer. En realidad, la única diferencia respecto al descodificador de los modelos de lengua es que el codificador no es autorregresivo, por lo que no es necesario aplicar las máscaras que anulan la atención a los tokens futuros. Todos los tokens de la entrada se procesan, por tanto, en paralelo y la salida del codificador es un conjunto de vectores de embeddings, uno para cada token de la entrada.

Apartado 11.2
{: .section}

Este apartado aborda la tarea de entrenar un codificador basado en transformer de forma que sus representaciones sean lo más generales posibles, es decir, que no estén adaptadas a un problema concreto. Para ello, el modelo se entrena con una tarea *neutra* cuya resolución, aun así, implicará que el modelo ha aprendido a generar buenas representaciones profundas de los tokens. Esta tarea es de tipo *autosupervisado* en el sentido de que no necesita un texto etiquetado (por ejemplo, con la temática o la polaridad de sus frases), sino que la simple secuencia de tokens es el ingrediente fundamental del entrenamiento. Una de las tareas más utilizadas para ello es la de predecir los tokens que deliberadamente se han eliminado de la secuencia de entrada. Una vez entrenado un modelo profundo de esta manera, este puede distribuirse para que otros desarrolladores solo tenga que ajustar el modelo a su tarea concreta mediante el proceso de *fine-tuning* en el que solo se actualizan los pesos del clasificador que procesa los embeddings generados por el modelo preentrenado. Uno de los primeros modelos preentrenados basados en transformer fue BERT, que se presentó en 2018 y que se convirtió rápidamente en uno de los modelos preentrenados más utilizado. 

## Ecuaciones del transformer

A modo de resumen, se presentan aquí a modo de resumen las ecuaciones del transformer completo. El transformer usa la arquitectura codificador-descodificador para emitir de forma autoregresiva una secuencia de salida $\boldsymbol{y}= y_1, y_2,\ldots,y_n$ a partir de una secuencia de entrada $\boldsymbol{x}= x_1, x_2,\ldots,x_n$. Habitualmente cada $x_i$ será un embedding *no contextual* para el token correspondiente de la frase a procesar obtenido de una tabla de embeddings, y cada $y_i$ será el vector de probabilidades correspondiente al $i$-ésimo token de la frase de salida.

El codificador tiene $N$ capas idénticas, cada una formada a su vez por dos subcapas:

$$
\begin{align}
\underline{\boldsymbol{h}}^l &= \text{LN}\left(\text{SelfAtt}\left(\boldsymbol{h}^{l-1}\right) + \boldsymbol{h}^{l-1}\right) \\
\boldsymbol{h}^l &= \text{LN}\left(\text{FF}\left(\underline{\boldsymbol{h}}^l\right) + \underline{\boldsymbol{h}}^l\right)
\end{align}
$$

donde $\boldsymbol{h}^l = \{h_1^l,h_2^l,\ldots,h_n^l\}$ son las salidas de la capa $l$-ésima (una por cada token de la entrada). La salida de la primera capa es $\boldsymbol{h}^0= \boldsymbol{x}$. La función LN obtiene la normalización a nivel de capa, SelfAtt es el mecanismo de atención con múltiples cabezales y FF es una red hacia delante completamente conectada.

El descodificador sigue un planteamiento similar con un par de particularidades: el mecanismo de autoatención usa una máscara para no usar los embeddings de los tokens aún no generados y aparece una tercera subcapa responsable de la atención hacia el codificador:

$$
\begin{align}
\underline{\boldsymbol{s}}^l &= \text{LN}\left(\text{MaskedSelfAtt}\left(\boldsymbol{s}^{l-1}\right) + \boldsymbol{s}^{l-1}\right) \\
\underline{\underline{\boldsymbol{s}}}^l &= \text{LN}\left(\text{CrossAtt}\left(\underline{\boldsymbol{s}}^{l},\boldsymbol{h}^N\right) + \underline{\boldsymbol{s}}^{l}\right) \\
\boldsymbol{s}^l &= \text{LN}\left(\text{FF}\left(\underline{\underline{\boldsymbol{s}}}^l\right) + \underline{\underline{\boldsymbol{s}}}^l\right)
\end{align}
$$

donde $\boldsymbol{s}^l$ son las salidas de la capa $l$-ésima del descodificador. Los embeddings de la última capa $\boldsymbol{s}^M$ se pasan por una capa densa adicional seguida de una función softmax para obtener la estimación de la probabilidad del token correspondiente. La salida de la primera capa del descodificador $\boldsymbol{s}^0$ es, como en el codificador, un embedding no contextual del token anterior (por ejemplo, el token de mayor probabilidad emitido en el paso anterior).

## Las diferentes caras de la atención

En el siguiente análisis nos basaremos en la [discusión][dontloo] de *dontloo* en Cross Validated. En los sistemas *seq2seq* originales basados en redes recurrentes, la atención se calcula una media ponderada de las claves:

[dontloo]: https://stats.stackexchange.com/a/424127/240809

$$
\boldsymbol{c} = \sum_{j} \alpha_j \boldsymbol{h}_j   \qquad \mathrm{con} \,\, \sum_j \alpha_j = 1
$$

Si $\alpha$ fuera un vector one-hot, la atención se reduciría a recuperar aquel elemento de entre los distintos $\boldsymbol{h}_j$ en base al correspondiente índice; pero sabemos que $\alpha$ difícilmente será un vector unitario, por lo que se tratará más bien de una recuperación ponderada. En este caso, $\boldsymbol{c}\,$ puede considerarse como el valor resultante.

Hay una diferencia importante en cómo este vector de pesos con suma 1 se obtiene en las arquitecturas de *seq2seq* y la del *transformer*. En el primer caso, se usa una red neuronal *feedforward*, representada mediante la función $a$, que determina la *compatibilidad* entre la representación del token $i$-ésimo del descodificador $\boldsymbol{s}_i$ y la representación del token $j$-ésimo del codificador $\boldsymbol{h}_j$:

$$
e_{ij} = a(\boldsymbol{s}_i,\boldsymbol{h}_j)
$$

y de aquí:

$$
\alpha_{ij} = \frac{\mathrm{exp}(e_{ij})}{\sum_k \mathrm{exp}(e_{ik})}
$$

Supongamos que la longitud de la secuencia de entrada es $m$ y la de la salida generada hasta este momento es $n$. Un problema de este enfoque es que en cada paso del descodificador es necesario pasar por la red neuronal $a$ un total de $mn$ veces para computar todos los $e_{ij}$.

Existe una estrategia más eficiente que pasa por proyectar los $\boldsymbol{s}_i$ y los $\boldsymbol{h}_j$ a un espacio común (mediante, por ejemplo, sendas transformaciones lineales de una capa, $f$ y $g$) y usar entonces una medida de similitud (como el producto escalar) para obtener la puntuación $e_{ij}$:

$$
e_{ij} = f(\boldsymbol{s}_i) \cdot g(\boldsymbol{h}_j)^T
$$

Podemos considerar que el vector de proyección $f(\boldsymbol{s}_i)$ es la consulta realizada por el descodificador y el vector de proyección $g(\boldsymbol{h}_j)$ es la clave proveniente del descodificador. Ahora solo es necesario realizar $n$ llamadas a $f$ y $m$ llamadas a $g$, con lo que hemos reducido la complejidad a $m+n$. Además, hemos conseguido que los $e_{ij}$ puedan calcularse eficientemente mediante producto de matrices.

El mecanismo de atención del transformer establece las condiciones para que las proyecciones de consultas y claves y el cálculo de la similitud se puedan llevar a cabo. Cuando la atención se realiza desde y hacia vectores con el mismo origen (por ejemplo, dentro del codificador) se denomina *autoatención*. El transformer combina autoatención separada en codificador y descodificador con el otro mecanismo de atención *heterogénea* en el que $Q$ viene del descodificador y $K$ y $V$ vienen del codificador.

Como ya se ha visto, la ecuación básica de la autoatención en el transformer es esta:

$$
\mathrm{SelfAttn}(Q,K,V) = \mathrm{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

En realidad, las ecuaciones finales del transformer son ligeramente más complejas que esta al tener que considerar los múltiples cabezales.

## Ejercicios de repaso

1. Razona cómo se usaría una máscara $M$ similar a la que evita la atención a los tokens siguientes para enmascarar los tokens de relleno de un mini-batch de frases de entrenamiento.

2. Una matriz de atención representa en cada fila el nivel de atención que se presta a los embeddings calculados para cada token de la frase en la capa anterior (representados en las columnas) cuando se va a calcular el embedding del token de dicha fila en un determinado cabezal (head) del codificador de un transformer. Un color más oscuro representa mayor atención. Considera ahora un cabezal (head) de un codificador de un transformer que presta una atención aproximadamente monótona sobre los embeddings de la capa anterior. En particular, para cada token se atiende en un elevado grado al embedding de ese mismo token en la capa anterior y con mucha menor intensidad al token inmediatamente a su derecha; el resto de embeddings de la capa anterior no reciben atención. Dibuja aproximadamente la matriz de atención resultante sobre la misma frase que la de la figura.