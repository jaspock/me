<div class="content-2columns" markdown>
![](assets/imgs/math-triangle.png){: .rounded-title-img}

# Redes hacia delante: el modelo básico de computación neuronal
</div>

Las redes neuronales hacia delante se pueden ver como un modelo sencillo de aprendizaje automático. Aunque no las usaremos de forma independiente, son un componente necesario de los modelos más avanzados que veremos posteriormente. 

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Redes feed-forward

Tras las introducciones anteriores, estamos ahora preparados para abordar el estudio del modelo básico de red neuronal conocido como red neuronal hacia delante (*feed-forward neural network*) en el capítulo [:octicons-book-24:][neural] "[Neural Networks and Neural Language Models][neural]". En la mayoría de las tareas actuales de procesamiento del lenguaje natural no usaremos arquitecturas tan sencillas, pero las redes *feed-forward* aparecen como componente en los modelos avanzados basados en la arquitectura *transformer* que veremos más adelante.

Todo el capítulo es relevante, aunque probablemente se hable de cosas que ya conoces. Puedes saltar o leer con menos detenimiento las secciones 7.6.3 ("Computation Graphs") y 7.6.4 ("Backward differentiation on computation graphs").

[neural]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/7.pdf


## Anotaciones al libro

Este capítulo no debería plantear grandes retos si has comprendido bien los conceptos de capítulos anteriores, menos aún si ya habías estudiado redes neuronales en algún otro curso. En cualquier caso, aquí tienes algunas anotaciones que pueden ayudarte a entender mejor el capítulo.

Las ideas básicas de los regresores se mantienen en las redes hacia delante (*feedforward*) que se estudian en este capítulo. La red neuronal recibe una serie de números a la entrada y produce uno o más números a la salida. Las entradas se transforman en las salidas mediante una serie de operaciones matemáticas que pueden expresarse como productos de matrices junto con la aplicación adicional de algunas funciones no lineales como la sigmoide o la función softmax. En el caso de que las entradas representen palabras, sabemos que estas se pueden representar de forma numérica mediante los llamados embeddings. Por lo tanto, una manera de hacer que la red neuronal procese, por ejemplo, tres palabras cada vez sería representar cada palabra mediante su embedding (aprendido mediante un algoritmo como skip-grams, que ya vimos) y concatenar los embeddings para obtener un único vector, que sería la entrada de la red neuronal. A la salida de la red neuronal, podríamos usar una función tipo sigmoide si nos interesa tener un clasificador binario que, por ejemplo, nos dé un valor entre cero y uno que represente el grado de positividad de un texto. Si queremos que la red neuronal nos dé una clasificación entre varias categorías, podemos usar una función softmax para obtener una distribución de probabilidad sobre ellas.

Esta última interpretación de la salida abre una posibilidad interesante: la de usar las redes neuronales como *modelo de lengua*, es decir, como un modelo probabilístico que nos puede decir, dado un determinado texto, cuál es la probabilidad de que cada una de las palabras de una lista sea la que va a continuación de dicho texto. Esta red, una vez entrenada, será muy útil para generar una continuación a un prefijo dado. Supongamos que tengo el prefijo "Cuando miro por la ventana, puedo ver el" y que concateno los embeddings de las tres últimas palabras (*puedo*, *ver*, *el*) y se lo doy como entrada a la red neuronal. Esta emitirá un vector de probabilidades con tantos elementos como hayamos decidido tener en el vocabulario. Supongamos que la salida que más probabilidad recibe es la asociada a la palabra "mar". Ahora puedo introducir en la red neuronal los embeddings correspondientes a *ver*, *el* y *mar* para obtener la siguiente palabra. Y así sucesivamente. De esta manera, dado el prefijo anterior podría obtener una secuencia como "Cuando miro por la ventana, puedo ver el mar Mediterráneo frente a mí". Idealmente, si el prefijo es una pregunta, la red neuronal debería ser capaz de responderla como hacen los grandes modelos de lengua actuales. Este tipo de comportamientos *emergentes* se pueden conseguir con los modelos neuronales, pero tendremos que ir un poco más allá de las redes neuronales de este capítulo para conseguirlo.

¿Por qué las redes neuronales *feedforward* no son adecuadas para este tipo de tareas? Habrás observado que una gran limitación del planteamiento anterior viene dada por el hecho de haber elegido tres palabras como entrada a la red. En el contexto del prefijo completo, puede tener sentido que la palabra "mar" tenga una gran probabilidad de aparecer tras "puedo ver el", pero dado que las redes hacia delante no van a tener ningún tipo de memoria que les permita *recordar* el resto del prefijo, muy probablemente den más probabilidad a palabras como *problema* o *caso*, por decir algunas, que a *mar* u otras palabras que pueden tener más sentido cuando uno mira por la ventana. En principio, podríamos pensar que si aumentamos el número de embeddings concatenados a la entrada, ampliaríamos el contexto que el sistema es capaz de tener en cuenta a la hora de hacer su predicción. Y no deja de ser cierto. Pero ampliar el tamaño de la entrada implica ampliar la cantidad de pesos que es necesario aprender. Y en aprendizaje automático es bien conocida la llamada *maldición de la dimensionalidad*, que implica que cuando aumenta la cantidad de parámetros a aprender, la cantidad de datos necesarios para hacerlo crece exponencialmente. Además, hay algunos otros problemas que no vamos a analizar en detalle, como el hecho de que los parámetros que se aprendan para una palabra cuando esta aparece en la primera posición no sean aprovechables cuando esta ocupa otras posiciones, o que tengamos que decidir una longitud máxima para las secuencias que queramos procesar. El transformer, que veremos más adelante, supera muchos de estos problemas. Pero antes de llegar a él, veamos como funciona una red neuronal más sencilla.

En toda la discusión anterior, sin embargo, no hay nada nuevo respecto a los regresores que ya conocemos. ¿De dónde viene la superioridad computacional de las redes *feedforward*? La respuesta está en el hecho de que estas redes introducen capas adicionales de procesamiento. Si un regresor multinomial solo multiplica la entrada por una matriz y aplica softmax al resultado, una red neuronal multiplica la entrada por una matriz de pesos, aplica una función habitualmente no lineal (por ejemplo, una sigmoide) al vector resultante, multiplica el nuevo vector por otra matriz de pesos, aplica al resultado una nueva función (llamada *función de activación*), etc. El proceso se realiza durante varias capas hasta producir el vector de salida final.

El resto de elementos que rodean las redes neuronales *feedforward* son heredados de modelos como el regresor. Así, se busca maximizar la verosimilitud de los datos mediante descenso por gradiente estocástico y, para ello, la entropía cruzada se usa como función de error que permite determinar la magnitud del *tirón* que se da a los pesos en cada paso de entrenamiento. De la misma forma, no es necesario que las muestras se presenten de una en una, sino que se pueden agrupar en lotes (*mini-batch*); en este caso, el error se calcula como la media del error de cada muestra del lote.

Por último, aunque hemos dicho que los embeddings de cada palabra se pueden haber aprendido con anterioridad al entrenamiento de la red neuronal, lo cierto es que en la práctica estos se suelen inicializar aleatoriamente y se aprenden junto con el resto de parámetros de la red. El mecanismo para hacerlo ya lo conoces: se calcula la derivada de la función de error respecto a cada componente de los embeddings para así poder actualizarlos en cada paso de entrenamiento.

Apartado 7.1
{: .section}

Puede parecer diferente, pero este apartado no está sino repitiendo lo que ya conoces del regresor logístico binario. Dado que, en general, estas unidades de procesamiento no solo se van a utilizar en la capa final de la red, sino también en las intermedias, se presentan otras funciones de activación (como ReLU) que suelen utilizarse además de la sigmoide. El criterio para usar una u otra función suele ser empírico y basado en el problema o arquitectura neuronal concreta.


Apartado 7.2
{: .section}

Este apartado justifica desde un punto de vista computacional la necesidad de tener varias capas de procesamiento, al demostrar que con un regresor lineal no es posible aprender a clasificar datos que siguen un esquema tan simple como el de la función *o exclusiva*.


Apartado 7.3
{: .section}

Aquí ya tenemos un modelo que podría usarse para la tarea de predicción de la siguiente palabra que hemos comentado más arriba: el modelo recibe un vector de entrada y produce un vector de probabilidades, todo ello mediante las siguientes ecuaciones que definen las dos capas del modelo:

$$
  \begin{align} 
  \mathbf{h} &= \sigma(\mathbf{W} \mathbf{x} + \mathbf{b}) \nonumber \\ 
  \mathbf{y} &= \mathrm{softmax}(\mathbf{U} \mathbf{h}) \nonumber
  \end{align}
$$

Observa que en principio la entrada a la función softmax también podría incluir la suma de un segundo vector de umbrales, pero en la ecuación (7.12) se ha omitido para simplificar, aunque aparece de nuevo un poco más adelante. Es bastante interesante entender el razonamiento que se incluye al final de este apartado sobre la necesidad de funciones de activación no lineales ya que, de no usarlas, la red "colapasaría" a un regresor con menor poder de representación. De hecho, un conocido teorema (el conocido como *teorema de aproximación universal*) demuestra que, en principio, cualquier función continua puede ser aproximada por una red neuronal con una sola capa oculta y funciones de activación no lineales. No obstante, que la red neuronal exista, no quiere decir que sea fácil dar con ella para un problema dado (esto es, saber cuántos parámetros tiene y obtener sus valores). En la práctica, se suelen usar varias capas, pues se ha comprobado que el aprendizaje se simplifica de esta manera.

Apartado 7.4
{: .section}

Este apartado no introduce apenas elementos nuevos. Si vamos a usar una red neuronal para realizar tareas de clasificación sobre frases, podemos evitar la necesidad de tener que lidiar con una cantidad variable de embeddings de palabras y representar la frase mediante un único embedding obtenido como la media de los embeddings de las palabras que la componen. Dado que la operación de promediado es derivable, podemos seguir usando el algoritmo de descenso por gradiente estocástico para aprender los embeddings de cada una de las palabras.

El apartado también presenta cómo gestionar un mini-batch de entradas. Observa que, salvo por algún pequeño ajuste en las matrices implicadas, el proceso es el mismo que en el caso de un solo ejemplo y consiste básicamente en multiplicar la entrada (en este caso, una matriz en lugar de un único vector) por la matriz de pesos. Comprueba que el resultado es el mismo que si se hubiera multiplicado cada vector de entrada por la matriz de pesos y se hubieran concatenado los resultados en otra matriz.

Estudia bien las figuras de este y el resto de apartados del capítulo, ya que son muy útiles para afianzar los conceptos discutidos.


Apartado 7.5
{: .section}

De nuevo, este apartado solo asienta ideas anteriores al proponer un modelo de lengua basado en redes *feedforward*.

Aunque conceptualmente tiene sentido mostrar la selección de un embedding de entre todos los disponibles en la tabla de embeddings como el producto de un vector one-hot por la matriz de embeddings, en las implementaciones prácticas, sin embargo, se suele usar una representación más eficiente que permite acceder directamente al embedding deseado mediante un índice. Aunque esta operación no es estrictamente derivable, se puede seguir usando el algoritmo de descenso por gradiente: basta con actualizar únicamente los parámetros del embedding que se ha seleccionado, dejando los demás sin modificar.

Apartado 7.6
{: .section}

Una vez más, se reelaboran aquí conceptos ya conocidos del tema de regresores, en particular de la sección de regresores multinomiales. La entropía cruzada a minimizar se obtiene como el valor opuesto del logaritmo de la probabilidad de la palabra correcta. La forma analítica de esta función de pérdida nos permite obtener el gradiente respecto a cada uno de los parámetros del modelo, que es lo que se usa para actualizarlos en cada paso de entrenamiento.

Al haber varias capas, el cálculo de la derivada del error se hace más tedioso de hacer manualmente que en el caso de un regresor logístico. Por suerte, PyTorch y otros frameworks actuales nos permiten calcularlas de forma automática (a través de la técnica conocida como *automatic differentiation*). Las ideas intuitivas detrás de este cálculo se presentan en las secciones 7.6.3 y 7.6.4, que puedes saltar salvo que tengas un especial interés en el tema.

Apartado 7.7
{: .section}

Como ya hemos comentado, todos los parámetros, incluidos los valores de los embeddings, se actualizan en cada paso de entrenamiento mediante el algoritmo de descenso por gradiente estocástico. En este apartado se condensan a modo de colofón todas estas ideas. 
