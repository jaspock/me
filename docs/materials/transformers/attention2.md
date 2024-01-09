
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

En el bloque anterior, hemos estudiado todos los elementos del transformer, pero en algunas aplicaciones en las que una secuencia de tokens se transforma en otra secuencia de tokens, las capas del transformer se asocian a dos submodelos bien diferenciados y conectados entre ellos mediante mecanismos de atención: el codificador y el descodificador. El capítulo [:octicons-book-24:][mt] "[Machine translation][mt]" se centra en la arquitectura completa. En este capítulo solo es necesario que estudies por ahora las secciones 10.2 y 10.6.

[mt]: https://web.archive.org/web/20221104204739/https://web.stanford.edu/~jurafsky/slp3/10.pdf

El capítulo aborda el estudio de la arquitectura completa del transformer desde el punto de vista de la traducción automática (la primera aplicación para la que se usaron los transformers), pero los apartados en los que nos vamos a centrar son bastante generales y no asumen una aplicación concreta dentro de la tarea de transformar una secuencia de entrada en otra secuencia de salida.

Observa que ya vimos en el bloque anterior que tareas como la traducción automática o la obtención de resúmenes pueden también ser abordadas con modelos basados únicamente en el descodificador.

## Anotaciones al libro

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--recomendable-start-->"
   end="<!--recomendable-end-->"
%}

Apartado 10.2
{: .section}

Este apartado es muy breve y se limita a presentar la idea de la arquitectura codificador-descodificador, que no solo está presente en los transformers. Observa cómo el codificador genera una representación intermedia de la secuencia de entrada que es usada por el descodificador para guiar la generación de la secuencia de salida.

Apartado 10.6
{: .section}

Otro apartado breve donde se ve que las únicas novedades que aporta la arquitectura completa a lo que ya sabías son:

- el codificador no es autorregresivo (observa la figura 10.15), por lo que no es necesario aplicar las máscaras que anulan la atención a los tokens futuros; más adelante, veremos las ecuaciones completas del modelo bajo este enfoque, pero la ausencia de máscara es el único cambio reseñable.
- en la *atención cruzada* las claves y los valores se obtienen mediante sendas matrices a partir de los embeddings de la última capa del codificador mientras que las consultas se generan en el descodificador.

## Búsqueda y subpalabras

Hay algunos elementos adicionales a la definición de los diferentes modelos neuronales que son también relevantes cuando estos se aplican en el área del procesamiento del lenguaje natural. 

- Estudia el mecanismo de búsqueda en haz (*beam search*) descrito en la [:octicons-book-24:][mt] [sección 10.5][mt].
- Lee lo que se dice sobre la obtención de subpalabras en las secciones [:octicons-book-24:][mt] [10.7.1][mt] y [:octicons-book-24:][cap2] [2.4.3][cap2].

[cap2]: https://web.archive.org/web/20221026071429/https://web.stanford.edu/~jurafsky/slp3/2.pdf

Apartado 10.5
{: .section}

A la hora de generar la secuencia de salida a partir de los sucesivos vectores de probabilidad emitidos por el descodificador, hay muchos enfoques posibles. Uno de los más sencillos consiste en elegir el token con mayor probabilidad en cada paso, pero, como se ve en el libro, esta estrategia suele ser muy ineficiente. Un enfoque ligeramente más elaborado es el de la búsqueda en haz (*beam search*), que consiste en mantener en cada paso un conjunto de posibles secuencias de salida y elegir las mejores para continuar con el proceso.

Apartados 2.4.3 y 10.7.1
{: .section}

La mayoría de los modelos neuronales actuales de procesamiento de textos no consideran cada palabra del texto como una entrada diferente, sino que habitualmente dividen (*tokenizan*) el texto en unidades más pequeñas antes de procesarlo por la red neuronal. Dos de los algoritmos más usados para esta tarea son *BPE* y *SentencePiece*. El primero necesita que el texto esté dividido en palabras, mientras que el segundo trata el texto carácter a carácter sin que los posibles separadores (el espacio en blanco, por ejemplo) jueguen un papel especial. Ambos terminan haciendo que las palabras muy frecuentes se representen con una subpalabra única, mientras que las palabras menos frecuentes se representan con una secuencia de subpalabras. A la hora de determinar la segmentación que se va a usar en nuestro modelo, el desarrollador indica el número máximo de subpalabras a utilizar para obtener el vocabulario del problema.

## Modelos preentrenados

El capítulo [:octicons-book-24:][bert] "[Transfer Learning with Contextual Embeddings and Pre-trained language models][bert]" estudia los modelos preentrenados basados en codificador y cómo adaptarlos a nuestras necesidades. Para nuestro estudio son relevantes la introducción y las secciones 11.1, 11.2 y 11.3 (quitando el apartado 11.3.4). 

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

Apartado 11.3
{: .section}

Este apartado demuestra como adaptar un modelo preentrenado capaz de producir embeddings *neutros* a tareas concretas de clasificación. El proceso de adaptación se denomina *fine-tuning* y consiste sustituir los predictores de la última capa del modelo pre-entrenado por un predictor específico para la tarea que se quiere resolver. El modelo preentrenado se mantiene fijo durante el proceso de *fine-tuning* y solo se actualizan los pesos del nuevo predictor. Si la tarea de clasificación es a nivel de frase o de pares de frases (por ejemplo, determinar el sentimiento de una frase o determinar si una frase se deduce de otra) basta con entrenar un predictor sobre el token [CLS]. Si la tarea es a nivel de token (por ejemplo, determinar la categoría gramatical de cada token) se puede entrenar un predictor que actúe sobre todos los tokens de la frase.

## Multilingual models

So far, in our exploration on transformer-based natural language processing models, we have primarily considered systems that work on a single language or, at least, we have not explictly considered the possibility of using a single model for multiple languages. But, as we will see in this section, models can handle multiple languages at the same time and there are important advantages in doing so. One of these advantages comes from a phenomenon known as *transfer learning*, where knowledge gained about one language in the representations learned by the neural network can be applied to other languages. This is particularly valuable for *low-resource* languages, that is, languages with fewer training resources, as they may end up benefiting from the knowledge gained by the model from other languages.

Training multilingual models
{: .section}

Training a multilingual transformer model is not very different from training a monolingual model. The main difference is that the training data is a mixture of texts in different languages. The tokenizer is obtained by using the whole multilingual corpus with systems such as SentencePiece or byte pair encoding (BPE). If the languages are related, then some tokens may be shared which will help the neural network to learn *joint representations* also known as *cross-lingual embeddings*. Consider, for example, the case of *centro*, *centru*, *centre* and *center* in Spanish, Asturian, Catalan, or English, respectively; if these words are tokenized as *centr* and the remaining suffix, then the model may learn a representation for *centr* from Spanish, Catalan or English texts and use it with Asturian inputs, even though the word *centru* has not been seen in the Asturian training data. People or location names may also be easily shared across languages.In the usual case of data imbalance, where there are many more texts in some languages than in others, it is common to use a rebalancing formula to upsample languages with less data so that their texts end up being overrepresented in the training or evaluation datasets, as well as in the corpus used to train the tokenizer.

Interestingly, in spite of not using shared tokens, it may happen that words such as *atristuráu* (in Asturian) or *sad* (in English) get similar embeddings in some layers of the model, which is another example of the language-independent nature of some of the representations learned.

Additionally, depending on the task, language-specific tokens can be used as a first token to indicate the language of the text. This is not so common in the case of decoder-like models, but it is a standard practice in encoder-like models such as those used in named entity recognition or sentence classification. These are special tokens added to the vocabulary that are preprended to every sentence both during training and inference, and also learned during training in the same way as the *MASK* token is learned in BERT-like pretraining. 

Decoder-like language models are usually trained with multilingual data from the beginning, but they do not require language-specific tokens. The language used in the prompt is enough to guide the generation process towards the desired language.

Pre-trained multilingual models
{: .section}

The first pre-trained models such as BERT were English-centric. However, over time variants emerged for other languages, such as CamemBERT for French, GilBERTo for Italian, or BERTo for Spanish. Despite these advances, these models were still essentially monolingual. An inflection point was reached with the appearance of pre-trained multilingual models such as mBERT or XLM-R, which handle around a hundred languages or, more recently, Glot500, which arrives to 500 languages. As these models are self-supervisedly trained with neutral tasks such as masked language modeling, they are general-purpose that can then be fine-tuned for specific tasks in any of the languages ​​they handle. A notable phenomenon in this case is the *zero-shot generalization* ability of the model, meaning that, for example, in a named entity recognition task, a model can be fine-tuned only with English texts, but thanks to the multilingual representations learned during pre-training, it can be applied to texts in other languages with some success ​​without the need for labeled data in those languages. This is particularly useful for low-resource languages.

Challenges
{: .section}

Multilingual models often confront the challenge known as *the curse of multilinguality* (a term coined after the *curse of dimensionality* in statistics and machine learning). This phenomenon refers to the decline in performance for each individual language as the model expands to encompass a larger number of languages. It is also common that those languages with more resources are the ones that benefit the least from the multilingual model, even showing a lower performance than a monolingual model trained with the same data. 

A number of techniques have been proposed in order to mitigate the curse of multilinguality. One of them is the use of *adapters*, small, trainable modules (for example, a feedforward network), one for each of the languages, inserted at specific points into the layers of the transformer. When training the model, the common parameters are learned as usual, but the parameters of the adapters are only updated for the language they are associated with. This way, the model learns a mixture of cross-lingual and language-specific parameters. Adapters are also used to fine-tune pre-trained models to new tasks or languages without retraining the entire model, this way keeping the original weights of the model frozen and only updating the parameters of the adapters. 

Note that other alternatives for the use of monolingual models in multilingual scenarios, such as *translate-train* or *translate-test*, are also possible. In the first case, the training data only available in one language is translated into the other languages and a multilingual model is trained with both the source and the translated data. In the second case, the test data is translated into the language of a monolingual model at inference time. In both cases, the translation is performed with a machine translation system.

Finally, there are a number of multilingual datasets available for different tasks, such as the universal dependencies treebanks, the XNLI dataset for natural language inference (determining whether a sentence entails, contradicts or is neutral toward a hypothesis sentence), or the FLORES+ corpus for machine translation.
   
## Las diferentes caras de la atención

Este contenido es opcional. Puedes saltarlo directamente salvo que te hayan dicho expresamente que lo estudies. En el siguiente análisis nos basaremos en la [discusión][dontloo] de *dontloo* en Cross Validated. En los sistemas *seq2seq* originales basados en redes recurrentes, la atención se calcula una media ponderada de las claves:

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
