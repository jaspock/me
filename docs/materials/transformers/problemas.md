
<div class="content-2columns" markdown>
![](assets/imgs/robot-eureka.png){: .rounded-title-img}

# Problemas
</div>

## Problemas de regresores

Problema
{: .problema}

La disposición de los elementos en matrices y vectores puede ser diferente a la utilizada en la sección 5.2.3. Lo realmente importante es que se realice el productor escalar de cada una de las $m$ muestras con el vector de pesos $\mathbf{w}$. Indica qué tamaños deberían tener las matrices y vectores si en lugar de una ecuación como la (5.14), usamos una de la forma $\mathbf{y} = \mathbf{w} \mathbf{X} + \mathbf{b}$.

Problema
{: .problema}

Calcula la derivada de la función de coste respecto al umbral $b$. Si te basas en la derivada de la función de coste respecto a los pesos $w$, que está calculada en el libro, llegarás rápido a la solución.

Problema
{: .problema}

Tras entrenar un regresor logístico, le aplicamos una entrada $\mathbf{x}$ y calculamos la derivada $\partial \hat{y} / \partial \mathbf{x}_i$ para un cierto $i$. ¿Qué mide esta derivada? Piensa en el concepto básico de la derivada y en cómo mide la *sensibilidad* del valor de un función respecto a un cambio en una de sus variables.

## Problemas de embeddings

Problema
{: .problema}

Al buscar relaciones de analogía entre word embeddings intentamos, dadas las palabras A y B (relacionadas entre ellas), encontrar en el espacio de embeddings dos palabras, C y D (también relacionadas tanto entre ellas como con A y B) para los que tenga sentido afirmar que "A es a B como C es a D". Matemáticamente, se trata de encontrar cuatro palabras cuyos embeddings cumplan ciertas propiedades geométricas basadas en las distancias entre sus vectores. Por ejemplo, es habitual observar que estas propiedades se cumplen si A=man, B=woman, C=king y D=queen (diremos que "man es a woman como king es a queen"). Supongamos que hemos obtenido mediante el algoritmo skip-gram unos embeddings de palabras para el inglés que permiten encontrar este tipo de analogías. Considera la lista de palabras L = {planes, France, Louvre, UK, Italy, plane, people, man, woman, pilot, cloud}. Indica con qué palabra de la lista L tiene más sentido sustituir @1, @2, @3, @4, @5 y @6 en las siguientes expresiones para que se cumplan las analogías:

- A=Paris, B=@1, C=London, D=@2
- A=plane, B=@3, C=person, D=@4
- A=mother, B=father, C=@5, D=@6

## Problemas de transformers

Problema
{: .problema}

Argumenta la verdad o falsedad de la siguiente afirmación: si la consulta de un token $q_i$ es igual a su clave $k_i$, entonces el embedding computado por el mecanismo de autoatención para dicho token coincide con su valor $v_i$.

Problema
{: .problema}

La siguiente fórmula para el cálculo de la auto-atención en el descodificador de un transformer es ligeramente distinta a la habitual, ya que se ha añadido explícitamente la máscara que impide que, como hemos visto, la atención calculada para un token durante el entrenamiento tenga en cuenta los tokens que se encuentran posteriormente en la frase:

$$
\text{Atención}(Q,K,V,M) = \text{softmax} \left( \frac{Q K^T + M}{\sqrt{d_k}} \right) \, V
$$

Indica qué forma tiene la matriz de máscara $M$. Busca las operaciones de PyTorch que te permiten inicializar dicha matriz. Si necesitas usar valores de infinito en el código, razona si es suficiente con utilizar un número grande como $10^9$.

Problema
{: .problema}

Dada una secuencia de tokens de entrada, el transformer permite obtener un conjunto de embeddings para cada token de entrada. Estos embeddings son contextuales en todas sus capas menos en una. ¿En qué capa los embeddings no son contextuales? ¿Cómo se obtienen los valores finales de estos embeddings? ¿Y los valores iniciales?

Problema
{: .problema}

Supongamos que el embedding no contextual almacenado en la tabla de embeddings de un transformer para un determinado token viene definido por el vector $\mathbf{e} = \mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$. Consideremos el caso en el que dicho token aparece en dos posiciones diferentes de una frase de entrada. Si no se utilizaran embeddings posicionales, ¿cuál sería el valor del coseno entre los vectores de los dos embeddings no contextuales usados para el token? Llamemos ahora $\mathbf{p}$ y $\mathbf{p}’$ a los embeddings posicionales que se utilizan en cada una de las dos apariciones del token. El coseno del ángulo entre los vectores resultantes de sumar al embedding los embeddings posicionales sería:

$$
\cos(\alpha)=\frac{\sum_{i=1}^d(\mathbf{e}_i+\mathbf{p}_i)(\mathbf{e}_i+\mathbf{p}'_i)}{\sqrt{\left(\sum_{j=1}^d(\mathbf{e}_j+\mathbf{p}_j)^2\right)\left(\sum_{j=1}^d(\mathbf{e}_j+\mathbf{p}_j')^2\right)}}
$$

¿A qué valor se irá acercando el coseno anterior cuando la distancia que separa las dos apariciones del token en la frase vaya aumentando? Razona tu respuesta.
