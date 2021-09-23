# Приложение: Математика для глубокого обучения
:label:`chap_appendix_math`

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), и авторы этой книги


Одной из замечательных вещей современного глубокого обучения является то, что большую его часть можно понять и использовать без полного понимания лежащей в его основе математики. Это признак того, что область взрослеет. Подобно тому, как большинству разработчиков программного обеспечения больше не нужно беспокоиться о теории вычислимых функций, так и специалистам-практикам глубокого обучения не следует переживать о теоретических основах обучения с максимальным правдоподобием. 

Но мы еще не совсем там.

На практике вам иногда нужно понимать, как выбор архитектуры влияет на движение градиента, или на неявные предположения, которые вы делаете при обучении с определенной функцией потерь. Возможно, вам потребуется знать, что измеряет энтропия, и как это поможет вам понять, что в точности означает бит на символ в вашей модели. Все это требует более глубокого математического понимания.

This appendix aims to provide you the mathematical background you need to understand the core theory of modern deep learning, but it is not exhaustive.  We will begin with examining linear algebra in greater depth.  We develop a geometric understanding of all the common linear algebraic objects and operations that will enable us to visualize the effects of various transformations on our data.  A key element is the development of the basics of eigen-decompositions.

We next develop the theory of differential calculus to the point that we can fully understand why the gradient is the direction of steepest descent, and why back-propagation takes the form it does.  Integral calculus is then discussed to the degree needed to support our next topic, probability theory.

Problems encountered in practice frequently are not certain, and thus we need a language to speak about uncertain things.  We review the theory of random variables and the most commonly encountered distributions so we may discuss models probabilistically.  This provides the foundation for the naive Bayes classifier, a probabilistic classification technique.

Closely related to probability theory is the study of statistics.  While statistics is far too large a field to do justice in a short section, we will introduce fundamental concepts that all machine learning practitioners should be aware of, in particular: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.

Last, we turn to the topic of information theory, which is the mathematical study of information storage and transmission.  This provides the core language by which we may discuss quantitatively how much information a model holds on a domain of discourse.

Taken together, these form the core of the mathematical concepts needed to begin down the path towards a deep understanding of deep learning.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```

