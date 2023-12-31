# Библиотека для решения задач выпуклой оптимизации Абель
Я обожаю задачи оптимизации и давно ими занимаюсь (с конца 2021 года), в т. ч. методами их решения и некоторыми численными методами линейной алгебры. В качестве полноценной практики я решил написать свою мини-библиотеку, способную решать достаточно общие варианты задач, которые часто встречаются. Это такие задачи, как например линейная задача наименьших квадратов, минимизация квадратичных форм с ограничениями равенствами и неравенствами, выпуклая негладкая оптимизация (в случае, когда возможно посчитать проксимальный оператор, например **LASSO**-регрессия), задачи линейного и полуопределённого программирования. Также присутствуют реализации метода Ньютона и метода золотого сечения для поиска корня или минимума функций одного переменного.
<p align=center>
   <img src=https://github.com/N-Kulagin/Abel/assets/30264322/dc8a6984-8d4d-49ba-9184-2c1e6a625aea/>
</p>

## На данный момент реализованы: 
1. Прямо-двойственные методы внутренней точки для задач линейного, квадратичного и полуопределённого программирования, основанные на схеме **Predictor-Corrector** Санджея Мехротры (**Sanjay Mehrotra**)
2. Ускоренный градиентный спуск Нестерова с адаптивными рестартами и возможностью кастомизации (применение оператора проекции или проксимального оператора, выбор схемы подбора константы Липшица в зависимости от наличия выпуклости в задаче)
3. Обыкновенный метод Ньютона для решения произвольной выпуклой задачи оптимизации с линейными ограничениями-равенствами и выбором шага по правилу Армихо
4. Метод Нелдера-Мида для невыпуклой негладкой оптимизации с экономным вычислением значений функции
5. Проекции на 7 наиболее часто встречающихся множеств:
   - множество решений СЛАУ $Ax=b$
   - шары в $1,2,\infty-$нормах
   - "коробку" $a \leq x \leq b$
   - неотрицательный ортант $\mathbb{R}^n_+ = \\{x \in \mathbb{R}^n |\ x_i \geq 0\ \forall i=\overline{1,n}\\}$
   - симплекс $S_n = \\{x \in \mathbb{R}^n\ | \displaystyle \sum_\{i=1\}^\{n\} x_i = \alpha, x_i \geq 0\\ \forall i=\overline{1,n},\ \alpha > 0  \\}$
8. Проксимальные операторы для $\ell_1, \ell_2, \ell_\infty$ норм
9. Метод Ньютона для поиска корня функции одной переменной с эвристическим выбором шага или выбором по правилу Армихо (backtracking), метод золотого сечения для поиска минимума
10. Некоторые вспомогательные функции, позволяющие автоматически решать задачи линейного и квадратичного программирования, задачу **LASSO**-регрессии, а также симметрическая векторизация симметричной матрицы (**svec**) и обратная её операция - восстановление симметрической матрицы по вектору (**smat**).

### Точка Ферма-Торичелли (работа [Suman Vaze](https://sites.google.com/site/vazeart/theorems&constructions))
Даны $N$ точек $x_1, \dots, x_N \in \mathbb{R}^n$, необходимо найти точку $y \in \mathbb{R}^n$, сумма расстояний от которой до всех остальных будет минимальна.
Т. е. необходимо решить задачу оптимизации, решением которой является точка Ферма-Торичелли: $$\displaystyle \sum_\{i=1\}^\{N\} \lVert y-x_i\rVert_2 \to \min\_\{y \in \mathbb{R}^n\}$$

<p align=center>
  <img src=https://github.com/N-Kulagin/Abel/assets/30264322/84da3de1-a723-4fc3-99d3-4ec7fb2b948b/>
</p>
