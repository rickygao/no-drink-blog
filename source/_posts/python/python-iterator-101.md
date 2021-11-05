---
title: Python 的迭代器入门
date: 2021-11-05 20:00:00
tags: [Python, Iterator]
---

迭代器是很多流行的编程语言的标准配置，用以表示逻辑上有顺序的一系列元素，通常是一次性的和惰性的。尤其是在 Python 中，迭代器得益于其高效且易用的特性，以及如 `more-itertools` 等社区维护的优秀开源包，得到了广泛的应用。本文将对 Python 中的迭代器进行简单的介绍，并辅以示例帮助理解。

灵活地使用迭代器需要解决三个问题：迭代器从哪来、迭代器怎么变和迭代器得到啥。下面的代码块是一个典型的迭代器使用实例，三行代码分别对应了迭代器从哪来、迭代器怎么变、迭代器到哪去。

```python
numbers = range(42) # where iterators come
doubled = map(lambda n: n * 2, numbers) # how iterators transform
summary = sum(doubled) # what iterators do
```

当然，与之基本等价的更地道的写法是 `summary = sum(n * 2 for n in range(42))`，而这通常适用于较为简单的操作。

<!-- more -->

## 迭代器从哪来

这一部分介绍了迭代器的常见来源。

### 范围 `range`

`range` 是 Python 的内建函数，用于创建一个按顺序排列的 `int` 类型元素的迭代器。`range` 通常是每个 Python 使用者学会使用的第一个迭代器。

```python
range(7) # 0, 1, 2, 3, 4, 5, 6
range(3, 6) # 3, 4, 5
range(10, 15, 2) # 10, 12, 14
range(42, 30, -3) # 42, 39, 36, 33
```

小提示：当你尝试打印一个迭代器时，你可能会对其并不直观输出结果感到疑惑，这是因为迭代器通常作为程序的中间结果。我推荐在初学阶段首先使用 `list(iterator)` 获得一个列表，如：`list(range(7))`，再打印以获得直观的迭代器中的元素输出。

注意：迭代器并不总具有有限个元素，你可以使用 `take(iterator, n)` 来获得迭代器 `iterator` 的前 `n` 个元素组成的迭代器，而这在初学阶段并不常见。

### 列表 `list` 和元组 `tuple`

`list` 和 `tuple` 作为 Python 的内建容器类型，是天生的迭代器。

```python
[7, 42, 256] # a list
(255, 0, 0) # a tuple
```

### 字典 `dict` 及其方法

`dict` 也是 Python 的内建容器类型。需要注意的是，当 `dict` 实例直接作为迭代器使用时，其元素是全体键。你可以显式地调用 `keys`, `values` 和 `items` 方法获取以键、值和键值对（也即由键和值组成的元组）为元素的迭代器。

```python
a_dict = {'name': 'NoDrink', 'author': 'Ruijun Gao', 'year': 2020}
a_dict.keys() # 'name', 'author', 'year'
a_dict.values() # 'NoDrink', 'Ruijun Gao', 2020
a_dict.items() # ('name', 'NoDrink'), ('author', 'Ruijun Gao'), ('year', 2020)
```

### 字符串 `str`

`str` 是天生的迭代器，其元素是字符串中的个字符。

```python
'Come on' # 'C', 'o', 'm', 'e', ' ', 'o', 'n'
```

## 迭代器怎么变

这一部分介绍了如何对迭代器进行变换。

### 映射 `map`

`map` 是 Python 的内建函数，可以将迭代器中的每个元素进行映射，以获取一个新的迭代器。其第一个参数通常是 lambda 表达式，用以定义映射操作的内容，返回值将作为新迭代器中的元素，该参数也可以是任何可调用的对象（即实现了 `__call__` 魔术方法的 `Callable` 实例）。这一操作是惰性的，也就是说在你最终消费迭代器前，映射操作都不会被执行。

```python
map(lambda n: n ** 2, range(7)) # 0, 1, 4, 9, 16, 25, 36
map(str.upper, ['one', 'two', 'three']) # 'ONE', 'TWO', 'THREE'
```

### 过滤 `filter`

`filter` 也是 Python 的内建函数，可以对迭代器中的元素进行过滤，以获取一个新的迭代器。与 `map` 类似，第一个参数通常是 lambda 表达式，但返回值为 `bool` 类型，用以指示该元素是否会被保留。

```python
filter(lambda n: n < 0, [-1, 42, 9, 128, -7]) # -1, -7
filter(str.isalpha, 'B3s7')) # 'B', 's'
```

### 缝合 `zip`

`zip` 也是 Python 的内建函数，可以将多个迭代器中的元素依次缝合为一个元组，以获取一个新的迭代器。

```python
zip(range(3), 'ABC') # (0, 'A'), (1, 'B'), (2, 'C')
```

## 迭代器能干啥

这一部分介绍了通过迭代器可以干什么。

### 列表 `list` 和元组 `tuple`

你如果将迭代器直接传入 `list` 或 `tuple`，可以得到含有迭代器中元素的列表和元组。

```python
list(range(7)) # [0, 1, 2, 3, 4, 5, 6]
tuple('NoDrink') # ('N', 'o', 'D', 'r', 'i', 'n', 'k')
```

### 字典 `dict`

如果一个迭代器的元素是二元组，那么可以调用 `dict` 以获取一个字典，字典的键和值则是该迭代器全体元组元素的第一个和第二个元素。

```python
dict(zip('ABC', range(3))) # {'A': 0, 'B': 1, 'C': 2}
```

### 求和 `sum`

对于元素为数值类型的迭代器，你可以使用 `sum` 对其全部元素进行求和。类似的函数还有 `all` 和 `any` 等。

```python
sum(range(10)) # 45
sum([0, 1.0, 3.2]) # 4.2
sum([-4, 4 + 2j]) # 2j
```

## 综合使用

以下代码读取文本文件 `example.txt` 并计算了其中所有单词的出现频次，最后将结果写入到 `word_counts.txt`。

```python
from pathlib import Path
from collections import Counter

example_words = Path('example.txt').read_text().split()
word_counts = Counter(word.lower() for word in example_words if word.isalpha())
count_result = '\n'.join(f'{word}: {count}' for word, count in word_counts.most_common())
Path('word_counts.txt').write_text(count_result)
```

迭代器是处顺序组织数据的利器，本文仅对其进行了简单的介绍。而除了上述介绍的函数和方法外，Python 标准库 `itertools` 以及 `more-itertools` 等第三方包中包含了更多更易用的处理迭代器的工具。望读者在使用过程中多加探索，更加地道地使用迭代器。
