---
title: Python 的陷阱
date: 2020-11-23 12:00:00
tags: [Python]
---

本文介绍了 Python 中常见的一些陷阱。

<!-- more -->

## 喋喋不休生成器

```python
l = [0]
l += l # ok, l == [0, 0]
l += (e for e in l) # infinite loop
```

一条蛇试图吞掉自己的尾巴。

## 合并同类项字典

```python
d = dict()
d[     0] = 'a' # d == {0: 'a'}
d[   0.0] = 'b' # d == {0: 'b'}
d[0 + 0j] = 'c' # d == {0: 'c'}
```

数，生而平等。`dict` 根据 `==`(`__eq__`) 来判断是否为相同的键，而 `0 == 0.0 == 0 + 0j`.

## 铺张浪费缝合怪

```python
nums = [0, 1, 2, 3, 4, 5, 6]
head, tail = nums[:3], nums[3:]
# head == [0, 1, 2] and tail == [3, 4, 5, 6]
it = iter(nums)
list(zip(it, head)) # [(0, 0), (1, 1), (2, 2)]
list(zip(it, tail)) # [(4, 3), (5, 4), (6, 5)]
```

请将迭代器视为一次性用品。`zip` 迭代器会在 `zip` 的任一迭代器参数停止时停止，但它无法预知哪一迭代器会率先停止。于是，它将轮流地检查 `it` 和 `head`, 而 `it` 作为第一个迭代器参数，将会率先消费掉一个元素 `3`, 然后 `zip` 迭代器才会检查到 `head` 已经停止（即调用 `next(head)` 抛出 `StopIteration`）。

## 夹带私货默认值

```python
def with_default(val=0, ref=[]):
    val += 1
    ref.append(0)
    print(f'{val=}, {ref=}')

with_default() # val=1, ref=[0]
with_default() # val=1, ref=[0, 0]
with_default() # val=1, ref=[0, 0, 0]
```

在为可变的引用类型的参数设置默认值时，应当配合使用 `None` 和 `if` 或进行拷贝，即

```python
def with_default(val=0, ref=None):
    if ref is None:
        ref = []
    val += 1
    ref.append(0)
    print(f'{val=}, {ref=}')

with_default() # val=1, ref=[0]
with_default() # val=1, ref=[0]
with_default() # val=1, ref=[0]
```
