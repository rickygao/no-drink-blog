---
title: Python 的陷阱 0x03
date: 2020-11-25 11:00:00
tags: [Python, Trap]
---

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
