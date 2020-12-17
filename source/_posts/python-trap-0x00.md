---
title: Python 的陷阱 0x00
date: 2020-11-23 12:00:00
tags: [Python, Trap]
---

## 喋喋不休生成器

```python
l = [0]
l += l # ok, l == [0, 0]
l += (e for e in l) # infinite loop
```

一条蛇试图吞掉自己的尾巴。
