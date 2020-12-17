---
title: Python 的陷阱 0x02
date: 2020-11-23 18:00:00
tags: [Python, Trap]
---

## 铺张浪费缝合怪

```python
nums = [0, 1, 2, 3, 4, 5, 6]
head, tail = nums[:3], nums[3:]
# head == [0, 1, 2] and tail == [3, 4, 5, 6]
it = iter(nums)
list(zip(it, head)) # [(0, 0), (1, 1), (2, 2)]
list(zip(it, tail)) # [(4, 3), (5, 4), (6, 5)]
```

请将迭代器视为一次性用品。

`zip` 迭代器会在 `zip` 的任一迭代器参数停止时停止，但它无法预知哪一迭代器会率先停止。于是，它将轮流地检查 `it` 和 `head`，而 `it` 作为第一个迭代器参数，将会率先消费掉一个元素 `3`，然后 `zip` 迭代器才会检查到 `head` 已经停止（即调用 `next(head)` 抛出 `StopIteration`）。
