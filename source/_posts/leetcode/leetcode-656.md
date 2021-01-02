---
title: LeetCode 605. 种花问题
date: 2021-01-01 01:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/can-place-flowers/)

## 题目

假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。

给定一个花坛（表示为一个数组包含 `0` 和 `1`, 其中 `0` 表示没种植花，`1` 表示种植了花），和一个数 `n`. 能否在不打破种植规则的情况下种入 `n` 朵花？能则返回 `true`, 不能则返回 `false`.

### 示例

```raw
输入：flowerbed = [1, 0, 0, 0, 1], n = 1
输出：true
```

```raw
输入：flowerbed = [1, 0, 0, 0, 1], n = 2
输出：false
```

### 注意

- 数组内已种好的花不会违反种植规则；
- 输入的数组长度范围为 `[1, 20000]`;
- `n` 是非负整数，且不会超过输入数组的大小。

<!-- more -->

## 题解

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        return can_place_flowers(flowerbed, n)

from itertools import chain

def can_place_flowers(flowerbed: List[int], n: int) -> bool:
    r = c = 0
    for b in chain([0], flowerbed, [0, 1]):
        if b:
            r += (c - 1) // 2
            c = 0
        else:
            c += 1
    return r >= n
```
