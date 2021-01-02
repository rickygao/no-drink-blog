---
title: LeetCode 135. 分发糖果
date: 2020-12-24 4:15:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/candy/)

## 题目

老师想给孩子们分发糖果，有 `N` 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

- 每个孩子至少分配到 `1` 个糖果；
- 相邻的孩子中，评分高的孩子必须获得更多的糖果。

那么这样下来，老师至少需要准备多少颗糖果呢？

### 示例

```raw
输入：[1, 0, 2]
输出：5
解释：你可以分别给这三个孩子分发 2、1、2 颗糖果。
```

```raw
输入：[1, 2, 2]
输出：4
解释：你可以分别给这三个孩子分发 1、2、1 颗糖果。第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
```

<!-- more -->

## 题解

### 朴素贪心

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        return candy(ratings)

def candy(ratings: List[int]) -> int:
    n = len(ratings)

    l = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            l[i] = l[i - 1] + 1

    r = [1] * n
    for i in range(n - 1, 0, -1):
        if ratings[i - 1] > ratings[i]:
            r[i - 1] = r[i] + 1

    return sum(map(max, l, r))
```

### 优化贪心

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        return candy(ratings)

from operator import sub

def candy(ratings: List[int]) -> int:
    r, inc, dec, pre = 1, 1, 0, 1
    for d in map(sub, ratings[1:], ratings):
        if d >= 0:
            dec = 0
            pre = 1 if d == 0 else pre + 1
            r += pre
            inc = pre
        else:
            dec += 1
            if dec == inc:
                dec += 1
            r += dec
            pre = 1
    return r
```
