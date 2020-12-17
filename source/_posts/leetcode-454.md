---
title: LeetCode 454. 四数相加 II
date: 2020-11-27 12:30:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/4sum-ii/)

## 题目

给定四个包含整数的数组列表 `A`, `B`, `C`, `D`, 计算有多少个元组 `(i, j, k, l)`, 使得 `A[i] + B[j] + C[k] + D[l] = 0`.

为了使问题简单化，所有的 `A`, `B`, `C`, `D` 具有相同的长度 $N$，且 $0 \le N \le 500$. 所有整数的范围在 $-2^{28}$ 到 $2^{28}-1$ 之间，最终结果不会超过 $2^{31}-1$.

### 示例

```raw
输入:
A = [ 1,  2]
B = [-2, -1]
C = [-1,  2]
D = [ 0,  2]

输出:
2

解释:
两个元组如下:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
```

<!-- more -->

## 题解

二分计数查找。

```python
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        return sum_count(A, B, C, D)

from itertools import product
from collections import Counter

def sum_count(A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
    AB = Counter(a + b for a, b in product(A, B))
    return sum(AB[-(c + d)] for c, d in product(C, D))
```
