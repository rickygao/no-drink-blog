---
title: LeetCode 400. 第 N 个数字
date: 2020-12-06 22:00:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/nth-digit/)

## 题目

在无限的整数序列 $1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, \dots$ 中找到第 `n` 个数字。

### 注意

`n` 是正数且在 32 位整数范围内 ($0 < n < 2^{31}$).

### 示例

```raw
输入：3
输出：3
```

```raw
输入：11
输出：0
说明：第 11 个数字在序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 里是 0，它是 10 的一部分。
```

<!-- more -->

## 题解

数学题。所有 `k` 位数的数字总个数为 $9k\times{10}^{k-1}$, 用 `m` 代表 $9\times{10}^{k-1}$. 找到第 `n` 个数字位于整数 `t` 的自低位起的第 `i` 个十进制位（`i` 从 `0` 计起）。

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        return find_nth_digit(n)

def find_nth_digit(n: int) -> int:
    k, m = 1, 9
    while n > (b := k * m):
        n -= b
        k += 1
        m *= 10
    t = m // 9 + (n - 1) // k
    i = k - 1 - (n - 1) % k
    return t // (10 ** i) % 10
```
