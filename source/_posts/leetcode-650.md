---
title: LeetCode 650. 只有两个键的键盘
date: 2020-12-13 19:40:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/2-keys-keyboard/)

## 题目

最初在一个记事本上只有一个字符 `'A'`. 你每次可以对这个记事本进行两种操作：

1. `Copy All`（复制全部）: 你可以复制这个记事本中的所有字符（部分的复制是不允许的）；
2. `Paste`（粘贴）: 你可以粘贴你上一次复制的字符。

给定一个数字 `n`. 你需要使用最少的操作次数，在记事本中打印出恰好 `n` 个 `'A'`. 输出能够打印出 `n` 个 `'A'` 的最少操作次数。

### 示例

```raw
输入: 3
输出: 3
解释:
最初, 我们只有一个字符 'A'。
第 1 步, 我们使用 Copy All 操作。
第 2 步, 我们使用 Paste 操作来获得 'AA'.
第 3 步, 我们使用 Paste 操作来获得 'AAA'.
```

### 说明

- `1 <= n <= 1000`.

<!-- more -->

## 题解

贪心，质因数分解。对于 $n=\prod_i f_i$, 其中 $f_i$ 是质数，需要 $\sum_i f_i$ 次操作，即 $\prod_i (\mathrm{Copy\ All})(\mathrm{Paste})^{f_i-1}$.

```python
class Solution:
    def minSteps(self, n: int) -> int:
        return min_steps(n)

def min_steps(n: int) -> int:
    d, r = 2, 0
    while n > 1:
        while n % d == 0:
            r += d
            n //= d
        d += 1
    return r
```
