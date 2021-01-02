---
title: LeetCode 204. 计数质数
date: 2020-12-03 19:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/count-primes/)

## 题目

统计所有小于非负整数 `n` 的质数的数量。

### 示例

```raw
输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个，它们是 2, 3, 5, 7.
```

```raw
输入：n = 0
输出：0
```

```raw
输入：n = 1
输出：0
```

### 提示

- `0 <= n <= 5e6`

<!-- more -->

## 题解

筛法。

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        return count_primes(n)

def count_primes(n: int) -> int:
    if n < 2:
        return 0
    is_primes = [True] * n
    is_primes[0] = is_primes[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_primes[i]:
            k = ceil((n - (b := i * 2)) / i)
            is_primes[b:n:i] = [False] * k
    return sum(is_primes)
```
