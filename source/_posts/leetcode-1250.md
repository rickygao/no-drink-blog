---
title: LeetCode 1250. 检查「好数组」
date: 2021-01-01 15:30:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/check-if-it-is-a-good-array/)

## 题目

给你一个正整数数组 `nums`, 你需要从中任选一些子集，然后将子集中每一个数乘以一个**任意整数**，并求出他们的和。

假如存在一种选取情况使该和结果为 `1`, 那么原数组就是一个「好数组」，则返回 `true`; 否则请返回 `false`.

### 示例

```raw
输入：nums = [12, 5, 7, 23]
输出：true
解释：挑选数字 5 和 7. 5 * 3 + 7 * (-2) = 1.
```

```raw
输入：nums = [29, 6, 10]
输出：true
解释：挑选数字 29, 6 和 10. 29 * 1 + 6 * (-3) + 10 * (-1) = 1.
```

```raw
输入：nums = [3, 6]
输出：false
```

### 提示

- `1 <= len(nums) <= 1e5`;
- `1 <= nums[i] <= 1e9`.

<!-- more -->

## 题解

数学题。[裴蜀定理](https://zh.wikipedia.org/wiki/貝祖等式)的推广形式：若 $a_i\in\mathbb{Z}$ 且 $d=\gcd(a_i)$, 则关于 $x_i$ 的方程 $\sum_ia_ix_i=m$ 有整数解当且仅当 $d\mid m$, 其中 $i=1,2,\dots,n$; 特别地，$\sum_ia_ix_i=1$ 有整数解当且仅当 $a_i$ 互质。

```python
class Solution:
    def isGoodArray(self, nums: List[int]) -> bool:
        return is_good_array(nums)

from math import gcd

def is_good_array(nums: List[int]) -> bool:
    return gcd(*nums) == 1
```
