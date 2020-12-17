---
title: LeetCode 321. 拼接最大数
date: 2020-12-02 21:20:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/create-maximum-number/)

## 题目

给定长度分别为 `m` 和 `n` 的两个数组，其元素由 `0` 到 `9` 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 `k` (`k <= m + n`) 个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。

求满足该条件的最大数。结果返回一个表示该最大数的长度为 `k` 的数组。

### 说明

请尽可能地优化你算法的时间和空间复杂度。

### 示例

```raw
输入：nums1 = [3, 4, 6, 5], nums2 = [9, 1, 2, 5, 8, 3], k = 5
输出：[9, 8, 6, 5, 3]
```

```raw
输入：nums1 = [6, 7], nums2 = [6, 0, 4], k = 5
输出：[6, 7, 6, 0, 4]
```

```raw
输入：nums1 = [3, 9], nums2 = [8, 9], k = 3
输出：[9, 8, 9]
```

<!-- more -->

## 题解

- 分治；
- 单调栈。

```python
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        return max_number(nums1, nums2, k)

def max_number(nums1: List[int], nums2: List[int], k: int) -> List[int]:
    x_min, x_max = max(0, k - len(nums2)), min(k, len(nums1))
    return max((
        _merge(_max_number(nums1, x), _max_number(nums2, k - x))
        for x in range(x_min, x_max + 1)
    ), default=[])

def _max_number(nums: List[int], k: int) -> List[int]:
    s, to_drop = [], len(nums) - k
    for n in nums:
        while to_drop > 0 and s and s[-1] < n:
            to_drop -= 1
            s.pop()
        s.append(n)
    return s[:k]

def _merge(seq1: List[int], seq2: List[int]) -> List[int]:
    r = []
    while seq1 or seq2:
        s = max(seq1, seq2)
        r.append(s.pop(0))
    return r
```
