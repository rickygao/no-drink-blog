---
title: LeetCode 164. 最大间距
date: 2020-11-26 12:00:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/maximum-gap/)

## 题目

给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。

如果数组元素个数小于 `2`, 则返回 `0`.

### 示例

```raw
输入: [3, 6, 9, 1]
输出: 3
解释: 排序后的数组是 [1, 3, 6, 9], 其中相邻元素 (3, 6) 和 (6, 9) 之间都存在最大差值 3.
```

```raw
输入: [10]
输出: 0
解释: 数组元素个数小于 2, 因此返回 0.
```

### 说明

- 你可以假设数组中所有元素都是非负整数，且数值在 32 位有符号整数范围内；
- 请尝试在线性时间复杂度和空间复杂度的条件下解决此问题。

<!-- more -->

## 题解

### 排序

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        return maximum_gap(nums)

def maximum_gap(nums: List[int]) -> int:
    nums = sorted(nums)
    return max(map(int.__sub__, nums[1:], nums[:-1]), default=0)
```

### 桶

记整数列表的长度为 $l$, 最小值和最大值分别为 $m$, $n$. 则桶的大小为 $s=\frac{n-m}{l-1}$, 第 $i$ 个桶容纳 $[m+is,m+(i+1)s)$, 可保证最大间距不会出现在桶内。

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        return maximum_gap(nums)

def maximum_gap(nums: List[int]) -> int:
    if len(nums) < 2:
        return 0

    mi, ma = min(nums), max(nums)
    size_bucket = max((ma - mi) // (len(nums) - 1), 1)
    num_buckets = (ma - mi) // size_bucket + 1
    min_buckets = [...] * num_buckets
    max_buckets = [...] * num_buckets

    for n in nums:
        i_bucket = (n - mi) // size_bucket
        if min_buckets[i_bucket] is ... or n < min_buckets[i_bucket]:
            min_buckets[i_bucket] = n
        if max_buckets[i_bucket] is ... or n > max_buckets[i_bucket]:
            max_buckets[i_bucket] = n

    r, prev = 0, ...
    for i in range(num_buckets):
        # check if an empty bucket
        if min_buckets[i] is ...:
            continue
        if prev is ...:
            r = max(r, min_buckets[i] - max_buckets[prev])
        prev = i
    return r
```
