---
title: LeetCode 493. 翻转对
date: 2020-11-28 00:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/reverse-pairs/)

## 题目

给定一个数组 `nums`, 如果 `i < j` 且 `nums[i] > 2 * nums[j]` 我们就将 `(i, j)` 称作一个重要翻转对。

你需要返回给定数组中的重要翻转对的数量。

### 示例

```raw
输入：[1, 3, 2, 3, 1]
输出：2
```

```raw
输入：[2, 4, 3, 5, 1]
输出：3
```

<!-- more -->

## 题解

### 二分查找

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        return reverse_pairs(nums)

import bisect

def reverse_pairs(nums: List[int]) -> int:
    r, nums2 = 0, []
    for n in reversed(nums):
        r += bisect.bisect_left(nums2, n)
        bisect.insort_left(nums2, n * 2)
    return r
```

### 树状数组

- 离散化；
- 树状数组。

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        return reverse_pairs(nums)

from bisect import bisect_left

def reverse_pairs(nums: List[int]) -> int:
    # discretize
    nums12 = nums + [n * 2 for n in nums]
    nums12.sort()
    mapping = [(
        bisect_left(sorted_nums, n),
        bisect_left(sorted_nums, n * 2)
    ) for n in nums]

    r, bit= 0, [0] * (len(nums) * 2)
    for n1, n2 in reversed(mapping):
        # query
        while n1 > 0:
            r += bit[n1 - 1]
            n1 -= n1 & (-n1)

        # update
        n2 += 1
        while 0 < n2 <= len(bit):
            bit[n2 - 1] += 1
            n2 += n2 & (-n2)

    return r
```
