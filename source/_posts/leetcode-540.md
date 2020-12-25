---
title: LeetCode 540. 有序数组中的单一元素
date: 2020-12-18 18:15:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

## 题目

给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

### 示例

```raw
输入：[1, 1, 2, 3, 3, 4, 4, 8, 8]
输出：2
```

```raw
输入：[3, 3, 7, 7, 10, 11, 11]
输出：10
```

### 注意

您的方案应该在 $\mathrm{O}(\log n)$ 时间复杂度和 $\mathrm{O}(1)$ 空间复杂度中运行。

<!-- more -->

## 题解

### 二分查找

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        return single_non_duplicate(nums)

def single_non_duplicate(nums: List[int]) -> int:
    low, high = 0, len(nums)
    while low + 1 < high:
        mid = (low + high) // 2
        mid -= mid % 2
        if nums[mid] == nums[mid + 1]:
            low = mid + 2
        else:
            high = mid + 1
    return nums[low]
```
