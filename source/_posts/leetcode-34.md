---
title: LeetCode 34. 在排序数组中查找元素的第一个和最后一个位置
date: 2020-12-01 10:30:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

## 题目

给定一个按照升序排列的整数数组 `nums`, 和一个目标值 `target`. 找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`, 返回 `[-1, -1]`.

### 进阶

你可以设计并实现时间复杂度为 $\mathrm{O}(\log n)$ 的算法解决此问题吗？

### 示例

```raw
输入：nums = [5, 7, 7, 8, 8, 10], target = 8
输出：[3, 4]
```

```raw
输入：nums = [5, 7, 7, 8, 8, 10], target = 6
输出：[-1, -1]
```

```raw
输入：nums = [], target = 0
输出：[-1, -1]
```

### 提示

- `0 <= len(nums) <= 1e5`;
- `-1e9 <= nums[i], target <= 1e9`;
- `nums` 是一个非递减数组。

<!-- more -->

## 题解

二分查找。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        return search_range(nums, target)

from bisect import bisect_left, bisect_right

def search_range(nums: List[int], target: int) -> List[int]:
    l = bisect_left(nums, target)
    if l >= len(nums) or nums[l] != target:
        return [-1, -1]
    r = bisect_right(nums, target)
    return [l, r - 1]
```
