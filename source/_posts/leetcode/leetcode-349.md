---
title: LeetCode 349. 两个数组的交集
date: 2020-12-13 16:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

## 题目

给定两个数组，编写一个函数来计算它们的交集。

### 示例

```raw
输入：nums1 = [1, 2, 2, 1], nums2 = [2, 2]
输出：[2]
```

```raw
输入：nums1 = [4, 9, 5], nums2 = [9, 4, 9, 8, 4]
输出：[9, 4]
```

### 说明

- 输出结果中的每个元素一定是唯一的；
- 我们可以不考虑输出结果的顺序。

<!-- more -->

## 题解

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return intersection(nums1, nums2)

def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    return list(set(nums1) & set(nums2))
```
