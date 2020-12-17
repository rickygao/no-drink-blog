---
title: LeetCode 1. 两数之和
date: 2020-11-25 10:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/two-sum/)

## 题目

给定一个整数数组 `nums` 和一个目标值 `target`, 请你在该数组中找出和为目标值的那**两个**整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

### 示例

```raw
给定 nums = [2, 7, 11, 15], target = 9,
因为 nums[0] + nums[1] = 2 + 7 = 9,
所以返回 [0, 1].
```

<!-- more -->

## 题解

索引反查字典。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        return two_sum(nums, target)

def two_sum(nums: List[int], target: int) -> List[int]:
    seen = dict()
    for i, m in enumerate(nums):
        n = target - m
        if n in seen:
            return [i, seen[n]]
        seen[m] = i
    raise ValueError
```
