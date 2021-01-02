---
title: LeetCode 137. 只出现一次的数字 II
date: 2020-12-17 13:20:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/single-number-ii/)

## 题目

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

### 说明

你的算法应该具有线性时间复杂度。你可以不使用额外空间来实现吗？

### 示例

```raw
输入：[2, 2, 3, 2]
输出：3
```

```raw
输入：[0, 1, 0, 1, 0, 1, 99]
输出：99
```

<!-- more -->

## 题解

### 计数

最通用。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

from collections import Counter

def single_number(nums: List[int]) -> int:
    return next(n for n, c in Counter(nums).items() if c == 1)
```

### 求和

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

def single_number(nums: List[int]) -> int:
    return (sum(set(nums)) * 3 - sum(nums)) // 2
```

### 位运算

`once`, `twice` 记录了出现次数为模三余一和二的二进制位。空间复杂度为常数级。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

def single_number(nums: List[int]) -> int:
    once = twice = 0
    for n in nums:
        once = ~twice & (once ^ n)
        twice = ~once & (twice ^ n)
    return once
```
