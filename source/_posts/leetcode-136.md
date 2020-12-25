---
title: LeetCode 136. 只出现一次的数字
date: 2020-12-17 15:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/single-number/)

## 题目

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

### 说明

你的算法应该具有线性时间复杂度。你可以不使用额外空间来实现吗？

### 示例

```raw
输入：[2, 2, 1]
输出：1
```

```raw
输入：[4, 1, 2, 1, 2]
输出：4
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
    return sum(set(nums)) * 2 - sum(nums)
```

### 位运算

利用按位异或的对合性。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

def single_number(nums: List[int]) -> int:
    return accumulate(int.__xor__, nums)
```
