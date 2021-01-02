---
title: LeetCode 217. 存在重复元素
date: 2020-12-13 00:50:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/contains-duplicate/)

## 题目

给定一个整数数组，判断是否存在重复元素。

如果任意一值在数组中出现至少两次，函数返回 `true`. 如果数组中每个元素都不相同，则返回 `false`.

### 示例

```raw
输入：[1, 2, 3, 1]
输出：true
```

```raw
输入：[1, 2, 3, 4]
输出：false
```

```raw
输入：[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
输出：true
```

<!-- more -->

## 题解

### 高效

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return contains_duplicate(nums)

def contains_duplicate(nums: List[int]) -> bool:
    seen = set()
    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False
```

### 简洁

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return contains_duplicate(nums)

def contains_duplicate(nums: List[int]) -> bool:
    return len(set(nums)) < len(nums)
```
