---
title: LeetCode 229. 求众数 II
date: 2020-12-13 10:50:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/majority-element-ii/)

## 题目

给定一个大小为 `n` 的整数数组，找出其中所有出现超过 $\lfloor\frac{n}{3}\rfloor$ 次的元素。

### 进阶

尝试设计时间复杂度为 $\mathrm{O}(n)$, 空间复杂度为 $\mathrm{O}(1)$ 的算法解决此问题。

### 示例

```raw
输入：[3, 2, 3]
输出：[3]
```

```raw
输入：nums = [1]
输出：[1]
```

```raw
输入：[1, 1, 1, 3, 3, 2, 2, 2]
输出：[1, 2]
```

### 提示

- `1 <= len(nums) <= 5e4`;
- `-1e9 <= nums[i] <= 1e9`.

<!-- more -->

## 题解

### 通用计数

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        return majority_element(nums, 3)

from collections import Counter

def majority_element(nums: List[int], f: int) -> List[int]:
    return [
        n for n, c in Counter(nums).most_common(f - 1)
        if c > len(nums) // f
    ]
```

### 通用摩尔投票

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        return majority_element(nums, 3)

def majority_element(nums: List[int], f: int) -> List[int]:
    m = f - 1
    r, c = [None] * m, [0] * m
    for n in nums:
        for i in range(m):
            if r[i] == n:
                c[i] += 1
                break
        else:
            for i in range(m):
                if c[i] == 0:
                    r[i], c[i] = n, 1
                    break
            else:
                for i in range(m):
                    c[i] -= 1

    c = [0] * m
    for n in nums:
        for i in range(m):
            if r[i] == n:
                c[i] += 1
                break
    
    l = len(nums) // f
    return [ri for ri, ci in zip(r, c) if ci > l]
```

### 特化摩尔投票

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        return majority_element(nums)

def majority_element(nums: List[int]) -> List[int]:
    r1 = r2 = None
    c1 = c2 = 0
    for n in nums:
        if r1 == n:
            c1 += 1
        elif r2 == n:
            c2 += 1
        elif c1 == 0:
            r1, c1 = n, 1
        elif c2 == 0:
            r2, c2 = n, 1
        else:
            c1 -= 1
            c2 -= 1

    c1 = c2 = 0
    for n in nums:
        if r1 == n:
            c1 += 1
        elif r2 == n:
            c2 += 1

    r, l = [], len(nums) // 3
    if c1 > l:
        r.append(r1)
    if c2 > l:
        r.append(r2)
    return r
```
