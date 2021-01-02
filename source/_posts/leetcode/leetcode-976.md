---
title: LeetCode 976. 三角形的最大周长
date: 2020-11-29 09:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/largest-perimeter-triangle/)

## 题目

给定由一些正数（代表长度）组成的数组 `A`, 返回由其中三个长度组成的、**面积不为零**的三角形的最大周长。

如果不能形成任何面积不为零的三角形，返回 `0`.

### 示例

```raw
输入：[2, 1, 2]
输出：5
```

```raw
输入：[1, 2, 1]
输出：0
```

```raw
输入：[3, 2, 3, 4]
输出：10
```

```raw
输入：[3, 6, 2, 3]
输出：8
```

### 提示

- `3 <= len(A) <= 10000`;
- `1 <= A[i] <= 1e6`.

<!-- more -->

## 题解

排序，贪心。对于每条边 `a`, 如果以该边为最长边，则最有可能形成合法三角形且周长最大的是 `b`, `c` 选取策略是一致的，即选取次长的两条边。

```python
class Solution:
    def largestPerimeter(self, A: List[int]) -> int:
        return largest_perimeter(A)

def largest_perimeter(sides: List[int]) -> int:
    sides.sort(reverse=True)
    return max((
        a + b + c
        for a, b, c in zip(sides, sides[1:], sides[2:])
        if b + c > a
    ), default=0)
```
