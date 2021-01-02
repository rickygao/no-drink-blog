---
title: LeetCode 84. 柱状图中最大的矩形
date: 2020-12-26 11:20:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

## 题目

给定 `n` 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 `1`.

求在该柱状图中，能够勾勒出来的矩形的最大面积。

### 示例

```raw
输入：[2, 1, 5, 6, 2, 3]
输出：10
```

<!-- more -->

## 题解

单调栈。每次出栈都对应找到了一组 `(li, mh, ri)`, 代表了一个高度为 `mh` 的柱，左右最近矮于它的柱的索引分别是 `li` 和 `ri`. 从而确定了一个局部最大矩形，高度为 `mh`, 宽度为左右矮柱所夹部分 `ri - li - 1`.

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        return largest_rectangle_area(heights)

from itertools import chain

def largest_rectangle_area(heights: List[int]) -> int:
    r, s = 0, [(-1, -1)]
    for ri, rh in enumerate(chain(heights, [0])):
        while s and rh <= (mh := s[-1][1]):
            s.pop()
            li = s[-1][0]
            r = max(r, (ri - li - 1) * mh)
        s.append((ri, rh))
    return r
```
