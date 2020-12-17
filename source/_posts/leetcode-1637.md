---
title: LeetCode 1637. 两点之间不包含任何点的最宽垂直面积
date: 2020-12-12 22:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/widest-vertical-area-between-two-points-containing-no-points/)

## 题目

给你 `n` 个二维平面上的点 `points`, 其中 `points[i] = [xi, yi]`, 请你返回两点之间内部不包含任何点的**最宽垂直面积**的宽度。

**垂直面积**的定义是固定宽度，而 `y` 轴上无限延伸的一块区域（也就是高度为无穷大）。**最宽垂直面积**为宽度最大的一个垂直面积。

请注意，垂直区域**边上**的点**不在**区域内。

### 示例

{% asset_img points.png 200 269 "'示例' '示例'" %}

```raw
输入：points = [[8, 7], [9, 9], [7, 4], [9, 7]]
输出：1
解释：红色区域和蓝色区域都是最优区域。
```

```raw
输入：points = [[3, 1], [9, 0], [1, 0], [1, 4], [5, 3], [8, 8]]
输出：3
```

### 提示

- `n == len(points)`;
- `2 <= n <= 1e5`;
- `len(points[i]) == 2`;
- `0 <= xi, yi <= 1e9`.

<!-- more -->

## 题解

排序，最大差值。

```python
class Solution:
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        return max_width_of_vertical_area(points)

def max_width_of_vertical_area(points: List[List[int]]) -> int:
    xs = sorted({p[0] for p in points})
    return max(map(int.__sub__, xs[1:], xs), default=0)
```
