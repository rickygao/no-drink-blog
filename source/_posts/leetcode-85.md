---
title: LeetCode 85. 最大矩形
date: 2020-12-26 12:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/maximal-rectangle/)

## 题目

给定一个仅包含 `'0'` 和 `'1'`, 大小为 `(rows, cols)` 的二维二进制矩阵，找出只包含 `'1'` 的最大矩形，并返回其面积。

### 示例

```raw
输入：matrix = [
    ['1', '0', '1', '0', '0'],
    ['1', '0', '1', '1', '1'],
    ['1', '1', '1', '1', '1'],
    ['1', '0', '0', '1', '0']
]
输出：6
```

```raw
输入：matrix = []
输出：0
```

```raw
输入：matrix = [['0']]
输出：0
```

```raw
输入：matrix = [['1']]
输出：1
```

```raw
输入：matrix = [['0', '0']]
输出：0
```

### 提示

- `rows == len(matrix)`;
- `cols == len(matrix[0])`;
- `0 <= row, cols <= 200`;
- `matrix[i][j]` 为 `'0'` 或 `'1'`.

<!-- more -->

## 题解

将矩阵逐行转化为柱状图，再利用{% post_link leetcode-84 %}的方法求解。

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        return maximal_rectangle(matrix)

from itertools import chain

def maximal_rectangle(matrix: List[List[str]]) -> int:
    if not matrix:
        return 0

    r, hs = 0, [0] * len(matrix[0])
    for row in matrix:
        hs = [h + 1 if int(e) else 0 for h, e in zip(hs, row)]
        r = max(r, maximal_rectangle_histogram(hs))
    return r

def maximal_rectangle_histogram(heights: List[int]) -> int:
    r, s = 0, [(-1, -1)]
    for ri, rh in enumerate(chain(heights, [0])):
        while s and rh <= (mh := s[-1][1]):
            s.pop()
            li = s[-1][0]
            r = max(r, (ri - li - 1) * mh)
        s.append((ri, rh))
    return r
```
