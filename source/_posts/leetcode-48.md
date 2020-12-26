---
title: LeetCode 48. 旋转图像
date: 2020-12-19 10:50:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/rotate-image/)

## 题目

给定一个 $n\times n$ 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

### 说明

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

### 示例

```raw
输入：matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

输出：matrix = [
    [7, 4, 1],
    [8, 5, 2],
    [9, 6, 3],
]
```

```raw
输入：matrix = [
    [ 5,  1,  9, 11],
    [ 2,  4,  8, 10],
    [13,  3,  6,  7],
    [15, 14, 12, 16],
]

输出：matrix = [
    [15, 13,  2,  5],
    [14,  3,  4,  1],
    [12,  6,  8,  9],
    [16,  7, 10, 11],
]
```

<!-- more -->

## 题解

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        return rotate(matrix)

def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    for i in range(n // 2):
        for j in range((n + 1) // 2):
            matrix[        i][        j], matrix[        j][n - i - 1],  \
            matrix[n - i - 1][n - j - 1], matrix[n - j - 1][        i] = \
            matrix[n - j - 1][        i], matrix[        i][        j],  \
            matrix[        j][n - i - 1], matrix[n - i - 1][n - j - 1]
```
