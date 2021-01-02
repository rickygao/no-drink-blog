---
title: LeetCode 861. 翻转矩阵后的得分
date: 2020-12-07 13:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/score-after-flipping-matrix/)

## 题目

有一个二维矩阵 `A` 其中每个元素的值为 `0` 或 `1`.

移动是指选择任一行或列，并转换该行或列中的每一个值：将所有 `0` 都更改为 `1`, 将所有 `1` 都更改为 `0`.

在做出任意次数的移动后，将该矩阵的每一行都按照二进制数来解释，矩阵的得分就是这些数字的总和。

返回尽可能高的分数。

### 示例

```raw
输入：[[0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0]]
输出：39
解释：转换为 [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], 0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39.
```

### 提示

- `1 <= len(A), len(A[i]) <= 20`;
- `A[i][j] in (0, 1)`.

<!-- more -->

## 题解

### 贪心

```python
class Solution:
    def matrixScore(self, A: List[List[int]]) -> int:
        return matrix_score(A)

def matrix_score(matrix: List[List[int]]) -> int:
    nr, nc = len(matrix), len(matrix[0])

    # hflip the rows w/ a leading zero
    for i in range(nr):
        if matrix[i][0] == 0:
            for j in range(nc):
                matrix[i][j] = 1 - matrix[i][j]

    # vflip the columns w/ zeros more than ones
    for j in range(nc):
        if sum(matrix[i][j] for i in range(nr)) < nr / 2:
            for i in range(nr):
                matrix[i][j] = 1 - matrix[i][j]

    return sum(sum(matrix[i][j] for i in range(nr)) * 2 ** (nc - 1 - j) for j in range(nc))
```

### 优化

```python
class Solution:
    def matrixScore(self, A: List[List[int]]) -> int:
        return matrix_score(A)

from functools import reduce

def matrix_score(matrix: List[List[int]]) -> int:
    nr, nc = len(matrix), len(matrix[0])
    return reduce(
        lambda r, j: r * 2 + max(
            s := sum(matrix[i][j] ^ matrix[i][0] for i in range(nr)),
            nr - s
        ), range(1, nc), nr
    )
```
