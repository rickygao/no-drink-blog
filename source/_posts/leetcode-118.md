---
title: LeetCode 118. 杨辉三角
date: 2020-12-06 13:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/pascals-triangle/)

## 题目

给定一个非负整数 `numRows`, 生成杨辉三角的前 `numRows` 行。

{% asset_img pascals_triangle.gif 200 185 "'杨辉三角' '杨辉三角'" %}

在杨辉三角中，每个数是它左上方和右上方的数的和。

### 示例

```raw
输入：5
输出：[
    [1],
    [1, 1],
    [1, 2, 1],
    [1, 3, 3, 1],
    [1, 4, 6, 4, 1]
]
```

<!-- more -->

## 题解

不如来写一个生成器吧！

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        return generate(numRows)

from itertools import islice

def generate(num_rows: int) -> List[List[int]]:
    return list(islice(pascals_triangle(), num_rows))

def pascals_triangle() -> Iterator[List[int]]:
    row = [1]
    yield row
    while True:
        row = [1, *map(int.__add__, row, row[1:]), 1]
        yield row
```
