---
title: LeetCode 62. 不同路径
date: 2020-12-09 10:00:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/unique-paths/)

## 题目

一个机器人位于一个 $m\times n$ 网格的左上角 （起始点在下图中标记为 "Start"）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 "Finish"）。

问总共有多少条不同的路径？

{% asset_img robot_maze.png 200 92 "'机器人迷宫' '机器人迷宫'" %}

例如，上图是一个 $7\times3$ 的网格。有多少可能的路径？

### 示例

```raw
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
```

```raw
输入：m = 7, n = 3
输出：28
```

### 提示

- `1 <= m, n <= 100`;
- 题目数据保证答案小于等于 `2e9`.

<!-- more -->

## 题解

数学题，组合计数。共需行动 $m+n-2$ 步，其中 $m-1$ 步为向右移动，则有 $\binom{m+n-2}{m-1}$ 种选择。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return unique_paths(m, n)

from math import comb

def unique_paths(m: int, n: int) -> int:
    return comb(m + n - 2, m - 1)
```
