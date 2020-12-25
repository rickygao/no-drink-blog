---
title: LeetCode 5625. 比赛中的配对次数
date: 2020-12-13 13:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/count-of-matches-in-tournament/)

## 题目

给你一个整数 `n`, 表示比赛中的队伍数。比赛遵循一种独特的赛制：

- 如果当前队伍数是**偶数**，那么每支队伍都会与另一支队伍配对。总共进行 `n / 2` 场比赛，且产生 `n / 2` 支队伍进入下一轮。
- 如果当前队伍数为**奇数**，那么将会随机轮空并晋级一支队伍，其余的队伍配对。总共进行 `(n - 1) / 2` 场比赛，且产生 `(n - 1) / 2 + 1` 支队伍进入下一轮。

返回在比赛中进行的配对次数，直到决出获胜队伍为止。

### 示例

```raw
输入：n = 7
输出：6
解释：比赛详情：
- 第 1 轮：队伍数 = 7, 配对次数 = 3, 4 支队伍晋级。
- 第 2 轮：队伍数 = 4, 配对次数 = 2, 2 支队伍晋级。
- 第 3 轮：队伍数 = 2, 配对次数 = 1, 决出 1 支获胜队伍。
总配对次数 = 3 + 2 + 1 = 6
```

```raw
输入：n = 14
输出：13
解释：比赛详情：
- 第 1 轮：队伍数 = 14, 配对次数 = 7, 7 支队伍晋级。
- 第 2 轮：队伍数 = 7, 配对次数 = 3, 4 支队伍晋级。 
- 第 3 轮：队伍数 = 4, 配对次数 = 2, 2 支队伍晋级。
- 第 4 轮：队伍数 = 2, 配对次数 = 1, 决出 1 支获胜队伍。
总配对次数 = 7 + 3 + 2 + 1 = 13
```

### 提示

- `1 <= n <= 200`.

<!-- more -->

## 题解

### 模拟

```python
class Solution:
    def numberOfMatches(self, n: int) -> int:
        return number_of_matches(n)

def number_of_matches(n: int) -> int:
    r = 0
    while n > 1:
        r += n // 2
        n = (n + 1) // 2
    return r
```

### 计算

每场比赛淘汰一支队伍，共淘汰 `n - 1` 支队伍，故需进行 `n - 1` 场比赛。

```python
class Solution:
    def numberOfMatches(self, n: int) -> int:
        return number_of_matches(n)

def number_of_matches(n: int) -> int:
    return n - 1
```
