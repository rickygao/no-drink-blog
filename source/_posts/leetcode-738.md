---
title: LeetCode 738. 单调递增的数字
date: 2020-12-15 12:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/monotone-increasing-digits/)

## 题目

给定一个非负整数 `N`, 找出小于或等于 `N` 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。

（当且仅当每个相邻位数上的数字 `x` 和 `y` 满足 `x <= y` 时，我们称这个整数是单调递增的。）

### 示例

```raw
输入：N = 10
输出：9
```

```raw
输入：N = 1234
输出：1234
```

```raw
输入：N = 332
输出：299
```

### 说明

- `0 <= N <= 1e9`.

<!-- more -->

## 题解

如果 `N` 不满足要求，则结果应为以若干 `9` 结尾的整数，位于破坏单调性的首个位置之前的不会破坏单调性的首个可退位之后；可退位进行退位；之前的位保留。

```python
class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        return monotone_increasing_digits(N)

from itertools import accumulate

def monotone_increasing_digits(n: int) -> int:
    s = str(n)

    for i in range(1, len(s)):
        if s[i - 1] > s[i]:
            break
    else:
        return n

    for j in range(i - 1, 0, -1):
        if s[j - 1] < s[j] and s[j] > '0':
            break
    else:
        j = 0

    return int(s[:j] + chr(ord(s[j]) - 1) + '9' * (len(s) - j - 1))
```
