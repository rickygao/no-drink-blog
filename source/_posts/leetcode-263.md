---
title: LeetCode 263. 丑数
date: 2020-12-06 20:30:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/ugly-number/)

## 题目

编写一个程序判断给定的数是否为丑数。

丑数就是只包含质因数 `2`, `3`, `5` 的正整数。

### 示例

```raw
输入：6
输出：true
解释：6 = 2 * 3
```

```raw
输入：8
输出：true
解释：8 = 2 * 2 * 2
```

```raw
输入：14
输出：false
解释：14 不是丑数，因为它包含了另外一个质因数 7.
```

### 说明

- `1` 是丑数；
- 输入不会超过 32 位有符号整数的范围: $[−2^{31}, 2^{31}−1]$.

<!-- more -->

## 题解

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        return is_ugly(num)

def is_ugly(num: int) -> bool:
    if num == 0:
        return False
    for f in [2, 3, 5]:
        while num % f == 0:
            num //= f
    return num == 1
```
