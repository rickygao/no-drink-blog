---
title: LeetCode 1641. 统计字典序元音字符串的数目
date: 2020-12-01 16:40:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/count-sorted-vowel-strings/)

## 题目

给你一个整数 `n`, 请返回长度为 `n`, 仅由元音 `('a', 'e', 'i', 'o', 'u')` 组成且按**字典序排列**的字符串数量。

字符串 `s` 按**字典序排列**需要满足：对于所有有效的 `i`, `s[i]` 在字母表中的位置总是与 `s[i+1]` 相同或在 `s[i+1]` 之前。

### 示例

```raw
输入：n = 1
输出：5
解释：仅由元音组成的 5 个字典序字符串为 ["a", "e", "i", "o", "u"].
```

```raw
输入：n = 2
输出：15
解释：仅由元音组成的 15 个字典序字符串为
["aa", "ae", "ai", "ao", "au", "ee", "ei", "eo", "eu", "ii", "io", "iu", "oo", "ou", "uu"].
注意，"ea" 不是符合题意的字符串，因为 'e' 在字母表中的位置比 'a' 靠后。
```

```raw
输入：n = 33
输出：66045
```

### 提示

`1 <= n <= 50`.

<!-- more -->

## 题解

### 分治

- 二分缩小问题；
- 记忆化。

```python
class Solution:
    def countVowelStrings(self, n: int) -> int:
        return count_vowel_strings(n, 5)

from functools import lru_cache

@lru_cache(maxsize=None)  # use `functools.cache` instead since Python 3.9
def count_vowel_strings(n: int, v: int) -> int:
    if n == 0:
        return 1
    if n == 1:
        return v
    l = n // 2
    r = n - l - 1
    return sum(
        count_vowel_strings(l, i) * count_vowel_strings(r, v - i + 1)
        for i in range(1, v + 1)
    )
```

### 组合计数

数学题。$\binom{n+4}{4}$, 即为 `n + 4` 个空位安置 `4` 个挡板，从而将 `n` 个空位划分为 `5` 组连续空位，分别装填 `(a, e, i, o, u)`.

```python
class Solution:
    def countVowelStrings(self, n: int) -> int:
        return count_vowel_strings(n, 5)

from math import comb

def count_vowel_strings(n: int, v: int) -> int:
    return comb(n + v - 1, v - 1)
```

### 查表

```python
TABLE = [
    1, 5, 15, 35, 70, 126, 210, 330, 495, 715, 1001,
    1365, 1820, 2380, 3060, 3876, 4845, 5985, 7315, 8855, 10626,
    12650, 14950, 17550, 20475, 23751, 27405, 31465, 35960, 40920, 46376,
    52360, 58905, 66045, 73815, 82251, 91390, 101270, 111930, 123410, 135751,
    148995, 163185, 178365, 194580, 211876, 230300, 249900, 270725, 292825, 316251
]

class Solution:
    def countVowelStrings(self, n: int) -> int:
        return TABLE[n]
```
