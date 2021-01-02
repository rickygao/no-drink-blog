---
title: LeetCode 1663. 具有给定数值的最小字符串
date: 2020-12-16 20:40:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/smallest-string-with-a-given-numeric-value/)

## 题目

**小写字符**的**数值**是它在字母表中的位置（从 `1` 开始），因此 `'a'` 的数值为 `1`, `'b'` 的数值为 `2`, `'c'` 的数值为 `3`, 以此类推。

字符串由若干小写字符组成，**字符串的数值**为各字符的数值之和。例如，字符串 `"abe"` 的数值等于 `1 + 2 + 5 = 8`.

给你两个整数 `n` 和 `k`. 返回**长度**等于 `n` 且**数值**等于 `k` 的**字典序最小**的字符串。

注意，如果字符串 `x` 在字典排序中位于 `y` 之前，就认为 `x` 字典序比 `y` 小，有以下两种情况：

- `x` 是 `y` 的一个前缀；
- 如果 `i` 是 `x[i] != y[i]` 的第一个位置，且 `x[i]` 在字母表中的位置比 `y[i]` 靠前。

### 示例

```raw
输入：n = 3, k = 27
输出："aay"
解释：字符串的数值为 1 + 1 + 25 = 27, 它是数值满足要求且长度等于 3 字典序最小的字符串。
```

```raw
输入：n = 5, k = 73
输出："aaszz"
```

### 提示

- `1 <= n <= 1e5`;
- `n <= k <= 26 * n`.

<!-- more -->

## 题解

贪心。构造形如 `AmZ` 的串，`A` 由 `'a'` 组成，`Z` 由 `'z'` 组成，`m` 是某个小写字母。

```python
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        return get_smallest_string(n, k)

def get_smallest_string(n: int, k: int) -> str:
    k -= n
    z, m = divmod(k, 25)
    if n <= z:
        return 'z' * n

    a = n - z - 1
    return 'a' * a + chr(m + ord('a')) + 'z' * z
```
