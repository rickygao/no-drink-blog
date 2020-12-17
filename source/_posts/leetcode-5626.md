---
title: LeetCode 5626. 十、二进制数的最少数目
date: 2020-12-13 13:35:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/)

## 题目

如果一个十进制数字不含任何前导零，且每一位上的数字不是 `0` 就是 `1`, 那么该数字就是一个**十、二进制数**。例如，`101` 和 `1100` 都是**十、二进制数**，而 `112` 和 `3001` 不是。

给你一个表示十进制整数的字符串 `n`, 返回和为 `n` 的**十、二进制数**的最少数目。

### 示例

```raw
输入：n = "32"
输出：3
解释：10 + 11 + 11 = 32
```

```raw
输入：n = "82734"
输出：8
```

```raw
输入：n = "27346209830709182346"
输出：9
```

### 提示

- `1 <= len(n) <= 1e5`;
- `n` 仅由数字组成；
- `n` 不含任何前导零并总是表示正整数。

<!-- more -->

## 题解

只需找到最大的十进制数字。

```python
class Solution:
    def minPartitions(self, n: str) -> int:
        return min_partitions(n)

def min_partitions(n: str) -> int:
    return ord(max(n)) - ord('0')
```
