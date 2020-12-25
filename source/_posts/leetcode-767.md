---
title: LeetCode 767. 重构字符串
date: 2020-11-30 14:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/reorganize-string/)

## 题目

给定一个字符串 `S`, 检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

### 示例

```raw
输入：S = "aab"
输出："aba"
```

```raw
输入：S = "aaab"
输出：""
```

### 注意

`S` 只包含小写字母并且长度在 `[1, 500]` 区间内。

<!-- more -->

## 题解

- 不可行的充要条件是最大出现次数大于长度的一半向上取整。
- 贪心。先填充奇数位置，再填充偶数位置。

```python
class Solution:
    def reorganizeString(self, S: str) -> str:
        return reorganize_string(S)

from collections import Counter

def reorganize_string(s: str) -> str:
    counter = Counter(s)
    if max(counter.values(), default=0) > (len(s) + 1) // 2:
        return ''
    result = [None] * len(s)
    even, odd, half = 0, 1, len(s) // 2
    for element, count in counter.items():
        if count <= half:
            while count > 0 and odd < len(result):
                result[odd] = element
                count -= 1
                odd += 2
        while count > 0:
            result[even] = element
            count -= 1
            even += 2
    return ''.join(result)
```
