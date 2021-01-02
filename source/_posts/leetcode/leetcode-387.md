---
title: LeetCode 387. 字符串中的第一个唯一字符
date: 2020-12-23 13:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

## 题目

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 `-1`.

### 示例

```raw
输入：s = "leetcode"
输出：0
```

```raw
输入：s = "loveleetcode"
输出：2
``` 

### 提示

你可以假定该字符串只包含小写字母。

<!-- more -->

## 题解

计数，迭代。

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        return first_unique_character(s)

from collections import Counter

def first_unique_character(s: str) -> int:
    counter = Counter(s)
    return next((i for i, c in enumerate(s) if counter[c] == 1), -1)
```
