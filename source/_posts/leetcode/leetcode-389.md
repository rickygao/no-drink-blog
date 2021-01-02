---
title: LeetCode 389. 找不同
date: 2020-12-18 12:20:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/find-the-difference/)

## 题目

给定两个字符串 `s` 和 `t`, 它们只包含小写字母。

字符串 `t` 由字符串 `s` 随机重排，然后在随机位置添加一个字母。

请找出在 `t` 中被添加的字母。

### 示例

```raw
输入：s = "abcd", t = "abcde"
输出："e"
解释：'e' 是那个被添加的字母。
```

```raw
输入：s = "", t = "y"
输出："y"
```

```raw
输入：s = "a", t = "aa"
输出："a"
```

```raw
输入：s = "ae", t = "aea"
输出："a"
```

### 提示

- `0 <= len(s) <= 1000`;
- `len(t) == len(s) + 1`;
- `s` 和 `t` 只包含小写字母。

<!-- more -->

## 题解

### 计数

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return find_the_difference(s, t)

from collections import Counter

def find_the_difference(s: str, t: str) -> str:
    for k in Counter(t) - Counter(s):
        return k
```

### 求和

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return find_the_difference(s, t)

def find_the_difference(s: str, t: str) -> str:
    return chr(sum(map(ord, t)) - sum(map(ord, s)))
```
