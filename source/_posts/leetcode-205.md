---
title: LeetCode 205. 同构字符串
date: 2020-12-27 12:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/isomorphic-strings/)

## 题目

给定两个字符串 `s` 和 `t`, 判断它们是否是同构的。

如果 `s` 中的字符可以被替换得到 `t`, 那么这两个字符串是同构的。

所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

### 示例

```raw
输入：s = "egg", t = "add"
输出：true
```

```raw
输入：s = "foo", t = "bar"
输出：false
```

```raw
输入：s = "paper", t = "title"
输出：true
```

### 说明

你可以假设 `s` 和 `t` 具有相同的长度。

<!-- more -->

## 题解

### 高效

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        return is_isomorphix(s, t)

def is_isomorphix(s: str, t: str) -> bool:
    s2t, tcs = {}, set()
    for sc, tc in zip(s, t):
        if sc in s2t:
            if s2t[sc] != tc:
                return False
        else:
            if tc in tcs:
                return False
            s2t[sc] = tc
            tcs.add(tc)
    return True
```

### 简洁

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        return is_isomorphix(s, t)

def is_isomorphix(s: str, t: str) -> bool:
    return len(set(s)) == len(set(t)) == len(set(zip(s, t)))
```