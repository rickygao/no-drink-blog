---
title: LeetCode 49. 字母异位词分组
date: 2020-12-14 21:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/group-anagrams/)

## 题目

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

### 示例

```raw
输入：["eat", "tea", "tan", "ate", "nat", "bat"]
输出：[["ate", "eat", "tea"], ["nat", "tan"], ["bat"]]
```

### 说明

- 所有输入均为小写字母；
- 不考虑答案输出的顺序。

<!-- more -->

## 题解

### 排序

有序字符串作为键。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        return group_anagrams(strs)

from collections import defaultdict

def group_anagrams(strs: List[str]) -> List[List[str]]:
    r = defaultdict(list)
    for s in strs:
        k = ''.join(sorted(s))
        r[k].append(s)
    return list(r.values())
```

### 计数

计数元组作为键。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        return group_anagrams(strs)

from collections import defaultdict

def group_anagrams(strs: List[str]) -> List[List[str]]:
    r = defaultdict(list)
    for s in strs:
        c = [0] * 26
        for ch in s:
            c[ord(ch) - ord('a')] += 1
        r[tuple(c)].append(s)
    return list(r.values())
```

### 质数

利用质数因数分解设计次序不敏感的键。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        return group_anagrams(strs)

from collections import defaultdict
from math import prod

_b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]

def group_anagrams(strs: List[str]) -> List[List[str]]:
    r = defaultdict(list)
    for s in strs:
        k = prod(_b[ord(ch) - ord('a')] for ch in s)
        r[k].append(s)
    return list(r.values())
```
