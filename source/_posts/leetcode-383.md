---
title: LeetCode 383. 赎金信
date: 2020-12-18 13:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/ransom-note/)

## 题目

给定一个赎金信 `ransom` 字符串和一个杂志 `magazine` 字符串，判断第一个字符串 `ransom` 能不能由第二个字符串 `magazines` 里面的字符构成。如果可以构成，返回 `true`; 否则返回 `false`.

### 说明

为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。

### 注意

你可以假设两个字符串均只含有小写字母。

### 示例

```raw
canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true
```

<!-- more -->

## 题解

### 计数

```python
class Solution:
    def canConstruct(self, ransom: str, magazine: str) -> bool:
        return can_construct(ransom, magazine)

from collections import Counter

def can_construct(ransom: str, magazine: str) -> bool:
    return not (Counter(ransom) - Counter(magazine))
```
