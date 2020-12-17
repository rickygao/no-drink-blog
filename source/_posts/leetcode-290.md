---
title: LeetCode 290. 单词规律
date: 2020-12-16 00:40:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/word-pattern/)

## 题目

给定一种规律 `pattern` 和一个字符串 `str`, 判断 `str` 是否遵循相同的规律。

这里的**遵循**指完全匹配，例如，`pattern` 里的每个字母和字符串 `str` 中的每个非空单词之间存在着双向连接的对应规律。

### 示例

```raw
输入：pattern = "abba", str = "dog cat cat dog"
输出：true
```

```raw
输入：pattern = "abba", str = "dog cat cat fish"
输出：false
```

```raw
输入：pattern = "aaaa", str = "dog cat cat dog"
输出：false
```

```raw
输入：pattern = "abba", str = "dog dog dog dog"
输出：false
```

### 说明

你可以假设 `pattern` 只包含小写字母, `str` 包含了由单个空格分隔的小写字母。

<!-- more -->

## 题解

两个字典记录双射。

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        return wrod_pattern(pattern, s)

def wrod_pattern(pattern: str, s: str) -> bool:
    words = s.split()
    if len(words) != len(pattern):
        return False

    p2w, w2p = {}, {}
    for p, w in zip(pattern, words):
        if (p in p2w and p2w[p] != w) or (w in w2p and w2p[w] != p):
            return False
        if p not in p2w:
            p2w[p], w2p[w] = w, p

    return True
```
