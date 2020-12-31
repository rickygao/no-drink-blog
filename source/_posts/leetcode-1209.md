---
title: LeetCode 1209. 删除字符串中的所有相邻重复项 II
date: 2020-12-19 16:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string-ii/)

## 题目

给你一个字符串 `s`,「k 倍重复项删除操作」将会从 `s` 中选择 `k` 个相邻且相等的字母，并删除它们，使被删去的字符串的左侧和右侧连在一起。

你需要对 `s` 重复进行无限次这样的删除操作，直到无法继续为止。

在执行完所有删除操作后，返回最终得到的字符串。

本题答案保证唯一。

### 示例

```raw
输入：s = "abcd", k = 2
输出："abcd"
解释：没有要删除的内容。
```

```raw
输入：s = "deeedbbcccbdaa", k = 3
输出："aa"
解释：
先删除 "eee" 和 "ccc", 得到 "ddbbbdaa"
再删除 "bbb", 得到 "dddaa"
最后删除 "ddd", 得到 "aa"
```

```raw
输入：s = "pbbcggttciiippooaais", k = 2
输出："ps"
```

### 提示

- `1 <= len(s) <= 1e5`;
- `2 <= k <= 1e4`;
- `s` 中只含有小写英文字母。

<!-- more -->

## 题解

栈。每一字符与栈顶消重。是{% post_link leetcode-1047 %}的推广。

```python
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        return remove_duplicates(s, k)

from operator import mul

def remove_duplicates(s: str, k: int) -> str:
    r, cnt = [], []
    for c in s:
        if not r or r[-1] != c:
            r.append(c)
            cnt.append(1)
        elif cnt[-1] < k - 1:
            cnt[-1] += 1
        else:
            r.pop()
            cnt.pop()
    return ''.join(map(mul, r, cnt))
```
