---
title: LeetCode 1047. 删除字符串中的所有相邻重复项
date: 2020-12-11 16:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

## 题目

给出由小写字母组成的字符串 `S`, 重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 `S` 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

### 示例

```raw
输入："abbaca"
输出："ca"
解释：例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca".
```

### 提示

- `1 <= len(S) <= 20000`;
- `S` 仅由小写英文字母组成。

<!-- more -->

## 题解

栈。每一字符与栈顶消重。

```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        return remove_duplicates(S)

def remove_duplicates(s: str) -> str:
    r = []
    for c in s:
        if r and r[-1] == c:
            r.pop()
        else:
            r.append(c)
    return ''.join(r)
```
