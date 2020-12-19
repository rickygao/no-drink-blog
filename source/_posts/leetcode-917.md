---
title: LeetCode 917. 仅仅反转字母
date: 2020-12-19 16:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/reverse-only-letters/)

## 题目

给定一个字符串 `S`, 返回 “反转后的” 字符串，其中不是字母的字符都保留在原地，而所有字母的位置发生反转。

### 示例

```raw
输入："ab-cd"
输出："dc-ba"
```

```raw
输入："a-bC-dEf-ghIj"
输出："j-Ih-gfE-dCba"
```

```raw
输入："Test1ng-Leet=code-Q!"
输出："Qedo1ct-eeLg=ntse-T!"
```

### 提示

- `len(S) <= 100`;
- `33 <= ord(S[i]) <= 122`;
- `S` 中不包含 `'\'` or `'"'`.

<!-- more -->

## 题解

双指针。

```python
class Solution:
    def reverseOnlyLetters(self, S: str) -> str:
        return reverse_only_letters(S)

def reverse_only_letters(s: str) -> str:
    s = list(s)
    i, j = 0, len(s) - 1
    while i < j:
        if s[i].isalpha() and s[j].isalpha():
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
        elif not s[i].isalpha():
            i += 1
        elif not s[j].isalpha():
            j -= 1
    return ''.join(s)
```
