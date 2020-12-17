---
title: LeetCode 1190. 反转每对括号间的子串
date: 2020-12-08 14:10:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

## 题目

给出一个字符串 `s`（仅含有小写英文字母和括号）。

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意，您的结果中**不应**包含任何括号。

### 示例

```raw
输入：s = "(abcd)"
输出："dcba"
```

```raw
输入：s = "(u(love)i)"
输出："iloveu"
```

```raw
输入：s = "(ed(et(oc))el)"
输出："leetcode"
```

```raw
输入：s = "a(bcdefghijkl(mno)p)q"
输出："apmnolkjihgfedcbq"
```

### 提示

- `0 <= len(s) <= 2000`;
- `s` 中只有小写英文字母和括号；
- 我们确保所有括号都是成对出现的。

<!-- more -->

## 题解

栈，每当括号闭合时翻转栈顶字符串并合并。

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        return reverse_parentheses(s)

def reverse_parentheses(s: str) -> str:
    r = ['']
    for c in s:
        if c == '(':
            r.append('')
        elif c == ')':
            t = r.pop()
            r[-1] += t[::-1]
        else:
            r[-1] += c
    return r[0]
```
