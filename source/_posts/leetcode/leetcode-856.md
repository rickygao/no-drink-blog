---
title: LeetCode 856. 括号的分数
date: 2020-12-27 15:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/score-of-parentheses/)

## 题目

给定一个平衡括号字符串 `S`, 按下述规则计算该字符串的分数：

- `'()'` 得 `1` 分；
- `A + B` 得 `A`, `B` 的分数之和分，其中 `A` 和 `B` 是平衡括号字符串；
- `'(' + A + ')'` 得 `A` 的分数的二倍分，其中 `A` 是平衡括号字符串。

### 示例

```raw
输入："()"
输出：1
```

```raw
输入："(())"
输出：2
```

```raw
输入："()()"
输出：2
```

```raw
输入："(()(()))"
输出：6
```

### 提示

- `S` 是平衡括号字符串，且只含有 `'('` 和 `')'`;
- `2 <= len(S) <= 50`.

<!-- more -->

## 题解

```python
class Solution:
    def scoreOfParentheses(self, S: str) -> int:
        return score_of_parentheses(S)

def score_of_parentheses(s: str) -> int:
    stack = [0]
    for c in s:
        if c == '(':
            stack.append(0)
        else:
            t = stack.pop()
            stack[-1] += t * 2 if t else 1
    return stack.pop()
```
