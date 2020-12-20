---
title: LeetCode 316. 去除重复字母
date: 2020-12-20 12:20:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/remove-duplicate-letters/)

## 题目

给你一个字符串 `s`, 请你去除字符串中重复的字母，使得每个字母只出现一次。需保证**返回结果的字典序最小**（要求不能打乱其他字符的相对位置）。

### 注意

该题与 [1081](https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters) 相同。

### 示例

```raw
输入：s = "bcabc"
输出："abc"
```

```raw
输入：s = "cbacdcbc"
输出："acdb"
```

### 提示

- `1 <= len(s) <= 1e4`;
- `s` 由小写英文字母组成。

<!-- more -->

## 题解

贪心，单调栈。集合 `included` 记录已经加入到栈 `stack` 中的字符。如果栈顶字符 `top` 与当前字符 `c` 出现逆序 `top >= c`, 且未来还有机会遇到栈顶字符，即 `remaining[top] > 0`, 则不断出栈。

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        return remove_duplicate_letters(s)

from collections import Counter

def remove_duplicate_letters(s: str) -> str:
    stack, included, remaining = [], set(), Counter(s)
    for c in s:
        if c not in included:
            while stack and (top := stack[-1]) >= c and remaining[top]:
                stack.pop()
                included.remove(top)
            stack.append(c)
            included.add(c)
        remaining[c] -= 1
    return ''.join(stack)
```
