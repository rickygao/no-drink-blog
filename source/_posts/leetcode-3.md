---
title: LeetCode 3. 无重复字符的最长子串
date: 2020-11-25 22:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

## 题目

给定一个字符串，请你找出其中不含有重复字符的**最长子串**的长度。

### 示例

```raw
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc", 所以其长度为 3.
```

```raw
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b", 所以其长度为 1.
```

```raw
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是子串的长度，"pwke" 是一个子序列，不是子串.
```

```raw
输入: s = ""
输出: 0
```

### 提示

- `0 <= len(s) <= 5e4`;
- `s` 由英文字母、数字、符号和空格组成。

<!-- more -->

## 题解

- `[i, j]` 滑动窗口；
- 利用字典进行 `i` 跳跃。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        return length_of_longest_substring(s)

def length_of_longest_substring(s: str) -> int:
    c2p = dict()
    i = mlen = 0
    for j, c in enumerate(s):
        p = c2p.get(c, -1)
        if p >= i:
            mlen = max(j - i, mlen)
            i = p + 1
        c2p[c] = j
    mlen = max(len(s) - i, mlen)
    return mlen
```
