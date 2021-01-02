---
title: LeetCode 1370. 上升下降字符串
date: 2020-11-25 00:40:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/increasing-decreasing-string/)

## 题目

给你一个字符串 `s`, 请你根据下面的算法重新构造字符串：

1. 从 `s` 中选出**最小**的字符，将它**接在**结果字符串的后面；
2. 从 `s` 剩余字符中选出**最小**的字符，且该字符比上一个添加的字符大，将它**接在**结果字符串后面；
3. 重复步骤 2, 直到你没法从 `s` 中选择字符；
4. 从 `s` 中选出**最大**的字符，将它**接在**结果字符串的后面；
5. 从 `s` 剩余字符中选出**最大**的字符，且该字符比上一个添加的字符小，将它**接在**结果字符串后面；
6. 重复步骤 5, 直到你没法从 `s` 中选择字符；
7. 重复步骤 1 到 6, 直到 `s` 中所有字符都已经被选过。

在任何一步中，如果最小或者最大字符不止一个，你可以选择其中任意一个，并将其添加到结果字符串。

请你返回将 `s` 中字符重新排序后的**结果字符串**。

### 示例

```raw
输入：s = "aaaabbbbcccc"
输出："abccbaabccba"
解释：第一轮的步骤 1, 2, 3 后，结果字符串为 result = "abc".
第一轮的步骤 4, 5, 6 后，结果字符串为 result = "abccba".
第一轮结束，现在 s = "aabbcc", 我们再次回到步骤 1.
第二轮的步骤 1, 2, 3 后，结果字符串为 result = "abccbaabc".
第二轮的步骤 4, 5, 6 后，结果字符串为 result = "abccbaabccba".
```

```raw
输入：s = "rat"
输出："art"
解释：单词 "rat" 在上述算法重排序以后变成 "art".
```

```raw
输入：s = "leetcode"
输出："cdelotee"
```

```raw
输入：s = "ggggggg"
输出："ggggggg"
```

```raw
输入：s = "spo"
输出："ops"
```

### 提示

- `1 <= len(s) <= 500`;
- `s` 只包含小写英文字母。

<!-- more -->

## 题解

桶计数，往复遍历。

```python
class Solution:
    def sortString(self, s: str) -> str:
        return sort_string(s)

def sort_string(s: str) -> str:
    ord_a = ord('a')

    counter = [0] * 26
    for c in s:
        counter[ord(c) - ord_a] += 1

    turn, result = 0, []
    desc = False
    for _ in range(max(counter)):
        it = range(len(counter))
        if desc:
            it = reversed(it)

        for i in it:
            if counter[i] > turn:
                result.append(chr(i + ord_a))

        turn += 1
        desc = not desc

    return ''.join(result)
```
