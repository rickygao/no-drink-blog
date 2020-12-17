---
title: LeetCode 1160. 拼写单词
date: 2020-12-13 01:20:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/find-words-that-can-be-formed-by-characters/)

## 题目

给你一份「词汇表」（字符串数组）`words` 和一张「字母表」（字符串）`chars`.

假如你可以用 chars 中的「字母」（字符）拼写出 `words` 中的某个「单词」（字符串），那么我们就认为你掌握了这个单词。

注意：每次拼写（指拼写词汇表中的一个单词）时，`chars` 中的每个字母都只能用一次。

返回词汇表 `words` 中你掌握的所有单词的**长度之和**。

### 示例

```raw
输入：words = ["cat", "bt", "hat", "tree"], chars = "atach"
输出：6
解释：可以形成字符串 "cat" 和 "hat", 所以答案是 3 + 3 = 6.
```

```raw
输入：words = ["hello", "world", "leetcode"], chars = "welldonehoneyr"
输出：10
解释：可以形成字符串 "hello" 和 "world", 所以答案是 5 + 5 = 10.
```

### 提示

- `1 <= len(words) <= 1000`;
- `1 <= len(words[i]), len(chars) <= 100`;
- 所有字符串中都仅包含小写英文字母。

<!-- more -->

## 题解

计数。

```python
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        return count_characters(words, chars)

from collections import Counter

def count_characters(words: List[str], chars: str) -> int:
    chars = Counter(chars)
    return sum(
        len(word) for word in words
        if all(chars[ch] >= c for ch, c in Counter(word).items())
    )
```
