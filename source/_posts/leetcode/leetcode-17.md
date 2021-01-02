---
title: LeetCode 17. 电话号码的字母组合
date: 2020-11-30 22:20:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

## 题目

给定一个仅包含数字 `2` 到 `9` 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 `1` 不对应任何字母。

{% asset_img telephone-keypad.png 200 181 "'手机键盘' '手机键盘'" %}

### 示例

```raw
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

### 说明

尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。

<!-- more -->

## 题解

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        return letter_combinations(digits)

from itertools import product

MAPPING = {
    '2': 'abc', '3': 'def',
    '4': 'ghi', '5': 'jkl', '6': 'mno',
    '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
}

def letter_combinations(digits: str) -> List[str]:
    if digits == '':
        return []

    mappings = (MAPPING[digit] for digit in digits)
    return [''.join(c) for c in product(*mappings)]
```
