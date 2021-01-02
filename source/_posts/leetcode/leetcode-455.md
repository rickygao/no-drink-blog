---
title: LeetCode 455. 分发饼干
date: 2020-12-25 14:40:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/assign-cookies/)

## 题目

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 `i`, 都有一个胃口值 `g[i]`, 这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 `j`, 都有一个尺寸 `s[j]`. 如果 `s[j] >= g[i]`, 我们可以将这个饼干 `j` 分配给孩子 `i`, 这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

### 示例

```raw
输入：g = [1, 2, 3], s = [1, 1]
输出：1
解释：
你有三个孩子和两块小饼干，3 个孩子的胃口值分别是 1, 2, 3.
虽然你有两块小饼干，由于他们的尺寸都是 1, 你只能让胃口值是 1 的孩子满足。
所以你应该输出 1.
```

```raw
输入：g = [1, 2], s = [1, 2, 3]
输出：2
解释：
你有两个孩子和三块小饼干，2 个孩子的胃口值分别是 1, 2.
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出 2.
```

### 提示

- `1 <= len(g) <= 3e4`;
- `0 <= len(s) <= 3e4`;
- `1 <= g[i], s[j] <= 2 ** 31 - 1`.

<!-- more -->

## 题解

排序，贪心。使用最小的代价满足每个胃口小的孩子。

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        return find_content_children(g, s)

def find_content_children(g: List[int], s: List[int]) -> int:
    g, s, i = sorted(g), sorted(s), 0
    for t in s:
        if i >= len(g):
            break
        if g[i] <= t:
            i += 1
    return i
```
