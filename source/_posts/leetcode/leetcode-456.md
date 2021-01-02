---
title: LeetCode 456. 132 模式
date: 2020-12-04 22:00:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/132-pattern/)

## 题目

给定一个整数序列：$a_1, a_2, \dots, a_n$, 一个 132 模式的子序列 $a_i, a_j, a_k$ 被定义为：当 $i < j < k$ 时，$a_i < a_k < a_j$. 设计一个算法，当给定有 `n` 个数字的序列时，验证这个序列中是否含有 132 模式的子序列。

### 注意

`n` 的值小于 `15000`.

### 示例

```raw
输入：[1, 2, 3, 4]
输出：false
解释：序列中不存在 132 模式的子序列。
```

```raw
输入：[3, 1, 4, 2]
输出：true
解释：序列中有 1 个 132 模式的子序列 [1, 4, 2].
```

```raw
输入：[-1, 3, 2, 0]
输出：true
解释：序列中有 3 个 132 模式的的子序列 [-1, 3, 2], [-1, 3, 0], [-1, 2, 0].
```

<!-- more -->

## 题解

- `s` 是单调递减栈；
- 通过 `s[-1] < a_j` 的条件为 `a_k` 赋值，保证了对于一个尽量大的 `a_k` 会存在 `a_j` 比它更大；
- 只需存在一个 `a_i < a_k` 小，即可断言成功。

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        return find_132_pattern(nums)

def find_132_pattern(nums: List[int]) -> bool:
    s, a_k = [], float('-inf')
    for a_i in reversed(nums):
        if a_i < a_k:
            return True
        # treat a_i as a_j
        while s and s[-1] < a_i:
            a_k = s.pop()
        s.append(a_i)
    return False
```
