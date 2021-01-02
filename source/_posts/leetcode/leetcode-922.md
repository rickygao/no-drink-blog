---
title: LeetCode 922. 按奇偶排序数组 II
date: 2020-11-26 17:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/sort-array-by-parity-ii/)

## 题目

给定一个非负整数数组 `A`, `A` 中一半整数是奇数，一半整数是偶数。

对数组进行排序，以便当 `A[i]` 为奇数时，`i` 也是奇数；当 `A[i]` 为偶数时, `i` 也是偶数。

你可以返回任何满足上述条件的数组作为答案。

### 示例

```raw
输入：[4, 2, 5, 7]
输出：[4, 5, 2, 7]
解释：[4, 7, 2, 5], [2, 5, 4, 7], [2, 7, 4, 5] 也会被接受。
```

### 提示

- `2 <= len(A) <= 20000`;
- `len(A) % 2 == 0`;
- `0 <= A[i] <= 1000`.

<!-- more -->

## 题解

双指针，一个遍历偶数位置，一个遍历奇数位置，找到每一个需要交换的数对。

```python
class Solution:
    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        return sort_array_by_parity_ii(A)

def sort_array_by_parity_ii(l: List[int]) -> List[int]:
    i = 0
    j = 1
    while i < len(l) and j < len(l):
        while i < len(l) and l[i] % 2 == 0:
            i += 2
        while j < len(l) and l[j] % 2 == 1:
            j += 2
        if i < len(l) and j < len(l):
            tmp = l[i]
            l[i] = l[j]
            l[j] = tmp
    return l
```
