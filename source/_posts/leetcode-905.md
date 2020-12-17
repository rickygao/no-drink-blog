---
title: LeetCode 905. 按奇偶排序数组
date: 2020-11-26 16:40:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/sort-array-by-parity/)

## 题目

给定一个非负整数数组 `A`, 返回一个数组，在该数组中，`A` 的所有偶数元素之后跟着所有奇数元素。

你可以返回满足此条件的任何数组作为答案。

### 示例

```raw
输入：[3, 1, 2, 4]
输出：[2, 4, 3, 1]
输出 [4, 2, 3, 1], [2, 4, 1, 3] 和 [4, 2, 1, 3] 也会被接受。
```

### 提示

- `1 <= len(A) <= 5000`;
- `0 <= A[i] <= 5000`.

<!-- more -->

## 题解

双指针，一个向后遍历，一个向前遍历，找到每一个需要交换的数对。

```python
class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        return sort_array_by_parity(A)

def sort_array_by_parity(l: List[int]) -> List[int]:
    i = 0
    j = len(l) - 1
    while i < j:
        while l[i] % 2 == 0 and i < j:
            i += 1
        while l[j] % 2 == 1 and i < j:
            j -= 1
        if i < j:
            tmp = l[i]
            l[i] = l[j]
            l[j] = tmp
    return l
```
