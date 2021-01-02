---
title: LeetCode 1122. 数组的相对排序
date: 2020-12-13 01:40:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/relative-sort-array/)

## 题目

给你两个数组，`arr1` 和 `arr2`.

- `arr2` 中的元素各不相同；
- `arr2` 中的每个元素都出现在 `arr1` 中。

对 `arr1` 中的元素进行排序，使 `arr1` 中项的相对顺序和 `arr2` 中的相对顺序相同。未在 `arr2` 中出现过的元素需要按照升序放在 `arr1` 的末尾。

### 示例

```raw
输入：arr1 = [2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19], arr2 = [2, 1, 4, 3, 9, 6]
输出：[2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]
```

### 提示

- `1 <= len(arr1), len(arr2) <= 1000`;
- `0 <= arr1[i], arr2[i] <= 1000`;
- `arr2` 中的元素 `arr2[i]` 各不相同；
- `arr2` 中的每个元素 `arr2[i]` 都出现在 `arr1` 中。

<!-- more -->

## 题解

反查位置。

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        return relative_sort_array(arr1, arr2)

def relative_sort_array(arr1: List[int], arr2: List[int]) -> List[int]:
    pos, mpos = {v: i for i, v in enumerate(arr2)}, len(arr2)
    return sorted(arr1, key=lambda v: (pos.get(v, mpos), v))
```
