---
title: LeetCode 1338. 数组大小减半
date: 2020-12-12 15:45:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/reduce-array-size-to-the-half/)

## 题目

给你一个整数数组 `arr`. 你可以从中选出一个整数集合，并删除这些整数在数组中的每次出现。

返回**至少**能删除数组中的一半整数的整数集合的最小大小。

### 示例

```raw
输入：arr = [3, 3, 3, 3, 5, 5, 5, 2, 2, 7]
输出：2
解释：选择 {3, 7} 使得结果数组为 [5, 5, 5, 2, 2], 长度为 5（原数组长度的一半）。
大小为 2 的可行集合有 {3, 5}, {3, 2}, {5, 2}.
选择 {2, 7} 是不可行的，它的结果数组为 [3, 3, 3, 3, 5, 5, 5], 新数组长度大于原数组的二分之一。
```

```raw
输入：arr = [7, 7, 7, 7, 7, 7]
输出：1
解释：我们只能选择集合 {7}, 结果数组为空。
```

```raw
输入：arr = [1, 9]
输出：1
```

```raw
输入：arr = [1000, 1000, 3, 7]
输出：1
```

```raw
输入：arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
输出：5
```

### 提示

- `1 <= len(arr) <= 1e5`;
- `len(arr)` 为偶数；
- `1 <= arr[i] <= 1e5`.

<!-- more -->

## 题解

贪心。

```python
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        return min_set_size(arr)

from collections import Counter
from itertools import accumulate

def min_set_size(arr: List[int]) -> int:
    half = ceil(len(arr) / 2)
    return next(i + 1 for i, a in enumerate(accumulate(
        sorted(Counter(arr).values(), reverse=True)
    )) if a >= half)
```
