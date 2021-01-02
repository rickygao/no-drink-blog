---
title: LeetCode 1577. 数的平方等于两数乘积的方法数
date: 2020-12-06 23:15:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/number-of-ways-where-square-of-number-is-equal-to-product-of-two-numbers/)

## 题目

给你两个整数数组 `nums1` 和 `nums2`, 请你返回根据以下规则形成的三元组的数目（类型 1 和类型 2）：

类型 1：三元组 `(i, j, k)`, 如果 `nums1[i] ** 2 == nums2[j] * nums2[k]` 其中 `0 <= i < len(nums1)` 且 `0 <= j < k < len(nums2)`;

类型 2：三元组 `(i, j, k)`, 如果 `nums2[i] ** 2 == nums1[j] * nums1[k]` 其中 `0 <= i < len(nums2)` 且 `0 <= j < k < len(nums1)`.

### 示例

```raw
输入：nums1 = [7, 4], nums2 = [5, 2, 8, 9]
输出：1
解释：
类型 1：(1, 1, 2), nums1[1] ** 2 = nums2[1] * nums2[2] (4 ** 2 = 2 * 8)
```

```raw
输入：nums1 = [1, 1], nums2 = [1, 1, 1]
输出：9
解释：所有三元组都符合题目要求，因为 1 ** 2 = 1 * 1.
类型 1：(0, 0, 1), (0, 0, 2), (0, 1, 2), (1, 0, 1), (1, 0, 2), (1, 1, 2), nums1[i] ** 2 = nums2[j] * nums2[k];
类型 2：(0, 0, 1), (1, 0, 1), (2, 0, 1), nums2[i] ** 2 = nums1[j] * nums1[k].
```

```raw
输入：nums1 = [7, 7, 8, 3], nums2 = [1, 2, 9, 7]
输出：2
解释：有两个符合题目要求的三元组。
类型 1：(3, 0, 2), nums1[3] ** 2 = nums2[0] * nums2[2];
类型 2：(3, 0, 1), nums2[3] ** 2 = nums1[0] * nums1[1].
```

```raw
输入：nums1 = [4, 7, 9, 11, 23], nums2 = [3, 5, 1024, 12, 18]
输出：0
解释：不存在符合题目要求的三元组。
```

### 提示

- `1 <= len(nums1), len(nums2) <= 1000`;
- `1 <= nums1[i], nums2[i] <= 1e5`.

<!-- more -->

## 题解

计数，求交，相乘。

```python
class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        return num_triplets(nums1, nums2) + num_triplets(nums2, nums1)

from itertools import combinations
from collections import Counter

def num_triplets(nums1: List[int], nums2: List[int]) -> int:
    squares = Counter(n_i * n_i for n_i in nums1)
    products = Counter(n_j * n_k for n_j, n_k in combinations(nums2, 2))
    equals = squares.keys() & products.keys()
    return sum(squares[r] * products[r] for r in equals)
```
