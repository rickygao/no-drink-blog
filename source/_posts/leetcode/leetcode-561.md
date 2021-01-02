---
title: LeetCode 561. 数组拆分 I
date: 2020-12-30 11:15:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/array-partition-i/)

## 题目

给定长度为 $2n$ 的整数数组 `nums`, 你的任务是将这些数分成 $n$ 对, 例如 $(a_1,b_1),(a_2,b_2),\dots,(a_n,b_n)$, 使得 $\sum_{i=1}^n\min(a_i,b_i)$ 最大。

返回该**最大总和**。

### 示例

```raw
输入：nums = [1, 4, 3, 2]
输出：4
解释：所有可能的分法（忽略元素顺序）为：
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3;
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3;
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4;
所以最大总和为 4.
```

```raw
输入：nums = [6, 2, 6, 5, 1, 2]
输出：9
解释：最优的分法为 (2, 1), (2, 5), (6, 6). min(2, 1) + min(2, 5) + min(6, 6) = 1 + 2 + 6 = 9.
```

### 提示

- `1 <= n <= 1e4`;
- `len(nums) == 2 * n`;
- `-1e4 <= nums[i] <= 1e4`.

<!-- more -->

## 题解

贪心。排序后即 $[a_1,b_1,a_2,b_2,\dots,a_n,b_n]$ 且 $a_i<b_i$, 于是求 $\sum_{i=1}^na_i$ 即可。

```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        return array_pair_sum(nums)

def array_pair_sum(nums: List[int]) -> int:
    return sum(sorted(nums)[::2])
```
