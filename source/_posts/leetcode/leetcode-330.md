---
title: LeetCode 330. 按要求补齐数组
date: 2020-12-29 15:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/patching-array/)

## 题目

给定一个已排序的正整数数组 `nums`, 和一个正整数 `n`. 从 `[1, n]` 区间内选取任意个数字补充到 `nums` 中，使得 `[1, n]` 区间内的任何数字都可以用 `nums` 中某几个数字的和来表示。请输出满足上述要求的最少需要补充的数字个数。

### 示例

```raw
输入：nums = [1, 3], n = 6
输出：1
解释：
根据 nums 里现有的组合 [1], [3], [1, 3]，可以得出 1, 3, 4.
现在如果我们将 2 添加到 nums 中， 组合变为: [1], [2], [3], [1, 3], [2, 3], [1, 2, 3].
其和可以表示数字 1, 2, 3, 4, 5, 6, 能够覆盖 [1, 6] 区间里所有的数。
所以我们最少需要添加一个数字。
```

```raw
输入：nums = [1, 5, 10], n = 20
输出：2
解释：我们需要添加 [2, 4].
```

```raw
输入：nums = [1, 2, 2], n = 5
输出：0
```

<!-- more -->

## 题解

贪心。当区间 `[1, m)` 中的整数可以被表示时，可以利用 `k <= m` 的数进行扩张，从而表示区间 `[1, m + k)` 中的整数。当 `nums` 中没有合适的 `k` 时，最节省的扩张是直接补充 `m`, 这样可以表示区间 `[1, 2 * m)` 中的整数，此时需要补充的个数 `r += 1`.

```python
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        return min_patches(nums, n)

def min_patches(nums: List[int], n: int) -> int:
    m, i, r = 1, 0, 0
    while m <= n:
        if i < len(nums) and (k := nums[i]) <= m:
            m += k
            i += 1
        else:
            m *= 2
            r += 1
    return r
```
