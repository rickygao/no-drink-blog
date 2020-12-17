---
title: LeetCode 659. 分割数组为连续子序列
date: 2020-12-04 00:15:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/)

## 题目

给你一个按升序排序的整数数组 `num`（可能包含重复数字），请你将它们分割成一个或多个子序列，其中每个子序列都由连续整数组成且长度至少为 `3`.

如果可以完成上述分割，则返回 `true`; 否则，返回 `false`.

### 示例

```raw
输入：[1, 2, 3, 3, 4, 5]
输出：true
解释：你可以分割出这样两个连续子序列 [1, 2, 3], [3, 4, 5].
```

```raw
输入：[1, 2, 3, 3, 4, 4, 5, 5]
输出：true
解释：你可以分割出这样两个连续子序列 [1, 2, 3, 4, 5], [3, 4, 5].
```

```raw
输入: [1, 2, 3, 4, 4, 5]
输出: false
```

### 提示

输入的数组长度范围为 `[1, 10000]`.

<!-- more -->

## 题解

### 贪心模拟

贪心策略：优先考虑最短的序列，如无合适的序列则新建。

```python
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        return is_possible(nums)

def is_possible(nums: List[int]) -> bool:
    seqs = []
    for n in nums:
        for seq in reversed(seqs):
            if n == seq[-1] + 1:
                seq.append(n)
                break
        else:
            seqs.append([n])
    return all(len(seq) >= 3 for seq in seqs)
```

### 贪心优化

- `s1`, `s2`, `s3` 分别代表长度为 `1`, 长度为 `2`, 长度大于等于 `3` 的末尾为 `prev` 的序列，存储它们的个数；
- 贪心策略：优先满足 `s1`, `s2` 的增长需求，如无法满足则失败；然后满足 `s3` 的增长需求；最后再考虑新建序列。

```python
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        return is_possible(nums)

from itertools import groupby

def is_possible(nums: List[int]) -> bool:
    prev, s1, s2, s3 = None, 0, 0, 0
    for n, group in groupby(nums):
        if prev is not None and n - prev != 1:
            if s1 or s2:
                return False
            else:
                s3 = 0
        prev = n

        count = sum(1 for _ in group) - s1 - s2
        if count < 0:
            return False

        a3 = min(count, s3)
        s1, s2, s3 = count - a3, s1, s2 + a3
    return not (s1 or s2)
```
