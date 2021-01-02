---
title: LeetCode 697. 数组的度
date: 2020-11-26 21:50:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/degree-of-an-array/)

## 题目

给定一个非空且只包含非负数的整数数组 `nums`, 数组的度的定义是指数组里任一元素出现频数的最大值。

你的任务是找到与 `nums` 拥有相同大小的度的最短连续子数组，返回其长度。

### 示例

```raw
输入：[1, 2, 2, 3, 1]
输出：2
解释：输入数组的度是 2, 因为元素 1 和 2 的出现频数最大，均为 2.
连续子数组里面拥有相同度的有如下所示：
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2].
最短连续子数组 [2, 2] 的长度为 2, 所以返回 2.
```

```raw
输入：[1, 2, 2, 3, 1, 4, 2]
输出：6
```

### 注意

- `1 <= len(nums) <= 50000`;
- `0 <= nums[i] <= 49999`.

<!-- more -->

## 题解

字典计数，记录出现区间。频次最大的数的出现区间中长度最短的即为所求。

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        return find_shortest_sub_array(nums)

def find_shortest_sub_array(nums: List[int]) -> int:
    counter = dict()
    starts = dict()
    stops = dict()

    for i, n in enumerate(nums):
        count = counter.get(n, 0)
        counter[n] = count + 1
        if count == 0:
            starts[n] = i
        stops[n] = i

    max_count = max(counter.values())
    min_slice = min((
        stops[n] - starts[n] + 1
        for n, count in counter.items()
        if count == max_count
    ), default=0)

    return min_slice
```
