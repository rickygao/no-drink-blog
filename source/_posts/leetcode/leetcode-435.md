---
title: LeetCode 435. 无重叠区间
date: 2020-12-31 11:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/non-overlapping-intervals/)

## 题目

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

### 注意

- 可以认为区间的终点总是大于它的起点；
- 区间 `[1, 2]` 和 `[2, 3]` 的边界相互“接触”，但没有相互重叠。

### 示例

```raw
输入：[[1, 2], [2, 3], [3, 4], [1, 3]]
输出：1
解释：移除 [1, 3] 后，剩下的区间没有重叠。
```

```raw
输入：[[1, 2], [1, 2], [1, 2]]
输出：2
解释：你需要移除两个 [1, 2] 来使剩下的区间没有重叠。
```

```raw
输入：[[1, 2], [2, 3]]
输出：0
解释：你不需要移除任何区间，因为它们已经是无重叠的了。
```

<!-- more -->

## 题解

贪心。从 `intervals` 逐一选出能保留下来的区间，对于当前所有可以保留的区间，选择右端点最小的。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        return erase_overlap_intervals(intervals)

from operator import itemgetter

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    intervals = sorted(intervals, key=itemgetter(1))
    last, reserved = float('-inf'), 0
    for l, r in intervals:
        if l >= last:
            last = r
            reserved += 1
    return len(intervals) - reserved
```
