---
title: LeetCode 57. 插入区间
date: 2020-11-27 13:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/insert-interval/)

## 题目

给出一个无重叠的，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

### 示例

```raw
输入：intervals = [[1, 3], [6, 9]], newInterval = [2, 5]
输出：[[1, 5], [6, 9]]
```

```raw
输入：intervals = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], newInterval = [4, 8]
输出：[[1, 2], [3, 10], [12, 16]]
解释：这是因为新的区间 [4, 8] 与 [3, 5], [6, 7], [8, 10] 重叠。
```

<!-- more -->

## 题解

- 处理边界条件；
- 查找重叠区间窗口 `[i, j)`, 并处理窗口边界的区间合并。

### 线性查找

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        return insert(intervals, newInterval)

def insert(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    if not intervals:
        return [new_interval]

    if intervals[0][0] > new_interval[1]:
        return [new_interval, *intervals]

    if intervals[-1][1] < new_interval[0]:
        return [*intervals, new_interval]

    i = 0
    while intervals[i][1] < new_interval[0]:
        i += 1

    j = i
    while j < len(intervals) and intervals[j][0] <= new_interval[1]:
        j += 1

    new_interval = [min(intervals[i][0], new_interval[0]), max(intervals[j - 1][1], new_interval[1])]
    return [*intervals[:i], new_interval, *intervals[j:]]
```

### 二分查找

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        return insert(intervals, newInterval)

def insert(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    if not intervals:
        return [new_interval]

    if intervals[0][0] > new_interval[1]:
        return [new_interval, *intervals]

    if intervals[-1][1] < new_interval[0]:
        return [*intervals, new_interval]

    low, high = 0, len(intervals)
    while low < high:
        mid = (low + high) // 2
        if intervals[mid][1] < new_interval[0]:
            low = mid + 1
        else:
            high = mid
    i = low

    low, high= i, len(intervals)
    while low < high:
        mid = (low + high) // 2
        if intervals[mid][0] <= new_interval[1]:
            low = mid + 1
        else:
            high = mid
    j = low

    new_interval = [min(intervals[i][0], new_interval[0]), max(intervals[j - 1][1], new_interval[1])]
    return [*intervals[:i], new_interval, *intervals[j:]]
```

### 二分查找（原地）

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        return insert(intervals, newInterval)

def insert(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    if not intervals:
        intervals.append(new_interval)

    elif intervals[0][0] > new_interval[1]:
        intervals.insert(0, new_interval)

    elif intervals[-1][1] < new_interval[0]:
        intervals.append(new_interval)

    else:
        low, high = 0, len(intervals)
        while low < high:
            mid = (low + high) // 2
            if intervals[mid][1] < new_interval[0]:
                low = mid + 1
            else:
                high = mid
        i = low

        low, high = i, len(intervals)
        while low < high:
            mid = (low + high) // 2
            if intervals[mid][0] <= new_interval[1]:
                low = mid + 1
            else:
                high = mid
        j = low

        new_interval[0] = min(intervals[i][0], new_interval[0])
        new_interval[1] = max(intervals[j - 1][1], new_interval[1])
        intervals[i: j] = [new_interval]

    return intervals
```
