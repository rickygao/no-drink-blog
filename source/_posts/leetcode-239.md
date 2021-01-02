---
title: LeetCode 239. 滑动窗口最大值
date: 2021-01-02 11:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/sliding-window-maximum/)

## 题目

给你一个整数数组 `nums`, 有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

### 示例

```raw
输入：nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
输出：[3, 3, 5, 5, 6, 7]
解释：滑动窗口的位置和最大值如下。
----------------------------+---
 [1  3  -1] -3  5  3  6  7  | 3
  1 [3  -1  -3] 5  3  6  7  | 3
  1  3 [-1  -3  5] 3  6  7  | 5
  1  3  -1 [-3  5  3] 6  7  | 5
  1  3  -1  -3 [5  3  6] 7  | 6
  1  3  -1  -3  5 [3  6  7] | 7
----------------------------+---
```

```raw
输入：nums = [1], k = 1
输出：[1]
```

```raw
输入：nums = [1, -1], k = 1
输出：[1, -1]
```

```raw
输入：nums = [9, 11], k = 2
输出：[11]
```

```raw
输入：nums = [4, -2], k = 2
输出：[4]
```

### 提示

- `1 <= len(nums) <= 1e5`;
- `-1e4 <= nums[i] <= 1e4`;
- `1 <= k <= len(nums)`.

<!-- more -->

## 题解

单调队列。

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        return max_sliding_window(nums, k)

from collections import deque

def max_sliding_window(nums: List[int], k: int) -> List[int]:
    return list(generate_max_sliding_window(nums, k))

def generate_max_sliding_window(nums: Iterator[int], k: int) -> Iterator[int]:
    q = deque()
    for i, n in enumerate(nums):
        if q and q[0] <= i - k:
            q.popleft()
        while q and nums[q[-1]] < n:
            q.pop()
        q.append(i)
        if i >= k - 1:
            yield nums[q[0]]
```
