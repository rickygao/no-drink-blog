---
title: LeetCode 1010. 总持续时间可被 60 整除的歌曲
date: 2020-12-31 21:40:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/)

## 题目

在歌曲列表中，第 `i` 首歌曲的持续时间为 `time[i]` 秒。

返回其总持续时间（以秒为单位）可被 `60` 整除的歌曲对的数量。形式上，我们希望索引的数字 `i` 和 `j` 满足，`i < j` 且有 `(time[i] + time[j]) % 60 == 0`.

### 示例

```raw
输入：[30, 20, 150, 100, 40]
输出：3
解释：这三对的总持续时间可被 60 整除，
(time[0] = 30, time[2] = 150) 总持续时间 180;
(time[1] = 20, time[3] = 100) 总持续时间 120;
(time[1] = 20, time[4] = 40) 总持续时间 60.
```

```raw
输入：[60, 60, 60]
输出：3
解释：所有三对的总持续时间都是 120, 可以被 60 整除。
```

### 提示

- `1 <= len(time) <= 60000`;
- `1 <= time[i] <= 500`.

<!-- more -->

## 题解

计数。

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        return num_pairs_divisible_by_60(time)

from collections import Counter

def num_pairs_divisible_by_60(time: List[int]) -> int:
    c = Counter(t % 60 for t in time)
    return sum(c[t] * c[60 - t] for t in c if t < 30) \
        + (c[0] * (c[0] - 1) + c[30] * (c[30] - 1)) // 2
```
