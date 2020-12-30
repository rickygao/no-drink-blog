---
title: LeetCode 1046. 最后一块石头的重量
date: 2020-12-30 10:50:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/last-stone-weight/)

## 题目

有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块**最重的**石头，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`, 且 `x <= y`. 那么粉碎的可能结果如下：

- 如果 `x == y`, 那么两块石头都会被完全粉碎；
- 如果 `x != y`, 那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y - x`.

最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 `0`.

### 示例

```raw
输入：[2, 7, 4, 1, 8, 1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2, 4, 1, 1, 1];
再选出 2 和 4，得到 2，所以数组转换为 [2, 1, 1, 1];
接着是 2 和 1，得到 1，所以数组转换为 [1, 1, 1];
最后选出 1 和 1，得到 0，最终数组转换为 [1], 这就是最后剩下那块石头的重量。
```

### 提示

- `1 <= len(stones) <= 30`;
- `1 <= stones[i] <= 1000`.

<!-- more -->

## 题解

### 排序

模拟。排序，二分插入。

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        return last_stone_weight(stones)

from bisect import insort

def last_stone_weight(stones: List[int]) -> int:
    stones = sorted(stones)
    while len(stones) >= 2:
        x = stones.pop()
        y = stones.pop()
        if x > y:
            insort(stones, x - y)
    return stones.pop() if stones else 0
```

### 堆

模拟。最小堆。通过取负数转化成最小堆。

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        return last_stone_weight(stones)

from heapq import heapify, heappop, heappush

def last_stone_weight(stones: List[int]) -> int:
    stones = [-s for s in stones]
    heapify(stones)
    while len(stones) >= 2:
        x = heappop(stones)
        y = heappop(stones)
        if x < y:
            heappush(stones, x - y)
    return -heappop(stones) if stones else 0
```
