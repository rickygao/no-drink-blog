---
title: LeetCode 746. 使用最小花费爬楼梯
date: 2020-12-21 11:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

## 题目

数组的每个索引作为一个阶梯，第 `i` 个阶梯对应着一个非负数的体力花费值 `cost[i]`（索引从 0 开始）。

每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。

您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 `0` 或 `1` 的元素作为初始阶梯。

### 示例

```raw
输入：cost = [10, 15, 20]
输出：15
解释：最低花费是从 cost[1] 开始，然后走两步即可到阶梯顶，一共花费 15.
```

```raw
输入：cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出：6
解释：最低花费方式是从 cost[0] 开始，逐个经过那些 1, 跳过 cost[3], 一共花费 6.
```

### 注意

- `cost` 的长度将会在 `[2, 1000]`;
- 每一个 `cost[i]` 将会是一个 `int` 类型，范围为 `[0, 999]`.

<!-- more -->

## 题解

动态规划。

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        return min_cost_climbing_stairs(cost)

def min_cost_climbing_stairs(cost: List[int]) -> int:
    r1 = r2 = 0
    for c in cost:
        r1, r2 = min(r1, r2) + c, r1
    return min(r1, r2)
```
