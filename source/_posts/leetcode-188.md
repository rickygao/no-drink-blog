---
title: LeetCode 188. 买卖股票的最佳时机 IV
date: 2020-12-28 11:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

## 题目

给定一个整数数组 `prices`, 它的第 `i` 个元素 `prices[i]` 是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 `k` 笔交易。

### 注意

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

### 示例

```raw
输入：k = 2, prices = [2, 4, 1]
输出：2
解释：
在第 1 天（股票价格 = 2）的时候买入，在第 2 天（股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4 - 2 = 2.
```

```raw
输入：k = 2, prices = [3, 2, 6, 5, 0, 3]
输出：7
解释：
在第 2 天（股票价格 = 2）的时候买入，在第 3 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 2 = 4.
随后，在第 5 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出, 这笔交易所能获得利润 = 3 - 0 = 3.
```

### 提示

- `0 <= k <= 1e9`;
- `0 <= len(prices) <= 1000`;
- `0 <= prices[i] <= 1000`.

<!-- more -->

## 题解

- 动态规划。仿照{% post_link leetcode-714 %}，其中 `p[j * 2]` 表示最多进行 `j` 次交易且当前为空仓的利润，`p[j * 2 + 1]` 表示最多进行 `j` 次交易且当前为开仓的利润，这也暗示 `p[:(j + 1) * 2]` 可以完整地表示 `j` 次交易的最大利润；
- 最多进行 `len(prices) // 2` 次有效的交易（买卖股票），故而可缩小总共需申请的状态空间。考虑第 `i` 支股票时，最多进行 `(i + 1) // 2` 次有效的交易，故而可缩小每次迭代需更新的状态空间。

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        return max_profit(k, prices)

def max_profit(k: int, prices: List[int]) -> int:
    p = [0] + [float('-inf')] * (min(k, len(prices) // 2) * 2)
    for i, price in enumerate(prices):
        for j in range(1, min(len(p), ((i + 1) // 2 + 1) * 2)):
            p[j] = max(p[j], p[j - 1] + price * (-1 if j % 2 else 1))
    return p[-1]
```
