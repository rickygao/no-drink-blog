---
title: LeetCode 714. 买卖股票的最佳时机含手续费
date: 2020-12-17 01:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

## 题目

给定一个整数数组 `prices`, 其中第 `i` 个元素代表了第 `i` 天的股票价格；非负整数 `fee` 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

### 示例

```raw
输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
输出：8
解释：能够达到的最大利润：
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润：((8 - 1) - 2) + ((9 - 4) - 2) = 8.
```

- `0 < len(prices) <= 50000`;
- `0 < prices[i] < 50000`;
- `0 <= fee < 50000`.

<!-- more -->

## 题解

动态规划。`cl`, `op` 分别记录平仓或持仓的情况下迄今的最大盈利。

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        return max_profit(prices, fee)

def max_profit(prices: List[int], fee: int) -> int:
    cl, op = 0, -prices[0]
    for price in prices[1:]:
        cl, op = max(cl, op + price - fee), max(op, cl - price)
    return cl
```
