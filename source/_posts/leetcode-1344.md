---
title: LeetCode 1344. 时钟指针的夹角
date: 2020-11-26 21:00:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/angle-between-hands-of-a-clock/)

## 题目

给你两个数 `hour` 和 `minutes`. 请你返回在时钟上，由给定时间的时针和分针组成的较小角的角度（60 单位制）。

### 示例

```raw
输入：hour = 12, minutes = 30
输出：165
```

```raw
输入：hour = 3, minutes = 30
输出；75
```

```raw
输入：hour = 3, minutes = 15
输出：7.5
```

```raw
输入：hour = 4, minutes = 50
输出：155
```

```raw
输入：hour = 12, minutes = 0
输出：0
```

### 提示

- `1 <= hour <= 12`;
- `0 <= minutes <= 59`;
- 与标准答案误差在 ${10}^{-5}$ 以内的结果都被视为正确结果。

<!-- more -->

## 题解

数学题。

```python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        return angle_clock(hour, minutes)

def angle_clock(hour: int, minutes: int) -> float:
    angle_hour = (hour + minutes / 60) / 12
    angle_minutes = minutes / 60
    delta_angle = abs(angle_hour - angle_minutes)
    if delta_angle > .5:
        delta_angle = 1 - delta_angle
    return delta_angle * 360
```
