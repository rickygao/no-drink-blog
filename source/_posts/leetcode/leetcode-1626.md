---
title: LeetCode 1626. 无矛盾的最佳球队
date: 2020-11-28 13:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/best-team-with-no-conflicts/)

## 题目

假设你是球队的经理。对于即将到来的锦标赛，你想组合一支总体得分最高的球队。球队的得分是球队中所有球员的分数**总和**。

然而，球队中的矛盾会限制球员的发挥，所以必须选出一支**没有矛盾**的球队。如果一名年龄较小球员的分数**严格大于**一名年龄较大的球员，则存在矛盾。同龄球员之间不会发生矛盾。

给你两个列表 `scores` 和 `ages`, 其中每组 `scores[i]` 和 `ages[i]` 表示第 `i` 名球员的分数和年龄。请你返回**所有可能的无矛盾球队中得分最高那支的分数**。

### 示例

```raw
输入：scores = [1, 3, 5, 10, 15], ages = [1, 2, 3, 4, 5]
输出：34
解释：你可以选中所有球员。
```

```raw
输入：scores = [4, 5, 6, 5], ages = [2, 1, 2, 1]
输出：16
解释：最佳的选择是后 3 名球员。注意，你可以选中多个同龄球员。
```

```raw
输入：scores = [1, 2, 3, 5], ages = [8, 9, 10, 1]
输出：6
解释：最佳的选择是前 3 名球员。
```

### 提示

- `1 <= len(scores) == len(ages) <= 1000`;
- `1 <= scores[i] <= 1e6`, `1 <= ages[i] <= 1e3`.

<!-- more -->

## 题解

排序，动态规划。`m[j]` 表示前 `j` 个球员进行组合且第 `j` 个球员出场的最高得分。

```python
class Solution:
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        return best_team_score(scores, ages)

def best_team_score(scores: List[int], ages: List[int]) -> int:
    players, m = sorted(zip(ages, scores)), []
    for i, (ai, si) in enumerate(players):
        m.append(max(si, max((
            m[j] + si
            for j, (aj, sj) in enumerate(players[:i])
            if not (ai > aj and si < sj)
        ), default=0)))
    return max(m, default=0)
```
