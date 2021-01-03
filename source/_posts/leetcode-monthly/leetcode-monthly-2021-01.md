---
title: LeetCode 月报 2021 年 1 月
date: 2021-01-01 00:00:00
tags: [LeetCode]
mathjax: true
---

新年新气象，跑步逃离魔幻的 2020 年。

<!-- more -->

## LeetCode 605 种花问题

[:link: 来源](https://leetcode-cn.com/problems/can-place-flowers/)

### 题目

假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。

给定一个花坛（表示为一个数组包含 `0` 和 `1`, 其中 `0` 表示没种植花，`1` 表示种植了花），和一个数 `n`. 能否在不打破种植规则的情况下种入 `n` 朵花？能则返回 `true`, 不能则返回 `false`.

#### 示例

```raw
输入：flowerbed = [1, 0, 0, 0, 1], n = 1
输出：true
```

```raw
输入：flowerbed = [1, 0, 0, 0, 1], n = 2
输出：false
```

#### 注意

- 数组内已种好的花不会违反种植规则；
- 输入的数组长度范围为 `[1, 20000]`;
- `n` 是非负整数，且不会超过输入数组的大小。

### 题解

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        return can_place_flowers(flowerbed, n)

from itertools import chain

def can_place_flowers(flowerbed: List[int], n: int) -> bool:
    r = c = 0
    for b in chain([0], flowerbed, [0, 1]):
        if b:
            r += (c - 1) // 2
            c = 0
        else:
            c += 1
    return r >= n
```

## LeetCode 1250 检查「好数组」

[:link: 来源](https://leetcode-cn.com/problems/check-if-it-is-a-good-array/)

### 题目

给你一个正整数数组 `nums`, 你需要从中任选一些子集，然后将子集中每一个数乘以一个**任意整数**，并求出他们的和。

假如存在一种选取情况使该和结果为 `1`, 那么原数组就是一个「好数组」，则返回 `true`; 否则请返回 `false`.

#### 示例

```raw
输入：nums = [12, 5, 7, 23]
输出：true
解释：挑选数字 5 和 7. 5 * 3 + 7 * (-2) = 1.
```

```raw
输入：nums = [29, 6, 10]
输出：true
解释：挑选数字 29, 6 和 10. 29 * 1 + 6 * (-3) + 10 * (-1) = 1.
```

```raw
输入：nums = [3, 6]
输出：false
```

#### 提示

- `1 <= len(nums) <= 1e5`;
- `1 <= nums[i] <= 1e9`.

### 题解

数学题。[裴蜀定理](https://zh.wikipedia.org/wiki/貝祖等式)的推广形式：若 $a_i\in\mathbb{Z}$ 且 $d=\gcd(a_i)$, 则关于 $x_i$ 的方程 $\sum_ia_ix_i=m$ 有整数解当且仅当 $d\mid m$, 其中 $i=1,2,\dots,n$; 特别地，$\sum_ia_ix_i=1$ 有整数解当且仅当 $a_i$ 互质。

```python
class Solution:
    def isGoodArray(self, nums: List[int]) -> bool:
        return is_good_array(nums)

from math import gcd

def is_good_array(nums: List[int]) -> bool:
    return gcd(*nums) == 1
```

## LeetCode 239 滑动窗口最大值

[:link: 来源](https://leetcode-cn.com/problems/sliding-window-maximum/)

### 题目

给你一个整数数组 `nums`, 有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

#### 示例

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

#### 提示

- `1 <= len(nums) <= 1e5`;
- `-1e4 <= nums[i] <= 1e4`;
- `1 <= k <= len(nums)`.

### 题解

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

## LeetCode 86 分隔链表

[:link: 来源](https://leetcode-cn.com/problems/partition-list/)

### 题目

给你一个链表和一个特定值 `x`, 请你对链表进行分隔，使得所有小于 `x` 的节点都出现在大于或等于 `x` 的节点之前。

你应当保留两个分区中每个节点的初始相对位置。

#### 示例

```raw
输入：head = 1 -> 4 -> 3 -> 2 -> 5 -> 2, x = 3
输出：1 -> 2 -> 2 -> 4 -> 3 -> 5
```

### 题解

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        return partition(head, x)

def partition(head: ListNode, x: int) -> ListNode:
    l = l_dummy = ListNode()
    g = g_dummy = ListNode()

    n = head
    while n:
        if n.val < x:
            l.next = l = n
        else:
            g.next = g = n
        n = n.next
    l.next, g.next = g_dummy.next, None

    return l_dummy.next
```
