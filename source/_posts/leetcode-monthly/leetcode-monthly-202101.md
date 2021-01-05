---
title: LeetCode 月报 202101
date: 2021-01-01 00:00:00
tags: [LeetCode]
mathjax: true
---

新年新气象，跑步逃离魔幻的 2020 年。

<!-- more -->

## 605. 种花问题

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

## 1250. 检查「好数组」

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

## 239. 滑动窗口最大值

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

## 86. 分隔链表

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

## 509. 斐波那契数

[:link: 来源](https://leetcode-cn.com/problems/fibonacci-number/)

### 题目

斐波那契数，通常用 $F(n)$ 表示，形成的序列称为**斐波那契数列**。该数列由 $0$ 和 $1$ 开始，后面的每一项数字都是前面两项数字的和。也就是：

$$
F(n)=\begin{cases}
    0,             & \text{if } n=0,\\
    1,             & \text{if } n=1,\\
    F(n-1)+F(n-2), & \text{if } n\ge2.
\end{cases}
$$

给你 $n$，请计算 $F(n)$.

#### 示例

```rraw
输入：2
输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1.
```

```raw
输入：3
输出：2
解释：F(3) = F(2) + F(1) = 1 + 1 = 2.
```

```raw
输入：4
输出：3
解释：F(4) = F(3) + F(2) = 2 + 1 = 3.
```

#### 提示

- `0 <= n <= 30`.

### 题解

#### 迭代

```python
class Solution:
    def fib(self, n: int) -> int:
        return fib(n)

def fib(n: int) -> int:
    f1, f2 = 0, 1
    for i in range(n):
        f1, f2 = f2, f1 + f2
    return f1
```

#### 通项

$$
\begin{aligned}
    F(n)&=\frac{\phi^n-\psi^n}{\phi-\psi},\\
    \text{where }\phi&=\frac{1+\sqrt5}2,\psi=\frac{1-\sqrt5}2.\\
\end{aligned}
$$

```python
class Solution:
    def fib(self, n: int) -> int:
        return fib(n)

from math import sqrt

def fib(n: int) -> int:
    sqrt5 = sqrt(5)
    phi, psi = (1 + sqrt5) / 2, (1 - sqrt5) / 2
    f = (phi ** n - psi ** n) / (phi - psi)
    return round(f)
```

#### 查表

```python
class Solution:
    def fib(self, n: int) -> int:
        return fib(n)

FIB = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
    610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657,
    46368, 75025, 121393, 196418, 317811, 514229, 832040
]

def fib(n: int) -> int:
    return FIB[n]
```

## 830. 较大分组的位置

[:link: 来源](https://leetcode-cn.com/problems/positions-of-large-groups/)

### 题目

在一个由小写字母构成的字符串 `s` 中，包含由一些连续的相同字符所构成的分组。

例如，在字符串 `s = "abbxxxxzyy"` 中，就含有 `"a"`, `"bb"`, `"xxxx"`, `"z"` 和 `"yy"` 这样的一些分组。

分组可以用区间 `[begin, end]` 表示，其中 `begin` 和 `end` 分别表示该分组的起始和终止位置的下标。上例中的 `"xxxx"` 分组用区间表示为 `[3, 6]`.

我们称所有包含大于或等于三个连续字符的分组为**较大分组**。

找到每一个**较大分组**的区间，按起始位置下标递增顺序排序后，返回结果。

#### 示例

```raw
输入：s = "abbxxxxzzy"
输出：[[3, 6]]
解释："xxxx" 是一个起始于 3 且终止于 6 的较大分组。
```

```raw
输入：s = "abc"
输出：[]
解释："a", "b" 和 "c" 均不是符合要求的较大分组。
```

```raw
输入：s = "abcdddeeeeaabbbcd"
输出：[[3, 5], [6, 9], [12, 14]]
解释：较大分组为 "ddd", "eeee" 和 "bbb".
```

```raw
输入：s = "aba"
输出：[]
```

#### 提示

- `1 <= len(s) <= 1000`;
- `s` 仅含小写英文字母。

### 题解

```python
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        return large_group_positions(s)

from itertools import chain

def large_group_positions(s: str) -> List[List[int]]:
    r, p = [], None
    for i, c in enumerate(chain(s, [None])):
        if p != c:
            if p and i - b >= 3:
                r.append([b, i - 1])
            p, b = c, i
    return r
```
