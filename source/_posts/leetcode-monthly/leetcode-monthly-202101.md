---
title: LeetCode 月报 202101
date: 2021-01-01 00:00:00
tags: [LeetCode]
mathjax: true
---

新年新气象，跑步逃离魔幻的 2020 年。

<!-- more -->

## 605. 种花问题{#leetcode-605}

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

## 1250. 检查「好数组」{#leetcode-1250}

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

数学题。[裴蜀定理](https://zh.wikipedia.org/wiki/貝祖等式)的推广形式：若 $a_i\in\mathbb{Z}$ 且 $d=\gcd(a_i)$, 则关于 $x_i$ 的方程 $\sum_ia_ix_i=m$ 有整数解当且仅当 $d\mid m$, 其中 $i=1, 2, \dots, n$; 特别地，$\sum_ia_ix_i=1$ 有整数解当且仅当 $a_i$ 互质。

```python
class Solution:
    def isGoodArray(self, nums: List[int]) -> bool:
        return is_good_array(nums)

from math import gcd

def is_good_array(nums: List[int]) -> bool:
    return gcd(*nums) == 1
```

## 239. 滑动窗口最大值{#leetcode-239}

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

## 86. 分隔链表{#leetcode-86}

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

## 509. 斐波那契数{#leetcode-509}

[:link: 来源](https://leetcode-cn.com/problems/fibonacci-number/)

### 题目

斐波那契数，通常用 $F(n)$ 表示，形成的序列称为**斐波那契数列**。该数列由 $0$ 和 $1$ 开始，后面的每一项数字都是前面两项数字的和。也就是：

$$
F(n)=\begin{cases}
    0,             & \text{if } n=0, \\
    1,             & \text{if } n=1, \\
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
    F(n)&=\frac{\phi^n-\psi^n}{\phi-\psi}, \\
    \text{where }\phi&=\frac{1+\sqrt5}2, \psi=\frac{1-\sqrt5}2.\\
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

## 830. 较大分组的位置{#leetcode-830}

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

## 399. 除法求值{#leetcode-399}

[:link: 来源](https://leetcode-cn.com/problems/evaluate-division/)

### 题目

给你一个变量对数组 `equations` 和一个实数值数组 `values` 作为已知条件，其中 `equations[i] = [a_i, b_i]` 和 `values[i]` 共同表示等式 `a_i / b_i = values[i]`. 每个 `a_i` 或 `b_i` 是一个表示单个变量的字符串。

另有一些以数组 `queries` 表示的问题，其中 `queries[j] = [c_j, d_j]` 表示第 `j` 个问题，请你根据已知条件找出 `c_j / d_j` 的结果作为答案。

返回**所有问题的答案**。如果存在某个无法确定的答案，则用 `-1.0` 替代这个答案。

#### 注意

输入总是有效的。你可以假设除法运算中不会出现除数为 `0` 的情况，且不存在任何矛盾的结果。

#### 示例

```raw
输入：equations = [["a", "b"], ["b", "c"]], values = [2.0, 3.0], queries = [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]
输出：[6.00000, 0.50000, -1.00000, 1.00000, -1.00000]
解释：
条件：a / b = 2.0, b / c = 3.0;
问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?;
结果：[6.0, 0.5, -1.0, 1.0, -1.0].
```

```raw
输入：equations = [["a", "b"], ["b", "c"], ["bc", "cd"]], values = [1.5, 2.5, 5.0], queries = [["a", "c"], ["c", "b"], ["bc", "cd"], ["cd", "bc"]]
输出：[3.75000, 0.40000, 5.00000, 0.20000]
```

```raw
输入：equations = [["a", "b"]], values = [0.5], queries = [["a", "b"], ["b", "a"], ["a", "c"], ["x", "y"]]
输出：[0.50000, 2.00000, -1.00000, -1.00000]
```

#### 提示

- `1 <= len(equations) <= 20`, `len(equations[i]) == 2`, `1 <= len(a_i), len(b_i) <= 5`;
- `len(values) == len(equations)`, `0.0 < values[i] <= 20.0`;
- `1 <= len(queries) <= 20`, `len(queries[i]) == 2`, `1 <= len(c_j), len(d_j) <= 5`;
- `a_i`, `b_i`, `c_j`, `d_j` 由小写英文字母与数字组成。

### 题解

#### 深度优先搜索

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        return calc_equation(equations, values, queries)

from collections import defaultdict

def calc_equation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    end_value_dict = defaultdict(list)
    for (begin, end), value in zip(equations, values):
        end_value_dict[begin].append((end, value))
        end_value_dict[end].append((begin, 1.0 / value))

    query_values = []
    for query_begin, query_end in queries:
        if not (query_begin in end_value_dict and query_end in end_value_dict):
            query_values.append(-1.0)
            continue
        seen, stack = set([query_begin]), [(query_begin, 1.0)]
        while stack:
            begin, query_value = stack.pop()
            if begin == query_end:
                query_values.append(query_value)
                break
            for end, value in end_value_dict[begin]:
                if end not in seen:
                    seen.add(end)
                    stack.append((end, value * query_value))
        else:
            query_values.append(-1.0)

    return query_values
```

#### 广度优先搜索

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        return calc_equation(equations, values, queries)

from collections import defaultdict, deque

def calc_equation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    end_value_dict = defaultdict(list)
    for (begin, end), value in zip(equations, values):
        end_value_dict[begin].append((end, value))
        end_value_dict[end].append((begin, 1.0 / value))

    query_values = []
    for query_begin, query_end in queries:
        if not (query_begin in end_value_dict and query_end in end_value_dict):
            query_values.append(-1.0)
            continue
        seen, queue = set([query_begin]), deque([(query_begin, 1.0)])
        while queue:
            begin, query_value = queue.popleft()
            if begin == query_end:
                query_values.append(query_value)
                break
            for end, value in end_value_dict[begin]:
                if end not in seen:
                    seen.add(end)
                    queue.append((end, value * query_value))
        else:
            query_values.append(-1.0)

    return query_values
```

## 547. 省份数量{#leetcode-547}

[:link: 来源](https://leetcode-cn.com/problems/number-of-provinces/)

### 题目

有 `n` 个城市，其中一些彼此相连，另一些没有相连。如果城市 `a` 与城市 `b` 直接相连，且城市 `b` 与城市 `c` 直接相连，那么城市 `a` 与城市 `c` 间接相连。

**省份**是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 $n\times n$ 的矩阵 `isConnected`, 其中 `isConnected[i][j] = 1` 表示第 `i` 个城市和第 `j` 个城市直接相连，而 `isConnected[i][j] = 0` 表示二者不直接相连。

返回矩阵中**省份**的数量。

#### 示例

```raw
输入：isConnected = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
输出：2
```

```raw
输入：isConnected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
输出：3
```

### 题解

#### 并查集

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        return find_circle_num(isConnected)

def find(parents: List[int], i: int) -> int:
    if (p := parents[i]) != i:
        parents[i] = find(parents, p)
    return parents[i]

def union(parents: List[int], i: int, j: int):
    parents[find(parents, i)] = find(parents, j)

def find_circle_num(is_connected: List[List[int]]) -> int:
    cities = len(is_connected)
    parents = list(range(cities))
    for i in range(cities):
        for j in range(i + 1, cities):
            if is_connected[i][j]:
                union(parents, i, j)
    return sum(i == p for i, p in enumerate(parents))
```

#### 深度优先搜索

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        return find_circle_num(isConnected)

def find_circle_num(is_connected: List[List[int]]) -> int:
    r, seen = 0, set()
    for i in range(len(is_connected)):
        if i not in seen:
            r += 1
            stack = [i]
            seen.add(i)
            while stack:
                i = stack.pop()
                for j, c in enumerate(is_connected[i]):
                    if c and j not in seen:
                        stack.append(j)
                        seen.add(j)
    return r
```

#### 广度优先搜索

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        return find_circle_num(isConnected)

from collections import deque

def find_circle_num(is_connected: List[List[int]]) -> int:
    r, seen = 0, set()
    for i in range(len(is_connected)):
        if i not in seen:
            r += 1
            queue = deque([i])
            seen.add(i)
            while queue:
                i = queue.popleft()
                for j, c in enumerate(is_connected[i]):
                    if c and j not in seen:
                        queue.append(j)
                        seen.add(j)
    return r
```

## 189. 旋转数组{#leetcode-189}

[:link: 来源](https://leetcode-cn.com/problems/rotate-array/)

### 题目

给定一个数组，将数组中的元素向右移动 `k` 个位置，其中 `k` 是非负数。

#### 示例

```raw
输入：[1, 2, 3, 4, 5, 6, 7] 和 k = 3
输出：[5, 6, 7, 1, 2, 3, 4]
解释：
向右旋转 1 步：[7, 1, 2, 3, 4, 5, 6];
向右旋转 2 步：[6, 7, 1, 2, 3, 4, 5];
向右旋转 3 步：[5, 6, 7, 1, 2, 3, 4].
```

```raw
输入：nums = [-1, -100, 3, 99], k = 2
输出：[3, 99, -1, -100]
解释：
向右旋转 1 步：[99, -1, -100, 3];
向右旋转 2 步：[3, 99, -1, -100].
```

#### 说明

- 尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题；
- 要求使用空间复杂度为 $\mathrm{O}(1)$ 的**原地**算法。

### 题解

#### 直接

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        rotate(nums, k)

def rotate(nums: List[int], k: int) -> None:
    k %= len(nums)
    nums[:] = nums[-k:] + nums[:-k]
```

#### 分组

将数组划分为 `gcd(n, k)` 个模 `k` 循环组，其中 `n = len(nums)`.

例如：`n = 10`, `k = 4`, `p = gcd(n, k) = 2`.
此时只需 `nums[0 -> 4 -> 8 -> 2 -> 6]`, `nums[1 -> 5 -> 9 -> 3 -> 5]` 即可完成旋转。

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        rotate(nums, k)

from math import gcd

def rotate(nums: List[int], k: int) -> None:
    n, p = len(nums), gcd(k, len(nums))
    for i in range(p):
        t = nums[i]
        for _ in range(n // p):
            i = (i + k) % n
            nums[i], t = t, nums[i]
```

### 翻转

三次翻转。

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        rotate(nums, k)

def reverse(nums: List[int], start: int, end: int) -> None:
    i, j = start, end - 1
    while i < j:
        nums[i], nums[j] = nums[j], nums[i]
        i, j = i + 1, j - 1

def rotate(nums: List[int], k: int) -> None:
    n = len(nums)
    s = n - k % n
    reverse(nums, 0, s)
    reverse(nums, s, n)
    reverse(nums, 0, n)
```

## 123. 买卖股票的最佳时机 III{#leetcode-123}

[:link: 来源](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

### 题目

给定一个数组，它的第 `i` 个元素是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成**两笔**交易。

#### 注意

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

#### 示例

```raw
输入：prices = [3, 3, 5, 0, 0, 3, 1, 4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3 - 0 = 3;
随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天（股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4 - 1 = 3.
```

```raw
输入：prices = [1, 2, 3, 4, 5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4.
注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```

```raw
输入：prices = [7, 6, 4, 3, 1]
输出：0 
解释：在这个情况下, 没有交易完成, 所以最大利润为 0.
```

```raw
输入：prices = [1]
输出：0
```

#### 提示

- `1 <= len(prices) <= 1e5`;
- `0 <= prices[i] <= 1e5`.

### 题解

#### 通用

本题是[188. 买卖股票的最佳时机 IV](/leetcode-monthly-202012/#leetcode-188)的特化，可直接套用。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        return max_profit(2, prices)

def max_profit(k: int, prices: List[int]) -> int:
    p = [0] + [float('-inf')] * (min(k, len(prices) // 2) * 2)
    for i, price in enumerate(prices):
        for j in range(1, min(len(p), ((i + 1) // 2 + 1) * 2)):
            p[j] = max(p[j], p[j - 1] + price * (-1 if j % 2 else 1))
    return p[-1]
```

#### 特化

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        return max_profit(prices)

def max_profit(prices: List[int]) -> int:
    cl0 = 0
    op1 = cl1 = op2 = cl2 = float('-inf')
    for price in prices:
        op1 = max(op1, cl0 - price)
        cl1 = max(cl1, op1 + price)
        op2 = max(op2, cl1 - price)
        cl2 = max(cl2, op2 + price)
    return cl2
```

## 228. 汇总区间

[:link: 来源](https://leetcode-cn.com/problems/summary-ranges/)

### 题目

给定一个无重复元素的有序整数数组 nums 。

返回**恰好覆盖数组中所有数字**的**最小有序**区间范围列表。也就是说，`nums` 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 `nums` 的数字 `x`.

列表中的每个区间范围 `[a, b]` 应该按如下格式输出：

- `"a->b"`, 如果 `a != b`;
- `"a"`, 如果 `a == b`.

#### 示例

```raw
输入：nums = [0, 1, 2, 4, 5, 7]
输出：["0->2", "4->5", "7"]
解释：区间范围是
[0, 2] => "0->2",
[4, 5] => "4->5",
[7, 7] => "7".
```

```raw
输入：nums = [0, 2, 3, 4, 6, 8, 9]
输出：["0", "2->4", "6", "8->9"]
解释：区间范围是
[0, 0] => "0",
[2, 4] => "2->4",
[6, 6] => "6",
[8, 9] => "8->9".
```

```raw
输入：nums = []
输出：[]
```

```raw
输入：nums = [-1]
输出：["-1"]
```

```raw
输入：nums = [0]
输出：["0"]
```

#### 提示

- `0 <= len(nums) <= 20`;
- `-2 ** 31 <= nums[i] <= 2 ** 31 - 1`;
- `nums` 中的所有值都**互不相同**；
- `nums` 按升序排列。

### 题解

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        return summary_ranges(nums)

def summary_ranges(nums: List[int]) -> List[str]:
    r = []
    if not nums:
        return r

    s = p = nums[0]
    for n in nums[1:]:
        if n > p + 1:
            r.append(f'{s}->{p}' if p > s else f'{s}')
            s = n
        p = n
    r.append(f'{s}->{p}' if p > s else f'{s}')
    return r
```

## 1202. 交换字符串中的元素

[:link: 来源](https://leetcode-cn.com/problems/smallest-string-with-swaps/)

### 题目

给你一个字符串 `s`, 以及该字符串中的一些「索引对」数组 `pairs`, 其中 `pairs[i] = [a, b]` 表示字符串中的两个索引（编号从 `0` 开始）。

你可以**任意多次交换**在 `pairs` 中任意一对索引处的字符。

返回在经过若干次交换后，`s` 可以变成的按字典序最小的字符串。

#### 示例

```raw
输入：s = "dcab", pairs = [[0, 3], [1, 2]]
输出："bacd"
解释： 
交换 s[0] 和 s[3], s = "bcad";
交换 s[1] 和 s[2], s = "bacd".
```

```raw
输入：s = "dcab", pairs = [[0, 3], [1, 2], [0, 2]]
输出："abcd"
解释：
交换 s[0] 和 s[3], s = "bcad";
交换 s[0] 和 s[2], s = "acbd";
交换 s[1] 和 s[2], s = "abcd".
```

```raw
输入：s = "cba", pairs = [[0, 1], [1, 2]]
输出："abc"
解释：
交换 s[0] 和 s[1], s = "bca";
交换 s[1] 和 s[2], s = "bac";
交换 s[0] 和 s[1], s = "abc".
```

#### 提示

- `1 <= len(s) <= 1e5`;
- `0 <= len(pairs) <= 1e5`;
- `0 <= pairs[i][0], pairs[i][1] < len(s)`;
- `s` 中只含有小写英文字母。

### 题解

并查集。先求连通分支，再排序 `s` 的每个连通分支子序列。

```python
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        return smallest_string_with_swaps(s, pairs)

from collections import defaultdict

def find(parents: List[int], i: int) -> int:
    if (p := parents[i]) != i:
        parents[i] = find(parents, p)
    return parents[i]

def union(parents: List[int], i: int, j: int):
    parents[find(parents, i)] = find(parents, j)

def smallest_string_with_swaps(s: str, pairs: List[List[int]]) -> str:
    parents = list(range(len(s)))
    for i, j in pairs:
        union(parents, i, j)

    subs = defaultdict(list)
    for i in range(len(s)):
        subs[find(parents, i)].append(i)

    r = [...] * len(s)
    for ss in subs.values():
        for i, j in zip(ss, sorted(ss, key=s.__getitem__)):
            r[i] = s[j]
    return ''.join(r)
```

## 1203. 项目管理

### 题目

公司共有 `n` 个项目和 `m` 个小组，每个项目要不无人接手，要不就由 `m` 个小组之一负责。

`group[i]` 表示第 `i` 个项目所属的小组，如果这个项目目前无人接手，那么 `group[i]` 就等于 `-1`.（项目和小组都是从零开始编号的）小组可能存在没有接手任何项目的情况。

请你帮忙按要求安排这些项目的进度，并返回排序后的项目列表：

- 同一小组的项目，排序后在列表中彼此相邻。
- 项目之间存在一定的依赖关系，我们用一个列表 `beforeItems` 来表示，其中 `beforeItems[i]` 表示在进行第 `i` 个项目前（位于第 `i` 个项目左侧）应该完成的所有项目。

如果存在多个解决方案，只需要返回其中任意一个即可。如果没有合适的解决方案，就请返回一个**空列表**。

#### 示例

```raw
输入：n = 8, m = 2, group = [-1, -1, 1, 0, 0, 1, 0, -1], beforeItems = [[], [6], [5], [6], [3, 6], [], [], []]
输出：[6, 3, 4, 1, 5, 2, 0, 7]
```

```raw
输入：n = 8, m = 2, group = [-1, -1, 1, 0, 0, 1, 0, -1], beforeItems = [[], [6], [5], [6], [3], [], [4], []]
输出：[]
解释：与示例 1 大致相同，但是在排序后的列表中，4 必须放在 6 的前面。
```

#### 提示

- `1 <= m <= n <= 3e4`;
- `len(group) == len(beforeItems) == n`, `0 <= len(beforeItems[i]) <= n - 1`;
- `0 <= beforeItems[i][j] <= n - 1`, `i != beforeItems[i][j]`;
- `-1 <= group[i] <= m - 1`;
- `beforeItems[i]` 不含重复元素。

### 题解

两次拓扑排序。

```python
class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        return sort_items(n, m, group, beforeItems)

from graphlib import TopologicalSorter, CycleError
from itertools import count

def sort_items(
    n: int, m: int, groups: List[int],
    before_items: List[List[int]]
) -> List[int]:
    groups = [
        group if group != -1 else fake_group
        for fake_group, group in zip(count(m), groups)
    ]
    all_groups = set(groups)

    before_groups = {group: set() for group in all_groups}
    before_items_inner = {item: set() for item in range(n)}
    for item, group, before_items_per_item in zip(count(), groups, before_items):
        for before_item in before_items_per_item:
            if (before_group := groups[before_item]) != group:
                before_groups[group].add(before_group)
            else:
                before_items_inner[item].add(before_item)

    try:
        sorted_items_inner = {group: [] for group in all_groups}
        for item in TopologicalSorter(before_items_inner).static_order():
            sorted_items_inner[groups[item]].append(item)

        sorted_items = []
        for group in TopologicalSorter(before_groups).static_order():
            sorted_items.extend(sorted_items_inner[group])
        return sorted_items
    except CycleError:
        return []
```
