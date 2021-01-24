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
- 输入的数组长度范围为 $\left[1,20000\right]$;
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

def union(parents: List[int], i: int, j: int) -> None:
    parents[find(parents, j)] = find(parents, i)

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

from math import inf

def max_profit(k: int, prices: List[int]) -> int:
    p = [0] + [-inf] * (min(k, len(prices) // 2) * 2)
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

from math import inf

def max_profit(prices: List[int]) -> int:
    cl0 = 0
    op1 = cl1 = op2 = cl2 = -inf
    for price in prices:
        op1 = max(op1, cl0 - price)
        cl1 = max(cl1, op1 + price)
        op2 = max(op2, cl1 - price)
        cl2 = max(cl2, op2 + price)
    return cl2
```

## 228. 汇总区间{#leetcode-228}

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

## 1202. 交换字符串中的元素{#leetcode-1202}

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

def union(parents: List[int], i: int, j: int) -> None:
    parents[find(parents, j)] = find(parents, i)

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

## 1203. 项目管理{#leetcode-1203}

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

## 684. 冗余连接{#leetcode-684}

[:link: 来源](https://leetcode-cn.com/problems/redundant-connection/)

### 题目

在本问题中, 树指的是一个连通且无环的无向图。

输入一个图，该图由一个有着 `N` 个节点（节点值不重复）的树及一条附加的边构成。附加的边的两个顶点包含在 `1` 到 `N` 中间，这条附加的边不属于树中已存在的边。

结果图是一个以边组成的二维数组。每一个边的元素是一对 `[u, v]`, 满足 `u < v`, 表示连接顶点 `u` 和 `v` 的无向图的边。

返回一条可以删去的边，使得结果图是一个有着 `N` 个节点的树。如果有多个答案，则返回二维数组中最后出现的边。答案边 `[u, v]` 应满足相同的格式 `u < v`.

#### 示例

```raw
输入: [[1, 2], [1, 3], [2, 3]]
输出: [2, 3]
解释: 给定的无向图为
  1
 / \
2 - 3
```

```raw
输入: [[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]
输出: [1, 4]
解释: 给定的无向图为
5 - 1 - 2
    |   |
    4 - 3
```

#### 注意

- 输入的二维数组大小在 `3` 到 `1000`;
- 二维数组中的整数在 `1` 到 `N` 之间，其中 `N` 是输入数组的大小。

### 题解

并查集。

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        return find_redundant_connection(edges)

def find(parents: List[int], i: int) -> int:
    if (p := parents[i]) != i:
        parents[i] = find(parents, p)
    return parents[i]

def union(parents: List[int], i: int, j: int) -> None:
    parents[find(parents, j)] = find(parents, i)

def find_redundant_connection(edges: List[List[int]]) -> List[int]:
    parents = list(range(len(edges) + 1))

    for i, j in edges:
        if find(parents, i) == find(parents, j):
            return [i, j]
        union(parents, i, j)
    return []
```

## 1018. 可被 5 整除的二进制前缀{#leetcode-1018}

[:link: 来源](https://leetcode-cn.com/problems/binary-prefix-divisible-by-5/)

### 题目

给定由若干 `0` 和 `1` 组成的数组 `A`. 我们定义 $N_i=\overline{A_0A_1\dots A_k\dots A_i}_{\left(2\right)}$, 即由 `A[0]` 到 `A[i]` 组成的第 `i` 个子数组被解释为一个二进制数（从最高有效位到最低有效位）。

返回布尔值列表 `answer`, 只有当 $N_i$ 可以被 `5` 整除时，答案 `answer[i]` 为 `true`, 否则为 `false`.

#### 示例

```raw
输入：[0, 1, 1]
输出：[true, false, false]
解释：输入数字为 0, 01, 011；也就是十进制中的 0, 1, 3.
只有第一个数可以被 5 整除，因此 answer[0] 为真。
```

```raw
输入：[1, 1, 1]
输出：[false, false, false]
```

```raw
输入：[0, 1, 1, 1, 1, 1]
输出：[true, false, false, false, true, false]
```

```raw
输入：[1, 1, 1, 0, 1]
输出：[false, false, false, false, false]
```

#### 提示

- `1 <= len(A) <= 30000`;
- `A[i] 为 0 或 1`.

### 题解

#### 循环

```python
class Solution:
    def prefixesDivBy5(self, A: List[int]) -> List[bool]:
        return prefixes_div_by_5(A)

def prefixes_div_by_5(a: List[int]) -> List[bool]:
    r, n = [], 0
    for ai in a:
        n = (2 * n + ai) % 5
        r.append(n == 0)
    return r
```

#### 简化

```python
class Solution:
    def prefixesDivBy5(self, A: List[int]) -> List[bool]:
        return prefixes_div_by_5(A)

from itertools import accumulate

def prefixes_div_by_5(a: List[int]) -> List[bool]:
    return [n == 0 for n in accumulate(
        a, lambda n, ai: (n * 2 + ai) % 5
    )]
```

## 947. 移除最多的同行或同列石头{#leetcode-947}

[:link: 来源](https://leetcode-cn.com/problems/most-stones-removed-with-same-row-or-column/)

### 题目

`n` 块石头放置在二维平面中的一些整数坐标点上。每个坐标点上最多只能有一块石头。

如果一块石头的**同行或者同列**上有其他石头存在，那么就可以移除这块石头。

给你一个长度为 `n` 的数组 `stones`, 其中 `stones[i] = [xi, yi]` 表示第 `i` 块石头的位置，返回**可以移除的石子**的最大数量。

#### 示例

```raw
输入：stones = [[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]]
输出：5
解释：一种移除 5 块石头的方法如下所示：
1. 移除石头 [2, 2], 因为它和 [2, 1] 同行；
2. 移除石头 [2, 1], 因为它和 [0, 1] 同列；
3. 移除石头 [1, 2], 因为它和 [1, 0] 同行；
4. 移除石头 [1, 0], 因为它和 [0, 0] 同列；
5. 移除石头 [0, 1], 因为它和 [0, 0] 同行；
石头 [0, 0] 不能移除，因为它没有与另一块石头同行、列。
```

```raw
输入：stones = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]
输出：3
解释：一种移除 3 块石头的方法如下所示：
1. 移除石头 [2, 2], 因为它和 [2, 0] 同行；
2. 移除石头 [2, 0], 因为它和 [0, 0] 同列；
3. 移除石头 [0, 2], 因为它和 [0, 0] 同行；
石头 [0, 0] 和 [1, 1] 不能移除，因为它们没有与另一块石头同行、列。
```

```raw
输入：stones = [[0, 0]]
输出：0
解释：[0, 0] 是平面上唯一一块石头，所以不可以移除它。
```

#### 提示

- `1 <= len(stones) <= 1000`;
- `0 <= xi, yi <= 1e4`;
- 不会有两块石头放在同一个坐标点上。

### 题解

并查集。同行、同列的点相连，最终最少剩余连通分支数个点（每一连通分支按任意遍历序的倒序移除点），最终结果（最多可移除的点数）为总点数减连通分支数。

```python
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        return remove_stones(stones)

from collections import defaultdict

def find(parents: List[int], i: int) -> int:
    if (p := parents[i]) != i:
        parents[i] = find(parents, p)
    return parents[i]

def union(parents: List[int], i: int, j: int) -> None:
    parents[find(parents, j)] = find(parents, i)

def remove_stones(stones: List[List[int]]) -> int:
    lines = defaultdict(list)
    for i, (x, y) in enumerate(stones):
        lines[(x, ...)].append(i)
        lines[(..., y)].append(i)

    parents = list(range(len(stones)))
    for points in lines.values():
        for point in points[1:]:
            union(parents, points[0], point)

    return sum(i != p for i, p in enumerate(parents))
```

## 1232. 缀点成线{#leetcode-1232}

[:link: 来源](https://leetcode-cn.com/problems/check-if-it-is-a-straight-line/)

### 题目

在一个 $xOy$ 坐标系中有一些点，我们用数组 `coordinates` 来分别记录它们的坐标，其中 `coordinates[i] = [x, y]` 表示横坐标为 `x`, 纵坐标为 `y` 的点。

请你来判断，这些点是否在该坐标系中属于同一条直线上，是则返回 `true`, 否则请返回 `false`.

#### 示例

```raw
输入：coordinates = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
输出：true
```

```raw
输入：coordinates = [[1, 1], [2, 2], [3, 4], [4, 5], [5, 6], [7, 7]]
输出：false
```

#### 提示

- `2 <= len(coordinates) <= 1000`, `len(coordinates[i]) == 2`;
- `-1e4 <= coordinates[i][0], coordinates[i][1] <= 1e4`;
- `coordinates` 中不含重复的点。

### 题解

数学题。

$$
\begin{aligned}
&\left\{\left(x_0,y_0\right),\left(x_1,y_1\right),\left(x_2,y_2\right)\right\}\text{ are collinear}\\
\Leftrightarrow
&\left(x_1-x_0\right)\left(y_2-y_0\right)=\left(y_1-y_0\right)\left(x_2-x_0\right)\\
\Leftrightarrow
&\left(y_0-y_1\right)x_2+\left(x_1-x_0\right)y_2+\left(x_0y_1-x_1y_0\right)=0.
\end{aligned}
$$

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        return check_straight_line(coordinates)

def check_straight_line(coordinates: List[List[int]]) -> bool:
    (x0, y0), (x1, y1) = coordinates[:2]
    dx, dy = x1 - x0, y1 - y0
    return all(dx * (y - y0) == dy * (x - x0) for x, y in coordinates[2:])
```

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        return check_straight_line(coordinates)

def check_straight_line(coordinates: List[List[int]]) -> bool:
    (x0, y0), (x1, y1) = coordinates[:2]
    a, b, c = y0 - y1, x1 - x0, x0 * y1 - x1 * y0
    return all(a * x + b * y + c == 0 for x, y in coordinates[2:])
```

## 721. 账户合并{#leetcode-721}

[:link: 来源](https://leetcode-cn.com/problems/accounts-merge/)

### 题目

给定一个列表 `accounts`, 每个元素 `accounts[i]` 是一个字符串列表，其中第一个元素 `accounts[i][0]` 是**名称**，其余元素是 `emails` 表示该账户的邮箱地址。

现在，我们想合并这些账户。如果两个账户都有一些共同的邮箱地址，则两个账户必定属于同一个人。请注意，即使两个账户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。一个人最初可以拥有任意数量的账户，但其所有账户都具有相同的名称。

合并账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是按顺序排列的邮箱地址。账户本身可以以任意顺序返回。

#### 示例

```raw
输入：accounts = [
    ["John", "johnsmith@mail.com", "john00@mail.com"],
    ["John", "johnnybravo@mail.com"],
    ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
    ["Mary", "mary@mail.com"]
]
输出：[
    ["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"],
    ["John", "johnnybravo@mail.com"],
    ["Mary", "mary@mail.com"]
]
解释：
第一个和第三个 John 是同一个人，因为他们有共同的邮箱地址 "johnsmith@mail.com"; 
第二个 John 和 Mary 是不同的人，因为他们的邮箱地址没有被其他帐户使用。
可以以任何顺序返回这些列表，例如答案 [["Mary"，"mary@mail.com"]，["John"，"johnnybravo@mail.com"]，
["John"，"john00@mail.com"，"john_newyork@mail.com"，"johnsmith@mail.com"]] 也是正确的。
```

#### 提示

- `1 <= len(accounts) <= 1000`;
- `1 <= len(accounts[i]) <= 10`;
- `1 <= len(accounts[i][j]) <= 30`.

### 题解

并查集。

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        return accounts_merge(accounts)

def find(parents: List[int], i: int) -> int:
    if (p := parents[i]) != i:
        parents[i] = find(parents, p)
    return parents[i]

def union(parents: List[int], i: int, j: int) -> None:
    parents[find(parents, j)] = find(parents, i)

def accounts_merge(accounts: List[List[str]]) -> List[List[str]]:
    parents, email2j = list(range(len(accounts))), {}
    for i, (name, *emails) in enumerate(accounts):
        for email in emails:
            union(parents, i, email2j.setdefault(email, i))

    p2account = {}
    for i, (name, *emails) in enumerate(accounts):
        p2account.setdefault(find(parents, i), (name, set()))[1].update(emails)
    return [[name, *sorted(emails)] for name, emails in p2account.values()]
```

## 1584. 连接所有点的最小费用{#leetcode-1584}

[:link: 来源](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/)

### 题目

给你一个 `points` 数组，表示 $xOy$ 平面上的一些点，其中 `points[i] = [xi, yi]`, 代表点 $\left(x_i,y_i\right)$.

连接点 $\left(x_i,y_i\right)$ 和点 $\left(x_j,y_j\right)$ 的费用为它们之间的**曼哈顿距离**

$$
d_{i,j}=\left|x_i-x_j\right|+\left|y_i-y_j\right|
$$

其中 $\left|v\right|$ 表示 $v$ 的绝对值。

请你返回将所有点连接的最小总费用。只有任意两点之间**有且仅有**一条简单路径时，才认为所有点都已连接。

#### 示例

```raw
输入：points = [[0, 0], [2, 2], [3, 10], [5, 2], [7, 0]]
输出：20
解释：我们可以按照上图所示连接所有点得到最小总费用，总费用为 20. 注意到任意两个点之间只有唯一一条路径互相到达。
```

```raw
输入：points = [[3, 12], [-2, 5], [-4, 1]]
输出：18
```

```raw
输入：points = [[0, 0], [1, 1], [1, 0], [-1, 1]]
输出：4
```

```raw
输入：points = [[-1000000, -1000000], [1000000, 1000000]]
输出：4000000
```

```raw
输入：points = [[0, 0]]
输出：0
```

#### 提示

- `1 <= len(points) <= 1000`;
- `-1e6 <= xi, yi <= 1e6`;
- 所有点 `(xi, yi)` 两两不同。

### 题解

#### Kruskal

并查集记录连通性。

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        return min_cost_connect_points(points)

from itertools import combinations

def find(parents: List[int], i: int) -> int:
    if (p := parents[i]) != i:
        parents[i] = find(parents, p)
    return parents[i]

def union(parents: List[int], i: int, j: int) -> None:
    parents[find(parents, j)] = find(parents, i)

def min_cost_connect_points(points: List[List[int]]) -> int:
    r, parents, edges = 0, list(range(len(points))), sorted(
        (abs(xi - xj) + abs(yi - yj), i, j)
        for (i, (xi, yi)), (j, (xj, yj))
        in combinations(enumerate(points), 2)
    )
    for d, i, j in edges:
        if find(parents, i) != find(parents, j):
            union(parents, i, j)
            r += d
    return r
```

#### Prim

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        return min_cost_connect_points(points)

from operator import itemgetter
from math import inf, isfinite

def min_cost_connect_points(points: List[List[int]]) -> int:
    r, p2d = 0, {(x, y): inf for x, y in points}
    while p2d:
        mp, d = min(p2d.items(), key=itemgetter(1))
        del p2d[mp]
        if isfinite(d):
            r += d
        for p in p2d:
            p2d[p] = min(p2d[p], abs(mp[0] - p[0]) + abs(mp[1] - p[1]))
    return r
```

## 628. 三个数的最大乘积{#leetcode-628}

[:link: 来源](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/)

### 题目

给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

#### 示例

```raw
输入：[1, 2, 3]
输出：6
```

```raw
输入：[1, 2, 3, 4]
输出：24
```

#### 注意

- 给定的整型数组长度范围是 $\left[3,{10}^4\right]$, 数组中所有的元素范围是 $\left[-1000,1000\right]$;
- 输入的数组中任意三个数的乘积不会超出 32 位有符号整数的范围。

### 题解

最大乘积在**最大的三个元素**或**最小的两个元素和最大的元素**取得。

#### 排序

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        return maximum_product(nums)

def maximum_product(nums: List[int]) -> int:
    sorted_nums = sorted(nums)
    (a, b), (x, y, z) = sorted_nums[:2], sorted_nums[-3:]
    return max(a * b * z, x * y * z)
```

#### 扫描

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        return maximum_product(nums)

from math import inf

def maximum_product(nums: List[int]) -> int:
    a = b = inf
    x = y = z = -inf

    for n in nums:
        if n < a:
            a, b = n, a
        elif n < b:
            b = n

        if n > z:
            x, y, z = y, z, n
        elif n > y:
            x, y = y, n
        elif n > x:
            x = n

    return max(a * b * z, x * y * z)
```

## 989. 数组形式的整数加法{#leetcode-989}

[:link: 来源](https://leetcode-cn.com/problems/add-to-array-form-of-integer/)

### 题目

对于非负整数 `X` 而言，`X` 的数组形式是每位数字按从左到右的顺序形成的数组。例如，如果 `X = 1231`, 那么其数组形式为 `[1, 2, 3, 1]`.

给定非负整数 `X` 的数组形式 `A`, 返回整数 `X + K` 的数组形式。

#### 示例

```raw
输入：A = [1, 2, 0, 0], K = 34
输出：[1, 2, 3, 4]
解释：1200 + 34 = 1234
```

```raw
输入：A = [2, 7, 4], K = 181
输出：[4, 5, 5]
解释：274 + 181 = 455
```

```raw
输入：A = [2, 1, 5], K = 806
输出：[1, 0, 2, 1]
解释：215 + 806 = 1021
```

```raw
输入：A = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9], K = 1
输出：[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
解释：9999999999 + 1 = 10000000000
```

#### 提示

- `1 <= len(A) <= 10000`;
- `0 <= A[i] <= 9`;
- `0 <= K <= 10000`;
- 如果 `len(A) > 1`, 那么 `A[0] != 0`.

### 题解

#### 列表

```python
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        return add_to_array_form(A, K)

def add_to_array_form(a: List[int], k: int) -> List[int]:
    r = []
    for i in reversed(a):
        k, b = divmod(i + k, 10)
        r.append(b)
    while k:
        k, b = divmod(k, 10)
        r.append(b)
    r.reverse()
    return r
```

#### 双端队列

```python
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        return add_to_array_form(A, K)

from collections import deque

def add_to_array_form(a: List[int], k: int) -> List[int]:
    r = deque()
    for i in reversed(a):
        k, b = divmod(i + k, 10)
        r.appendleft(b)
    while k:
        k, b = divmod(k, 10)
        r.appendleft(b)
    return r
```

## 1319. 连通网络的操作次数{#leetcode-1319}

[:link: 来源](https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/)

### 题目

用以太网线缆将 `n` 台计算机连接成一个网络，计算机的编号从 `0` 到 `n - 1`. 线缆用 `connections` 表示，其中 `connections[i] = [a, b]` 连接了计算机 `a` 和 `b`.

网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。

给你这个计算机网络的初始布线 `connections`, 你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 `-1`.

#### 示例

```raw
输入：n = 4, connections = [[0, 1], [0, 2], [1, 2]]
输出：1
解释：拔下计算机 1 和 2 之间的线缆，并将它插到计算机 1 和 3 上。
```

```raw
输入：n = 6, connections = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
输出：2
```

```raw
输入：n = 6, connections = [[0, 1], [0, 2], [0, 3], [1, 2]]
输出：-1
解释：线缆数量不足。
```

```raw
输入：n = 5, connections = [[0, 1], [0, 2], [3, 4], [2, 3]]
输出：0
```

#### 提示

- `1 <= n <= 1e5`;
- `1 <= len(connections) <= min(n * (n - 1) / 2, 1e5)`;
- `len(connections[i]) == 2`;
- `0 <= connections[i][0], connections[i][1] < n`;
- `connections[i][0] != connections[i][1]`;
- 没有重复的连接；两台计算机不会通过多条线缆连接。

### 题解

并查集。每次操作可合并两个连通分支，结果即连通分支数减一；若线缆数少于 `n - 1`, 则线缆数量不足。

```python
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        return make_connected(n, connections)

def find(parents: List[int], i: int) -> int:
    if (p := parents[i]) != i:
        parents[i] = find(parents, p)
    return parents[i]

def union(parents: List[int], i: int, j: int) -> None:
    parents[find(parents, j)] = find(parents, i)

def make_connected(n: int, connections: List[List[int]]) -> int:
    if len(connections) < n - 1:
        return -1

    parents = list(range(n))
    for i, j in connections:
        union(parents, i, j)
    return sum(find(parents, i) == i for i in range(n)) - 1
```

## 674. 最长连续递增序列{#leetcode-674}

[:link: 来源](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)

### 题目

给定一个未经排序的整数数组，找到最长且**连续递增的子序列**，并返回该序列的长度。

连续递增的子序列 可以由两个下标 `l < r` 确定，如果对于每个 `l <= i < r - 1`, 都有 `nums[i] < nums[i + 1]`, 那么子序列 `nums[l:r]` 就是连续递增子序列。

#### 示例

```raw
输入：nums = [1, 3, 5, 4, 7]
输出：3
解释：最长连续递增序列是 [1, 3, 5], 长度为 3. 尽管 [1, 3, 5, 7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 
```

```raw
输入：nums = [2, 2, 2, 2, 2]
输出：1
解释：最长连续递增序列是 [2], 长度为 1.
```

#### 提示

- `0 <= len(nums) <= 1e4`;
- `-1e9 <= nums[i] <= 1e9`.

### 题解

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        return find_length_of_lcis(nums)

from operator import sub

def find_length_of_lcis(nums: List[int]) -> int:
    if not nums:
        return 0

    m, r = 1, 0
    for d in map(sub, nums[1:], nums):
        if d > 0:
            m += 1
        else:
            r = max(m, r)
            m = 1
    return max(m, r)
```
