---
title: LeetCode 月报 202105
date: 2021-05-01 00:00:00
tags: [LeetCode]
mathjax: true
---

劳逸结合、快乐生活。

<!-- more -->

## 690. 员工的重要性{#leetcode-698}

[:link: 来源](https://leetcode-cn.com/problems/employee-importance/)

### 题目

给定一个保存员工信息的数据结构，它包含了员工**唯一的 `id`**、**重要度**和**直系下属的 `id`**。

比如，员工 `1` 是员工 `2` 的领导，员工 `2` 是员工 `3` 的领导。他们相应的重要度为 `15`、`10`、`5`。那么员工 `1` 的数据结构是 `[1, 15, [2]]`，员工 `2` 的数据结构是 `[2, 10, [3]]`，员工 `3` 的数据结构是 `[3, 5, []]`。注意虽然员工 `3` 也是员工 `1` 的一个下属，但是由于**并不是直系**下属，因此没有体现在员工 `1` 的数据结构中。

现在输入一个公司的所有员工信息，以及单个员工 `id`，返回这个员工和他所有下属的重要度之和。

#### 示例

```raw
输入：[[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
输出：11
解释：员工 1 自身的重要度是 5，他有两个直系下属 2 和 3，而且 2 和 3 的重要度均为 3。因此员工 1 的总重要度是 5 + 3 + 3 = 11。
```

#### 提示

- 一个员工最多有一个**直系**领导，但是可以有多个**直系**下属；
- 员工数量不超过 `2000`。

### 题解

深度优先搜索。

```python Python
"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution:
    def getImportance(self, employees: list['Employee'], id: int) -> int:
        return get_importance(employees, id)

def get_importance(employees: list['Employee'], id: int) -> int:
    employees = {employee.id: employee for employee in employees}
    stack, importance = [id], 0
    while stack:
        employee = employees[stack.pop()]
        importance += employee.importance
        stack += employee.subordinates
    return importance
```

## 554. 砖墙{#leetcode-554}

[:link: 来源](https://leetcode-cn.com/problems/brick-wall/)

### 题目

你的面前有一堵矩形的、由 `n` 行砖块组成的砖墙。这些砖块高度相同（也就是一个单位高）但是宽度不同。每一行砖块的宽度之和应该相等。

你现在要画一条**自顶向下**的、穿过**最少**砖块的垂线。如果你画的线只是从砖块的边缘经过，就不算穿过这块砖。你不能沿着墙的两个垂直边缘之一画线，这样显然是没有穿过一块砖的。

给你一个二维数组 `wall`，该数组包含这堵墙的相关信息。其中，`wall[i]` 是一个代表从左至右每块砖的宽度的数组。你需要找出怎样画才能使这条线**穿过的砖块数量最少**，并且返回**穿过的砖块数量**。

#### 示例

```raw
输入：wall = [[1, 2, 2, 1], [3, 1, 2], [1, 3, 2], [2, 4], [3, 1, 2], [1, 3, 1, 1]]
输出：2
```

```raw
输入：wall = [[1], [1], [1]]
输出：3
```

#### 提示

- `n == len(wall)`；
- `1 <= n <= 1e4`；
- `1 <= len(wall[i]) <= 1e4`；
- `1 <= sum(len(row) for row in wall[i]) <= 2e4`
- 对于每一行 `i`，`sum(wall[i])` 应当是相同的；
- `1 <= wall[i][j] <= 2 ** 31 - 1`。

### 题解

```rust Rust
impl Solution {
    pub fn least_bricks(wall: Vec<Vec<i32>>) -> i32 {
        let wall = wall.iter().map(|row| row.iter().map(|&brick| brick as usize));
        least_bricks(wall) as i32
    }
}

use std::collections::HashMap;

pub fn least_bricks<W, R>(wall: W) -> usize
where
    W: IntoIterator<Item = R>,
    R: IntoIterator<Item = usize>,
{
    let (mut bounds, mut count) = (HashMap::new(), 0);
    for row in wall {
        let mut bound = 0;
        for brick in row {
            *bounds.entry(bound).or_insert(0) += 1;
            bound += brick;
        }
        count += 1;
    }
    bounds.remove(&0);
    count - bounds.values().max().copied().unwrap_or(0)
}
```

```python Python
class Solution:
    def leastBricks(self, wall: list[list[int]]) -> int:
        return least_bricks(wall)

from collections import Counter
from itertools import accumulate, chain

def least_bricks(wall: list[list[int]]) -> int:
    bounds = Counter(chain.from_iterable(accumulate(row[:-1]) for row in wall))
    return len(wall) - max(bounds.values(), default=0)
```

## 7. 整数反转{#leetcode-7}

[:link: 来源](https://leetcode-cn.com/problems/reverse-integer/submissions/)

### 题目

给你一个 32 位的有符号整数 `x`，返回将 `x` 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围，则返回 `0`。

假设环境不允许存储 64 位整数（有符号或无符号）。

#### 示例

```raw
输入：x = 123
输出：321
```

```raw
输入：x = -123
输出：-321
```

```raw
输入：x = 120
输出：21
```

```raw
输入：x = 0
输出：0
```

#### 提示

- `-2 ** 31 <= x <= 2 ** 31 - 1`。

### 题解

```rust Rust
impl Solution {
    pub fn reverse(x: i32) -> i32 {
        checked_reverse(x).unwrap_or(0)
    }
}

pub fn checked_reverse(mut x: i32) -> Option<i32> {
    let mut r: i32 = 0;
    while x != 0 {
        r = r.checked_mul(10)?.checked_add(x % 10)?;
        x /= 10;
    }
    r.into()
}
```

## 1720. 解码异或后的数组{#leetcode-1720}

[:link: 来源](https://leetcode-cn.com/problems/decode-xored-array/)

### 题目

**未知**整数数组 `arr` 由 `n` 个非负整数组成。

经编码后变为长度为 `n - 1` 的另一个整数数组 `encoded`，其中 `encoded[i] = arr[i] ^ arr[i + 1]`。例如，`arr = [1, 0, 2, 1]` 经编码后得到 `encoded = [1, 2, 3]`。

给你编码后的数组 `encoded` 和原数组 `arr` 的第一个元素 `first`，也即 `arr[0]`。

请解码返回原数组 `arr`。可以证明答案存在并且是唯一的。

#### 示例

```raw
输入：encoded = [1, 2, 3], first = 1
输出：[1, 0, 2, 1]
解释：若 arr = [1, 0, 2, 1]，那么 first = 1 且 encoded = [1 ^ 0, 0 ^ 2, 2 ^ 1] = [1, 2, 3]
```

```raw
输入：encoded = [6, 2, 7, 3], first = 4
输出：[4, 2, 0, 7, 4]
```

#### 提示

- `2 <= n <= 1e4`；
- `encoded.length == n - 1`；
- `0 <= encoded[i] <= 1e5`；
- `0 <= first <= 1e5`。

### 题解

```rust Rust
impl Solution {
    pub fn decode(encoded: Vec<i32>, first: i32) -> Vec<i32> {
        let (mut decoded, mut last) = (Vec::with_capacity(encoded.len()), first);
        decoded.push(first);
        for n in encoded {
            last ^= n;
            decoded.push(last);
        }
        decoded
    }
}
```

```rust Rust
impl Solution {
    pub fn decode(encoded: Vec<i32>, first: i32) -> Vec<i32> {
        std::iter::once(first)
            .chain(encoded.into_iter().scan(first, |st, n| {
                *st ^= n;
                Some(*st)
            }))
            .collect()
    }
}
```

```python Python
class Solution:
    def decode(self, encoded: list[int], first: int) -> list[int]:
        return decode(encoded, first)

from operator import xor
from itertools import accumulate

def decode(encoded: list[int], first: int) -> list[int]:
    return list(accumulate(encoded, xor, initial=first))
```

## 1486. 数组异或操作{#leetcode-1486}

[:link: 来源](https://leetcode-cn.com/problems/xor-operation-in-an-array/)

### 题目

给你两个整数，`n` 和 `start`。

数组 `nums` 定义为：`nums[i] = start + 2 * i`（下标从 `0` 开始）且 `n == len(nums)`。

请返回 `nums` 中所有元素按位异或后得到的结果。

#### 示例

```raw
输入：n = 5, start = 0
输出：8
解释：数组 nums 为 [0, 2, 4, 6, 8]，其中 0 ^ 2 ^ 4 ^ 6 ^ 8 = 8。
```

```raw
输入：n = 4, start = 3
输出：8
解释：数组 nums 为 [3, 5, 7, 9]，其中 3 ^ 5 ^ 7 ^ 9 = 8。
```

```raw
输入：n = 1, start = 7
输出：7
```

```raw
输入：n = 10, start = 5
输出：2
```

#### 提示

- `1 <= n <= 1e3`；
- `0 <= start <= 1e3`；
- `n == len(nums)`。

### 题解

#### 模拟

```rust Rust
impl Solution {
    pub fn xor_operation(n: i32, start: i32) -> i32 {
        xor_operation(n as usize, start as usize) as i32
    }
}

use std::iter::successors;
use std::ops::BitXor;

pub fn xor_operation(n: usize, start: usize) -> usize {
    successors(start.into(), |p| p.checked_add(2))
        .take(n)
        .fold(0, BitXor::bitxor)
}
```

#### 数学

首先，将原异或和式转变为计算连续整数的异或和，即

$$
\bigoplus_{k=0}^{n-1}\left(start+2k\right)=
\overbrace{\bigoplus_{k=0}^{n-1}\left(s+k\right)}^{\text{high bits }r}\times2+
\overbrace{\left(start\bmod2\right)\left(n\bmod2\right)}^{\text{low bit }e},
$$

其中 $s=\left\lfloor\frac{start}{2}\right\rfloor$。这样一来，可以利用异或的对合律可知

$$
r=\left(\bigoplus_{k=0}^{s-1}k\right)\oplus\left(\bigoplus_{k=0}^{s+n-1}k\right).
$$

将 $\bigoplus_{k=0}^{n-1}k$ 记作 $f\left(n\right)$，利用异或的如下性质

$$
\forall i\in\mathbb{N},\,\bigoplus_{k=0}^3\left(4i+k\right)=0,
$$

可以快速计算

$$
f\left(n\right)=\begin{cases}
    0,   & \text{if } n\bmod4=0, \\
    n-1, & \text{if } n\bmod4=1, \\
    1,   & \text{if } n\bmod4=2, \\
    n,   & \text{if } n\bmod4=3. \\
\end{cases}
$$

综合上述结论有

$$
\bigoplus_{k=0}^{n-1}\left(start+2k\right)=\overbrace{f\left(n\right)\oplus f\left(s+n\right)}^{\text{high bits }r}\times2+
\overbrace{\left(start\bmod2\right)\left(n\bmod2\right)}^{\text{low bit }e}.
$$

```rust Rust
impl Solution {
    pub fn xor_operation(n: i32, start: i32) -> i32 {
        xor_operation(n as usize, start as usize) as i32
    }
}

pub fn xor_operation(n: usize, start: usize) -> usize {
    let (s, e) = (start >> 1, n & start & 1);
    let r = sum_xor(s) ^ sum_xor(s + n);
    r << 1 | e
}

fn sum_xor(n: usize) -> usize {
    match n % 4 {
        0 => 0,
        1 => n - 1,
        2 => 1,
        3 => n,
        _ => unreachable!(),
    }
}
```
