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
输入：employees = [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], id = 1
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
解释：若 arr = [1, 0, 2, 1]，那么 first = 1 且 encoded = [1 ^ 0, 0 ^ 2, 2 ^ 1] = [1, 2, 3]。
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
use std::iter::once;

impl Solution {
    pub fn decode(encoded: Vec<i32>, first: i32) -> Vec<i32> {
        once(first)
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

## 172. 阶乘后的零{#leetcode-172}

[:link: 来源](https://leetcode-cn.com/problems/factorial-trailing-zeroes/)

### 题目

给定一个整数 `n`，返回 `n!` 结果尾数中零的数量。

#### 示例

```raw
输入: 3
输出: 0
解释: 3! = 6，尾数中没有零。
```

```raw
输入: 5
输出: 1
解释: 5! = 120，尾数中有 1 个零。
```

#### 说明

你算法的时间复杂度应为 $\mathrm{O}(\log n)$。

### 题解

```rust Rust
impl Solution {
    pub fn trailing_zeroes(n: i32) -> i32 {
        trailing_zeroes(n as usize) as i32
    }
}

pub fn trailing_zeroes(n: usize) -> usize {
    prime_factor_order_to(n, 5)
}

fn prime_factor_order_to(mut n: usize, f: usize) -> usize {
    let mut p = 0;
    while n >= f {
        n /= f;
        p += n;
    }
    p
}
```

## 872. 叶子相似的树{#leetcode-872}

[:link: 来源](https://leetcode-cn.com/problems/leaf-similar-trees/)

### 题目

请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个**叶值序列**。

如果有两棵二叉树的叶值序列是相同，那么我们就认为它们是**叶相似**的。

如果给定的两个根结点分别为 `root1` 和 `root2` 的树是叶相似的，则返回 `true`；否则返回 `false`。

#### 示例

```raw
输入：root1 = [3, 5, 1, 6, 2, 9, 8, null, null, 7, 4], root2 = [3, 5, 1, 6, 7, 4, 2, null, null, null, null, null, null, 9, 8]
输出：true
```

```raw
输入：root1 = [1], root2 = [1]
输出：true
```

```raw
输入：root1 = [1], root2 = [2]
输出：false
```

```raw
输入：root1 = [1, 2], root2 = [2, 2]
输出：true
```

```raw
输入：root1 = [1, 2, 3], root2 = [1, 3, 2]
输出：false
```

#### 提示

- 给定的两棵树可能会有 `1` 到 `200` 个结点；
- 给定的两棵树上的值介于 `0` 到 `200` 之间。

### 题解

#### 深度优先搜索

```rust Rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }

use std::cell::RefCell;
use std::rc::Rc;

impl Solution {
    pub fn leaf_similar(
        root1: Option<Rc<RefCell<TreeNode>>>,
        root2: Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        leaves(root1) == leaves(root2)
    }
}

fn leaves(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    root.map(|root| {
        let (mut stack, mut leaves) = (vec![root], vec![]);
        while let Some(node) = stack.pop() {
            let node = node.borrow();
            node.left.as_ref().cloned().map(|l| stack.push(l));
            node.right.as_ref().cloned().map(|r| stack.push(r));
            if node.left.as_ref().or(node.right.as_ref()).is_none() {
                leaves.push(node.val);
            }
        }
        leaves
    }).unwrap_or_default()
}
```

#### 叶子迭代器

```rust Rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }

use std::cell::RefCell;
use std::rc::Rc;

impl Solution {
    pub fn leaf_similar(
        root1: Option<Rc<RefCell<TreeNode>>>,
        root2: Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        Iterator::eq(LeavesIter::new(root1), LeavesIter::new(root2))
    }
}

struct LeavesIter {
    stack: Vec<Rc<RefCell<TreeNode>>>,
}

impl LeavesIter {
    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        Self {
            stack: root.into_iter().collect(),
        }
    }
}

impl Iterator for LeavesIter {
    type Item = Rc<RefCell<TreeNode>>;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        while let Some(node) = self.stack.pop() {
            let inner = node.borrow();
            inner.left.as_ref().cloned().map(|l| self.stack.push(l));
            inner.right.as_ref().cloned().map(|r| self.stack.push(r));
            if inner.left.as_ref().or(inner.right.as_ref()).is_none() {
                return node.clone().into();
            }
        }
        return None;
    }
}
```

## 1734. 解码异或后的排列{#leetcode-1734}

[:link: 来源](https://leetcode-cn.com/problems/decode-xored-permutation/)

### 题目

给你一个整数数组 `perm`，它是前 `n` 个正整数的排列，且 `n` 是个**奇数**。

它被加密成另一个长度为 `n - 1` 的整数数组 `encoded`，满足 `encoded[i] = perm[i] ^ perm[i + 1]`。比方说，如果 `perm = [1, 3, 2]`，那么 `encoded = [2, 1]`。

给你 `encoded` 数组，请你返回原始数组 `perm`。题目保证答案存在且唯一。

#### 示例

```raw
输入：encoded = [3, 1]
输出：[1, 2, 3]
解释：如果 perm = [1, 2, 3]，那么 encoded = [1 ^ 2, 2 ^ 3] = [3, 1]。
```

```raw
输入：encoded = [6, 5, 4, 6]
输出：[2, 4, 1, 5, 3]
```

#### 提示

- `3 <= n < 1e5`；
- `n` 是奇数；
- `len(encoded) == n - 1`。

### 题解

正整数排列的全异或为

$$
\begin{aligned}
total&=\bigoplus_{k=1}^{n}k=\bigoplus_{k=0}^{n-1}{perm}_{k}\\
&={perm}_0\oplus\left(\bigoplus_{k=1}^{n-1}{perm}_{k}\right).
\end{aligned}
$$

编码结果的奇数项异或为

$$
\begin{aligned}
odd&=\bigoplus_{k=1}^{\left\lfloor n/2\right\rfloor}{encoded}_{2k-1}\\
&=\bigoplus_{k=1}^{\left\lfloor n/2\right\rfloor}\left({perm}_{2k-1}\oplus{perm}_{2k}\right)\\
&=\bigoplus_{k=1}^{n-1}{perm}_{k}.
\end{aligned}
$$

于是可得

$$
\begin{aligned}
total\oplus odd&={perm}_0\oplus\left(\bigoplus_{k=1}^{n-1}{perm}_{k}\right)\oplus\left(\bigoplus_{k=1}^{n-1}{perm}_{k}\right)\\
&={perm}_0.
\end{aligned}
$$

```python Python
class Solution:
    def decode(self, encoded: list[int]) -> list[int]:
        return decode(encoded)

from operator import xor
from itertools import accumulate
from functools import reduce

def decode(encoded: list[int]) -> list[int]:
    first = reduce(xor, range(len(encoded) + 2)) ^ reduce(xor, encoded[1::2])
    return list(accumulate(encoded, xor, initial=first))
```

```rust Rust
use std::iter::once;
use std::ops::BitXor;

impl Solution {
    pub fn decode(encoded: Vec<i32>) -> Vec<i32> {
        let n = encoded.len() + 1;
        let first = (1..=n as i32)
            .chain(encoded.iter().skip(1).step_by(2).copied())
            .fold(0, BitXor::bitxor);
        once(first)
            .chain(encoded.into_iter().scan(first, |st, e| {
                *st ^= e;
                Some(*st)
            }))
            .collect()
    }
}
```

## 1310. 子数组异或查询{#leetcode-1310}

[:link: 来源](https://leetcode-cn.com/problems/xor-queries-of-a-subarray/)

### 题目

有一个正整数数组 `arr`，现给你一个对应的查询数组 `queries`，其中 `queries[i] = [li, ri]`。

对于每个查询 `i`，请你计算从 `li` 到 `ri` 的异或值（即 `arr[li] ^ arr[li+1] ^ ... ^ arr[ri]`）作为本次查询的结果。

并返回一个包含给定查询 `queries` 所有结果的数组。

#### 示例

```raw
输入：arr = [1, 3, 4, 8], queries = [[0, 1], [1, 2], [0, 3], [3, 3]]
输出：[2, 7, 14, 8]
```

```raw
输入：arr = [4, 8, 2, 10], queries = [[2, 3], [1, 3], [0, 0], [0, 3]]
输出：[8, 0, 4, 4]
```

#### 提示

- `1 <= len(arr) <= 3e4`；
- `1 <= arr[i] <= 1e9`；
- `1 <= len(queries) <= 3e4`；
- `len(queries[i]) == 2`；
- `0 <= queries[i][0] <= queries[i][1] < len(arr)`。

### 题解

```rust Rust
impl Solution {
    pub fn xor_queries(arr: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let queries: Vec<_> = queries
            .into_iter()
            .map(|queries| (queries[0] as usize, queries[1] as usize))
            .collect();
        xor_queries(&arr, &queries)
    }
}

pub fn xor_queries(arr: &[i32], queries: &[(usize, usize)]) -> Vec<i32> {
    let prefix: Vec<_> = arr
        .iter()
        .scan(0, |st, n| {
            *st ^= n;
            Some(*st)
        })
        .collect();

    queries
        .iter()
        .map(|&(l, r)| {
            if l > 0 {
                prefix[l - 1] ^ prefix[r]
            } else {
                prefix[r]
            }
        })
        .collect()
}
```

```python Python
class Solution:
    def xorQueries(self, arr: list[int], queries: list[list[int]]) -> list[int]:
        return xor_queries(arr, queries)

from operator import xor
from itertools import accumulate

def xor_queries(arr: list[int], queries: list[tuple[int, int]]) -> list[int]:
    prefix = list(accumulate(arr, xor, initial=0))
    return [prefix[l] ^ prefix[r + 1] for l, r in queries]
```

## 1679. K 和数对的最大数目{#leetcode-1679}

[:link: 来源](https://leetcode-cn.com/problems/max-number-of-k-sum-pairs/)

### 题目

给你一个整数数组 `nums` 和一个整数 `k`。

每一步操作中，你需要从数组中选出和为 `k` 的两个整数，并将它们移出数组。

返回你可以对数组执行的最大操作数。

#### 示例

```raw
输入：nums = [1, 2, 3, 4], k = 5
输出：2
```

```raw
输入：nums = [3, 1, 3, 4, 3], k = 6
输出：1
```

#### 提示

- `1 <= len(nums) <= 1e5`；
- `1 <= nums[i] <= 1e9`；
- `1 <= k <= 1e9`。

### 题解

```rust Rust
impl Solution {
    pub fn max_operations(nums: Vec<i32>, k: i32) -> i32 {
        max_operations(&nums, k) as i32
    }
}

use std::collections::HashMap;

pub fn max_operations(nums: &[i32], k: i32) -> usize {
    let (mut counter, mut result) = (HashMap::new(), 0);
    for &n in nums {
        match counter.get_mut(&(k - n)) {
            Some(count) if *count > 0 => {
                *count -= 1;
                result += 1;
            }
            _ => *counter.entry(n).or_insert(0) += 1,
        }
    }
    result
}
```
