---
title: LeetCode 月报 202103
date: 2021-03-01 00:00:00
tags: [LeetCode]
mathjax: true
---

假期正式结束了！这是新学期的第一份月报。

<!-- more -->

## 303. 区域和检索 - 数组不可变{#leetcode-303}

[:link: 来源](https://leetcode-cn.com/problems/range-sum-query-immutable/)

### 题目

给定一个整数数组 `nums`，求出数组从索引 `i` 到 `j`（`i <= j`）范围内元素的总和，包含 `i`、`j` 两点。

实现 `NumArray` 类：

- `NumArray(int[] nums)` 使用数组 `nums` 初始化对象；
- `int sumRange(int i, int j)` 返回数组 `nums` 从索引 `i` 到 `j`（`i <= j`）范围内元素的总和，包含 `i`、`j` 两点（也就是 `sum(nums[i..=j]`）。

#### 示例

```raw
输入：
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]

输出：
[null, 1, -1, -3]

解释：
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return 1 ((-2) + 0 + 3)
numArray.sumRange(2, 5); // return -1 (3 + (-5) + 2 + (-1)) 
numArray.sumRange(0, 5); // return -3 ((-2) + 0 + 3 + (-5) + 2 + (-1))
```

#### 提示

- `0 <= len(nums) <= 1e4`；
- `-1e5 <= nums[i] <= 1e5`；
- `0 <= i <= j < len(nums)`；
- 最多调用 `1e4` 次 `sumRange` 方法。

### 题解

```rust Rust
struct NumArray {
    prefix_sum: Vec<i32>,
}

impl NumArray {
    fn new(nums: Vec<i32>) -> Self {
        Self {
            prefix_sum: nums
                .into_iter()
                .scan(0, |s, n| {
                    *s += n;
                    Some(*s)
                })
                .collect(),
        }
    }

    fn sum_range(&self, i: i32, j: i32) -> i32 {
        self.real_sum_range(i as usize, j as usize)
    }

    fn real_sum_range(&self, i: usize, j: usize) -> i32 {
        let mut r = self.prefix_sum[j];
        if i > 0 {
            r -= self.prefix_sum[i - 1];
        }
        r
    }
}
```

## 304. 二维区域和检索 - 矩阵不可变{#leetcode-304}

[:link: 来源](https://leetcode-cn.com/problems/range-sum-query-2d-immutable/)

### 题目

给定一个二维矩阵，计算其子矩形范围内元素的总和，该子矩阵的左上角为 `(row1, col1)`，右下角为 `(row2, col2)`。

#### 提示

- 你可以假设矩阵不可变；
- 会多次调用 `sumRegion` 方法；
- 你可以假设 `row1 <= row2` 且 `col1 <= col2`。

### 题解

```rust Rust
struct NumMatrix {
    prefix_sum: Vec<Vec<i32>>,
}

impl NumMatrix {
    fn new(matrix: Vec<Vec<i32>>) -> Self {
        let l = matrix.first().map_or(0, |v| v.len());
        Self {
            prefix_sum: matrix
                .into_iter()
                .scan(vec![0; l], |t, r| {
                    t.iter_mut()
                        .zip(r.into_iter().scan(0, |s, n| {
                            *s += n;
                            Some(*s)
                        }))
                        .for_each(|(s, n)| *s += n);
                    Some(t.clone())
                })
                .collect(),
        }
    }

    fn sum_region(&self, row1: i32, col1: i32, row2: i32, col2: i32) -> i32 {
        self.real_sum_region(
            (row1 as usize, col1 as usize),
            (row2 as usize, col2 as usize),
        )
    }

    fn real_sum_region(&self, i: (usize, usize), j: (usize, usize)) -> i32 {
        let mut r = self.prefix_sum[j.0][j.1];
        if i.0 > 0 {
            r -= self.prefix_sum[i.0 - 1][j.1];
        }
        if i.1 > 0 {
            r -= self.prefix_sum[j.0][i.1 - 1];
        }
        if i.0 > 0 && i.1 > 0 {
            r += self.prefix_sum[i.0 - 1][i.1 - 1];
        }
        r
    }
}
```

## 338. 比特位计数{#leetcode-338}

[:link: 来源](https://leetcode-cn.com/problems/counting-bits/)

### 题目

给定一个非负整数 `num`。对于 `0 <= i <= num` 范围中的每个数字 `i`，计算其二进制数中的 `1` 的数目并将它们作为数组返回。

#### 示例

```raw
输入：2
输出：[0, 1, 1]
```

```raw
输入：5
输出：[0, 1, 1, 2, 1, 2]
```

### 题解

#### 直接

```rust Rust
impl Solution {
    pub fn count_bits(num: i32) -> Vec<i32> {
        (0..=num).map(|n| n.count_ones() as i32).collect()
    }
}
```

#### 动态规划

将求 `i` 中 `1` 比特的个数转化为求 `j < i` 中 `1` 比特的个数。

```rust Rust
impl Solution {
    pub fn count_bits(num: i32) -> Vec<i32> {
        count_bits(num as usize + 1).into_iter().map(|n| n as i32).collect()
    }
}

pub fn count_bits(num: usize) -> Vec<usize> {
    let mut r = Vec::with_capacity(num);
    r.push(0);
    for i in 1..num {
        r.push(r[i >> 1] + (i & 1)); // or r.push(r[i & (i - 1)] + 1);
    }
    r
}
```
