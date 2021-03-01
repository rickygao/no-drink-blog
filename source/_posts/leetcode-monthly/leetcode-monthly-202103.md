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
