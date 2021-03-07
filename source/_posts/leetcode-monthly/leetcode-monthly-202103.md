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

## 300. 最长递增子序列{#leetcode-300}

[:link: 来源](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

### 题目

给你一个整数数组 `nums`，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3, 6, 2, 7]` 是数组 `[0, 3, 1, 6, 2, 2, 7]` 的子序列。

#### 示例

```raw
输入：nums = [10, 9, 2, 5, 3, 7, 101, 18]
输出：4
解释：最长递增子序列是 [2, 3, 7, 101]，因此长度为 4。
```

```raw
输入：nums = [0, 1, 0, 3, 2, 3]
输出：4
```

```raw
输入：nums = [7, 7, 7, 7, 7, 7, 7]
输出：1
```

#### 提示

- `1 <= len(nums) <= 2500`；
- `-1e4 <= nums[i] <= 1e4`。

#### 进阶

- 你可以设计时间复杂度为 $\mathrm{O}(n^2)$ 的解决方案吗？
- 你能将算法的时间复杂度降低到 $\mathrm{O}(n\log n)$ 吗？

### 题解

#### 动态规划

记 `nums` 为 $a$ 且其长度为 $n$，有递推方程：$l_0=1,l_j=\max_{(0\le i\lt j)\wedge(a_j>a_i)}l_i+1$，最终结果为 $\max_{0<i<n} l_i$。时间复杂度为 $\mathrm{O}(n^2)$。

```rust Rust
impl Solution {
    pub fn length_of_lis(nums: Vec<i32>) -> i32 {
        length_of_lis(&nums) as i32
    }
}

pub fn length_of_lis(nums: &[impl Ord]) -> usize {
    let mut len = Vec::with_capacity(nums.len());
    for num in nums {
        len.push(
            Iterator::zip(len.iter(), nums.iter())
                .filter_map(|(l, n)| if num > n { Some(l) } else { None })
                .max()
                .copied()
                .unwrap_or(0)
                + 1,
        );
    }
    len.iter().max().copied().unwrap_or(0)
}
```

#### 二分贪心

```rust Rust
impl Solution {
    pub fn length_of_lis(nums: Vec<i32>) -> i32 {
        length_of_lis(&nums) as i32
    }
}

pub fn length_of_lis(nums: &[impl Ord]) -> usize {
    let mut len = 0;
    let mut mono = Vec::with_capacity(nums.len());
    for num in nums {
        if mono.last().map_or(true, |&last| last < num) {
            len += 1;
            mono.push(num);
        } else if let Err(pos) = mono.binary_search(&num) {
            mono[pos] = num;
        }
    }
    len
}
```

## 354. 俄罗斯套娃信封问题{#leetcode-354}

[:link: 来源](https://leetcode-cn.com/problems/russian-doll-envelopes/)

### 题目

给定一些标记了宽度和高度的信封，宽度和高度以整数对形式 `(w, h)` 出现。当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算最多能有多少个信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

#### 说明

不允许旋转信封。

#### 示例

```raw
输入: envelopes = [[5, 4], [6, 4], [6, 7], [2, 3]]
输出: 3 
解释: 最多信封的个数为 3，组合为：[2, 3] => [5, 4] => [6, 7]。
```

### 题解

按照“宽度升序、宽度相同时高度降序”的原则排序，转化为[上题](#leetcode-300)。

```rust Rust
impl Solution {
    pub fn max_envelopes(envelopes: Vec<Vec<i32>>) -> i32 {
        let envelopes = envelopes
                .into_iter()
                .map(|v| (v[0] as usize, v[1] as usize))
                .collect::<Vec<_>>();
        max_envelopes(&envelopes) as i32
    }
}

pub fn max_envelopes(envelopes: &[(usize, usize)]) -> usize {
    let mut envelopes: Vec<_> = envelopes.iter().copied().collect();
    envelopes.sort_unstable_by(|a, b| Ord::cmp(&a.0, &b.0).then_with(|| Ord::cmp(&b.1, &a.1)));
    let heights: Vec<_> = envelopes.into_iter().map(|(_, h)| h).collect();
    length_of_lis(&heights)
}

pub fn length_of_lis(nums: &[impl Ord]) -> usize {
    let mut len = 0;
    let mut mono = Vec::with_capacity(nums.len());
    for num in nums {
        if mono.last().map_or(true, |&last| last < num) {
            len += 1;
            mono.push(num);
        } else if let Err(pos) = mono.binary_search(&num) {
            mono[pos] = num;
        }
    }
    len
}
```

## 232. 用栈实现队列{#leetcode-232}

[:link: 来源](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

### 题目

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列的支持的所有操作（`push`、`pop`、`peek`、`empty`）。

实现 `MyQueue` 类：

- `void push(int x)` 将元素 `x` 推到队列的末尾；
- `int pop()` 从队列的开头移除并返回元素；
- `int peek()` 返回队列开头的元素；
- `boolean empty()` 如果队列为空，返回 `true`；否则，返回 `false`。

#### 说明

- 你只能使用标准的栈操作 —— 也就是只有 `push`、`pop`、`peek`、`size` 和 `is_empty` 操作是合法的；
- 你所使用的语言也许不支持栈。你可以使用 `list` 或者 `deque`（双端队列）来模拟一个栈，只要是标准的栈操作即可。

#### 进阶

你能否实现每个操作均摊时间复杂度为 $\mathrm{O}(1)$ 的队列？换句话说，执行 `n` 个操作的总时间复杂度为 $\mathrm{O}(n)$，即使其中一个操作可能花费较长时间。

#### 示例

```raw
输入：["MyQueue", "push", "push", "peek", "pop", "empty"], [[], [1], [2], [], [], []]
输出：[null, null, null, 1, 1, false]
解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
```

#### 提示

- `1 <= x <= 9`；
- 最多调用 `100` 次 `push`、`pop`、`peek` 和 `empty`；
- 假设所有操作都是有效的（例如，一个空的队列不会调用 `pop` 或者 `peek` 操作）。

### 题解

利用 `std::cell::RefCell` 实现[内部可变性（Interior Mutability](https://doc.rust-lang.org/book/ch15-05-interior-mutability.html)。

```rust Rust
use std::cell::RefCell;

struct MyQueue {
    front: RefCell<Vec<i32>>,
    back: RefCell<Vec<i32>>,
}

impl MyQueue {
    fn new() -> Self {
        Self {
            front: Vec::new().into(),
            back: Vec::new().into(),
        }
    }

    fn push(&mut self, x: i32) {
        self.back.get_mut().push(x);
    }

    fn pop(&mut self) -> i32 {
        self.rotate();
        self.front.get_mut().pop().unwrap()
    }

    fn peek(&self) -> i32 {
        self.rotate();
        *self.front.borrow().last().unwrap()
    }

    fn rotate(&self) {
        let (mut front, mut back) = (self.front.borrow_mut(), self.back.borrow_mut());
        if front.is_empty() {
            while let Some(e) = back.pop() {
                front.push(e);
            }
        }
    }

    fn empty(&self) -> bool {
        self.front.borrow().is_empty() && self.back.borrow().is_empty()
    }
}
```

## 503. 下一个更大元素 II{#leetcode-503}

[:link: 来源](https://leetcode-cn.com/problems/next-greater-element-ii/)

### 题目

给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 `x` 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 `-1`。

#### 示例

```raw
输入：[1, 2, 1]
输出：[2, -1, 2]
解释：第一个 1 的下一个更大的数是 2；数字 2 找不到下一个更大的数；第二个 1 的下一个最大的数需要循环搜索，结果也是 2。
```

#### 注意

输入数组的长度不会超过 `1e4`。

### 题解

单调栈。

```rust Rust
impl Solution {
    pub fn next_greater_elements(nums: Vec<i32>) -> Vec<i32> {
        next_greater_elements(&nums)
    }
}

pub fn next_greater_elements(nums: &[i32]) -> Vec<i32> {
    let mut greater = vec![-1; nums.len()];
    let mut mono = Vec::with_capacity(2 * nums.len());
    nums.iter().count();
    for (i, n) in Iterator::chain(
        nums.iter().copied().enumerate(),
        nums.iter().copied().enumerate(),
    ) {
        while let Some(&(j, m)) = mono.last() {
            if m >= n {
                break;
            }
            mono.pop();
            greater[j] = n;
        }
        mono.push((i, n));
    }
    greater
}
```

## 131. 分割回文串{#leetcode-131}

[:link: 来源](https://leetcode-cn.com/problems/palindrome-partitioning/)

### 题目

给定一个字符串 `s`，将 `s` 分割成一些子串，使每个子串都是回文串。

返回 `s` 所有可能的分割方案。

#### 示例

```raw
输入："aab"
输出：[["aa", "b"], ["a", "a", "b"]]
```

### 题解

深度优先搜索。

```rust Rust
impl Solution {
    pub fn partition(s: String) -> Vec<Vec<String>> {
        partition(s.as_bytes())
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .map(|s| std::str::from_utf8(s).unwrap().to_owned())
                    .collect()
            })
            .collect()
    }
}

pub fn partition<'a>(s: &'a [u8]) -> Vec<Vec<&'a [u8]>> {
    let mut result = Vec::new();
    for i in 0..s.len() {
        let (front, back) = s.split_at(i);
        if is_palindrome(back) {
            if front.is_empty() {
                result.push(vec![back]);
            } else {
                let mut front_result = partition(front);
                front_result.iter_mut().for_each(|v| v.push(back));
                result.append(&mut front_result);
            }
        }
    }
    result
}

fn is_palindrome(s: &[impl Ord]) -> bool {
    Iterator::zip(s.iter(), s.iter().rev()).take(s.len() / 2).all(|c| c.0 == c.1)
}
```
