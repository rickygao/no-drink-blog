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

## 132. 分割回文串 II{#leetcode-132}

[:link: 来源](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

### 题目

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文。

返回符合要求的**最少分割次数**。

#### 示例

```raw
输入：s = "aab"
输出：1
解释：只需一次分割就可将 s 分割成 ["aa", "b"] 这样两个回文子串。
```

```raw
输入：s = "a"
输出：0
```

```raw
输入：s = "ab"
输出：1
```

#### 提示

- `1 <= len(s) <= 2e3`；
- `s` 仅由小写英文字母组成。

### 题解

动态规划。

```rust Rust
impl Solution {
    pub fn min_cut(s: String) -> i32 {
        min_cut(s.as_bytes()) as i32
    }
}

pub fn min_cut(s: &[u8]) -> usize {
    let n = s.len();
    let (mut is_palindrome, mut cut_count) = (vec![true; n], vec![0; n]);

    // is_palindrome[j] in loop #i means s[j..=i] is palindrome.
    for i in 0..n {
        for j in 0..i {
            // s[j..=i] is palindrome if s[j + 1..=i - 1] is palindrome and s[j] == s[i]
            is_palindrome[j] = is_palindrome[j + 1] && s[j] == s[i];
        }
        if !is_palindrome[0] {
            cut_count[i] = usize::MAX;
            for j in 1..=i {
                if is_palindrome[j] {
                    cut_count[i] = cut_count[i].min(cut_count[j - 1] + 1);
                }
            }
        }
    }
    cut_count.last().copied().unwrap_or(0)
}
```

## 1047. 删除字符串中的所有相邻重复项{#leetcode-1047}

[:link: 来源](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

### 题目

给出由小写字母组成的字符串 `S`，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 `S` 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

#### 示例

```raw
输入："abbaca"
输出："ca"
解释：例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
```

#### 提示

- `1 <= len(S) <= 2e4`；
- `S` 仅由小写英文字母组成。

### 题解

```rust Rust
impl Solution {
    pub fn remove_duplicates(s: String) -> String {
        remove_duplicates(&s)
    }
}

pub fn remove_duplicates(s: &str) -> String {
    let mut buf = Vec::new();
    for c in s.chars() {
        match buf.last() {
            Some(&d) if c == d => {
                buf.pop();
            }
            _ => buf.push(c),
        }
    }
    buf.into_iter().collect()
}
```

## 224. 基本计算器{#leetcode-224}

[:link: 来源](https://leetcode-cn.com/problems/basic-calculator/)

### 题目

实现一个基本的计算器来计算一个简单的字符串表达式 `s` 的值。

#### 示例

```raw
输入：s = "1 + 1"
输出：2
```

```raw
输入：s = " 2-1 + 2 "
输出：3
```

```raw
输入：s = "(1+(4+5+2)-3)+(6+8)"
输出：23
```

#### 提示

- `1 <= len(s) <= 3e5`；
- `s` 由数字、`'+'`、`'-'`、`'('`、`')'`、和 `' '` 组成；
- `s` 表示一个有效的表达式。

### 题解

```rust Rust
impl Solution {
    pub fn calculate(s: String) -> i32 {
        let (mut ops, mut sign) = (vec![1], 1);
        let mut result = 0;
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '+' => sign = ops.last().copied().unwrap(),
                '-' => sign = -ops.last().copied().unwrap(),
                '(' => ops.push(sign),
                ')' => {
                    ops.pop();
                }
                '0'..='9' => {
                    let mut buf = c.to_digit(10).unwrap();
                    while let Some(c) = chars.peek() {
                        match c.to_digit(10) {
                            Some(d) => buf = buf * 10 + d,
                            None => break,
                        }
                        chars.next();
                    }
                    result += sign * (buf as i32);
                }
                _ => {}
            }
        }
        result
    }
}
```

## 227. 基本计算器 II{#leetcode-227}

[:link: 来源](https://leetcode-cn.com/problems/basic-calculator-ii/)

### 题目

给你一个字符串表达式 `s`，请你实现一个基本计算器来计算并返回它的值。

整数除法仅保留整数部分。

#### 示例

```raw
输入：s = "3+2*2"
输出：7
```

```raw
输入：s = " 3/2 "
输出：1
```

```raw
输入：s = " 3+5 / 2 "
输出：5
```

#### 提示

- `1 <= len(s) <= 3e5`；
- `s` 由整数和算符（`'+'`、`'-'`、`'*'`、`'/'`）组成，中间由一些空格隔开；
- `s` 表示一个**有效表达式**；
- 表达式中的所有整数都是非负整数，且在范围 `range(0, 2 ** 31)` 内；
- 题目数据保证答案是一个 32 位整数。

### 题解

```rust Rust
impl Solution {
    pub fn calculate(s: String) -> i32 {
        let (mut operands, mut operators) = (vec![], vec![]);
        let mut buf = None;

        fn priority(operator: &char) -> u8 {
            match operator {
                '(' => u8::MIN,
                '+' => 1,
                '-' => 1,
                '*' => 2,
                '/' => 2,
                ')' => u8::MAX,
                _ => unimplemented!(),
            }
        }

        fn operate(operator: char, operands: &mut Vec<i32>) {
            let rhs = operands.pop().expect("missed rhs");
            let lhs = operands.pop().expect("missed lhs");
            operands.push(match operator {
                '+' => lhs + rhs,
                '-' => lhs - rhs,
                '*' => lhs * rhs,
                '/' => lhs / rhs,
                _ => unimplemented!(),
            });
        }

        fn balance_or_push(operator: char, operators: &mut Vec<char>) {
            if operator == ')' {
                operators.pop().filter(|&left| left == '(').expect("unpaired parentheses");
            } else {
                operators.push(operator);
            }
        }

        for ch in s.chars() {
            if ch.is_whitespace() {
                continue;
            }
            if let Some(digit) = ch.to_digit(10) {
                let n = buf.get_or_insert(0);
                *n = *n * 10 + digit;
                continue;
            }
            if let Some(num) = buf.take() {
                operands.push(num as i32);
            }
            let current_priority = priority(&ch);
            while operators.last().map_or(false, |op| priority(op) >= current_priority) {
                operate(operators.pop().unwrap(), &mut operands);
            }
            balance_or_push(ch, &mut operators);
        }
        if let Some(num) = buf.take() {
            operands.push(num as i32);
        }
        while operators.last().is_some() {
            operate(operators.pop().unwrap(), &mut operands);
        }
        operands.pop().unwrap()
    }
}
```

## 331. 验证二叉树的前序序列化{#leetcode-331}

[:link: 来源](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/)

### 题目

序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 `'#'`。

```raw
     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #
```

例如，上面的二叉树可以被序列化为字符串 `"9,3,4,#,#,1,#,#,2,#,6,#,#"`，其中 `'#'` 代表一个空节点。

给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。

每个以逗号分隔的字符或为一个整数或为一个表示 `null` 指针的 `'#'`。

你可以认为输入格式总是有效的，例如它永远不会包含两个连续的逗号，比如 `"1,,3"`。

#### 示例

```raw
输入："9,3,4,#,#,1,#,#,2,#,6,#,#"
输出：true
```

```raw
输入："1,#"
输出：false
```

```raw
输入："9,#,#,1"
输出：false
```

### 题解

#### 栈

在栈顶替换模式 `[?, #, #]"` 为 `[#]`。

```rust Rust
impl Solution {
    pub fn is_valid_serialization(preorder: String) -> bool {
        is_valid_serialization(&preorder)
    }
}

pub fn is_valid_serialization(preorder: &str) -> bool {
    let mut stack = vec![];
    for leaf in preorder.split(',').map(|e| e == "#") {
        stack.push(leaf);
        while let [.., false, true, true] = stack.as_slice() {
            stack.truncate(stack.len() - 2);
            *stack.last_mut().unwrap() = true;
        }
    }
    stack == [true]
}
```

#### 计数

记录当前节点空余槽位。

```rust Rust
impl Solution {
    pub fn is_valid_serialization(preorder: String) -> bool {
        is_valid_serialization(&preorder)
    }
}

pub fn is_valid_serialization(preorder: &str) -> bool {
    let mut slots = 1;
    for leaf in preorder.split(',').map(|e| e == "#") {
        if slots <= 0 {
            return false;
        }
        slots = if leaf { slots - 1 } else { slots + 1 };
    }
    slots == 0
}
```

## 705. 设计哈希集合{#leetcode-705}

[:link: 来源](https://leetcode-cn.com/problems/design-hashset/)

### 题目

不使用任何内建的哈希表库设计一个哈希集合（`HashSet`）。

实现 `MyHashSet` 类：

- `void add(key)` 向哈希集合中插入值 `key`；
- `bool contains(key)` 返回哈希集合中是否存在这个值 `key`；
- `void remove(key)` 将给定值 `key` 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

#### 示例

```raw
输入：
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]

输出：
[null, null, null, true, false, null, true, null, false]

解释：
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // 返回 true
myHashSet.contains(3); // 返回 false（未找到）
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // 返回 true
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // 返回 false（已移除）
```

#### 提示

- `0 <= key <= 1e6`；
- 最多调用 `1e4` 次 `add`、`remove` 和 `contains`。

#### 进阶

你可以不使用内建的哈希集合库解决此问题吗？

### 题解

```rust Rust
struct MyHashSet {
    table: Vec<Vec<i32>>,
    base: usize,
}

impl MyHashSet {
    const DEFAULT_BASE: usize = 1009;

    pub fn new() -> Self {
        Self::new_with_base(Self::DEFAULT_BASE)
    }

    pub fn new_with_base(base: usize) -> Self {
        Self {
            table: vec![vec![]; base],
            base,
        }
    }

    pub fn add(&mut self, key: i32) {
        let bucket = self.bucket_mut(key);
        if let Err(index) = bucket.binary_search(&key) {
            bucket.insert(index, key);
        }
    }

    pub fn remove(&mut self, key: i32) {
        let bucket = self.bucket_mut(key);
        if let Ok(index) = bucket.binary_search(&key) {
            bucket.remove(index);
        }
    }

    pub fn contains(&self, key: i32) -> bool {
        let bucket = self.bucket(key);
        bucket.binary_search(&key).is_ok()
    }

    fn hash(&self, key: i32) -> usize {
        key.rem_euclid(self.base as i32) as usize
    }

    fn bucket(&self, key: i32) -> &Vec<i32> {
        let hash = self.hash(key);
        &self.table[hash]
    }

    fn bucket_mut(&mut self, key: i32) -> &mut Vec<i32> {
        let hash = self.hash(key);
        &mut self.table[hash]
    }
}
```

## 706. 设计哈希映射{#leetcode-706}

[:link: 来源](https://leetcode-cn.com/problems/design-hashmap/)

### 题目

不使用任何内建的哈希表库设计一个哈希映射（`HashMap`）。

实现 `MyHashMap` 类：

- `void put(int key, int value)` 向哈希映射插入一个键值对 `(key, value)`。如果 `key` 已经存在于映射中，则更新其对应的值 `value`；
- `int get(int key)` 返回特定的 `key` 所映射的 `value`。如果映射中不包含 `key` 的映射，返回 `-1`；
- `void remove(key)` 如果映射中存在 `key` 的映射，则移除 `key` 和它所对应的 `value`。

#### 示例

```raw
输入：
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]

输出：
[null, null, null, 1, -1, null, 1, null, -1]

解释：
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // map = [(1, 1)]
myHashMap.put(2, 2); // map = [(1, 1), (2, 2)]
myHashMap.get(1);    // 返回 1
myHashMap.get(3);    // 返回 -1（未找到）
myHashMap.put(2, 1); // map = [(1, 1), (2, 1)]（更新已有的值）
myHashMap.get(2);    // 返回 1
myHashMap.remove(2); // map = [(1, 1)]
myHashMap.get(2);    // 返回 -1（未找到）
```

#### 提示

- `0 <= key, value <= 1e6`；
- 最多调用 `1e4` 次 `put`、`get` 和 `remove` 方法。

### 题解

修改一下[上题](#leetcode-705)的方法签名和搜索实现。`MyHashMap::put` 方法中考虑键已经存在的情况，并使用 `[T]::binary_search_by_key` 代替 `[T]::binary_search`。

```rust Rust
struct MyHashMap {
    table: Vec<Vec<(i32, i32)>>,
    base: usize,
}

impl MyHashMap {
    const DEFAULT_BASE: usize = 1009;

    pub fn new() -> Self {
        Self::new_with_base(Self::DEFAULT_BASE)
    }

    pub fn new_with_base(base: usize) -> Self {
        Self {
            table: vec![vec![]; base],
            base,
        }
    }

    pub fn put(&mut self, key: i32, value: i32) {
        let bucket = self.bucket_mut(key);
        match bucket.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(index) => bucket[index].1 = value,
            Err(index) => bucket.insert(index, (key, value)),
        }
    }

    pub fn get(&self, key: i32) -> i32 {
        let bucket = self.bucket(key);
        match bucket.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(index) => bucket[index].1,
            Err(_) => -1,
        }
    }

    pub fn remove(&mut self, key: i32) {
        let bucket = self.bucket_mut(key);
        if let Ok(index) = bucket.binary_search_by_key(&key, |&(k, _)| k) {
            bucket.remove(index);
        }
    }

    fn hash(&self, key: i32) -> usize {
        key.rem_euclid(self.base as i32) as usize
    }

    fn bucket(&self, key: i32) -> &Vec<(i32, i32)> {
        let hash = self.hash(key);
        &self.table[hash]
    }

    fn bucket_mut(&mut self, key: i32) -> &mut Vec<(i32, i32)> {
        let hash = self.hash(key);
        &mut self.table[hash]
    }
}
```

## 54. 螺旋矩阵{#leetcode-54}

[:link: 来源](https://leetcode-cn.com/problems/spiral-matrix/)

### 题目

给你一个 `m` 行 `n` 列的矩阵 `matrix`，请按照**顺时针螺旋顺序**，返回矩阵中的所有元素。

#### 示例

```raw
输入：matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
输出：[1, 2, 3, 6, 9, 8, 7, 4, 5]
```

```raw
输入：matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
输出：[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

#### 提示

- `m == len(matrix)`；
- `n == len(matrix[i])`；
- `1 <= m, n <= 10`；
- `-100 <= matrix[i][j] <= 100`。

### 题解

```rust Rust
impl Solution {
    pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
        let (m, n) = (matrix.len(), matrix.first().map_or(0, |row| row.len()));
        let s = Ord::min(m, n);
        let c = s / 2;
        let mut v = Vec::with_capacity(m * n);
        for i in 0..c {
            for j in i..n - i - 1 {
                v.push(matrix[i][j]);
            }
            for j in i..m - i - 1 {
                v.push(matrix[j][n - i - 1]);
            }
            for j in i..n - i - 1 {
                v.push(matrix[m - i - 1][n - j - 1]);
            }
            for j in i..m - i - 1 {
                v.push(matrix[m - j - 1][i]);
            }
        }
        if s % 2 == 1 {
            if m <= n {
                for j in c..n - c {
                    v.push(matrix[c][j]);
                }
            } else {
                for j in c..m - c {
                    v.push(matrix[j][c]);
                }
            }
        }
        v
    }
}
```

## 59. 螺旋矩阵 II{#leetcode-59}

[:link: 来源](https://leetcode-cn.com/problems/spiral-matrix-ii/)

### 题目

给你一个正整数 `n`，生成一个包含 `1` 到 `n * n` 所有整数，且整数按顺时针顺序螺旋排列的 $n\times n$ 正方形矩阵 `matrix`。

#### 示例

```raw
输入：n = 3
输出：[[1, 2, 3], [8, 9, 4], [7, 6, 5]]
```

```raw
输入：n = 1
输出：[[1]]
```

#### 提示

- `1 <= n <= 20`。

### 题解

```rust Rust
impl Solution {
    pub fn generate_matrix(n: i32) -> Vec<Vec<i32>> {
        let n = n as usize;
        let (c, mut k) = (n / 2, 1);
        let mut m = vec![vec![0; n]; n];
        for i in 0..c {
            for j in i..n - i - 1 {
                m[i][j] = k;
                k += 1;
            }
            for j in i..n - i - 1 {
                m[j][n - i - 1] = k;
                k += 1;
            }
            for j in i..n - i - 1 {
                m[n - i - 1][n - j - 1] = k;
                k += 1;
            }
            for j in i..n - i - 1 {
                m[n - j - 1][i] = k;
                k += 1;
            }
        }
        if n % 2 == 1 {
            m[c][c] = k;
        }
        m
    }
}
```

## 115. 不同的子序列{#leetcode-115}

[:link: 来源](https://leetcode-cn.com/problems/distinct-subsequences/)

### 题目

给定一个字符串 `s` 和一个字符串 `t`，计算在 `s` 的子序列中 `t` 出现的个数。

字符串的一个**子序列**是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，`"ACE"` 是 `"ABCDE"` 的一个子序列，而 `"AEC"` 不是）

题目数据保证答案符合 32 位带符号整数范围。

#### 示例

```raw
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
（上箭头符号 ^ 表示选取的字母）
rabbbit
^^^^ ^^
rabbbit
^^ ^^^^
rabbbit
^^^ ^^^
```

```raw
输入：s = "babgbag", t = "bag"
输出：5
解释：如下图所示, 有 5 种可以从 s 中得到 "bag" 的方案。
（上箭头符号 ^ 表示选取的字母）
babgbag
^^ ^
babgbag
^^    ^
babgbag
^    ^^
babgbag
  ^  ^^
babgbag
    ^^^
```

#### 提示

- `0 <= len(s), len(t) <= 1e3`；
- `s` 和 `t` 由英文字母组成。

### 题解

动态规划。状态压缩。

```rust Rust
impl Solution {
    pub fn num_distinct(s: String, t: String) -> i32 {
        num_distinct(&s, &t) as i32
    }
}

pub fn num_distinct(s: &str, t: &str) -> usize {
    let t: Vec<_> = t.chars().collect();
    let mut f = vec![0; t.len()];
    for c in s.chars() {
        for (j, d) in t.iter().copied().enumerate().rev() {
            if c == d {
                f[j] += f.get(j - 1).copied().unwrap_or(1);
            }
        }
    }
    f.last().copied().unwrap_or(0)
}
```

## 1603. 设计停车系统{#leetcode-1603}

[:link: 来源](https://leetcode-cn.com/problems/design-parking-system/)

### 题目

请你给一个停车场设计一个停车系统。停车场总共有三种不同大小的车位：大、中、小，每种尺寸分别有固定数目的车位。

请你实现 `ParkingSystem` 类：

- `ParkingSystem(int big, int medium, int small)` 初始化 `ParkingSystem` 类，三个参数分别对应每种停车位的数目；
- `bool addCar(int carType)` 检查是否有 `carType` 对应的停车位。 `carType` 有三种类型：大、中、小，分别用数字 `1`、`2`、`3` 表示。一辆车只能停在 `carType` 对应尺寸的停车位中。如果没有空车位，请返回 `false`，否则将该车停入车位并返回 `true`。

#### 示例

```raw
输入：
["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
[[1, 1, 0], [1], [2], [3], [1]]

输出：
[null, true, true, false, false]

解释：
ParkingSystem parkingSystem = new ParkingSystem(1, 1, 0);
parkingSystem.addCar(1); // 返回 true，因为有 1 个空的大车位
parkingSystem.addCar(2); // 返回 true，因为有 1 个空的中车位
parkingSystem.addCar(3); // 返回 false，因为没有空的小车位
parkingSystem.addCar(1); // 返回 false，因为没有空的大车位，唯一一个大车位已经被占据了
```

#### 提示

- `0 <= big, medium, small <= 1e3`；
- `carType` 取值为 `1`、`2` 或 `3`；
- 最多会调用 `addCar` 函数 `1e3` 次。

### 题解

```rust Rust
struct ParkingSystem([usize; 3]);

impl ParkingSystem {
    fn new(big: i32, medium: i32, small: i32) -> Self {
        Self([big as usize, medium as usize, small as usize])
    }

    fn add_car(&mut self, car_type: i32) -> bool {
        self.0.get_mut((car_type - 1) as usize).map_or(false, |c| {
            if *c > 0 {
                *c -= 1;
                true
            } else {
                false
            }
        })
    }
}
```

## 150. 逆波兰表达式求值{#leetcode-150}

[:link: 来源](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

### 题目

根据[逆波兰表示法](https://zh.wikipedia.org/wiki/逆波兰表示法)，求表达式的值。

有效的算符包括 `+`、`-`、`*`、`/`。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

#### 说明

- 整数除法只保留整数部分；
- 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 `0` 的情况。

#### 示例

```raw
输入：tokens = ["2", "1", "+", "3", "*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
```

```raw
输入：tokens = ["4", "13", "5", "/", "+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
```

```raw
输入：tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
输出：22
解释：该算式转化为常见的中缀算术表达式为
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

#### 提示

- `1 <= len(tokens) <= 1e4`；
- `tokens[i]` 要么是一个算符（`"+"`、`"-"`、`"*"`、`"/"`），要么是一个在范围 $[-200,200]$ 内的整数。

### 题解

写一个泛型版本。

```rust Rust
impl Solution {
    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        eval_rpn(&tokens).unwrap()
    }
}

use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

pub fn eval_rpn<T>(tokens: &[String]) -> Option<T>
where
    T: FromStr + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let mut stack = Vec::with_capacity((tokens.len() + 1) / 2);
    for token in tokens {
        if let Ok(num) = token.parse::<T>() {
            stack.push(num)
        } else {
            let rhs = stack.pop()?;
            let lhs = stack.pop()?;
            stack.push(match token.as_str() {
                "+" => lhs + rhs,
                "-" => lhs - rhs,
                "*" => lhs * rhs,
                "/" => lhs / rhs,
                _ => return None,
            })
        }
    }
    if stack.len() > 1 {
        return None;
    }
    stack.pop()
}
```

## 73. 矩阵置零{#leetcode-73}

[:link: 来源](https://leetcode-cn.com/problems/set-matrix-zeroes/)

### 题目

给定一个 $m\times n$ 的矩阵，如果一个元素为 `0`，则将其所在行和列的所有元素都设为 `0`。请使用**原地**算法。

#### 进阶

- 一个直观的解决方案是使用 $\mathrm{O}(mn)$ 的额外空间，但这并不是一个好的解决方案；
- 一个简单的改进方案是使用 $\mathrm{O}(m+n)$ 的额外空间，但这仍然不是最好的解决方案；
- 你能想出一个仅使用常量空间的解决方案吗？

#### 示例

```raw
输入：matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
输出：[[1, 0, 1], [0, 0, 0], [1, 0, 1]]
```

```raw
输入：matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
输出：[[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
```

#### 提示

- `m == len(matrix)`；
- `n == len(matrix[0])`；
- `1 <= m, n <= 200`；
- `-2 ** 31 <= matrix[i][j] <= 2 ** 31 - 1`。

### 题解

```rust Rust
impl Solution {
    pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
        let (m, n) = (matrix.len(), matrix.get(0).map_or(0, |row| row.len()));
        if m == 0 || n == 0 {
            return;
        }
        let first_row = matrix[0].contains(&0);
        for j in 0..n {
            for i in 0..m {
                if matrix[i][j] == 0 {
                    matrix[0][j] = 0;
                    break;
                }
            }
        }
        for i in 1..m {
            if matrix[i].contains(&0) {
                for j in 0..n {
                    matrix[i][j] = 0;
                }
            }
        }
        for j in 0..n {
            if matrix[0][j] == 0 {
                for i in 1..m {
                    matrix[i][j] = 0;
                }
            }
        }
        if first_row {
            for j in 0..n {
                matrix[0][j] = 0;
            }
        }
    }
}
```

## 190. 颠倒二进制位{#leetcode-190}

[:link: 来源](https://leetcode-cn.com/problems/reverse-bits/)

### 题目

颠倒给定的 32 位无符号整数的二进制位。

#### 示例

```raw
输入: 0b00000010100101000001111010011100
输出: 0b00111001011110000010100101000000
```

```raw
输入：0b11111111111111111111111111111101
输出：0b10111111111111111111111111111111
```

```raw
输入：0b00000010100101000001111010011100
输出：0b00111001011110000010100101000000
```

```raw
输入：0b11111111111111111111111111111101
输出：0b10111111111111111111111111111111
```

#### 提示

- 输入是一个 32 位无符号整数。

### 题解

分治。

```rust
impl Solution {
    pub fn reverse_bits(mut x: u32) -> u32 {
        x = (x >> 16) | (x << 16);
        x = ((x >> 8) & 0x00ff00ff) | ((x & 0x00ff00ff) << 8);
        x = ((x >> 4) & 0x0f0f0f0f) | ((x & 0x0f0f0f0f) << 4);
        x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
        x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
        x
    }
}
```
