---
title: LeetCode 月报 202104
date: 2021-04-01 00:00:00
tags: [LeetCode]
mathjax: true
---

心到神知、上供人吃。共完成 17 题。

<!-- more -->

## 1006. 笨阶乘{#leetcode-1006}

[:link: 来源](https://leetcode-cn.com/problems/clumsy-factorial/)

### 题目

通常，正整数 `n` 的阶乘是所有小于或等于 `n` 的正整数的乘积。例如，`factorial(10) = 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1`。

相反，我们设计了一个笨阶乘 `clumsy`：在整数的递减序列中，我们以一个固定顺序的操作符序列来依次替换原有的乘法操作符：乘法（`*`），除法（`/`），加法（`+`）和减法（`-`）。

例如，`clumsy(10) = 10 * 9 / 8 + 7 - 6 * 5 / 4 + 3 - 2 * 1`。然而，这些运算仍然使用通常的算术运算顺序：我们在任何加、减步骤之前执行所有的乘法和除法步骤，并且按从左到右处理乘法和除法步骤。

另外，我们使用的除法是地板除法（floor division），所以 `10 * 9 / 8 = 11`。这保证结果是一个整数。

实现上面定义的笨函数：给定一个整数 `n`，它返回 `n` 的笨阶乘。

#### 示例

```raw
输入：4
输出：7
解释：7 = 4 * 3 / 2 + 1
```

```raw
输入：10
输出：12
解释：12 = 10 * 9 / 8 + 7 - 6 * 5 / 4 + 3 - 2 * 1
```

#### 提示

- `1 <= n <= 1e5`；
- 答案保证符合 32 位整数。

### 题解

数学题。首先确认 `m * (m − 1) / (m − 2) = m + 1`，再分类讨论求通项。

```rust Rust
impl Solution {
    pub fn clumsy(n: i32) -> i32 {
        if n <= 4 {
            [1, 2, 6, 7][n as usize - 1]
        } else {
            n + [1, 2, 2, -1][n as usize % 4]
        }
    }
}
```

## 1143. 最长公共子序列{#leetcode-1143}

[:link: 来源](https://leetcode-cn.com/problems/longest-common-subsequence/)

### 题目

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长**公共子序列**的长度。如果不存在**公共子序列**，返回 `0`。

一个字符串的**子序列**是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的**公共子序列**是这两个字符串所共同拥有的子序列。

#### 示例

```raw
输入：text1 = "abcde", text2 = "ace"
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
```

```raw
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。
```

```raw
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。
```

#### 提示

- `1 <= len(text1), len(text2) <= 1e3`；
- `text1` 和 `text2` 仅由小写英文字符组成。

### 题解

动态规划。设 $s=\overline{s_1s_2\dots s_m}$, $t=\overline{t_1t_2\dots t_n}$, $f_{i,j}=\mathrm{lcs}(\overline{s_1s_2\dots s_i},\overline{t_1t_2\dots t_j})$，则算法应返回 $f_{m,n}=\mathrm{lcs}(s,t)$，则有递推公式

$$
f_{i,j}=\begin{cases}
    0,                         & \text{if } i=0\vee j=0,\\
    f_{i-1,j-1},               & \text{if } i>0\wedge j>0\wedge s_i=s_j,\\
    \max(f_{i-1,j},f_{i,j-1}), & \text{otherwise}.
\end{cases}
$$

#### 二维状态

```rust Rust
impl Solution {
    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        longest_common_subsequence(&text1, &text2) as i32
    }
}

impl Solution {
    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        longest_common_subsequence(&text1, &text2) as i32
    }
}

pub fn longest_common_subsequence(s: &str, t: &str) -> usize {
    let (m, n) = (s.chars().count(), t.chars().count());
    let mut f = vec![vec![0; n + 1]; m + 1];
    for (i, c) in s.chars().enumerate() {
        for (j, d) in t.chars().enumerate() {
            f[i + 1][j + 1] = if c == d {
                f[i][j] + 1
            } else {
                Ord::max(f[i][j + 1], f[i + 1][j])
            };
        }
    }
    f[m][n]
}
```

#### 状态压缩

由于内层循环会覆盖掉左上方元素，需要通过 `ul` 记录。

```rust Rust
impl Solution {
    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        longest_common_subsequence(&text1, &text2) as i32
    }
}

pub fn longest_common_subsequence(s: &str, t: &str) -> usize {
    let n = t.chars().count();
    let mut f = vec![0; n + 1];
    for c in s.chars() {
        let mut ul = f[0];
        for (j, d) in t.chars().enumerate() {
            let tmp = f[j + 1];
            f[j + 1] = if c == d {
                ul + 1
            } else {
                Ord::max(f[j + 1], f[j])
            };
            ul = tmp;
        }
    }
    f[n]
}
```

## 781. 森林中的兔子{#leetcode-781}

[:link: 来源](https://leetcode-cn.com/problems/rabbits-in-forest/)

### 题目

森林中，每个兔子都有颜色。其中一些兔子（可能是全部）告诉你还有多少其他的兔子和自己有相同的颜色。我们将这些回答放在 `answers` 数组里。

返回森林中兔子的最少数量。

#### 示例

```raw
输入：answers = [1, 1, 2]
输出：5
解释：
两只回答了 1 的兔子可能有相同的颜色，设为红色；
之后回答了 2 的兔子不会是红色，否则他们的回答会相互矛盾，设回答了 2 的兔子为蓝色；
此外，森林中还应有另外 2 只蓝色兔子的回答没有包含在数组中。
因此森林中兔子的最少数量是 5，3 只回答的和 2 只没有回答的。
```

```raw
输入：answers = [10, 10, 10]
输出：11
```

```raw
输入：answers = []
输出：0
```

#### 说明

- `len(answers) < 1e3`；
- `0 <= answers[i] < 1e3`。

### 题解

贪心。对于每 $k+1$ 只回答了 $k$ 的兔子，至少要分配一种颜色。则对于回答了 $k$ 的 $v$ 只兔子，至少有 $\lceil\frac{v}{k+1}\rceil$ 种颜色，即至少有 $\lceil\frac{v}{k+1}\rceil(k+1)$ 只兔子。换言之，如果 $(k+1)\nmid v$，则至少有 $(k+1)-v\bmod(k+1)$ 只兔子没有回答。

```rust Rust
impl Solution {
    pub fn num_rabbits(answers: Vec<i32>) -> i32 {
        let answers: Vec<_> = answers.into_iter().map(|e| e as usize).collect();
        num_rabbits(&answers) as i32
    }
}

use std::collections::HashMap;

pub fn num_rabbits(answers: &[usize]) -> usize {
    let mut counts = HashMap::new();
    answers
        .iter()
        .copied()
        .for_each(|k| *counts.entry(k + 1).or_insert(0) += 1);
    counts
        .into_iter()
        .map(|(k, v)| if v % k == 0 { v } else { v + k - v % k })
        .sum()
}
```

## 80. 删除有序数组中的重复项 II{#leetcode-80}

[:link: 来源](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)

### 题目

给你一个有序数组 `nums`，请你**原地**删除重复出现的元素，使每个元素**最多出现两次**，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在**原地**修改输入数组，并在使用 $\mathrm{O}(1)$ 额外空间的条件下完成。

#### 示例

```raw
输入：nums = [1, 1, 1, 2, 2, 3]
输出：5, nums = [1, 1, 2, 2, 3]
解释：函数应返回新长度 5，并且原数组的前五个元素被修改为 1, 1, 2, 2, 3。不需要考虑数组中超出新长度后面的元素。
```

```raw
输入：nums = [0, 0, 1, 1, 1, 1, 2, 3, 3]
输出：7, nums = [0, 0, 1, 1, 2, 3, 3]
解释：函数应返回新长度 7，并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3。不需要考虑数组中超出新长度后面的元素。
```

#### 提示

- `0 <= len(nums) <= 3e4`；
- `-1e4 <= nums[i] <= 1e4`；
- `nums` 已按升序排列。

### 题解

双指针检查重复并交换位置。`nums[i] = nums[j]` 赋值是拷贝语义，而 `Vec<T>::swap` 是移动语义不要求 `T: Copy`。

```rust Rust
impl Solution {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        remove_duplicates(nums, 2) as i32
    }
}

pub fn remove_duplicates(nums: &mut [impl Eq], k: usize) -> usize {
    let (mut i, mut j) = (k, k);
    while j < nums.len() {
        if nums[i - k] != nums[j] {
            nums.swap(i, j);
            i += 1;
        }
        j += 1;
    }
    i.min(nums.len())
}
```

## 198. 打家劫舍{#leetcode-198}

[:link: 来源](https://leetcode-cn.com/problems/house-robber/)

### 题目

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你**不触动警报装置的情况下**，一夜之内能够偷窃到的最高金额。

#### 示例

```raw
输入：[1, 2, 3, 1]
输出：4
解释：偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。偷窃到的最高金额 = 1 + 3 = 4。
```

```raw
输入：[2, 7, 9, 3, 1]
输出：12
解释：偷窃 1 号房屋（金额 = 2），偷窃 3 号房屋（金额 = 9），接着偷窃 5 号房屋（金额 = 1）。偷窃到的最高金额 = 2 + 9 + 1 = 12。
```

#### 提示

- `0 <= len(nums) <= 1e2`；
- `0 <= nums[i] <= 4e2`。

### 题解

动态规划。

```rust Rust
impl Solution {
    pub fn rob(nums: Vec<i32>) -> i32 {
        nums.into_iter().fold((0, 0), |(a, b), n| (a.max(n + b), a)).0
    }
}
```

## 213. 打家劫舍 II{#leetcode-213}

[:link: 来源](https://leetcode-cn.com/problems/house-robber-ii/)

### 题目

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都**围成一圈**，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你**在不触动警报装置的情况下**，能够偷窃到的最高金额。

#### 示例

```raw
输入：nums = [2, 3, 2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2），因为他们是相邻的。
```

```raw
输入：nums = [1, 2, 3, 1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。偷窃到的最高金额 = 1 + 3 = 4 。
```

```raw
输入：nums = [0]
输出：0
```

#### 提示

- `1 <= len(nums) <= 1e2`；
- `0 <= nums[i] <= 1e3`。

### 题解

动态规划。

```rust Rust
impl Solution {
    pub fn rob(nums: Vec<i32>) -> i32 {
        let nums: Vec<_> = nums.into_iter().map(|n| n as usize).collect();
        rob_circular(&nums) as i32
    }
}

pub fn rob(nums: &[usize]) -> usize {
    nums.iter().fold((0, 0), |(a, b), n| (a.max(n + b), a)).0
}

pub fn rob_circular(nums: &[usize]) -> usize {
    match nums {
        [] => 0,
        [single] => *single,
        _ => usize::max(
            rob(nums.split_first().unwrap().1),
            rob(nums.split_last().unwrap().1),
        ),
    }
}
```

## 263. 丑数{#leetcode-263}

[:link: 来源](https://leetcode-cn.com/problems/ugly-number/)

### 题目

给你一个整数 `n`，请你判断 `n` 是否为**丑数**。如果是，返回 `true`；否则，返回 `false`。

**丑数**就是只包含质因数 `2`、`3`、`5` 的正整数。

#### 示例

```raw
输入：n = 6
输出：true
解释：6 = 2 * 3
```

```raw
输入：n = 8
输出：true
解释：8 = 2 * 2 * 2
```

```raw
输入：n = 14
输出：false
解释：14 不是丑数，因为它包含了另外一个质因数 7。
```

```raw
输入：n = 1
输出：true
解释：1 通常被视为丑数。
```

#### 提示

- `-2 ** 31 <= n <= 2 ** 31 - 1`。

### 题解

```rust Rust
impl Solution {
    pub fn is_ugly(n: i32) -> bool {
        if n <= 0 {
            return false;
        }
        is_ugly(n as usize, &[2, 3, 5])
    }
}

pub fn is_ugly(mut n: usize, factors: &[usize]) -> bool {
    for factor in factors {
        while n % factor == 0 {
            n /= factor;
        }
    }
    n == 1
}
```

## 264. 丑数 II{#leetcode-264}

[:link: 来源](https://leetcode-cn.com/problems/ugly-number-ii/)

### 题目

给你一个整数 `n`，请你找出并返回第 `n` 个**丑数**。

**丑数**就是只包含质因数 `2`、`3`、`5` 的正整数。

#### 示例

```raw
输入：n = 10
输出：12
解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
```

```raw
输入：n = 1
输出：1
解释：1 通常被视为丑数。
```

#### 提示

- `1 <= n <= 1690`。

### 题解

```rust Rust
impl Solution {
    pub fn nth_ugly_number(n: i32) -> i32 {
        nth_ugly_number(n as usize, &[2, 3, 5]) as i32
    }
}

pub fn nth_ugly_number(n: usize, factors: &[usize]) -> usize {
    let mut ugly_numbers = Vec::with_capacity(n);
    ugly_numbers.push(1);
    let mut indices = vec![0; factors.len()];
    let mut candidates = Vec::new();
    while ugly_numbers.len() < n {
        candidates.clear();
        candidates.extend(
            indices
                .iter()
                .map(|&i| ugly_numbers[i])
                .zip(factors)
                .map(|(i, f)| i * f),
        );
        let next = candidates.iter().min().copied().unwrap();
        ugly_numbers.push(next);
        Iterator::zip(indices.iter_mut(), candidates.iter())
            .filter_map(|(i, c)| if *c == next { Some(i) } else { None })
            .for_each(|i| *i += 1);
    }
    ugly_numbers.last().copied().unwrap()
}

// Once [const generics](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md) is stablized.
// pub fn nth_ugly_number<const N: usize>(n: usize, factors: &[usize; N]) -> usize {
//     let mut ugly_numbers = Vec::with_capacity(n);
//     ugly_numbers.push(1);
//     let (mut indices, mut candidates) = ([0; N], [1; N]);
//     while ugly_numbers.len() < n {
//         for i in 0..N {
//             candidates[i] = ugly_numbers[indices[i]] * factors[i];
//         }
//         let next = candidates.iter().min().copied().unwrap();
//         ugly_numbers.push(next);
//         for i in 0..N {
//             if candidates[i] == next {
//                 indices[i] += 1;
//             }
//         }
//     }
//     ugly_numbers.last().copied().unwrap()
// }
```

## 179. 最大数{#leetcode-179}

[:link: 来源](https://leetcode-cn.com/problems/largest-number/)

### 题目

给定一组非负整数 `nums`，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

#### 注意

输出结果可能非常大，所以你需要返回一个字符串而不是整数。

#### 示例

```raw
输入：nums = [10, 2]
输出："210"
```

```raw
输入：nums = [3, 30, 34, 5, 9]
输出："9534330"
```

```raw
输入：nums = [1]
输出："1"
```

```raw
输入：nums = [10]
输出："10"
```

```raw
输入：nums = [0, 0, 0]
输出："0"
```

#### 提示

- `1 <= len(nums) <= 1e2`；
- `0 <= nums[i] <= 1e9`。

### 题解

贪心。

#### 字符串拼接比较

```rust Rust
impl Solution {
    pub fn largest_number(nums: Vec<i32>) -> String {
        let mut nums: Vec<_> = nums.into_iter().map(|n| n.to_string()).collect();
        nums.sort_unstable_by(|a, b| Ord::cmp(&(b.clone() + &a), &(a.clone() + &b)));
        if let None | Some("0") = nums.first().map(String::as_str) {
            "0".to_string()
        } else {
            nums.concat()
        }
    }
}
```

#### 迭代器拼接比较

```rust Rust
impl Solution {
    pub fn largest_number(nums: Vec<i32>) -> String {
        let mut nums: Vec<_> = nums.into_iter().map(|n| n.to_string()).collect();
        nums.sort_unstable_by(|a, b| {
            Iterator::cmp(
                b.as_bytes().iter().chain(a.as_bytes()),
                a.as_bytes().iter().chain(b.as_bytes()),
            )
        });
        if let None | Some("0") = nums.first().map(String::as_str) {
            "0".to_string()
        } else {
            nums.concat()
        }
    }
}
```

#### 整数拼接比较

```rust Rust
impl Solution {
    pub fn largest_number(mut nums: Vec<i32>) -> String {
        nums.sort_unstable_by(|&a, &b| {
            let (a, mut s) = (a as u64, 10);
            while s <= a {
                s *= 10;
            }
            let (b, mut t) = (b as u64, 10);
            while t <= b {
                t *= 10;
            }
            Ord::cmp(&(s * b + a), &(t * a + b))
        });
        if let None | Some(0) = nums.first() {
            "0".to_string()
        } else {
            nums.into_iter().map(|x| x.to_string()).collect()
        }
    }
}
```

## 1680. 连接连续二进制数字{#leetcode-1680}

[:link: 来源](https://leetcode-cn.com/problems/concatenation-of-consecutive-binary-numbers/)

### 题目

给你一个整数 `n`，请你将 `1` 到 `n` 的二进制表示连接起来，并返回连接结果对应的**十进制**数对 `1e9 + 7` 取余的结果。

#### 示例

```raw
输入：n = 1
输出：1
解释：二进制的 "1" 对应着十进制的 1。
```

```raw
输入：n = 3
输出：27
解释：
二进制下，1、2、3 分别对应 "1"、"10"、"11"；
将它们依次连接，我们得到 "11011"，对应着十进制的 27。
```

```raw
输入：n = 12
输出：505379714
解释：
连接结果为 "1101110010111011110001001101010111100"；
对应的十进制数字为 118505380540；
对 1e9 + 7 取余后，结果为 505379714。
```

#### 提示

- `1 <= n <= 1e5`。

### 题解

#### 直接模拟

```rust Rust
impl Solution {
    pub fn concatenated_binary(n: i32) -> i32 {
        concatenated_binary(n as usize, 1_000_000_007) as i32
    }
}

pub fn concatenated_binary(n: usize, m: usize) -> usize {
    let mut r = 0;
    for i in 1..=n {
        r = (r * (i + 1).next_power_of_two() + i) % m;
    }
    r
}
```

#### 优化模拟

```rust Rust
impl Solution {
    pub fn concatenated_binary(n: i32) -> i32 {
        concatenated_binary(n as usize, 1_000_000_007) as i32
    }
}

pub fn concatenated_binary(n: usize, m: usize) -> usize {
    let (mut r, mut b) = (0, 0);
    for i in 1..=n {
        // or check if i & (i - 1) == 0
        if i.is_power_of_two() {
            p += 1;
        }
        r = ((r << b) + i) % m;
    }
    r
}
```

## 208. 实现 Trie (前缀树){#leetcode-208}

[:link: 来源](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

### 题目

Trie（发音类似 "try"）或者说**前缀树**是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 `Trie` 类：

- `Trie()` 初始化前缀树对象；
- `void insert(String word)` 向前缀树中插入字符串 `word`；
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即在检索之前已经插入），否则，返回 `false`；
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix`，返回 `true`，否则，返回 `false`。

#### 示例

```raw
输入：
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]

输出：
[null, null, true, false, true, null, true]

解释：
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 true
trie.search("app");     // 返回 false
trie.startsWith("app"); // 返回 true
trie.insert("app");
trie.search("app");     // 返回 true
```

#### 提示

- `1 <= len(word), len(prefix) <= 2e3`；
- `word` 和 `prefix` 仅由小写英文字母组成；
- `insert`、`search` 和 `startsWith` 调用次数**总计**不超过 `3e4` 次。

### 题解

```rust Rust
use std::collections::HashMap;

#[derive(Default)]
pub struct Trie(Option<Node>);

#[derive(Default)]
struct Node {
    children: HashMap<char, Node>,
    contained: bool,
}

impl Trie {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert(&mut self, word: String) {
        let mut node = self.0.get_or_insert_with(Default::default);
        for c in word.chars() {
            node = node.children.entry(c).or_default();
        }
        node.contained = true;
    }

    pub fn search(&self, word: String) -> bool {
        self.node(&word).map_or(false, |w| w.contained)
    }

    pub fn starts_with(&self, prefix: String) -> bool {
        self.node(&prefix).is_some()
    }

    fn node(&self, s: &str) -> Option<&Node> {
        let mut node = self.0.as_ref()?;
        for c in s.chars() {
            node = node.children.get(&c)?;
        }
        node.into()
    }
}
```

## 87. 扰乱字符串{#leetcode-87}

[:link: 来源](https://leetcode-cn.com/problems/scramble-string/)

### 题目

使用下面描述的算法可以扰乱字符串 `s` 得到字符串 `t`：

- 如果字符串的长度为 `1`，算法停止；
- 如果字符串的长度大于 `1`，执行下述步骤：
  1. 在一个**随机**下标处将字符串分割成两个非空的子字符串。即如果已知字符串 `s`，则可以将其分成两个子字符串 `x` 和 `y`，且满足 `s = x + y`；
  2. **随机**决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即在执行这一步骤之后，`s` 可能是 `s = x + y` 或者 `s = y + x`。
  3. 在 `x` 和 `y` 这两个子字符串上继续从步骤 1 开始递归执行此算法。

给你两个**长度相等**的字符串 `s1` 和 `s2`，判断 `s2` 是否是 `s1` 的扰乱字符串。如果是，返回 `true`，否则返回 `false`。

#### 示例

```raw
输入：s1 = "great", s2 = "rgeat"
输出：true
解释：
(1) "great" -> "gr", "eat"
(2) "gr" -> "r", "g"; "eat" -> "e", "at"
(3) "r"; "g"; "e"; "at" -> "a", "t"
这是一种能够扰乱 s1 得到 s2 的情形，可以认为 s2 是 s1 的扰乱字符串，返回 true。
```

```raw
输入：s1 = "abcde", s2 = "caebd"
输出：false
```

```raw
输入：s1 = "a", s2 = "a"
输出：true
```

#### 提示

- `len(s1) == len(s2)`；
- `1 <= len(s1) <= 30`；
- `s1` 和 `s2` 由小写英文字母组成。

### 题解

记忆化搜索。

```rust Rust
impl Solution {
    pub fn is_scramble(s1: String, s2: String) -> bool {
        is_scramble(&s1, &s2)
    }
}

struct Memo<'a> {
    v: Vec<Option<bool>>,
    l: usize,
    s: &'a [char],
    t: &'a [char],
}

impl<'a> Memo<'a> {
    pub fn new(s: &'a [char], t: &'a [char]) -> Memo<'a> {
        let l = s.len() + 1;
        let v = vec![None; l * l * l * l];
        Memo { v, l, s, t }
    }

    fn idx(&self, s_low: usize, s_high: usize, t_low: usize, t_high: usize) -> usize {
        let mut idx = 0;
        idx = idx * self.l + s_low;
        idx = idx * self.l + s_high;
        idx = idx * self.l + t_low;
        idx = idx * self.l + t_high;
        idx
    }

    pub fn search(&mut self, s_low: usize, s_high: usize, t_low: usize, t_high: usize) -> bool {
        let idx = self.idx(s_low, s_high, t_low, t_high);
        self.v[idx].unwrap_or_else(|| {
            let len = s_high - s_low;
            let r = match len {
                0 => true,
                1 => self.s[s_low] == self.t[t_low],
                _ => (1..len).any(|p| {
                    self.search(s_low, s_low + p, t_low, t_low + p)
                        && self.search(s_low + p, s_high, t_low + p, t_high)
                        || self.search(s_low, s_low + p, t_high - p, t_high)
                            && self.search(s_low + p, s_high, t_low, t_high - p)
                }),
            };
            self.v[idx] = r.into();
            r
        })
    }
}

pub fn is_scramble(s: &str, t: &str) -> bool {
    let (ref s, ref t) = (s.chars().collect::<Vec<_>>(), t.chars().collect::<Vec<_>>());
    Memo::new(s, t).search(0, s.len(), 0, t.len())
}
```

## 849. 到最近的人的最大距离{#leetcode-849}

[:link: 来源](https://leetcode-cn.com/problems/maximize-distance-to-closest-person/)

### 题目

给你一个数组 `seats` 表示一排座位，其中 `seats[i] == 1` 代表有人坐在第 `i` 个座位上，`seats[i] == 0` 代表座位 `i` 上是空的（下标从 `0` 开始）。

至少有一个空座位，且至少有一人已经坐在座位上。

亚历克斯希望坐在一个能够使他与离他最近的人之间的距离达到最大化的座位上。

返回他到离他最近的人的最大距离。

#### 示例

```raw
输入：seats = [1, 0, 0, 0, 1, 0, 1]
输出：2
解释：如果亚历克斯坐在第二个空位（seats[2]）上，他到离他最近的人的距离为 2。如果亚历克斯坐在其它任何一个空位上，他到离他最近的人的距离为 1。因此，他到离他最近的人的最大距离是 2。
```

```raw
输入：seats = [1, 0, 0, 0]
输出：3
解释：如果亚历克斯坐在最后一个座位上，他离最近的人有 3 个座位远。这是可能的最大距离，所以答案是 3。
```

```raw
输入：seats = [0, 1]
输出：1
```

#### 提示

- `2 <= len(seats) <= 2e4`；
- `seats[i] in (0, 1)`；
- 至少有一个**空座位**；
- 至少有一个**座位上有人**。

### 题解

```rust Rust
impl Solution {
    pub fn max_dist_to_closest(seats: Vec<i32>) -> i32 {
        max_dist_to_closest(&seats.into_iter().map(|s| s != 0).collect::<Vec<_>>()) as i32
    }
}

pub fn max_dist_to_closest(seats: &[bool]) -> usize {
    let seat_indices = seats.iter().enumerate().filter_map(|(i, &s)| if s { Some(i) } else { None });
    let (last, max) = seat_indices.fold((None, 0), |(pre, max), now| {
        (Some(now), pre.map_or(now, |pre| max.max((now - pre) / 2)))
    });
    max.max(seats.len() - last.unwrap() - 1)
}
```

## 377. 组合总和 Ⅳ{#leetcode-377}

[:link: 来源](https://leetcode-cn.com/problems/combination-sum-iv/)

### 题目

给你一个由**不同**整数组成的数组 `nums`，和一个目标整数 `target`。请你从 `nums` 中找出并返回总和为 `target` 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。

#### 示例

```raw
输入：nums = [1, 2, 3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。
```

```raw
输入：nums = [9], target = 3
输出：0
```

#### 提示

- `1 <= len(nums) <= 2e2`;
- `1 <= nums[i] <= 1e3`；
- `nums` 中的所有元素**互不相同**；
- `1 <= target <= 1e3`。

### 题解

动态规划。

```rust Rust
impl Solution {
    pub fn combination_sum4(nums: Vec<i32>, target: i32) -> i32 {
        let nums: Vec<_> = nums.into_iter().map(|n| n as usize).collect();
        combination_sum4(&nums, target as usize) as i32
    }
}

pub fn combination_sum4(nums: &[usize], target: usize) -> usize {
    let mut memo = Vec::with_capacity(target + 1);
    memo.push(1);
    for i in 1..=target {
        memo.push(nums.iter().filter_map(|n| memo.get(i - n)).sum());
    }
    memo[target]
}
```

## 938. 二叉搜索树的范围和{#leetcode-938}

[:link: 来源](https://leetcode-cn.com/problems/range-sum-of-bst/)

### 题目

给定二叉搜索树的根结点 `root`，返回值位于范围 $[low,high]$ 之间的所有结点的值的和。

#### 示例

```raw
输入：root = [10, 5, 15, 3, 7, null, 18], low = 7, high = 15
输出：32
```

```raw
输入：root = [10, 5, 15, 3, 7, 13, 18, 1, null, 6], low = 6, high = 10
输出：23
```

#### 提示

- 树中节点数目在范围 $[1,2\times{10}^4]$ 内；
- `1 <= node.val <= 1e5`；
- `1 <= low <= high <= 1e5`；
- 所有 `node.val` 互不相同。

### 题解

深度优先搜索，剪枝。

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
    pub fn range_sum_bst(root: Option<Rc<RefCell<TreeNode>>>, low: i32, high: i32) -> i32 {
        let (mut stack, mut sum) = (root.into_iter().collect::<Vec<_>>(), 0);
        while let Some(node) = stack.pop() {
            let node = node.borrow();
            let (left, right) = (node.val >= low, node.val <= high);
            if left && right {
                sum += node.val;
            }
            if left {
                stack.extend(node.left.clone());
            }
            if right {
                stack.extend(node.right.clone());
            }
        }
        sum
    }
}
```

## 633. 平方数之和{#leetcode-633}

[:link: 来源](https://leetcode-cn.com/problems/sum-of-square-numbers/)

### 题目

给定一个非负整数 `c`，你要判断是否存在两个整数 `a` 和 `b`，使得 $a^2+b^2=c$。

#### 示例

```raw
输入：c = 5
输出：true
解释：1 * 1 + 2 * 2 = 5
```

```raw
输入：c = 3
输出：false
```

```raw
输入：c = 4
输出：true
```

```raw
输入：c = 2
输出：true
```

```raw
输入：c = 1
输出：true
```

#### 提示

- `0 <= c <= 2 ** 31 - 1`。

### 题解

```rust Rust
impl Solution {
    pub fn judge_square_sum(c: i32) -> bool {
        judge_square_sum(c as usize)
    }
}

use std::cmp::Ordering;

pub fn judge_square_sum(c: usize) -> bool {
    let (mut a, mut b) = (0, (c as f64).sqrt() as usize);
    while a <= b {
        match Ord::cmp(&c, &(a * a + b * b)) {
            Ordering::Greater => a += 1,
            Ordering::Less => b -= 1,
            Ordering::Equal => return true,
        }
    }
    false
}
```

## 403. 青蛙过河{#leetcode-403}

[:link: 来源](https://leetcode-cn.com/problems/frog-jump/)

### 题目

一只青蛙想要过河。假定河流被等分为若干个单元格，并且在每一个单元格内都有可能放有一块石子（也有可能没有）。青蛙可以跳上石子，但是不可以跳入水中。

给你石子的位置列表 `stones`（用单元格序号**升序**表示），请判定青蛙能否成功过河（即能否在最后一步跳至最后一块石子上）。

开始时，青蛙默认已站在第一块石子上，并可以假定它第一步只能跳跃一个单位（即只能从单元格 1 跳至单元格 2）。

如果青蛙上一步跳跃了 `k` 个单位，那么它接下来的跳跃距离只能选择为 `k - 1`、`k` 或 `k + 1` 个单位。另请注意，青蛙只能向前方（终点的方向）跳跃。

#### 示例

```raw
输入：stones = [0, 1, 3, 5, 6, 8, 12, 17]
输出：true
解释：青蛙可以成功过河，按照如下方案跳跃：跳 1 个单位到第 2 块石子，然后跳 2 个单位到第 3 块石子，接着 跳 2 个单位到第 4 块石子，然后跳 3 个单位到第 6 块石子，跳 4 个单位到第 7 块石子，最后，跳 5 个单位到第 8 个石子（即最后一块石子）。
```

```raw
输入：stones = [0, 1, 2, 3, 4, 8, 9, 11]
输出：false
解释：这是因为第 5 和第 6 个石子之间的间距太大，没有可选的方案供青蛙跳跃过去。
```

#### 提示

- `2 <= len(stones) <= 2e3`；
- `0 <= stones[i] <= 2 ** 31 - 1`；
- `stones[0] == 0`。

### 题解

模拟递推。

```rust Rust
impl Solution {
    pub fn can_cross(stones: Vec<i32>) -> bool {
        let stones: Vec<_> = stones.into_iter().map(|s| s as usize).collect();
        can_cross(&stones, &[-1, 0, 1])
    }
}

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

pub fn can_cross(stones: &[usize], deltas: &[isize]) -> bool {
    let steps_per_stone: HashMap<_, RefCell<HashSet<_>>> = stones
        .iter()
        .map(|&stone| (stone, Default::default()))
        .collect();
    steps_per_stone[&0].borrow_mut().insert(1);
    for &stone in stones {
        for &step in steps_per_stone[&stone].borrow().iter() {
            steps_per_stone.get(&(stone + step)).map(|steps| {
                steps.borrow_mut().extend(deltas.iter().filter_map(|delta| {
                    let next_step = step as isize + delta;
                    if next_step > 0 {
                        Some(next_step as usize)
                    } else {
                        None
                    }
                }))
            });
        }
    }
    stones.last().map_or(true, |last_stone| {
        steps_per_stone[last_stone].borrow().len() > 0
    })
}
```
