---
title: LeetCode 月报 202104
date: 2021-04-01 00:00:00
tags: [LeetCode]
mathjax: true
---

心到神知、上供人吃。

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
