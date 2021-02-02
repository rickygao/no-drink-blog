---
title: LeetCode 月报 202102
date: 2021-02-01 00:00:00
tags: [LeetCode]
mathjax: true
---

快过年了。重启「Rust 从入门到放弃」计划。

<!-- more -->

## 888. 公平的糖果棒交换{#leetcode-888}

[:link: 来源](https://leetcode-cn.com/problems/fair-candy-swap/)

### 题目

爱丽丝和鲍勃有不同大小的糖果棒：`A[i]` 是爱丽丝拥有的第 `i` 根糖果棒的大小，`B[j]` 是鲍勃拥有的第 `j` 根糖果棒的大小。

因为他们是朋友，所以他们想交换一根糖果棒，这样交换后，他们都有相同的糖果总量。（一个人拥有的糖果总量是他们拥有的糖果棒大小的总和）

返回一个整数数组 `ans`，其中 `ans[0]` 是爱丽丝必须交换的糖果棒的大小，`ans[1]` 是 Bob 必须交换的糖果棒的大小。

如果有多个答案，你可以返回其中任何一个。保证答案存在。

#### 示例

```raw
输入：A = [1, 1], B = [2, 2]
输出：[1, 2]
```

```raw
输入：A = [1, 2], B = [2, 3]
输出：[1, 2]
```

```raw
输入：A = [2], B = [1, 3]
输出：[2, 3]
```

```raw
输入：A = [1, 2, 5], B = [2, 4]
输出：[5, 4]
```

#### 提示

- `1 <= len(A), len(B) <= 1e4`；
- `1 <= A[i], B[i] <= 1e5`；
- `sum(A) != sum(B)`；
- 答案肯定存在。

### 题解

```python Python
class Solution:
    def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
        return fair_candy_swap(A, B)

def fair_candy_swap(a: List[int], b: List[int]) -> List[int]:
    d = (sum(a) - sum(b)) // 2
    a = set(a)
    for y in b:
        if (x := y + d) in a:
            return [x, y]
```

```rust Rust
impl Solution {
    pub fn fair_candy_swap(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {
        _fair_candy_swap(&a, &b).map(|(x, y)| vec![x, y]).unwrap()
    }
}

use std::collections::HashSet;

pub fn fair_candy_swap(a: &[i32], b: &[i32]) -> Option<(i32, i32)> {
    let s: i32 = a.iter().sum();
    let t: i32 = b.iter().sum();
    let d = s - t;
    if d % 2 == 1 {
        return None;
    }
    let d = d / 2;
    let a: HashSet<_> = a.iter().collect();
    for &y in b {
        let x = y + d;
        if a.contains(&x) {
            return Some((x, y));
        }
    }
    None
}
```

## 424. 替换后的最长重复字符{#leetcode-424}

[:link: 来源](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

### 题目

给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 `k` 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

#### 注意

字符串长度和 `k` 不会超过 `1e4`。

#### 示例

```raw
输入：s = "ABAB", k = 2
输出：4
解释：用两个 'A' 替换为两个 'B'，反之亦然。
```

```raw
输入：s = "AABABBA", k = 1
输出：4
解释：将中间的一个 'A' 替换为 'B'，字符串变为 "AABBBBA"。子串 "BBBB" 有最长重复字母，答案为 4。
```

### 题解

双指针。

```python Python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        return character_replacement(s, k)

from collections import Counter

def character_replacement(s: str, k: int) -> int:
    c = Counter()
    i = j = m = 0
    while j < len(s):
        c[s[j]] += 1
        m = max(m, c[s[j]])
        j += 1
        if j - i - m > k:
            c[s[i]] -= 1
            i += 1
    return j - i
```

```rust Rust
impl Solution {
    pub fn character_replacement(s: String, k: i32) -> i32 {
        character_replacement(&s, k as usize) as i32
    }
}

use std::collections::HashMap;

pub fn character_replacement(s: &str, k: usize) -> usize {
    let mut c: HashMap<_, usize> = HashMap::new();
    let (mut m, mut l) = (0, 0);
    let (mut p, mut q) = (s.chars(), s.chars());
    while let Some(j) = q.next() {
        m = m.max(*c.entry(j).and_modify(|cj| *cj += 1).or_insert(1));
        if l >= k + m {
            *c.get_mut(&p.next().unwrap()).unwrap() -= 1;
        } else {
            l += 1;
        }
    }
    l
}
```
