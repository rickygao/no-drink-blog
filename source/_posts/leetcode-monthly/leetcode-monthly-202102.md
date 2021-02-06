---
title: LeetCode 月报 202102
date: 2021-02-01 00:00:00
tags: [LeetCode]
mathjax: true
---

快过年了。重启「Rust 从入门到放弃」计划。

本月大概是滑动窗口月了。

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

## 480. 滑动窗口中位数{#leetcode-480}

[:link: 来源](https://leetcode-cn.com/problems/sliding-window-median/)

### 题目

中位数是有序序列最中间的那个数。如果序列的大小是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

例如：

- `[2, 3, 4]` 的中位数是 `3`；
- `[2, 3]` 的中位数是 `(2 + 3) / 2 = 2.5`。

给你一个数组 `nums`，有一个大小为 `k` 的窗口从最左端滑动到最右端。窗口中有 `k` 个数，每次窗口向右移动一位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。

#### 示例

```raw
输入：nums = [1, 3, -1,- 3, 5, 3, 6, 7], k = 3
输出：[1, -1, -1, 3, 5, 6]
解释：滑动窗口的位置和中位数如下。
----------------------------+----
 [1  3  -1] -3  5  3  6  7  |  1
  1 [3  -1  -3] 5  3  6  7  | -1
  1  3 [-1  -3  5] 3  6  7  | -1
  1  3  -1 [-3  5  3] 6  7  |  3
  1  3  -1  -3 [5  3  6] 7  |  5
  1  3  -1  -3  5 [3  6  7] |  6
----------------------------+----
```

#### 提示

- 你可以假设 `k` 始终有效，即 `k` 始终小于输入的非空数组的元素个数；
- 与真实值误差在 `1e-5` 以内的答案将被视作正确答案。

### 题解

二分查找。维护有序窗口。然后，写一个生成器吧！

```python Python
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        return list(generate_median_sliding_window(nums, k))

from bisect import bisect_left
from collections import deque

def generate_median_sliding_window(nums: Iterator[int], k: int) -> Iterator[int]:
    m, r = divmod(k, 2)
    w, sw = deque(), []
    for n in nums:
        w.append(n)
        sw.insert(bisect.bisect_left(sw, n), n)
        if len(w) < k:
            continue
        if len(w) > k:
            sw.pop(bisect.bisect_left(sw, w.popleft()))
        yield sw[m] if r else (sw[m] + sw[m - 1]) / 2
```

尚难驾驭 Rust 的类型体操和生命周期，暂时不写迭代器适配器。

```rust Rust
impl Solution {
    pub fn median_sliding_window(nums: Vec<i32>, k: i32) -> Vec<f64> {
        median_sliding_window(&nums, k as usize)
    }
}

pub fn median_sliding_window(nums: &[i32], k: usize) -> Vec<f64> {
    if nums.len() < k || k == 0 {
        return Vec::new();
    }
    let mut sorted_window = nums[..k].iter().collect::<Vec<_>>();
    sorted_window.sort_unstable();
    let median = |sorted_window: &Vec<&i32>| if k % 2 == 1 {
        *sorted_window[k / 2] as f64
    } else {
        (*sorted_window[k / 2] as f64 + *sorted_window[k / 2 - 1] as f64) / 2f64
    };
    let mut medians = Vec::with_capacity(nums.len() - k + 1);
    medians.push(median(&sorted_window));
    for (m, n) in Iterator::zip(nums.iter(), nums[k..].iter()) {
        sorted_window.remove(sorted_window.binary_search(&m).unwrap());
        sorted_window.insert(sorted_window.binary_search(&n).unwrap_or_else(|i| i), n);
        medians.push(median(&sorted_window));
    }
    medians
}
```

## 643. 子数组最大平均数 I{#leetcode-643}

[:link: 来源](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

### 题目

给定 `n` 个整数，找出平均数最大且长度为 `k` 的连续子数组，并输出该最大平均数。

#### 示例

```raw
输入：[1, 12, -5, -6, 50, 3], k = 4
输出：12.75
解释：最大平均数 (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75
```

#### 提示

- `1 <= k <= n <= 3e4`；
- 所给数据范围 $[-{10}^4，{10}^4]$。

### 题解

```python Python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        return find_max_average(nums, k)

from collections import deque

def find_max_average(nums: List[int], k: int) -> float:
    return max(generate_sum_sliding_window(nums, k)) / k

def generate_sum_sliding_window(nums: Iterator[int], k: int) -> Iterator[int]:
    w, s = deque(), 0
    for n in nums:
        w.append(n)
        s += n
        if len(w) < k:
            continue
        if len(w) > k:
            s -= w.popleft()
        yield s
```

```rust Rust
impl Solution {
    pub fn find_max_average(nums: Vec<i32>, k: i32) -> f64 {
        find_max_average(&nums, k as usize).unwrap()
    }
}

pub fn find_max_average(nums: &[i32], k: usize) -> Option<f64> {
    if nums.len() < k || k == 0 {
        return None;
    }
    let mut sum = nums[..k].iter().sum::<i32>();
    let mut max = sum;
    for (m, n) in Iterator::zip(nums.iter(), nums[k..].iter()) {
        sum = sum - m + n;
        max = max.max(sum);
    }
    Some((max as f64) / (k as f64))
}
```

## 1208. 尽可能使字符串相等{#leetcode-1208}

[:link: 来源](https://leetcode-cn.com/problems/get-equal-substrings-within-budget/)

### 题目

给你两个长度相同的字符串，`s` 和 `t`。

将 `s` 中的第 `i` 个字符变到 `t` 中的第 `i` 个字符需要 `abs(s[i] - t[i])` 的开销（开销可能为 `0`），也就是两个字符的 ASCII 码值的差的绝对值。

用于变更字符串的最大预算是 `maxCost`。在转化字符串时，总开销应当小于等于该预算，这也意味着字符串的转化可能是不完全的。

如果你可以将 `s` 的子字符串转化为它在 `t` 中对应的子字符串，则返回可以转化的最大长度。

如果 `s` 中没有子字符串可以转化成 `t` 中对应的子字符串，则返回 `0`。

#### 示例

```raw
输入：s = "abcd", t = "bcdf", maxCost = 3
输出：3
解释：s 中的 "abc" 可以变为 "bcd"。开销为 3，所以最大长度为 3。
```

```raw
输入：s = "abcd", t = "cdef", maxCost = 3
输出：1
解释：s 中的任一字符要想变成 t 中对应的字符，其开销都是 2。因此，最大长度为 1。
```

```raw
输入：s = "abcd", t = "acde", maxCost = 0
输出：1
解释：你无法作出任何改动，所以最大长度为 1。
```

#### 提示

- `1 <= len(s), len(t) <= 1e5`；
- `0 <= maxCost <= 1e6`；
- `s` 和 `t` 都只含小写英文字母。

### 题解

```python Python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        return equal_substring(s, t, maxCost)

def equal_substring(s: str, t: str, max_cost: int) -> int:
    w, c, l = deque(), 0, 0
    for d in map(lambda si, ti: abs(ord(si) - ord(ti)), s, t):
        w.append(d)
        c += d
        while c > max_cost:
            c -= w.popleft()
        l = max(l, len(w))
    return l
```

```rust Rust
impl Solution {
    pub fn equal_substring(s: String, t: String, max_cost: i32) -> i32 {
        equal_substring(&s, &t, max_cost as u32) as i32
    }
}

use std::collections::VecDeque;

pub fn equal_substring(s: &str, t: &str, max_cost: u32) -> usize {
    let mut window = VecDeque::new();
    let (mut cost, mut max) = (0, 0);
    for distance in Iterator::zip(s.chars(), t.chars()).map(
        |(si, ti)| (si as i32 - ti as i32).abs() as u32
    ) {
        window.push_back(distance);
        cost += distance;
        while cost > max_cost {
            cost -= window.pop_front().unwrap();
        }
        max = max.max(window.len());
    }
    max
}
```

## 1423. 可获得的最大点数{#leetcode-1423}

[:link: 来源](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/)

### 题目

几张卡牌**排成一行**，每张卡牌都有一个对应的点数。点数由整数数组 `cardPoints` 给出。

每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 `k` 张卡牌。

你的点数就是你拿到手中的所有卡牌的点数之和。

给你一个整数数组 `cardPoints` 和整数 `k`，请你返回可以获得的最大点数。

#### 示例

```raw
输入：cardPoints = [1, 2, 3, 4, 5, 6, 1], k = 3
输出：12
解释：第一次行动，不管拿哪张牌，你的点数总是 1。但是，先拿最右边的卡牌将会最大化你的可获得点数。最优策略是拿右边的三张牌，最终点数为 1 + 6 + 5 = 12。
```

```raw
输入：cardPoints = [2, 2, 2], k = 2
输出：4
解释：无论你拿起哪两张卡牌，可获得的点数总是 4。
```

```raw
输入：cardPoints = [9, 7, 7, 9, 7, 7, 9], k = 7
输出：55
解释：你必须拿起所有卡牌，可以获得的点数为所有卡牌的点数之和。
```

```raw
输入：cardPoints = [1, 1000, 1], k = 1
输出：1
解释：你无法拿到中间那张卡牌，所以可以获得的最大点数为 1。
```

```raw
输入：cardPoints = [1, 79, 80, 1, 1, 1, 200, 1], k = 3
输出：202
```

#### 提示

- `1 <= len(cardPoints) <= 1e5`；
- `1 <= cardPoints[i] <= 1e4`；
- `1 <= k <= len(cardPoints)`。

### 题解

滑动窗口。逆向思维，窗口大小为 `len(cardPoints) - k`。

```python Python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        return max_score(cardPoints, k)

def max_score(card_points: List[int], k: int) -> int:
    ws = len(card_points) - k
    ms = s = sum(card_points[:ws])
    for p, q in zip(card_points, card_points[ws:]):
        s = s - p + q
        ms = min(ms, s)
    return sum(card_points) - ms
```

```rust Rust
impl Solution {
    pub fn max_score(card_points: Vec<i32>, k: i32) -> i32 {
        max_score(&card_points, k as usize)
    }
}

pub fn max_score(card_points: &[i32], k: usize) -> i32 {
    if k == 0 {
        return 0;
    }
    let window_size = card_points.len() - k;
    let mut sum = card_points[..window_size].iter().sum::<i32>();
    let mut min = sum;
    for (m, n) in Iterator::zip(card_points.iter(), card_points[window_size..].iter()) {
        sum = sum - m + n;
        min = min.min(sum);
    }
    card_points.iter().sum::<i32>() - min
}
```

使用 `Iterator::fold` 的版本，这样写的可读性有些许下降。

```rust Rust
impl Solution {
    pub fn max_score(card_points: Vec<i32>, k: i32) -> i32 {
        max_score(&card_points, k as usize)
    }
}

pub fn max_score(card_points: &[i32], k: usize) -> i32 {
    if k == 0 {
        return 0;
    }
    let window_size = card_points.len() - k;
    let sum = card_points[..window_size].iter().sum::<i32>();
    card_points.iter().sum::<i32>() - Iterator::zip(
        card_points.iter(), card_points[window_size..].iter()
    ).fold((sum, sum), |(sum, min), (m, n)| {
        let sum = sum - m + n;
        (sum, min.min(sum))
    }).1
}
```
