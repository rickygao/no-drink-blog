---
title: LeetCode 月报 202102
date: 2021-02-01 00:00:00
tags: [LeetCode, Sliding Window]
mathjax: true
---

快过年了。重启「Rust 从入门到放弃」计划。

本月大概是滑动窗口月了，战绩 30 题。

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
    def fairCandySwap(self, A: list[int], B: list[int]) -> list[int]:
        return fair_candy_swap(A, B)

def fair_candy_swap(a: list[int], b: list[int]) -> list[int]:
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
    let mut c = HashMap::new();
    let (mut m, mut l) = (0, 0);
    let mut p = s.chars();
    for j in s.chars() {
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
    def medianSlidingWindow(self, nums: list[int], k: int) -> list[float]:
        return list(generate_median_sliding_window(nums, k))

from typing import Iterator
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
    def findMaxAverage(self, nums: list[int], k: int) -> float:
        return find_max_average(nums, k)

from typing import Iterator
from collections import deque

def find_max_average(nums: list[int], k: int) -> float:
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
    def maxScore(self, cardPoints: list[int], k: int) -> int:
        return max_score(cardPoints, k)

def max_score(card_points: list[int], k: int) -> int:
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

## 665. 非递减数列{#leetcode-665}

[:link: 来源](https://leetcode-cn.com/problems/non-decreasing-array/)

### 题目

给你一个长度为 `n` 的整数数组，请你判断在**最多**改变 `1` 个元素的情况下，该数组能否变成一个非递减数列。

我们是这样定义一个非递减数列的：对于数组中所有的 `0 < i < n`，总满足 `nums[i - 1] <= nums[i]`。

#### 示例

```raw
输入：nums = [4, 2, 3]
输出：true
解释：你可以通过把第一个 4 变成 1 来使得它成为一个非递减数列。
```

```raw
输入：nums = [4, 2, 1]
输出：false
解释：你不能在只改变一个元素的情况下将其变为非递减数列。
```

#### 说明

- `1 <= n <= 1e4`；
- `-1e5 <= nums[i] <= 1e5`。

### 题解

```python Python
class Solution:
    def checkPossibility(self, nums: list[int]) -> bool:
        return check_possibility(nums)

from math import inf

def check_possibility(nums: list[int]) -> bool:
    p, q, t = -inf, -inf, 0
    for n in nums:
        if n >= p:
            p, q = n, p
        else:
            if t > 0:
                return False
            t += 1
            if n >= q:
                p = n
            else:
                q = n
    return True
```

```rust Rust
impl Solution {
    pub fn check_possibility(nums: Vec<i32>) -> bool {
        check_possibility(&nums)
    }
}

pub fn check_possibility(nums: &[i32]) -> bool {
    let (mut p, mut q, mut t) = (i32::MIN, i32::MIN, 0usize);
    for &n in nums {
        if n >= p {
            q = p;
            p = n;
        } else {
            if t > 0 {
                return false
            }
            t += 1;
            if n >= q {
                p = n;
            } else {
                q = n;
            }
        }
    }
    true
}
```

## 50. Pow(x, n){#leetcode-50}

[:link: 来源](https://leetcode-cn.com/problems/powx-n/)

### 题目

实现 `pow(x, n)`，即计算 x 的 n 次幂函数（即，`x ** n`）。

#### 示例

```raw
输入：x = 2.00000, n = 10
输出：1024.00000
```

```raw
输入：x = 2.10000, n = 3
输出：9.26100
```

```raw
输入：x = 2.00000, n = -2
输出：0.25000
解释：2 ** -2 = 1 / 2 ** 2 = 1 / 4 = 0.25
```

#### 提示

- `-1e2 < x < 1e2`；
- `-2 ** 31 <= n <= 2 ** 31 - 1`；
- `-1e4 <= x ** n <= 1e4`。

### 题解

快速幂。

```python Python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        return my_pow(x, n)

def my_pow(x: float, n: int) -> float:
    if n < 0:
        x = 1 / x
        n = -n
    r = 1
    while n:
        if n % 2:
            r *= x
        x *= x
        n //= 2
    return r
```

```rust Rust
impl Solution {
    pub fn my_pow(mut x: f64, mut n: i32) -> f64 {
        if n < 0 {
            x = 1f64 / x;
        }
        let mut r = 1f64;
        while n != 0 {
            if n & 1 == 1 {
                r *= x;
            }
            x *= x;
            n /= 2;
        }
        r
    }
}
```

## 978. 最长湍流子数组{#leetcode-978}

[:link: 来源](https://leetcode-cn.com/problems/longest-turbulent-subarray/)

### 题目

当 A 的子数组 `A[i:j]` 满足下列条件之一时，我们称其为湍流子数组：

- 若 `i < k < j`，当 `k` 为奇数时，`A[k - 1] > A[k]`，且当 `k` 为偶数时，`A[k - 1] < A[k]`；
- 若 `i < k < j`，当 `k` 为偶数时，`A[k - 1] > A[k]`，且当 `k` 为奇数时，`A[k - 1] < A[k]`。

也就是说，如果比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组。

返回 `A` 的最大湍流子数组的长度。

#### 示例

```raw
输入：[9, 4, 2, 10, 7, 8, 8, 1, 9]
输出：5
解释：(A[1] > A[2] < A[3] > A[4] < A[5])
```

```raw
输入：[4, 8, 12, 16]
输出：2
```

```raw
输入：[100]
输出：1
```

#### 提示

- `1 <= len(A) <= 4e4`；
- `0 <= A[i] <= 1e9`。

### 题解

```python Python
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        return max_turbulence_size(arr)

from operator import sub

def max_turbulence_size(nums: List[int]) -> int:
    if not nums:
        return 0
    p, l, m = 0, 1, 1
    for d in map(sub, nums, nums[1:]):
        if d == 0:
            m = max(m, l)
            l = 1
        elif (d > 0) == (p > 0):
            m = max(m, l)
            l = 2
        else:
            l += 1
        p = d
    return max(m, l)
```

```rust Rust
impl Solution {
    pub fn max_turbulence_size(arr: Vec<i32>) -> i32 {
        max_turbulence_size(&arr) as i32
    }
}

use std::cmp::Ordering;

pub fn max_turbulence_size(slice: &[impl Ord]) -> usize {
    if slice.is_empty() {
        return 0;
    }
    let (mut pre, mut len, mut max) = (Ordering::Equal, 1, 1);
    for (m, n) in Iterator::zip(slice.iter(), slice[1..].iter()) {
        let ord = Ord::cmp(m, n);
        if ord == Ordering::Equal {
            max = max.max(len);
            len = 1;
        } else if ord == pre {
            max = max.max(len);
            len = 2; 
        } else {
            len += 1;
        }
        pre = ord;
    }
    max.max(len)
}

```

## 992. K 个不同整数的子数组{#leetcode-992}

[:link: 来源](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)

### 题目

给定一个正整数数组 `A`，如果 `A` 的某个子数组中不同整数的个数恰好为 `K`，则称 `A` 的这个连续、不一定独立的子数组为**好子数组**。（例如，`[1, 2, 3, 1, 2]` 中有 3 个不同的整数：`1`、`2`、`3`）

返回 `A` 中好子数组的数目。

#### 示例

```raw
输入：A = [1, 2, 1, 2, 3], K = 2
输出：7
解释：恰好由 2 个不同整数组成的子数组：[1, 2]、[2, 1]、[1, 2]、[2, 3]、[1, 2, 1]、[2, 1, 2]、[1, 2, 1, 2]。
```

```raw
输入：A = [1, 2, 1, 3, 4], K = 3
输出：3
解释：恰好由 3 个不同整数组成的子数组：[1, 2, 1, 3]、[2, 1, 3]、[1, 3, 4]。
```

#### 提示

- `1 <= len(A) <= 2e4`；
- `1 <= A[i] <= len(A)`；
- `1 <= K <= len(A)`。

### 题解

```rust Rust
impl Solution {
    pub fn subarrays_with_k_distinct(a: Vec<i32>, k: i32) -> i32 {
        subarrays_with_k_distinct(&a, k as usize) as i32
    }
}

use std::collections::HashMap;
use std::hash::Hash;

pub fn subarrays_with_k_distinct(a: &[impl Eq + Hash], k: usize) -> usize {
    if k == 0 {
        return 0;
    }
    subarrays_with_at_most_k_distinct(a, k) - subarrays_with_at_most_k_distinct(a, k - 1)
}

pub fn subarrays_with_at_most_k_distinct(a: &[impl Eq + Hash], k: usize) -> usize {
    let mut c = HashMap::new();
    let (mut s, mut l, mut r) = (0, 0, 0);
    let mut p = a.iter();
    for i in a.iter() {
        l += 1;
        if *c.entry(i).and_modify(|ci| *ci += 1).or_insert(1) == 1 {
            s += 1;
        }
        while s > k {
            let j = p.next().unwrap();
            l -= 1;
            let cj = c.get_mut(j).unwrap();
            *cj -= 1;
            if *cj == 0 {
                s -= 1;
            }
        }
        r += l;
    }
    r
}
```

## 567. 字符串的排列{#leetcode-567}

[:link: 来源](https://leetcode-cn.com/problems/permutation-in-string/)

### 题目

给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1` 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的子串。

#### 示例

```raw
输入：s1 = "ab", s2 = "eidbaooo"
输出：true
解释：s2 包含 s1 的排列之一（"ba"）。
```

```raw
输入：s1 = "ab", s2 = "eidboaoo"
输出：false
```

#### 注意

- 输入的字符串只包含小写字母；
- `1 <= len(s1), len(s2) <= 1e4`。

### 题解

滑动窗口。

```rust Rust
impl Solution {
    pub fn check_inclusion(s1: String, s2: String) -> bool {
        check_inclusion(&s1, &s2)
    }
}

use std::collections::HashMap;

pub fn check_inclusion(s1: &str, s2: &str) -> bool {
    let mut c = HashMap::new();
    for i in s1.chars() {
        c.entry(i).and_modify(|ci| *ci += 1).or_insert(1);
    }
    let mut r = c.keys().len();
    fn window_push(i: char, c: &mut HashMap<char, isize>, r: &mut usize) {
        c.entry(i).and_modify(|ci| {
            *ci -= 1;
            match *ci {
                0 => *r -= 1,
                -1 => *r += 1,
                _ => (),
            }
        });
    }
    fn window_pop(i: char, c: &mut HashMap<char, isize>, r: &mut usize) {
        c.entry(i).and_modify(|ci| {
            *ci += 1;
            match *ci {
                0 => *r -= 1,
                1 => *r += 1,
                _ => (),
            }
        });
    }
    for k in s2.chars().take(s1.len()) {
        window_push(k, &mut c, &mut r);
    }
    if r == 0 {
        return true;
    }
    for (j, k) in Iterator::zip(s2.chars(), s2.chars().skip(s1.len())) {
        window_pop(j, &mut c, &mut r);
        window_push(k, &mut c, &mut r);
        if r == 0 {
            return true;
        }
    }
    false
}
```

## 703. 数据流中的第 K 大元素{#leetcode-703}

[:link: 来源](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/)

### 题目

设计一个找到数据流中第 `k` 大元素的类（class）。注意是排序后的第 `k` 大元素，不是第 `k` 个不同的元素。

请实现 `KthLargest` 类：

- `KthLargest(k, nums)` 使用整数 `k` 和整数流 `nums` 初始化对象；
- `add(val)` 将 `val` 插入数据流 `nums` 后，返回当前数据流中第 `k` 大的元素。

#### 示例

```raw
输入：["KthLargest", "add", "add", "add", "add", "add"], [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
输出：[null, 4, 5, 5, 8, 8]
```

#### 提示

- `1 <= k <= 1e4`；
- `0 <= len(nums) <= 1e4`；
- `-1e4 <= nums[i] <= 1e4`；
- `-1e4 <= val <= 1e4`；
- 最多调用 `add` 方法 `1e4` 次；
- 题目数据保证，在查找第 `k` 大元素时，数组中至少有 `k` 个元素。

### 题解

```rust Rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

struct KthLargest {
    k: usize,
    h: BinaryHeap<Reverse<i32>>,
}

impl KthLargest {
    pub fn new(k: i32, mut nums: Vec<i32>) -> Self {
        let mut r = KthLargest {
            h: BinaryHeap::new(),
            k: k as usize,
        };
        for n in nums {
            r.add(n);
        }
        r
    }

    pub fn add(&mut self, val: i32) -> i32 {
        let val = Reverse(val);
        if self.h.len() < self.k {
            self.h.push(val);
        } else if val < *self.h.peek().unwrap() {
            self.h.pop();
            self.h.push(val);
        }
        self.h.peek().unwrap().0
    }
}
```

## 119. 杨辉三角 II{#leetcode-119}

[:link: 来源](https://leetcode-cn.com/problems/pascals-triangle-ii/)

### 题目

给定一个非负索引 `k`，其中 `k <= 33`，返回杨辉三角的第 `k` 行。

{% asset_img pascals_triangle.gif 200 185 "'杨辉三角' '杨辉三角'" %}

在杨辉三角中，每个数是它左上方和右上方的数的和。

#### 示例

```raw
输入：3
输出：[1, 3, 3, 1]
```

#### 进阶

你可以优化你的算法到 $\mathrm{O}(k)$ 空间复杂度吗？

### 题解

另见 [118. 杨辉三角 I](/leetcode-monthly-202012/#leetcode-118)。

```rust Rust
impl Solution {
    pub fn get_row(row_index: i32) -> Vec<i32> {
        get_row(row_index as usize).into_iter().map(|i| i as i32).collect()
    }
}

pub fn get_row(row_index: usize) -> Vec<usize> {
    let mut r = Vec::with_capacity(row_index + 1);
    for i in 0..=row_index {
        r.push(r.last().map(|p| p * (row_index - i + 1) / i).unwrap_or(1));
    }
    r
}
```

## 448. 找到所有数组中消失的数字{#leetcode-448}

[:link: 来源](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

### 题目

给定一个范围在 `1 <= a[i] <= n`，`n == len(a)` 的**整型数组**，数组中的元素一些出现了两次，另一些只出现一次。

找到所有在 `1` 至 `n` 范围内没有出现在数组中的数字。

您能在不使用额外空间且时间复杂度为 $\mathrm{O}(n)$ 的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

#### 示例

```raw
输入：[4, 3, 2, 7, 8, 2, 3, 1]
输出：[5, 6]
```

### 题解

```rust Rust
impl Solution {
    pub fn find_disappeared_numbers(nums: Vec<i32>) -> Vec<i32> {
        find_disappeared_numbers(&nums)
    }
}

pub fn find_disappeared_numbers(nums: &[i32]) -> Vec<i32> {
    let mut r: Vec<_> = (1..=nums.len() as i32).collect();
    nums.iter().for_each(|&i| r[i as usize - 1] = 0);
    r.into_iter().filter(|&i| i > 0).collect()
}
```

## 91. 解码方法{#leetcode-91}

[:link: 来源](https://leetcode-cn.com/problems/decode-ways/)

### 题目

一条包含字母 `'A'` 至 `'Z'` 的消息通过以下映射进行了**编码**：

- `'A'` -> `"1"`；
- `'B'` -> `"2"`；
- ……
- `'Z'` -> `"26"`。

要**解码**已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，`"111"` 可以将 `"1"` 中的每个 `"1"` 映射为 `'A'`，从而得到 `"AAA"`，或者可以将 `"11"` 和 `"1"`（分别为 `'K'` 和 `'A'`）映射为 `"KA"`。注意，`"06"` 不能映射为 `'F'`，因为 `"6"` 和 `"06"` 不同。

给你一个只含数字的**非空**字符串 `s`，请计算并返回**解码**方法的**总数**。

题目数据保证答案肯定是一个 32 位的整数。

#### 示例

```raw
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（"1" + "2"）或者 "L"（"12"）。
```

```raw
输入：s = "226"
输出：3
解释：它可以解码为 "BZ"（"2" + "26"）、"VF"（"22" + "6"）或者 "BBF"（"2" + "2" + "6"）。
```

```raw
输入：s = "0"
输出：0
解释：没有字符映射到以 '0' 开头的数字。含有 '0' 的有效映射是 'J' -> "10" 和 'T' -> "20"。由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。
```

```raw
输入：s = "06"
输出：0
解释："06" 不能解码为 'F'，因为字符串开头的 '0' 无法指向一个有效的字符。
```

#### 提示

- `1 <= len(s) <= 1e2`；
- `s` 只包含数字，并且可能包含前导零。

### 题解

```rust Rust
impl Solution {
    pub fn num_decodings(s: String) -> i32 {
        num_decodings(&s) as i32
    }
}

pub fn num_decodings(s: &str) -> usize {
    s.chars().skip(1).fold((1, 1, match s.chars().next() {
        None | Some('0') => return 0,
        Some(c) => c,
    }), |(p, q, c), d| {
        let mut r = 0;
        if d != '0' {
            r += p;
        }
        if c == '1' || c == '2' && d <= '6' {
            r += q;
        }
        (r, p, d)
    }).0
}
```

## 765. 情侣牵手{#leetcode-765}

[:link: 来源](https://leetcode-cn.com/problems/couples-holding-hands/)

### 题目

$N$ 对情侣坐在连续排列的 $2N$ 个座位上，想要牵到对方的手。计算最少交换座位的次数，以便每对情侣可以并肩坐在一起。一次交换可选择任意两人，让他们站起来交换座位。

人和座位用 $0$ 到 $2N-1$ 的整数表示，情侣们按顺序编号，第一对是 $(0,1)$，第二对是 $(2,3)$，以此类推，最后一对是 $(2N-2,2N-1)$。

这些情侣的初始座位 `row[i]` 是由最初始坐在第 `i` 个座位上的人决定的。

#### 示例

```raw
输入：row = [0, 2, 1, 3]
输出：1
解释：我们只需要交换 row[1] 和 row[2] 的位置即可。
```

```raw
输入：row = [3, 2, 0, 1]
输出：0
解释：无需交换座位，所有的情侣都已经可以手牵手了。
```

#### 说明

- `len(row)` 是偶数且 `4 <= len(row) <= 60`；
- 可以保证 `row` 是 `range(len(row))` 的一个全排列。

### 题解

并查集。

```rust Rust
impl Solution {
    pub fn min_swaps_couples(row: Vec<i32>) -> i32 {
        let row: Vec<_> = row.into_iter().map(|i| i as usize).collect();
        min_swaps_couples(&row) as i32
    }
}

use std::cell::Cell;
use std::fmt;

pub struct UnionFind {
    parents: Vec<Cell<usize>>,
}

impl UnionFind {
    #[inline]
    pub fn new(len: usize) -> UnionFind {
        UnionFind {
            parents: (0..len).map(Cell::new).collect(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    pub fn find(&self, i: usize) -> usize {
        let pc = &self.parents[i];
        let p = pc.get();
        if p != i {
            pc.set(self.find(p));
        }
        pc.get()
    }

    pub fn union(&mut self, i: usize, j: usize) -> bool {
        let (pi, pj) = (self.find(i), self.find(j));
        if pi == pj {
            return false;
        }
        self.parents[pj].set(pi);
        true
    }
}

impl fmt::Debug for UnionFind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.parents.iter().map(Cell::get)).finish()
    }
}

pub fn min_swaps_couples(row: &[usize]) -> usize {
    let mut r = row.len() / 2;
    let mut uf = UnionFind::new(r);
    for c in row.chunks(2) {
        if !uf.union(c[0] / 2, c[1] / 2) {
            r -= 1;
        }
    }
    r
}
```

## 485. 最大连续1的个数{#leetcode-485}

[:link: 来源](https://leetcode-cn.com/problems/max-consecutive-ones/)

### 题目

给定一个二进制数组，计算其中最大连续 `1` 的个数。

#### 示例

```raw
输入：[1, 1, 0, 1, 1, 1]
输出：3
解释：开头的两位和最后的三位都是连续 1，所以最大连续 1 的个数是 3。
```

#### 注意

- 输入的数组只包含 `0` 和 `1`；
- 输入数组的长度是正整数，且不超过 `1e4`。

### 题解

```rust Rust
impl Solution {
    pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
        find_max_consecutive_ones(&nums) as i32
    }
}

pub fn find_max_consecutive_ones(nums: &[i32]) -> usize {
    let (mut c, mut m) = (0, 0);
    for n in nums {
        if *n == 1 {
            c += 1;
        } else {
            m = m.max(c);
            c = 0;
        }
    }
    m.max(c)
}
```

## 561. 数组拆分 I{#leetcode-561}

[:link: 来源](https://leetcode-cn.com/problems/array-partition-i/)

### 题目

给定长度为 $2n$ 的整数数组 $nums$，你的任务是将这些数分成 $n$ 对, 例如 $(a_1,b_1),(a_2,b_2),\dots,(a_n,b_n)$，使得从 $\sum_{i=1}^n\min(a_i,b_i)$ 最大。

返回该**最大总和**。

#### 示例

```raw
输入：nums = [1, 4, 3, 2]
输出：4
解释：所有可能的分法（忽略元素顺序）为：
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3；
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3；
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4。
```

```raw
输入：nums = [6, 2, 6, 5, 1, 2]
输出：9
解释：最优的分法为 (2, 1), (2, 5), (6, 6)。min(2, 1) + min(2, 5) + min(6, 6) = 1 + 2 + 6 = 9。
```

#### 提示

- `1 <= n <= 1e4`；
- `len(nums) == 2 * n`；
- `-1e4 <= nums[i] <= 1e4`。

### 题解

排序，贪心。

```python Python
class Solution:
    def arrayPairSum(self, nums: list[int]) -> int:
        return array_pair_sum(nums)

def array_pair_sum(nums: list[int]) -> int:
    return sum(sorted(nums)[::2])
```

```rust Rust
impl Solution {
    pub fn array_pair_sum(nums: Vec<i32>) -> i32 {
        array_pair_sum(&nums)
    }
}

pub fn array_pair_sum(nums: &[i32]) -> i32 {
    let mut v = nums.to_vec();
    v.sort_unstable();
    v.into_iter().step_by(2).sum()
}
```

## 566. 重塑矩阵{#leetcode-566}

[:link: 来源](https://leetcode-cn.com/problems/reshape-the-matrix/)

### 题目

在 MATLAB 中，有一个非常有用的函数 `reshape`，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。

给出一个由二维数组表示的矩阵，以及两个正整数 `r` 和 `c`，分别表示想要的重构的矩阵的行数和列数。

重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。

如果具有给定参数的 `reshape` 操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

#### 示例

```raw
输入：nums = [[1, 2], [3, 4]], r = 1, c = 4
输出：[[1, 2, 3, 4]]
解释：行遍历 nums 的结果是 [1, 2, 3, 4]。新的矩阵是 1 * 4 矩阵，用之前的元素值一行一行填充新矩阵。
```

```raw
输入：nums = [[1, 2], [3, 4]], r = 2, c = 4
输出：[[1, 2], [3, 4]]
解释：没有办法将 2 * 2 矩阵转化为 2 * 4 矩阵。所以输出原矩阵。
```

#### 注意

- 给定矩阵的宽和高范围在 $[1,{10}^2]$；
- 给定的 `r` 和 `c` 都是正数。

### 题解

```rust Rust
use std::iter::{repeat, IntoIterator};

impl Solution {
    pub fn matrix_reshape(nums: Vec<Vec<i32>>, r: i32, c: i32) -> Vec<Vec<i32>> {
        let (r, c) = (r as usize, c as usize);
        if nums.len() * nums.first().map_or(0, |v| v.len()) != r * c {
            nums
        } else {
            let s = &mut nums.into_iter().flatten();
            repeat(()).take(r).map(|_| s.into_iter().take(c).collect::<Vec<_>>()).collect::<Vec<_>>()
        }
    }
}
```

## 995. K 连续位的最小翻转次数{#leetcode-995}

[:link: 来源](https://leetcode-cn.com/problems/minimum-number-of-k-consecutive-bit-flips/)

### 题目

在仅包含 `0` 和 `1` 的数组 `A` 中，一次 `K` 位翻转包括选择一个长度为 `K` 的（连续）子数组，同时将子数组中的每个 `0` 更改为 `1`，而每个 `1` 更改为 `0`。

返回所需的 `K` 位翻转的最小次数，以便数组没有值为 `0` 的元素。如果不可能，返回 `-1`。

#### 示例

```raw
输入：A = [0, 1, 0], K = 1
输出：2
解释：先翻转 A[0]，然后翻转 A[2]。
```

```raw
输入：A = [1, 1, 0], K = 2
输出：-1
解释：无论我们怎样翻转大小为 2 的子数组，我们都不能使数组变为 [1, 1, 1]。
```

```raw
输入：A = [0, 0, 0, 1, 0, 1, 1, 0], K = 3
输出：3
解释：
翻转 A[0]、A[1]、A[2]，A 变成 [1, 1, 1, 1, 0, 1, 1, 0]；
翻转 A[4]、A[5]、A[6]，A 变成 [1, 1, 1, 1, 1, 0, 0, 0]；
翻转 A[5]、A[6]、A[7]，A 变成 [1, 1, 1, 1, 1, 1, 1, 1]。
```

#### 提示

- `1 <= len(A) <= 3e4`；
- `1 <= K <= len(A)`。

### 题解

贪心，滑动窗口。

```python Python
class Solution:
    def minKBitFlips(self, A: list[int], K: int) -> int:
        return min_bit_flips(A, K)

from collections import deque
from itertools import islice

def min_bit_flips(a: list[int], k: int) -> int:
    futures, flip, count = deque([False] * k), False, 0
    for bit in a:
        flip = flip != futures.popleft()
        need = bit == flip
        if need:
            flip = not flip
            count += 1
        futures.append(need)
    return -1 if any(islice(futures, 1, None)) else count
```

```rust Rust
impl Solution {
    pub fn min_k_bit_flips(a: Vec<i32>, k: i32) -> i32 {
        min_k_bit_flips(&a, k as usize).map_or(-1, |i| i as i32)
    }
}

use std::collections::VecDeque;
use std::iter::repeat;

pub fn min_k_bit_flips(a: &[i32], k: usize) -> Option<usize> {
    let mut futures = repeat(false).take(k).collect::<VecDeque<_>>();
    let mut flip = false;
    let mut count = 0;
    for bit in a.iter().cloned().map(|bit| bit == 1) {
        flip ^= futures.pop_front().unwrap();
        let need = !(bit ^ flip);
        if need {
            flip = !flip;
            count += 1;
        }
        futures.push_back(need);
    }
    if futures.into_iter().skip(1).any(|bit| bit) {
        None
    } else {
        Some(count)
    }
}
```

## 1004. 最大连续 1 的个数 III{#leetcode-1004}

[:link: 来源](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)

### 题目

给定一个由若干 `0` 和 `1` 组成的数组 `A`，我们最多可以将 `K` 个值从 `0` 变成 `1`。

返回仅包含 `1` 的最长（连续）子数组的长度。

#### 示例

```raw
输入：A = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], K = 2
输出：6
解释：[1, 1, 1, 0, 0, {(1), 1, 1, 1, 1, (1)}]，最长的子数组长度为 6。
```

```raw
输入：A = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], K = 3
输出：10
解释：[0, 0, {1, 1, (1), (1), 1, 1, 1, (1), 1, 1}, 0, 0, 0, 1, 1, 1, 1]，最长的子数组长度为 10。
```

#### 提示

- `1 <= len(A) <= 2e4`;
- `0 <= K <= len(A)`；
- `A[i] in (0, 1)`。

### 题解

滑动窗口。

```rust Rust
impl Solution {
    pub fn longest_ones(a: Vec<i32>, k: i32) -> i32 {
        longest_ones(&a, k as usize) as i32
    }
}

use std::iter::{repeat, once};
use std::collections::VecDeque;

pub fn longest_ones(a: &[i32], k: usize) -> usize {
    let mut max = 0;
    let mut begins = repeat(0).take(k + 1).collect::<VecDeque<_>>();
    a.iter()
        .enumerate()
        .filter_map(|(i, &bit)| if bit == 0 { Some(i) } else { None })
        .chain(once(a.len()))
        .for_each(|i| {
            max = max.max(i - begins.pop_front().unwrap());
            begins.push_back(i + 1);
        });
    max
}
```

```python Python
class Solution:
    def longestOnes(self, A: list[int], K: int) -> int:
        return longest_ones(A, K)

from itertools import chain
from collections import deque

def longest_ones(a: list[int], k: int) -> int:
    starts, m = deque([0] * (k + 1)), 0
    for i in chain((i for i, bit in enumerate(a) if not bit), [len(a)]):
        m = max(m, i - starts.popleft())
        starts.append(i + 1)
    return m
```

## 1438. 绝对差不超过限制的最长连续子数组{#leetcode-1438}

[:link: 来源](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

### 题目

给你一个整数数组 `nums`，和一个表示限制的整数 `limit`，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 `limit`。

如果不存在满足条件的子数组，则返回 `0`。

#### 示例

```raw
输入：nums = [8, 2, 4, 7], limit = 4
输出：2 
解释：所有子数组如下：
[8         ] 最大绝对差 abs(8 - 8) = 0 <= 4；
[8, 2      ] 最大绝对差 abs(8 - 2) = 6 >  4；
[8, 2, 4   ] 最大绝对差 abs(8 - 2) = 6 >  4；
[8, 2, 4, 7] 最大绝对差 abs(8 - 2) = 6 >  4；
[   2      ] 最大绝对差 abs(2 - 2) = 0 <= 4；
[   2, 4   ] 最大绝对差 abs(2 - 4) = 2 <= 4；
[   2, 4, 7] 最大绝对差 abs(2 - 7) = 5 >  4；
[      4   ] 最大绝对差 abs(4 - 4) = 0 <= 4；
[      4, 7] 最大绝对差 abs(4 - 7) = 3 <= 4；
[         7] 最大绝对差 abs(7 - 7) = 0 <= 4。
因此，满足题意的最长子数组的长度为 2。
```

```raw
输入：nums = [10, 1, 2, 4, 7, 2], limit = 5
输出：4 
解释：满足题意的最长子数组是 [2, 4, 7, 2]，其最大绝对差 abs(2 - 7) = 5 <= 5。
```

```raw
输入：nums = [4, 2, 2, 2, 4, 4, 2, 2], limit = 0
输出：3
```

#### 提示

- `1 <= len(nums) <= 1e5`；
- `1 <= nums[i] <= 1e9`；
- `0 <= limit <= 1e9`。

### 题解

滑动窗口，单调队列。维护窗口内的最小值和最大值。

```rust Rust
impl Solution {
    pub fn longest_subarray(nums: Vec<i32>, limit: i32) -> i32 {
        longest_subarray(&nums, limit) as i32
    }
}

use std::collections::VecDeque;
use std::ops::Sub;

pub fn longest_subarray<'a, E, I>(nums: &'a I, limit: E) -> usize
where
    E: 'a + Ord,
    I: 'a + ?Sized,
    &'a E: Sub<Output = E>,
    &'a I: IntoIterator<Item = &'a E>,
{
    let (mut mins, mut maxs) = (VecDeque::new(), VecDeque::new());
    let (mut p, mut l) = (nums.into_iter(), 0);
    for n in nums {
        while mins.back().map_or(false, |&x| x > n) {
            mins.pop_back();
        }
        while maxs.back().map_or(false, |&x| x < n) {
            maxs.pop_back();
        }
        mins.push_back(n);
        maxs.push_back(n);
        let (&min, &max) = (mins.front().unwrap(), maxs.front().unwrap());
        let distance = max - min;
        if distance > limit {
            let left = p.next().unwrap();
            if min >= left {
                mins.pop_front();
            }
            if max <= left {
                maxs.pop_front();
            }
        } else {
            l += 1;
        }
    }
    l
}
```

## 766. 托普利茨矩阵{#leetcode-766}

[:link: 来源](https://leetcode-cn.com/problems/toeplitz-matrix/)

### 题目

给你一个 $m\times n$ 的矩阵 `matrix`。如果这个矩阵是托普利茨矩阵，返回 `true`；否则，返回 `false`。

如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是**托普利茨矩阵**。

#### 示例

```raw
输入：matrix = [[1, 2, 3, 4], [5, 1, 2, 3], [9, 5, 1, 2]]
输出：true
解释：在上述矩阵中，其对角线为：[9], [5, 5], [1, 1, 1], [2, 2, 2], [3, 3], [4]。
各条对角线上的所有元素均相同，因此答案是 true。
```

```raw
输入：matrix = [[1, 2], [2, 2]]
输出：false
解释：对角线 [1, 2] 上的元素不同。
```

#### 提示

- `m == len(matrix)`；
- `n == len(matrix[i])`；
- `1 <= m, n <= 20`；
- `0 <= matrix[i][j] <= 99`。

#### 进阶

- 如果矩阵存储在磁盘上，并且内存有限，以至于一次最多只能将矩阵的一行加载到内存中，该怎么办？
- 如果矩阵太大，以至于一次只能将不完整的一行加载到内存中，该怎么办？

### 题解

小题大做。感受一下 Rust 中泛型和迭代器的威力。

```rust Rust
impl Solution {
    pub fn is_toeplitz_matrix(matrix: Vec<Vec<i32>>) -> bool {
        is_toeplitz_matrix(&matrix)
    }
}

pub fn is_toeplitz_matrix<'a, E, I, J>(matrix: &'a J) -> bool
where
    E: 'a + ?Sized,
    I: 'a + ?Sized,
    J: 'a + ?Sized,
    &'a E: Eq,
    &'a I: IntoIterator<Item = &'a E>,
    &'a J: IntoIterator<Item = &'a I>,
{
    Iterator::zip(matrix.into_iter(), matrix.into_iter().skip(1))
        .all(|(r, s)| Iterator::zip(r.into_iter(), s.into_iter().skip(1)).all(|(m, n)| m == n))
}
```

## 1052. 爱生气的书店老板{#leetcode-1052}

[:link: 来源](https://leetcode-cn.com/problems/grumpy-bookstore-owner/)

### 题目

今天，书店老板有一家店打算试营业 `len(customers)` 分钟。每分钟都有一些顾客（`customers[i]`）会进入书店，所有这些顾客都会在那一分钟结束后离开。

在某些时候，书店老板会生气。如果书店老板在第 `i` 分钟生气，那么 `grumpy[i] = 1`，否则 `grumpy[i] = 0`。当书店老板生气时，那一分钟的顾客就会不满意，不生气则他们是满意的。

书店老板知道一个秘密技巧，能抑制自己的情绪，可以让自己连续 `X` 分钟不生气，但却只能使用一次。

请你返回这一天营业下来，最多有多少客户能够感到满意的数量。

#### 示例

```raw
输入：customers = [1, 0, 1, 2, 1, 1, 7, 5], grumpy = [0, 1, 0, 1, 0, 1, 0, 1], X = 3
输出：16
解释：书店老板在最后 3 分钟保持冷静。感到满意的最大客户数量 = 1 + 1 + 1 + 1 + 7 + 5 = 16。
```

#### 提示

- `1 <= X <= len(customers) == len(grumpy) <= 2e4`；
- `0 <= customers[i] <= 1e3`;
- `0 <= grumpy[i] <= 1`。

### 题解

滑动窗口。同时计算老板生气和时不生气时两种情况的顾客数。生气时的求最大窗口和，不生气时的全部保留。

```rust Rust
impl Solution {
    pub fn max_satisfied(customers: Vec<i32>, grumpy: Vec<i32>, x: i32) -> i32 {
        let customers = customers.into_iter().map(|e| e as usize).collect::<Vec<_>>();
        let grumpy = grumpy.into_iter().map(|e| e == 1).collect::<Vec<_>>();
        max_satisfied(&customers, &grumpy, x as usize) as i32
    }
}

pub fn max_satisfied(customers: &[usize], grumpy: &[bool], x: usize) -> usize {
    let (mut extra, mut base) = (0, 0);
    Iterator::zip(customers[..x].iter(), grumpy[..x].iter()).for_each(|(&m, &g)| {
        if g {
            extra += m;
        } else {
            base += m;
        }
    });

    let mut max = extra;
    Iterator::zip(
        Iterator::zip(customers.iter(), grumpy.iter()),
        Iterator::zip(customers[x..].iter(), grumpy[x..].iter()),
    )
    .for_each(|((&m, &g), (&n, &h))| {
        if g {
            extra -= m;
        }
        if h {
            extra += n;
        } else {
            base += n;
        }
        max = max.max(extra);
    });

    max + base
}
```

## 832. 翻转图像{#leetcode-832}

[:link: 来源](https://leetcode-cn.com/problems/flipping-an-image/)

### 题目

给定一个二进制矩阵 `A`，我们想先水平翻转图像，然后反转图像并返回结果。

水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 `[1, 1, 0]` 的结果是 `[0, 1, 1]`。

反转图片的意思是图片中的 `0` 全部被 `1` 替换，`1` 全部被 `0` 替换。例如，反转 `[0, 1, 1]` 的结果是 `[1, 0, 0]`。

#### 示例

```raw
输入：[[1, 1, 0], [1, 0, 1], [0, 0, 0]]
输出：[[1, 0, 0], [0, 1, 0], [1, 1, 1]]
解释：
首先翻转每一行：[[0, 1, 1], [1, 0, 1], [0, 0, 0]]；
然后反转图片：[[1, 0, 0], [0, 1, 0], [1, 1, 1]]。
```

```raw
输入：[[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]]
输出：[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0]]
解释：
首先翻转每一行:[[0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 1]]；
然后反转图片：[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0]]。
```

#### 提示

- `1 <= len(A) = len(A) <= 20`；
- `0 <= A[i][j] <= 1`。

### 题解

#### 原地

```rust Rust
impl Solution {
    pub fn flip_and_invert_image(mut a: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        a.iter_mut().for_each(|r| {
            r.reverse();
            r.iter_mut().for_each(|e| *e ^= 1);
        });
        a
    }
}
```

#### 副本

```rust Rust
impl Solution {
    pub fn flip_and_invert_image(a: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        a.iter().map(|r| r.iter().rev().map(|e| e ^ 1).collect()).collect()
    }
}
```

## 867. 转置矩阵{#leetcode-867}

[:link: 来源](https://leetcode-cn.com/problems/transpose-matrix/)

### 题目

给你一个二维整数数组 `matrix`，返回 `matrix` 的**转置矩阵**。

矩阵的**转置**是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。

#### 示例

```raw
输入：matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
输出：[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```

```raw
输入：matrix = [[1, 2, 3], [4, 5, 6]]
输出：[[1, 4], [2, 5], [3, 6]]
```

#### 提示

- `m == len(matrix)`；
- `n == len(matrix[i])`；
- `1 <= m, n <= 1e3`；
- `1 <= m * n <= 1e5`；
- `-1e9 <= matrix[i][j] <= 1e9`。

### 题解

```rust Rust
impl Solution {
    pub fn transpose(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        (0..matrix.first().map_or(0, |r| r.len()))
            .map(|c| matrix.iter().map(|r| r[c]).collect())
            .collect()
    }
}
```

```python Python
class Solution:
    def transpose(self, matrix: list[list[int]]) -> list[list[int]]:
        return list(map(list, zip(*matrix)))
```

## 41. 缺失的第一个正数{#leetcode-41}

[:link: 来源](https://leetcode-cn.com/problems/first-missing-positive/)

### 题目

给你一个未排序的整数数组 `nums`，请你找出其中没有出现的最小的正整数。

#### 进阶

你可以实现时间复杂度为 $\mathrm{O}(n)$ 并且只使用常数级别额外空间的解决方案吗？

#### 示例

```raw
输入：nums = [1, 2, 0]
输出：3
```

```raw
输入：nums = [3, 4, -1, 1]
输出：2
```

```raw
输入：nums = [7, 8, 9, 11, 12]
输出：1
```

#### 提示

- `0 <= len(nums) <= 3e2`；
- `-2 ** 31 <= nums[i] <= 2 ** 31 - 1`。

### 题解

不能，除非注明可以原地修改。

```python Python
class Solution:
    def firstMissingPositive(self, nums: list[int]) -> int:
        return first_missing_positive(nums)

def first_missing_positive(nums: list[int]) -> int:
    s, i = set(nums), 1
    while i in s:
        i += 1
    return i
```

```rust Rust
impl Solution {
    pub fn first_missing_positive(nums: Vec<i32>) -> i32 {
        first_missing_positive(&nums)
    }
}

use std::collections::HashSet;

pub fn first_missing_positive(nums: &[i32]) -> i32 {
    let (s, mut i) = (nums.iter().collect::<HashSet<_>>(), 1);
    while s.contains(&i) {
        i += 1;
    }
    i
}
```

## 1178. 猜字谜{#leetcode-1178}

[:link: 来源](https://leetcode-cn.com/problems/number-of-valid-words-for-each-puzzle/)

### 题目

外国友人仿照中国字谜设计了一个英文版猜字谜小游戏，请你来猜猜看吧。

字谜的迷面 `puzzle` 按字符串形式给出，如果一个单词 `word` 符合下面两个条件，那么它就可以算作谜底：

- 单词 `word` 中包含谜面 `puzzle` 的第一个字母。
- 单词 `word` 中的每一个字母都可以在谜面 `puzzle` 中找到。例如，如果字谜的谜面是 `"abcdefg"`，那么可以作为谜底的单词有 `"faced"`、`"cabbage"` 和 `"baggage"`；而 `"beefed"`（不含字母 `'a'`）以及 `"based"`（其中的 `'s'` 没有出现在谜面中）。

返回一个答案数组 `answer`，数组中的每个元素 `answer[i]` 是在给出的单词列表 `words` 中可以作为字谜迷面 `puzzles[i]` 所对应的谜底的单词数目。

#### 示例

```raw
输入：words = ["aaaa", "asas", "able", "ability", "actt", "actor", "access"], puzzles = ["aboveyz", "abrodyz", "abslute", "absoryz", "actresz", "gaswxyz"]
输出：[1, 1, 3, 2, 4, 0]
解释：
1 个单词可以作为 "aboveyz" 的谜底："aaaa"；
1 个单词可以作为 "abrodyz" 的谜底："aaaa"；
3 个单词可以作为 "abslute" 的谜底："aaaa"、"asas"、"able"；
2 个单词可以作为 "absoryz" 的谜底："aaaa"、"asas"；
4 个单词可以作为 "actresz" 的谜底："aaaa"、"asas"、"actt"、"access"；
没有单词可以作为 "gaswxyz" 的谜底，因为列表中的单词都不含字母 'g'。
```

#### 提示

- `1 <= len(words) <= 1e5`；
- `4 <= len(words[i]) <= 50`；
- `1 <= len(puzzles) <= 1e4`；
- `len(puzzles[i]) == 7`；
- `words[i][j]`、`puzzles[i][j]` 都是小写英文字母。
- 每个 `puzzles[i]` 所包含的字符都不重复。

### 题解

位集。只利用了字符串仅包含小写英文字母的限制。

```rust Rust
impl Solution {
    pub fn find_num_of_valid_words(words: Vec<String>, puzzles: Vec<String>) -> Vec<i32> {
        find_num_of_valid_words(
            &words.iter().map(String::as_str).collect::<Vec<_>>(),
            &puzzles.iter().map(String::as_str).collect::<Vec<_>>(),
        )
        .into_iter()
        .map(|e| e as i32)
        .collect()
    }
}

pub fn find_num_of_valid_words(words: &[&str], puzzles: &[&str]) -> Vec<usize> {
    fn into_mask(a: u8) -> u32 {
        1 << (a - b'a')
    }
    fn into_set(s: &str) -> u32 {
        s.bytes().fold(0, |s, c| s | into_mask(c))
    }
    let words: Vec<u32> = words.iter().copied().map(into_set).collect();
    puzzles
        .iter()
        .copied()
        .map(|puzzle| (into_mask(puzzle.as_bytes()[0]), into_set(puzzle)))
        .map(|(first_mask, puzzle)| {
            words
                .iter()
                .copied()
                .filter(|&word| word & puzzle == word | first_mask)
                .count()
        })
        .collect()
}
```

## 395. 至少有K个重复字符的最长子串{#leetcode-395}

[:link: 来源](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

### 题目

找到给定字符串（由小写字符组成）中的最长子串 `T`，要求 `T` 中的每一字符出现次数都不少于 `k`。输出 `T` 的长度。

#### 示例

```raw
输入：s = "aaabb", k = 3
输出：3
解释：最长子串为 "aaa"，其中 'a' 重复了 3 次。
```

```raw
输入：s = "ababbc", k = 2
输出：5
解释：最长子串为 "ababb"，其中 'a' 重复了 2 次，'b' 重复了 3 次。
```

### 题解

分治。满足性质的最长字串不会跨越破坏性质的字符，于是以破坏性质的字符为界限，分而治之。

```rust Rust
impl Solution {
    pub fn longest_substring(s: String, k: i32) -> i32 {
        longest_substring(&s, k as usize) as i32
    }
}

use std::collections::HashMap;

pub fn longest_substring(s: &str, k: usize) -> usize {
    let mut counter = HashMap::new();
    for c in s.chars() {
        *counter.entry(c).or_insert(0) += 1;
    }

    let pattern = counter
        .into_iter()
        .filter_map(|(c, n)| if n < k { Some(c) } else { None })
        .collect::<Vec<_>>();

    if pattern.is_empty() {
        return s.chars().count();
    }

    s.split(pattern.as_slice())
        .map(|ss| longest_substring(ss, k))
        .max()
        .unwrap()
}
```

## 896. 单调数列{#leetcode-896}

[:link: 来源](https://leetcode-cn.com/problems/monotonic-array/)

### 题目

如果数组是单调递增或单调递减的，那么它是单调的。

如果对于所有 `i <= j`，`A[i] <= A[j]`，那么数组 `A` 是单调递增的。 如果对于所有 `i <= j`，`A[i] >= A[j]`，那么数组 `A` 是单调递减的。

当给定的数组 `A` 是单调数组时返回 `true`，否则返回 `false`。

#### 示例

```raw
输入：[1, 2, 2, 3]
输出：true
```

```raw
输入：[6, 5, 4, 4]
输出：true
```

```raw
输入：[1, 3, 2]
输出：false
```

```raw
输入：[1, 2, 4, 5]
输出：true
```

```raw
输入：[1, 1, 1]
输出：true
```

#### 提示

- `1 <= len(A) <= 5e4`；
- `-1e5 <= A[i] <= 1e5`。

### 题解

利用 `[T]::windows`、`Iterator::try_fold`、`PartialOrd::partial_cmp` 完成一个泛型版本。

```rust Rust
impl Solution {
    pub fn is_monotonic(a: Vec<i32>) -> bool {
        is_monotonic(&a)
    }
}

use std::cmp::Ordering::Equal;

pub fn is_monotonic(a: &[impl PartialOrd]) -> bool {
    a.windows(2).try_fold(Equal, |s, w| {
        let o = PartialOrd::partial_cmp(&w[0], &w[1])?;
        match (s, o) {
            (Equal, o) => Some(o),
            (s, Equal) => Some(s),
            (s, o) if s == o => Some(s),
            _ => None
        }
    }).is_some()
}
```
