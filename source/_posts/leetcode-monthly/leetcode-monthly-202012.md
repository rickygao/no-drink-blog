---
title: LeetCode 月报 202012
date: 2020-12-01 00:00:00
tags: [LeetCode]
mathjax: true
---

这个月也是顽张的一个月呢，共完成了 66 道题目。

<!-- more -->

## 34. 在排序数组中查找元素的第一个和最后一个位置

[:link: 来源](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

### 题目

给定一个按照升序排列的整数数组 `nums`, 和一个目标值 `target`. 找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`, 返回 `[-1, -1]`.

#### 进阶

你可以设计并实现时间复杂度为 $\mathrm{O}(\log n)$ 的算法解决此问题吗？

#### 示例

```raw
输入：nums = [5, 7, 7, 8, 8, 10], target = 8
输出：[3, 4]
```

```raw
输入：nums = [5, 7, 7, 8, 8, 10], target = 6
输出：[-1, -1]
```

```raw
输入：nums = [], target = 0
输出：[-1, -1]
```

#### 提示

- `0 <= len(nums) <= 1e5`;
- `-1e9 <= nums[i], target <= 1e9`;
- `nums` 是一个非递减数组。

### 题解

二分查找。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        return search_range(nums, target)

from bisect import bisect_left, bisect_right

def search_range(nums: List[int], target: int) -> List[int]:
    l = bisect_left(nums, target)
    if l >= len(nums) or nums[l] != target:
        return [-1, -1]
    r = bisect_right(nums, target)
    return [l, r - 1]
```

## 1641. 统计字典序元音字符串的数目

[:link: 来源](https://leetcode-cn.com/problems/count-sorted-vowel-strings/)

### 题目

给你一个整数 `n`, 请返回长度为 `n`, 仅由元音 `('a', 'e', 'i', 'o', 'u')` 组成且按**字典序排列**的字符串数量。

字符串 `s` 按**字典序排列**需要满足：对于所有有效的 `i`, `s[i]` 在字母表中的位置总是与 `s[i+1]` 相同或在 `s[i+1]` 之前。

#### 示例

```raw
输入：n = 1
输出：5
解释：仅由元音组成的 5 个字典序字符串为 ["a", "e", "i", "o", "u"].
```

```raw
输入：n = 2
输出：15
解释：仅由元音组成的 15 个字典序字符串为 ["aa", "ae", "ai", "ao", "au", "ee", "ei", "eo", "eu", "ii", "io", "iu", "oo", "ou", "uu"]. 注意，"ea" 不是符合题意的字符串，因为 'e' 在字母表中的位置比 'a' 靠后。
```

```raw
输入：n = 33
输出：66045
```

#### 提示

`1 <= n <= 50`.

### 题解

#### 分治

- 二分缩小问题；
- 记忆化。

```python
class Solution:
    def countVowelStrings(self, n: int) -> int:
        return count_vowel_strings(n, 5)

from functools import lru_cache

@lru_cache(maxsize=None)  # use `functools.cache` instead since Python 3.9
def count_vowel_strings(n: int, v: int) -> int:
    if n == 0:
        return 1
    if n == 1:
        return v
    l = n // 2
    r = n - l - 1
    return sum(
        count_vowel_strings(l, i) * count_vowel_strings(r, v - i + 1)
        for i in range(1, v + 1)
    )
```

#### 组合计数

数学题。$\binom{n+4}{4}$, 即为 `n + 4` 个空位安置 `4` 个挡板，从而将 `n` 个空位划分为 `5` 组连续空位，分别装填 `(a, e, i, o, u)`.

```python
class Solution:
    def countVowelStrings(self, n: int) -> int:
        return count_vowel_strings(n, 5)

from math import comb

def count_vowel_strings(n: int, v: int) -> int:
    return comb(n + v - 1, v - 1)
```

#### 查表

```python
TABLE = [
    1, 5, 15, 35, 70, 126, 210, 330, 495, 715, 1001,
    1365, 1820, 2380, 3060, 3876, 4845, 5985, 7315, 8855, 10626,
    12650, 14950, 17550, 20475, 23751, 27405, 31465, 35960, 40920, 46376,
    52360, 58905, 66045, 73815, 82251, 91390, 101270, 111930, 123410, 135751,
    148995, 163185, 178365, 194580, 211876, 230300, 249900, 270725, 292825, 316251
]

class Solution:
    def countVowelStrings(self, n: int) -> int:
        return TABLE[n]
```

## 451. 根据字符出现频率排序

[:link: 来源](https://leetcode-cn.com/problems/sort-characters-by-frequency/)

### 题目

给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

#### 示例

```raw
输入："tree"
输出："eert"
解释：'e' 出现两次，'r' 和 't' 都只出现一次。因此 'e' 必须出现在 'r' 和 't' 之前。
此外，"eetr" 也是一个有效的答案。
```

```raw
输入："cccaaa"
输出："cccaaa"
解释：'c' 和 'a' 都出现三次。此外，"aaaccc" 也是有效的答案。
注意 "cacaca" 是不正确的，因为相同的字母必须放在一起。
```

```raw
输入："Aabb"
输出："bbAa"
解释：此外，"bbaA" 也是一个有效的答案，但 "Aabb" 是不正确的。注意 'A' 和 'a' 被认为是两种不同的字符。
```

### 题解

计数，排序，生成字符串。

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        return frequency_sort(s)

from collections import Counter

def frequency_sort(s: str) -> str:
    return ''.join(c * n for c, n in Counter(s).most_common())
```

## 321. 拼接最大数

[:link: 来源](https://leetcode-cn.com/problems/create-maximum-number/)

### 题目

给定长度分别为 `m` 和 `n` 的两个数组，其元素由 `0` 到 `9` 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 `k` (`k <= m + n`) 个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。

求满足该条件的最大数。结果返回一个表示该最大数的长度为 `k` 的数组。

#### 说明

请尽可能地优化你算法的时间和空间复杂度。

#### 示例

```raw
输入：nums1 = [3, 4, 6, 5], nums2 = [9, 1, 2, 5, 8, 3], k = 5
输出：[9, 8, 6, 5, 3]
```

```raw
输入：nums1 = [6, 7], nums2 = [6, 0, 4], k = 5
输出：[6, 7, 6, 0, 4]
```

```raw
输入：nums1 = [3, 9], nums2 = [8, 9], k = 3
输出：[9, 8, 9]
```

### 题解

- 分治；
- 单调栈。

```python
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        return max_number(nums1, nums2, k)

def max_number(nums1: List[int], nums2: List[int], k: int) -> List[int]:
    x_min, x_max = max(0, k - len(nums2)), min(k, len(nums1))
    return max((
        _merge(_max_number(nums1, x), _max_number(nums2, k - x))
        for x in range(x_min, x_max + 1)
    ), default=[])

def _max_number(nums: List[int], k: int) -> List[int]:
    s, to_drop = [], len(nums) - k
    for n in nums:
        while to_drop > 0 and s and s[-1] < n:
            to_drop -= 1
            s.pop()
        s.append(n)
    return s[:k]

def _merge(seq1: List[int], seq2: List[int]) -> List[int]:
    r = []
    while seq1 or seq2:
        s = max(seq1, seq2)
        r.append(s.pop(0))
    return r
```

## 204. 计数质数

[:link: 来源](https://leetcode-cn.com/problems/count-primes/)

### 题目

统计所有小于非负整数 `n` 的质数的数量。

#### 示例

```raw
输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个，它们是 2, 3, 5, 7.
```

```raw
输入：n = 0
输出：0
```

```raw
输入：n = 1
输出：0
```

#### 提示

- `0 <= n <= 5e6`

### 题解

筛法。

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        return count_primes(n)

def count_primes(n: int) -> int:
    if n < 2:
        return 0
    is_primes = [True] * n
    is_primes[0] = is_primes[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_primes[i]:
            k = ceil((n - (b := i * 2)) / i)
            is_primes[b:n:i] = [False] * k
    return sum(is_primes)
```

## 101. 对称二叉树

[:link: 来源](https://leetcode-cn.com/problems/symmetric-tree/)

### 题目

给定一个二叉树，检查它是否是镜像对称的。

#### 示例

```raw
输入：
    1
   / \
  2   2
 / \ / \
3  4 4  3
输出：true
```

```raw
输入：
    1
   / \
  2   2
   \   \
    3   3
输出：false
```

#### 进阶

你可以运用递归和迭代两种方法解决这个问题吗？

### 题解

#### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return is_symmetric(root.left, root.right)

def is_symmetric(t1: TreeNode, t2: TreeNode) -> bool:
    if not (t1 or t2):
        return True
    if not (t1 and t2):
        return False
    if t1.val != t2.val:
        return False
    return is_symmetric(t1.left, t2.right) and is_symmetric(t1.right, t2.left)
```

#### 迭代

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return is_symmetric(root.left, root.right)

def is_symmetric(t1: TreeNode, t2: TreeNode) -> bool:
    s1, s2 = [t1], [t2]
    while s1 or s2:
        n1, n2 = s1.pop(), s2.pop()
        if not (n1 or n2):
            continue
        if not (n1 and n2):
            return False
        if n1.val != n2.val:
            return False
        s1 += [n1.left, n1.right]
        s2 += [n2.right, n2.left]
    return True
```

## 659. 分割数组为连续子序列

[:link: 来源](https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/)

### 题目

给你一个按升序排序的整数数组 `num`（可能包含重复数字），请你将它们分割成一个或多个子序列，其中每个子序列都由连续整数组成且长度至少为 `3`.

如果可以完成上述分割，则返回 `true`; 否则，返回 `false`.

#### 示例

```raw
输入：[1, 2, 3, 3, 4, 5]
输出：true
解释：你可以分割出这样两个连续子序列 [1, 2, 3], [3, 4, 5].
```

```raw
输入：[1, 2, 3, 3, 4, 4, 5, 5]
输出：true
解释：你可以分割出这样两个连续子序列 [1, 2, 3, 4, 5], [3, 4, 5].
```

```raw
输入：[1, 2, 3, 4, 4, 5]
输出：false
```

#### 提示

输入的数组长度范围为 `[1, 10000]`.

### 题解

#### 贪心模拟

贪心策略：优先考虑最短的序列，如无合适的序列则新建。

```python
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        return is_possible(nums)

def is_possible(nums: List[int]) -> bool:
    seqs = []
    for n in nums:
        for seq in reversed(seqs):
            if n == seq[-1] + 1:
                seq.append(n)
                break
        else:
            seqs.append([n])
    return all(len(seq) >= 3 for seq in seqs)
```

#### 贪心优化

- `s1`, `s2`, `s3` 分别代表长度为 `1`, 长度为 `2`, 长度大于等于 `3` 的末尾为 `prev` 的序列，存储它们的个数；
- 贪心策略：优先满足 `s1`, `s2` 的增长需求，如无法满足则失败；然后满足 `s3` 的增长需求；最后再考虑新建序列。

```python
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        return is_possible(nums)

from itertools import groupby

def is_possible(nums: List[int]) -> bool:
    prev, s1, s2, s3 = None, 0, 0, 0
    for n, group in groupby(nums):
        if prev is not None and n - prev != 1:
            if s1 or s2:
                return False
            else:
                s3 = 0
        prev = n

        count = sum(1 for _ in group) - s1 - s2
        if count < 0:
            return False

        a3 = min(count, s3)
        s1, s2, s3 = count - a3, s1, s2 + a3
    return not (s1 or s2)
```

## 456. 132 模式

[:link: 来源](https://leetcode-cn.com/problems/132-pattern/)

### 题目

给定一个整数序列：$a_1, a_2, \dots, a_n$, 一个 132 模式的子序列 $a_i, a_j, a_k$ 被定义为：当 $i < j < k$ 时，$a_i < a_k < a_j$. 设计一个算法，当给定有 `n` 个数字的序列时，验证这个序列中是否含有 132 模式的子序列。

#### 注意

`n` 的值小于 `15000`.

#### 示例

```raw
输入：[1, 2, 3, 4]
输出：false
解释：序列中不存在 132 模式的子序列。
```

```raw
输入：[3, 1, 4, 2]
输出：true
解释：序列中有 1 个 132 模式的子序列 [1, 4, 2].
```

```raw
输入：[-1, 3, 2, 0]
输出：true
解释：序列中有 3 个 132 模式的的子序列 [-1, 3, 2], [-1, 3, 0], [-1, 2, 0].
```

### 题解

- `s` 是单调递减栈；
- 通过 `s[-1] < a_j` 的条件为 `a_k` 赋值，保证了对于一个尽量大的 `a_k` 会存在 `a_j` 比它更大；
- 只需存在一个 `a_i < a_k` 小，即可断言成功。

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        return find_132_pattern(nums)

def find_132_pattern(nums: List[int]) -> bool:
    s, a_k = [], float('-inf')
    for a_i in reversed(nums):
        if a_i < a_k:
            return True
        # treat a_i as a_j
        while s and s[-1] < a_i:
            a_k = s.pop()
        s.append(a_i)
    return False
```

## 621. 任务调度器

[:link: 来源](https://leetcode-cn.com/problems/task-scheduler/)

### 题目

给你一个用字符数组 `tasks` 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 `1` 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。

然而，两个**相同种类**的任务之间必须有长度为整数 `n` 的冷却时间，因此至少有连续 `n + 1` 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

你需要计算完成所有任务所需要的**最短时间**。

#### 示例

```raw
输入：tasks = ["A", "A", "A", "B", "B", "B"], n = 2
输出：8
解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了 (待命) 状态。 
```

```raw
输入：tasks = ["A", "A", "A", "B", "B", "B"], n = 0
输出：6
解释：在这种情况下，任何大小为 6 的排列都可以满足要求，因为 n = 0.
["A", "A", "A", "B", "B", "B"]
["A", "B", "A", "B", "A", "B"]
["B", "B", "B", "A", "A", "A"]
...
诸如此类
```

```raw
输入：tasks = ["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], n = 2
输出：16
解释：一种可能的解决方案是 A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> (待命) -> (待命) -> A -> (待命) -> (待命) -> A.
```

#### 提示

- `1 <= len(tasks) <= 1e4`;
- `tasks[i]` 是大写英文字母；
- `n` 的取值范围为 `[0, 100]`.

### 题解

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        return least_interval(tasks, n)

from collections import Counter

def least_interval(tasks: List[str], n: int) -> int:
    times = Counter(tasks).values()
    num_tasks = len(tasks)
    max_times = max(times, default=0)
    max_count = sum(1 for t in times if t == max_times)
    return max((max_times - 1) * (n + 1) + max_count, num_tasks)
```

## 118. 杨辉三角

[:link: 来源](https://leetcode-cn.com/problems/pascals-triangle/)

### 题目

给定一个非负整数 `numRows`, 生成杨辉三角的前 `numRows` 行。

{% asset_img pascals_triangle.gif 200 185 "'杨辉三角' '杨辉三角'" %}

在杨辉三角中，每个数是它左上方和右上方的数的和。

#### 示例

```raw
输入：5
输出：[
    [1],
    [1, 1],
    [1, 2, 1],
    [1, 3, 3, 1],
    [1, 4, 6, 4, 1]
]
```

### 题解

不如来写一个生成器吧！

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        return pascals_triangle(numRows)

from itertools import islice
from operator import add

def pascals_triangle(num_rows: int) -> List[List[int]]:
    return list(islice(generate_pascals_triangle(), num_rows))

def generate_pascals_triangle() -> Iterator[List[int]]:
    row = [1]
    yield row
    while True:
        row = [1, *map(add, row, row[1:]), 1]
        yield row
```

## 263. 丑数

[:link: 来源](https://leetcode-cn.com/problems/ugly-number/)

### 题目

编写一个程序判断给定的数是否为丑数。

丑数就是只包含质因数 `2`, `3`, `5` 的正整数。

#### 示例

```raw
输入：6
输出：true
解释：6 = 2 * 3.
```

```raw
输入：8
输出：true
解释：8 = 2 * 2 * 2.
```

```raw
输入：14
输出：false
解释：14 不是丑数，因为它包含了另外一个质因数 7.
```

#### 说明

- `1` 是丑数；
- 输入不会超过 32 位有符号整数的范围: $[−2^{31}, 2^{31}−1]$.

### 题解

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        return is_ugly(num)

def is_ugly(num: int) -> bool:
    if num == 0:
        return False
    for f in [2, 3, 5]:
        while num % f == 0:
            num //= f
    return num == 1
```

## 400. 第 N 个数字

[:link: 来源](https://leetcode-cn.com/problems/nth-digit/)

### 题目

在无限的整数序列 $1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, \dots$ 中找到第 `n` 个数字。

#### 注意

`n` 是正数且在 32 位整数范围内 ($0 < n < 2^{31}$).

#### 示例

```raw
输入：3
输出：3
```

```raw
输入：11
输出：0
说明：第 11 个数字在序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 里是 0, 它是 10 的一部分。
```

### 题解

数学题。所有 `k` 位数的数字总个数为 $9k\times{10}^{k-1}$, 用 `m` 代表 $9\times{10}^{k-1}$. 找到第 `n` 个数字位于整数 `t` 的自低位起的第 `i` 个十进制位（`i` 从 `0` 计起）。

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        return find_nth_digit(n)

def find_nth_digit(n: int) -> int:
    k, m = 1, 9
    while n > (b := k * m):
        n -= b
        k += 1
        m *= 10
    p, q = divmod(n - 1, k)
    t = m // 9 + p
    i = k - 1 - q
    return t // (10 ** i) % 10
```

## 1577. 数的平方等于两数乘积的方法数

[:link: 来源](https://leetcode-cn.com/problems/number-of-ways-where-square-of-number-is-equal-to-product-of-two-numbers/)

### 题目

给你两个整数数组 `nums1` 和 `nums2`, 请你返回根据以下规则形成的三元组的数目（类型 1 和类型 2）：

类型 1：三元组 `(i, j, k)`, 如果 `nums1[i] ** 2 == nums2[j] * nums2[k]` 其中 `0 <= i < len(nums1)` 且 `0 <= j < k < len(nums2)`;

类型 2：三元组 `(i, j, k)`, 如果 `nums2[i] ** 2 == nums1[j] * nums1[k]` 其中 `0 <= i < len(nums2)` 且 `0 <= j < k < len(nums1)`.

#### 示例

```raw
输入：nums1 = [7, 4], nums2 = [5, 2, 8, 9]
输出：1
解释：
类型 1：(1, 1, 2), nums1[1] ** 2 = nums2[1] * nums2[2] (4 ** 2 = 2 * 8)
```

```raw
输入：nums1 = [1, 1], nums2 = [1, 1, 1]
输出：9
解释：所有三元组都符合题目要求，因为 1 ** 2 = 1 * 1.
类型 1：(0, 0, 1), (0, 0, 2), (0, 1, 2), (1, 0, 1), (1, 0, 2), (1, 1, 2), nums1[i] ** 2 = nums2[j] * nums2[k];
类型 2：(0, 0, 1), (1, 0, 1), (2, 0, 1), nums2[i] ** 2 = nums1[j] * nums1[k].
```

```raw
输入：nums1 = [7, 7, 8, 3], nums2 = [1, 2, 9, 7]
输出：2
解释：有两个符合题目要求的三元组。
类型 1：(3, 0, 2), nums1[3] ** 2 = nums2[0] * nums2[2];
类型 2：(3, 0, 1), nums2[3] ** 2 = nums1[0] * nums1[1].
```

```raw
输入：nums1 = [4, 7, 9, 11, 23], nums2 = [3, 5, 1024, 12, 18]
输出：0
解释：不存在符合题目要求的三元组。
```

#### 提示

- `1 <= len(nums1), len(nums2) <= 1000`;
- `1 <= nums1[i], nums2[i] <= 1e5`.

### 题解

计数，求交，相乘。

```python
class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        return num_triplets(nums1, nums2) + num_triplets(nums2, nums1)

from itertools import combinations
from collections import Counter

def num_triplets(nums1: List[int], nums2: List[int]) -> int:
    squares = Counter(n_i * n_i for n_i in nums1)
    products = Counter(n_j * n_k for n_j, n_k in combinations(nums2, 2))
    equals = squares.keys() & products.keys()
    return sum(squares[r] * products[r] for r in equals)
```

## 861. 翻转矩阵后的得分

[:link: 来源](https://leetcode-cn.com/problems/score-after-flipping-matrix/)

### 题目

有一个二维矩阵 `A` 其中每个元素的值为 `0` 或 `1`.

移动是指选择任一行或列，并转换该行或列中的每一个值：将所有 `0` 都更改为 `1`, 将所有 `1` 都更改为 `0`.

在做出任意次数的移动后，将该矩阵的每一行都按照二进制数来解释，矩阵的得分就是这些数字的总和。

返回尽可能高的分数。

#### 示例

```raw
输入：[[0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0]]
输出：39
解释：转换为 [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], 0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39.
```

#### 提示

- `1 <= len(A), len(A[i]) <= 20`;
- `A[i][j] in (0, 1)`.

### 题解

#### 贪心

```python
class Solution:
    def matrixScore(self, A: List[List[int]]) -> int:
        return matrix_score(A)

def matrix_score(matrix: List[List[int]]) -> int:
    nr, nc = len(matrix), len(matrix[0])

    # hflip the rows w/ a leading zero
    for i in range(nr):
        if matrix[i][0] == 0:
            for j in range(nc):
                matrix[i][j] = 1 - matrix[i][j]

    # vflip the columns w/ zeros more than ones
    for j in range(nc):
        if sum(matrix[i][j] for i in range(nr)) < nr / 2:
            for i in range(nr):
                matrix[i][j] = 1 - matrix[i][j]

    return sum(sum(matrix[i][j] for i in range(nr)) * 2 ** (nc - 1 - j) for j in range(nc))
```

#### 优化

```python
class Solution:
    def matrixScore(self, A: List[List[int]]) -> int:
        return matrix_score(A)

from functools import reduce

def matrix_score(matrix: List[List[int]]) -> int:
    nr, nc = len(matrix), len(matrix[0])
    return reduce(
        lambda r, j: r * 2 + max(
            s := sum(matrix[i][j] ^ matrix[i][0] for i in range(nr)),
            nr - s
        ), range(1, nc), nr
    )
```

## 842. 将数组拆分成斐波那契序列

[:link: 来源](https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/)

### 题目

给定一个数字字符串 `S`, 比如 `S = "123456579"`, 我们可以将它分成斐波那契式的序列 `[123, 456, 579]`.

形式上，斐波那契式序列是一个非负整数列表 `F` 且满足：

- `0 <= F[i] <= 2 ** 31 - 1`（也就是说，每个整数都符合 32 位有符号整数类型）；
- `len(F) >= 3`;
- 对于所有的 `0 <= i < len(F) - 2`, 都有 `F[i] + F[i + 1] = F[i + 2]` 成立。

另外，请注意，将字符串拆分成小块时，每个块的数字一定不要以零开头，除非这个块是数字 `0` 本身。

返回从 `S` 拆分出来的任意一组斐波那契式的序列块，如果不能拆分则返回 `[]`.

#### 示例

```raw
输入："123456579"
输出：[123, 456, 579]
```

```raw
输入："11235813"
输出：[1, 1, 2, 3, 5, 8, 13]
```

```raw
输入："112358130"
输出：[]
解释：这项任务无法完成。
```

```raw
输入："0123"
输出：[]
解释：每个块的数字不能以零开头，因此 "01", "2", "3" 不是有效答案。
```

```raw
输入："1101111"
输出：[110, 1, 111]
解释：输出 [11, 0, 11, 11] 也同样被接受。
```

#### 提示

- `1 <= len(S) <= 200`;
- 字符串 `S` 中只含有数字。

### 题解

- 当确定了斐波那契数列的起始两项时，后续的项也就确定了，枚举测试即可；
- 注意题目要求整数在 32 位有符号整数范围内。

```python
class Solution:
    def splitIntoFibonacci(self, S: str) -> List[int]:
        return split_into_fibonacci(S)

def split_into_fibonacci(s: str) -> List[int]:
    l, fm, fml = len(s), 2 ** 31, len(str(2 ** 31))

    candidates = (
        (i, j) for i in range(1, l) for j in range(i + 1, l)
        if (i <= fml) and (j <= i + fml) and (s[0] != '0' or i == 1) and (s[i] != '0' or j - i == 1)
    )

    for i, j in candidates:
        f = [int(s[:i]), int(s[i:j])]
        while (fk := f[-1] + f[-2]) < fm and s[j:].startswith(fks := str(fk)):
            f.append(fk)
            j += len(fks)
            if j == l:
                return f
    return []
```

## 1190. 反转每对括号间的子串

[:link: 来源](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

### 题目

给出一个字符串 `s`（仅含有小写英文字母和括号）。

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

注意，您的结果中**不应**包含任何括号。

#### 示例

```raw
输入：s = "(abcd)"
输出："dcba"
```

```raw
输入：s = "(u(love)i)"
输出："iloveu"
```

```raw
输入：s = "(ed(et(oc))el)"
输出："leetcode"
```

```raw
输入：s = "a(bcdefghijkl(mno)p)q"
输出："apmnolkjihgfedcbq"
```

#### 提示

- `0 <= len(s) <= 2000`;
- `s` 中只有小写英文字母和括号；
- 我们确保所有括号都是成对出现的。

### 题解

栈，每当括号闭合时翻转栈顶字符串并合并。

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        return reverse_parentheses(s)

def reverse_parentheses(s: str) -> str:
    r = ['']
    for c in s:
        if c == '(':
            r.append('')
        elif c == ')':
            t = r.pop()
            r[-1] += t[::-1]
        else:
            r[-1] += c
    return r[0]
```

## 492. 构造矩形

[:link: 来源](https://leetcode-cn.com/problems/construct-the-rectangle/)

### 题目

作为一位 Web 开发者， 懂得怎样去规划一个页面的尺寸是很重要的。现给定一个具体的矩形页面面积，你的任务是设计一个长度为 `L` 和宽度为 `W` 且满足以下要求的矩形的页面。要求：

1. 你设计的矩形页面必须等于给定的目标面积；
2. 宽度 `W` 不应大于长度 `L`, 换言之，要求 `L >= W`;
3. 长度 `L` 和宽度 `W` 之间的差距应当尽可能小。

你需要按顺序输出你设计的页面的长度 `L` 和宽度 `W`.

#### 示例

```raw
输入：4
输出：[2, 2]
解释：目标面积是 4, 所有可能的构造方案有 [1, 4], [2, 2], [4, 1].
但是根据要求 2, [1, 4] 不符合要求; 根据要求 3, [2, 2] 比 [4, 1] 更能符合要求. 所以输出长度 L 为 2, 宽度 W 为 2.
```

#### 说明

- 给定的面积不大于 `10_000_000` 且为正整数；
- 你设计的页面的长度和宽度必须都是正整数。

### 题解

```python
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        return construct_rectangle(area)

from math import isqrt

def construct_rectangle(area: int) -> List[int]:
    w = isqrt(area)
    while area % w != 0:
        w -= 1
    return [area // w, w]
```

## 62. 不同路径

[:link: 来源](https://leetcode-cn.com/problems/unique-paths/)

### 题目

一个机器人位于一个 $m\times n$ 网格的左上角 （起始点在下图中标记为 "Start"）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 "Finish"）。

问总共有多少条不同的路径？

{% asset_img robot_maze.png 200 92 "'机器人迷宫' '机器人迷宫'" %}

例如，上图是一个 $7\times3$ 的网格。有多少可能的路径？

#### 示例

```raw
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
```

```raw
输入：m = 7, n = 3
输出：28
```

#### 提示

- `1 <= m, n <= 100`;
- 题目数据保证答案小于等于 `2e9`.

### 题解

数学题，组合计数。共需行动 $m+n-2$ 步，其中 $m-1$ 步为向右移动，则有 $\binom{m+n-2}{m-1}$ 种选择。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return unique_paths(m, n)

from math import comb

def unique_paths(m: int, n: int) -> int:
    return comb(m + n - 2, m - 1)
```

## 1217. 玩筹码

[:link: 来源](https://leetcode-cn.com/problems/minimum-cost-to-move-chips-to-the-same-position/)

### 题目

数轴上放置了一些筹码，每个筹码的位置存在数组 `chips` 当中。

你可以对**任何筹码**执行下面两种操作之一（不限操作次数，0 次也可以）：

- 将第 `i` 个筹码向左或者右移动 2 个单位，代价为 **0**;
- 将第 `i` 个筹码向左或者右移动 1 个单位，代价为 **1**.

最开始的时候，同一位置上也可能放着两个或者更多的筹码。

返回将所有筹码移动到同一位置（任意位置）上所需要的最小代价。

#### 示例

```raw
输入：chips = [1, 2, 3]
输出：1
解释：第二个筹码移动到位置三的代价是 1, 第一个筹码移动到位置三的代价是 0, 总代价为 1.
```

```raw
输入：chips = [2, 2, 2, 3, 3]
输出：2
解释：第四和第五个筹码移动到位置二的代价都是 1, 所以最小总代价为 2.
```

#### 提示

- `1 <= len(chips) <= 100`;
- `1 <= chips[i] <= 1e9`.

### 题解

奇偶性相同的位置是等价的，只需求奇位置和偶位置的筹码数量的最小值。

```python
class Solution:
    def minCostToMoveChips(self, chips: List[int]) -> int:
        return min_cost_to_move_chips(chips)

def min_cost_to_move_chips(chips: List[int]) -> int:
    return min(odd := sum(p % 2 for p in chips), len(chips) - odd)
```

## 268. 丢失的数字

[:link: 来源](https://leetcode-cn.com/problems/missing-number/)

### 题目

给定一个包含 `[0, n]` 中 `n` 个数的数组 `nums`, 找出 `[0, n]` 这个范围内没有出现在数组中的那个数。

#### 进阶

你能否实现线性时间复杂度、仅使用额外常数空间的算法解决此问题？

#### 示例

```raw
输入：nums = [3, 0, 1]
输出：2
解释：n = 3, 因为有 3 个数字，所以所有的数字都在范围 [0, 3] 内。2 是丢失的数字，因为它没有出现在 nums 中。
```

```raw
输入：nums = [0, 1]
输出：2
解释：n = 2, 因为有 2 个数字，所以所有的数字都在范围 [0, 2] 内。2 是丢失的数字，因为它没有出现在 nums 中。
```

```raw
输入：nums = [9, 6, 4, 2, 3, 5, 7, 0, 1]
输出：8
解释：n = 9, 因为有 9 个数字，所以所有的数字都在范围 [0, 9] 内。8 是丢失的数字，因为它没有出现在 nums 中。
```

```raw
输入：nums = [0]
输出：1
解释：n = 1, 因为有 1 个数字，所以所有的数字都在范围 [0, 1] 内。1 是丢失的数字，因为它没有出现在 nums 中。
```

#### 提示

- `n == len(nums)`;
- `1 <= n <= 1e4`;
- `0 <= nums[i] <= n`;
- nums 中的所有数字都**独一无二**。

### 题解

利用按位异或的对合性。

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        return missing_number(nums)

from itertools import chain
from functools import reduce
from operator import xor

def missing_number(nums: List[int]) -> int:
    return reduce(xor, chain(nums, range(len(nums) + 1)))
```

## 860. 柠檬水找零

[:link: 来源](https://leetcode-cn.com/problems/lemonade-change/)

### 题目

在柠檬水摊上，每一杯柠檬水的售价为 `5` 美元。

顾客排队购买你的产品，（按账单 `bills` 支付的顺序）一次购买一杯。

每位顾客只买一杯柠檬水，然后向你付 `5` 美元、`10` 美元或 `20` 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 `5` 美元。

注意，一开始你手头没有任何零钱。

如果你能给每位顾客正确找零，返回 `true`, 否则返回 `false`.

#### 示例

```raw
输入：[5, 5, 5, 10, 20]
输出：true
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 true.
```

```raw
输入：[5, 5, 10]
输出：true
```

```raw
输入：[10, 10]
输出：false
```

```raw
输入：[5, 5, 10, 10, 20]
输出：false
解释：
前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
由于不是每位顾客都得到了正确的找零，所以答案是 false.
```

#### 提示

- `0 <= len(bills) <= 10000`;
- `bills[i] in (5, 10, 20)`.

### 题解

#### 通用贪心

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        return lemonade_change(bills)

def lemonade_change(bills: List[int]) -> bool:
    values = [20, 10, 5]
    notes = [0] * (n := len(values))
    for bill in bills:
        for i in range(n):
            if bill == values[i]:
                notes[i] += 1
                break
        bill -= 5
        for i in range(n):
            while bill > 0 and values[i] <= bill and notes[i] > 0:
                bill -= values[i]
                notes[i] -= 1
        if bill > 0:
            return False
    return True
```

#### 针对优化

`20` 美元无法用于找零。

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        return lemonade_change(bills)

def lemonade_change(bills: List[int]) -> bool:
    five = ten = 0
    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            ten += 1
            if five >= 1:
                five -= 1
            else:
                return False
        elif bill == 20:
            if five >= 1 and ten >= 1:
                five -= 1
                ten -= 1
            elif five >= 3:
                five -= 3
            else:
                return False
    return True
```

## 649. Dota2 参议院

[:link: 来源](https://leetcode-cn.com/problems/dota2-senate/)

### 题目

Dota2 的世界里有两个阵营：Radiant（天辉）和 Dire（夜魇）。

Dota2 参议院由来自两派的参议员组成。现在参议院希望对一个 Dota2 游戏里的改变作出决定。他们以一个基于轮为过程的投票进行。在每一轮中，每一位参议员都可以行使两项权利中的一项：

1. 禁止一名参议员的权利：参议员可以让另一位参议员在这一轮和随后的几轮中丧失所有的权利。
2. 宣布胜利：如果参议员发现有权利投票的参议员都是同一个阵营的，他可以宣布胜利并决定在游戏中的有关变化。

给定一个字符串代表每个参议员的阵营。字母 `'R'` 和 `'D'` 分别代表了 Radiant（天辉）和 Dire（夜魇）。然后，如果有 `n` 个参议员，给定字符串的大小将是 `n`.

以轮为基础的过程从给定顺序的第一个参议员开始到最后一个参议员结束。这一过程将持续到投票结束。所有失去权利的参议员将在过程中被跳过。

假设每一位参议员都足够聪明，会为自己的政党做出最好的策略，你需要预测哪一方最终会宣布胜利并在 Dota2 游戏中决定改变。输出应该是 `"Radiant"` 或 `"Dire"`.

#### 示例

```raw
输入："RD"
输出："Radiant"
解释：第一个参议员来自 Radiant 阵营并且他可以使用第一项权利让第二个参议员失去权利，因此第二个参议员将被跳过因为他没有任何权利。然后在第二轮的时候，第一个参议员可以宣布胜利，因为他是唯一一个有投票权的人。
```

```raw
输入："RDD"
输出："Dire"
解释：
第一轮中，第一个来自 Radiant 阵营的参议员可以使用第一项权利禁止第二个参议员的权利；
第二个来自 Dire 阵营的参议员会被跳过因为他的权利被禁止；
第三个来自 Dire 阵营的参议员可以使用他的第一项权利禁止第一个参议员的权利；
因此在第二轮只剩下第三个参议员拥有投票的权利，于是他可以宣布胜利。
```

#### 提示

- `0 <= n <= 10000`.

### 题解

贪心。每个参议员选择禁止当前最先要投票的敌方参议员的权利，直到有一方全部被禁止。

```python
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        return predict_party_victory(senate)

from collections import deque

def predict_party_victory(senate: str) -> str:
    n, radiant, dire = len(senate), deque(), deque()
    for t, s in enumerate(senate):
        (radiant if s == 'R' else dire).append(t)

    while radiant and dire:
        if (r := radiant.popleft()) < (d := dire.popleft()):
            radiant.append(r + n)
        else:
            dire.append(d + n)

    return 'Radiant' if radiant else 'Dire'
```

## 1047. 删除字符串中的所有相邻重复项

[:link: 来源](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

### 题目

给出由小写字母组成的字符串 `S`, 重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 `S` 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

#### 示例

```raw
输入："abbaca"
输出："ca"
解释：例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca", 其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca".
```

#### 提示

- `1 <= len(S) <= 20000`;
- `S` 仅由小写英文字母组成。

### 题解

栈。每一字符与栈顶消重。是{% post_link leetcode-1209 %}的特化。

```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        return remove_duplicates(S)

def remove_duplicates(s: str) -> str:
    r = []
    for c in s:
        if r and r[-1] == c:
            r.pop()
        else:
            r.append(c)
    return ''.join(r)
```

## 102. 二叉树的层序遍历

[:link: 来源](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

### 题目

给你一个二叉树，请你返回其按**层序遍历**得到的节点值。（即逐层地，从左到右访问所有节点）。

#### 示例

```raw
输入：
  3
 / \
9  20
  /  \
 15   7
输出：[[3], [9, 20], [15, 7]]
```

### 题解

层次遍历。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        return level_order(root)

from collections import deque

def level_order(root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    q, r = deque([root]), []
    while q:
        r.append([n.val for n in q])
        for _ in range(len(q)):
            n = q.popleft()
            if c := n.left:
                q.append(c)
            if c := n.right:
                q.append(c)
    return r
```

## 103. 二叉树的锯齿形层次遍历

[:link: 来源](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

### 题目

给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

#### 示例

```raw
输入： 
  3
 / \
9  20
  /  \
 15   7
输出：[[3], [20, 9], [15, 7]]
```

### 题解

层次遍历。奇数层翻转。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        return zigzag_level_order(root)

from collections import deque

def zigzag_level_order(root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    q, r = deque([root]), []
    while q:
        r.append([n.val for n in (
            reversed(q) if len(r) % 2 else q
        )])
        for _ in range(len(q)):
            n = q.popleft()
            if c := n.left:
                q.append(c)
            if c := n.right:
                q.append(c)
    return r
```

## 376. 摆动序列

[:link: 来源](https://leetcode-cn.com/problems/wiggle-subsequence/)

### 题目

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。

例如，`[1, 7, 4, 9, 2, 5]` 是一个摆动序列，因为差值 `(6, -3, 5, -7, 3)` 是正负交替出现的。相反, `[1, 4, 7, 2, 5]` 和 `[1, 7, 4, 5, 5]` 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。

给定一个整数序列，返回作为摆动序列的最长子序列的长度。 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。

#### 示例

```raw
输入：[1, 7, 4, 9, 2, 5]
输出：6 
解释：整个序列为摆动序列。
```

```raw
输入：[1, 17, 5, 10, 13, 15, 10, 5, 16, 8]
输出：7
解释：这个序列包含几个长度为 7 摆动序列，其中一个可为 [1, 17, 10, 13, 10, 16, 8].
```

```raw
输入：[1, 2, 3, 4, 5, 6, 7, 8, 9]
输出：2
```

#### 进阶

你能否用 $\mathrm{O}(n)$ 时间复杂度完成此题？

### 题解

贪心。统计波峰、波谷数量。

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        return wiggle_max_length(nums)

from operator import sub

def wiggle_max_length(nums: List[int]) -> int:
    if not nums:
        return 0

    inc, r = None, 1
    for diff in map(sub, nums[1:], nums):
        if diff and ((sign := diff > 0) != inc or inc is None):
            inc, r = sign, r + 1

    return r
```

## 1338. 数组大小减半

[:link: 来源](https://leetcode-cn.com/problems/reduce-array-size-to-the-half/)

### 题目

给你一个整数数组 `arr`. 你可以从中选出一个整数集合，并删除这些整数在数组中的每次出现。

返回**至少**能删除数组中的一半整数的整数集合的最小大小。

#### 示例

```raw
输入：arr = [3, 3, 3, 3, 5, 5, 5, 2, 2, 7]
输出：2
解释：选择 {3, 7} 使得结果数组为 [5, 5, 5, 2, 2], 长度为 5（原数组长度的一半）。
大小为 2 的可行集合有 {3, 5}, {3, 2}, {5, 2}.
选择 {2, 7} 是不可行的，它的结果数组为 [3, 3, 3, 3, 5, 5, 5], 新数组长度大于原数组的二分之一。
```

```raw
输入：arr = [7, 7, 7, 7, 7, 7]
输出：1
解释：我们只能选择集合 {7}, 结果数组为空。
```

```raw
输入：arr = [1, 9]
输出：1
```

```raw
输入：arr = [1000, 1000, 3, 7]
输出：1
```

```raw
输入：arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
输出：5
```

#### 提示

- `1 <= len(arr) <= 1e5`;
- `len(arr)` 为偶数；
- `1 <= arr[i] <= 1e5`.

### 题解

贪心。

```python
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        return min_set_size(arr)

from collections import Counter
from itertools import accumulate

def min_set_size(arr: List[int]) -> int:
    half = ceil(len(arr) / 2)
    return next(i + 1 for i, a in enumerate(accumulate(
        sorted(Counter(arr).values(), reverse=True)
    )) if a >= half)
```

## 1637. 两点之间不包含任何点的最宽垂直面积

[:link: 来源](https://leetcode-cn.com/problems/widest-vertical-area-between-two-points-containing-no-points/)

### 题目

给你 `n` 个二维平面上的点 `points`, 其中 `points[i] = [xi, yi]`, 请你返回两点之间内部不包含任何点的**最宽垂直面积**的宽度。

**垂直面积**的定义是固定宽度，而 `y` 轴上无限延伸的一块区域（也就是高度为无穷大）。**最宽垂直面积**为宽度最大的一个垂直面积。

请注意，垂直区域**边上**的点**不在**区域内。

#### 示例

{% asset_img points.png 200 269 "'示例' '示例'" %}

```raw
输入：points = [[8, 7], [9, 9], [7, 4], [9, 7]]
输出：1
解释：红色区域和蓝色区域都是最优区域。
```

```raw
输入：points = [[3, 1], [9, 0], [1, 0], [1, 4], [5, 3], [8, 8]]
输出：3
```

#### 提示

- `n == len(points)`;
- `2 <= n <= 1e5`;
- `len(points[i]) == 2`;
- `0 <= xi, yi <= 1e9`.

### 题解

排序，最大差值。

```python
class Solution:
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        return max_width_of_vertical_area(points)

from operator import sub

def max_width_of_vertical_area(points: List[List[int]]) -> int:
    xs = sorted({p[0] for p in points})
    return max(map(sub, xs[1:], xs), default=0)
```

## 1227. 飞机座位分配概率

[:link: 来源](https://leetcode-cn.com/problems/airplane-seat-assignment-probability/)

### 题目

有 `n` 位乘客即将登机，飞机正好有 `n` 个座位。第一位乘客的票丢了，他随便选了一个座位坐下。

剩下的乘客将会：

- 如果他们自己的座位还空着，就坐到自己的座位上；
- 当他们自己的座位被占用时，随机选择其他座位。

第 `n` 位乘客坐在自己的座位上的概率是多少？

#### 示例

```raw
输入：n = 1
输出：1.00000
解释：第一个人只会坐在自己的位置上。
```

```raw
输入：n = 2
输出：0.50000
解释：在第一个人选好座位坐下后，第二个人坐在自己的座位上的概率是 0.5.
```

#### 提示

- `1 <= n <= 1e5`.

### 题解

数学题。

```python
class Solution:
    def nthPersonGetsNthSeat(self, n: int) -> float:
        return nth_person_gets_nth_seat(n)

def nth_person_gets_nth_seat(n: int) -> float:
    return 1. if n == 1 else .5
```

## 217. 存在重复元素

[:link: 来源](https://leetcode-cn.com/problems/contains-duplicate/)

### 题目

给定一个整数数组，判断是否存在重复元素。

如果任意一值在数组中出现至少两次，函数返回 `true`. 如果数组中每个元素都不相同，则返回 `false`.

#### 示例

```raw
输入：[1, 2, 3, 1]
输出：true
```

```raw
输入：[1, 2, 3, 4]
输出：false
```

```raw
输入：[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
输出：true
```

### 题解

#### 高效

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return contains_duplicate(nums)

def contains_duplicate(nums: List[int]) -> bool:
    seen = set()
    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False
```

#### 简洁

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return contains_duplicate(nums)

def contains_duplicate(nums: List[int]) -> bool:
    return len(set(nums)) < len(nums)
```

## 1160. 拼写单词

[:link: 来源](https://leetcode-cn.com/problems/find-words-that-can-be-formed-by-characters/)

### 题目

给你一份「词汇表」（字符串数组）`words` 和一张「字母表」（字符串）`chars`.

假如你可以用 chars 中的「字母」（字符）拼写出 `words` 中的某个「单词」（字符串），那么我们就认为你掌握了这个单词。

注意：每次拼写（指拼写词汇表中的一个单词）时，`chars` 中的每个字母都只能用一次。

返回词汇表 `words` 中你掌握的所有单词的**长度之和**。

#### 示例

```raw
输入：words = ["cat", "bt", "hat", "tree"], chars = "atach"
输出：6
解释：可以形成字符串 "cat" 和 "hat", 所以答案是 3 + 3 = 6.
```

```raw
输入：words = ["hello", "world", "leetcode"], chars = "welldonehoneyr"
输出：10
解释：可以形成字符串 "hello" 和 "world", 所以答案是 5 + 5 = 10.
```

#### 提示

- `1 <= len(words) <= 1000`;
- `1 <= len(words[i]), len(chars) <= 100`;
- 所有字符串中都仅包含小写英文字母。

### 题解

计数。

```python
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        return count_characters(words, chars)

from collections import Counter

def count_characters(words: List[str], chars: str) -> int:
    chars = Counter(chars)
    return sum(
        len(word) for word in words
        if all(chars[ch] >= c for ch, c in Counter(word).items())
    )
```

## 1122. 数组的相对排序

[:link: 来源](https://leetcode-cn.com/problems/relative-sort-array/)

### 题目

给你两个数组，`arr1` 和 `arr2`.

- `arr2` 中的元素各不相同；
- `arr2` 中的每个元素都出现在 `arr1` 中。

对 `arr1` 中的元素进行排序，使 `arr1` 中项的相对顺序和 `arr2` 中的相对顺序相同。未在 `arr2` 中出现过的元素需要按照升序放在 `arr1` 的末尾。

#### 示例

```raw
输入：arr1 = [2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19], arr2 = [2, 1, 4, 3, 9, 6]
输出：[2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]
```

#### 提示

- `1 <= len(arr1), len(arr2) <= 1000`;
- `0 <= arr1[i], arr2[i] <= 1000`;
- `arr2` 中的元素 `arr2[i]` 各不相同；
- `arr2` 中的每个元素 `arr2[i]` 都出现在 `arr1` 中。

### 题解

反查位置。

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        return relative_sort_array(arr1, arr2)

def relative_sort_array(arr1: List[int], arr2: List[int]) -> List[int]:
    pos, mpos = {v: i for i, v in enumerate(arr2)}, len(arr2)
    return sorted(arr1, key=lambda v: (pos.get(v, mpos), v))
```

## 229. 求众数 II

[:link: 来源](https://leetcode-cn.com/problems/majority-element-ii/)

### 题目

给定一个大小为 `n` 的整数数组，找出其中所有出现超过 $\lfloor\frac{n}{3}\rfloor$ 次的元素。

#### 进阶

尝试设计时间复杂度为 $\mathrm{O}(n)$, 空间复杂度为 $\mathrm{O}(1)$ 的算法解决此问题。

#### 示例

```raw
输入：[3, 2, 3]
输出：[3]
```

```raw
输入：nums = [1]
输出：[1]
```

```raw
输入：[1, 1, 1, 3, 3, 2, 2, 2]
输出：[1, 2]
```

#### 提示

- `1 <= len(nums) <= 5e4`;
- `-1e9 <= nums[i] <= 1e9`.

### 题解

#### 通用计数

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        return majority_element(nums, 3)

from collections import Counter

def majority_element(nums: List[int], f: int) -> List[int]:
    return [
        n for n, c in Counter(nums).most_common(f - 1)
        if c > len(nums) // f
    ]
```

#### 通用摩尔投票

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        return majority_element(nums, 3)

def majority_element(nums: List[int], f: int) -> List[int]:
    m = f - 1
    r, c = [None] * m, [0] * m
    for n in nums:
        for i in range(m):
            if r[i] == n:
                c[i] += 1
                break
        else:
            for i in range(m):
                if c[i] == 0:
                    r[i], c[i] = n, 1
                    break
            else:
                for i in range(m):
                    c[i] -= 1

    c = [0] * m
    for n in nums:
        for i in range(m):
            if r[i] == n:
                c[i] += 1
                break
    
    l = len(nums) // f
    return [ri for ri, ci in zip(r, c) if ci > l]
```

#### 特化摩尔投票

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        return majority_element(nums)

def majority_element(nums: List[int]) -> List[int]:
    r1 = r2 = None
    c1 = c2 = 0
    for n in nums:
        if r1 == n:
            c1 += 1
        elif r2 == n:
            c2 += 1
        elif c1 == 0:
            r1, c1 = n, 1
        elif c2 == 0:
            r2, c2 = n, 1
        else:
            c1 -= 1
            c2 -= 1

    c1 = c2 = 0
    for n in nums:
        if r1 == n:
            c1 += 1
        elif r2 == n:
            c2 += 1

    r, l = [], len(nums) // 3
    if c1 > l:
        r.append(r1)
    if c2 > l:
        r.append(r2)
    return r
```

## 5625. 比赛中的配对次数

[:link: 来源](https://leetcode-cn.com/problems/count-of-matches-in-tournament/)

### 题目

给你一个整数 `n`, 表示比赛中的队伍数。比赛遵循一种独特的赛制：

- 如果当前队伍数是**偶数**，那么每支队伍都会与另一支队伍配对。总共进行 `n / 2` 场比赛，且产生 `n / 2` 支队伍进入下一轮。
- 如果当前队伍数为**奇数**，那么将会随机轮空并晋级一支队伍，其余的队伍配对。总共进行 `(n - 1) / 2` 场比赛，且产生 `(n - 1) / 2 + 1` 支队伍进入下一轮。

返回在比赛中进行的配对次数，直到决出获胜队伍为止。

#### 示例

```raw
输入：n = 7
输出：6
解释：比赛详情：
- 第 1 轮：队伍数 = 7, 配对次数 = 3, 4 支队伍晋级。
- 第 2 轮：队伍数 = 4, 配对次数 = 2, 2 支队伍晋级。
- 第 3 轮：队伍数 = 2, 配对次数 = 1, 决出 1 支获胜队伍。
总配对次数 = 3 + 2 + 1 = 6
```

```raw
输入：n = 14
输出：13
解释：比赛详情：
- 第 1 轮：队伍数 = 14, 配对次数 = 7, 7 支队伍晋级。
- 第 2 轮：队伍数 = 7, 配对次数 = 3, 4 支队伍晋级。 
- 第 3 轮：队伍数 = 4, 配对次数 = 2, 2 支队伍晋级。
- 第 4 轮：队伍数 = 2, 配对次数 = 1, 决出 1 支获胜队伍。
总配对次数 = 7 + 3 + 2 + 1 = 13
```

#### 提示

- `1 <= n <= 200`.

### 题解

#### 模拟

```python
class Solution:
    def numberOfMatches(self, n: int) -> int:
        return number_of_matches(n)

def number_of_matches(n: int) -> int:
    r = 0
    while n > 1:
        r += n // 2
        n = (n + 1) // 2
    return r
```

#### 计算

每场比赛淘汰一支队伍，共淘汰 `n - 1` 支队伍，故需进行 `n - 1` 场比赛。

```python
class Solution:
    def numberOfMatches(self, n: int) -> int:
        return number_of_matches(n)

def number_of_matches(n: int) -> int:
    return n - 1
```

## 5626. 十、二进制数的最少数目

[:link: 来源](https://leetcode-cn.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/)

### 题目

如果一个十进制数字不含任何前导零，且每一位上的数字不是 `0` 就是 `1`, 那么该数字就是一个**十、二进制数**。例如，`101` 和 `1100` 都是**十、二进制数**，而 `112` 和 `3001` 不是。

给你一个表示十进制整数的字符串 `n`, 返回和为 `n` 的**十、二进制数**的最少数目。

#### 示例

```raw
输入：n = "32"
输出：3
解释：10 + 11 + 11 = 32.
```

```raw
输入：n = "82734"
输出：8
```

```raw
输入：n = "27346209830709182346"
输出：9
```

#### 提示

- `1 <= len(n) <= 1e5`;
- `n` 仅由数字组成；
- `n` 不含任何前导零并总是表示正整数。

### 题解

只需找到最大的十进制数字。

```python
class Solution:
    def minPartitions(self, n: str) -> int:
        return min_partitions(n)

def min_partitions(n: str) -> int:
    return ord(max(n)) - ord('0')
```

## 349. 两个数组的交集

[:link: 来源](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

### 题目

给定两个数组，编写一个函数来计算它们的交集。

#### 示例

```raw
输入：nums1 = [1, 2, 2, 1], nums2 = [2, 2]
输出：[2]
```

```raw
输入：nums1 = [4, 9, 5], nums2 = [9, 4, 9, 8, 4]
输出：[9, 4]
```

#### 说明

- 输出结果中的每个元素一定是唯一的；
- 我们可以不考虑输出结果的顺序。

### 题解

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return intersection(nums1, nums2)

def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    return list(set(nums1) & set(nums2))
```

## 1374. 生成每种字符都是奇数个的字符串

[:link: 来源](https://leetcode-cn.com/problems/generate-a-string-with-characters-that-have-odd-counts/)

### 题目

给你一个整数 `n`, 请你返回一个含 `n` 个字符的字符串，其中每种字符在该字符串中都恰好出现**奇数次**。

返回的字符串必须只含小写英文字母。如果存在多个满足题目要求的字符串，则返回其中任意一个即可。

#### 示例

```raw
输入：n = 4
输出："pppz"
解释："pppz" 是一个满足题目要求的字符串，因为 'p' 出现 3 次，且 'z' 出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ohhh" 和 "love".
```

```raw
输入：n = 2
输出："xy"
解释："xy" 是一个满足题目要求的字符串，因为 'x' 和 'y' 各出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ag" 和 "ur".
```

```raw
输入：n = 7
输出："holasss"
```

#### 提示

- `1 <= n <= 500`.

### 题解

```python
class Solution:
    def generateTheString(self, n: int) -> str:
        return generate_the_string(n)

def generate_the_string(n: int) -> str:
    return 'r' * n if n % 2 else 'r' * (n - 1) + 'g'
```

## 650. 只有两个键的键盘

[:link: 来源](https://leetcode-cn.com/problems/2-keys-keyboard/)

### 题目

最初在一个记事本上只有一个字符 `'A'`. 你每次可以对这个记事本进行两种操作：

1. `Copy All`（复制全部）: 你可以复制这个记事本中的所有字符（部分的复制是不允许的）；
2. `Paste`（粘贴）: 你可以粘贴你上一次复制的字符。

给定一个数字 `n`. 你需要使用最少的操作次数，在记事本中打印出恰好 `n` 个 `'A'`. 输出能够打印出 `n` 个 `'A'` 的最少操作次数。

#### 示例

```raw
输入：3
输出：3
解释：
最初, 我们只有一个字符 'A'.
第 1 步, 我们使用 Copy All 操作。
第 2 步, 我们使用 Paste 操作来获得 'AA'.
第 3 步, 我们使用 Paste 操作来获得 'AAA'.
```

#### 说明

- `1 <= n <= 1000`.

### 题解

贪心，质因数分解。对于 $n=\prod_i f_i$, 其中 $f_i$ 是质数，需要 $\sum_i f_i$ 次操作，即 $\prod_i (\mathrm{Copy\ All})(\mathrm{Paste})^{f_i-1}$.

```python
class Solution:
    def minSteps(self, n: int) -> int:
        return min_steps(n)

def min_steps(n: int) -> int:
    d, r = 2, 0
    while n > 1:
        while n % d == 0:
            r += d
            n //= d
        d += 1
    return r
```

## 49. 字母异位词分组

[:link: 来源](https://leetcode-cn.com/problems/group-anagrams/)

### 题目

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

#### 示例

```raw
输入：["eat", "tea", "tan", "ate", "nat", "bat"]
输出：[["ate", "eat", "tea"], ["nat", "tan"], ["bat"]]
```

#### 说明

- 所有输入均为小写字母；
- 不考虑答案输出的顺序。

### 题解

#### 排序

有序字符串作为键。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        return group_anagrams(strs)

from collections import defaultdict

def group_anagrams(strs: List[str]) -> List[List[str]]:
    r = defaultdict(list)
    for s in strs:
        k = ''.join(sorted(s))
        r[k].append(s)
    return list(r.values())
```

#### 计数

计数元组作为键。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        return group_anagrams(strs)

from collections import defaultdict

def group_anagrams(strs: List[str]) -> List[List[str]]:
    r = defaultdict(list)
    for s in strs:
        c = [0] * 26
        for ch in s:
            c[ord(ch) - ord('a')] += 1
        r[tuple(c)].append(s)
    return list(r.values())
```

#### 质数

利用质数因数分解设计次序不敏感的键。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        return group_anagrams(strs)

from collections import defaultdict
from math import prod

_b = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]

def group_anagrams(strs: List[str]) -> List[List[str]]:
    r = defaultdict(list)
    for s in strs:
        k = prod(_b[ord(ch) - ord('a')] for ch in s)
        r[k].append(s)
    return list(r.values())
```

## 738. 单调递增的数字

[:link: 来源](https://leetcode-cn.com/problems/monotone-increasing-digits/)

### 题目

给定一个非负整数 `N`, 找出小于或等于 `N` 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。

（当且仅当每个相邻位数上的数字 `x` 和 `y` 满足 `x <= y` 时，我们称这个整数是单调递增的。）

#### 示例

```raw
输入：N = 10
输出：9
```

```raw
输入：N = 1234
输出：1234
```

```raw
输入：N = 332
输出：299
```

#### 说明

- `0 <= N <= 1e9`.

### 题解

如果 `N` 不满足要求，则结果应为以若干 `9` 结尾的整数，位于破坏单调性的首个位置之前的不会破坏单调性的首个可退位之后；可退位进行退位；之前的位保留。

```python
class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        return monotone_increasing_digits(N)

from itertools import accumulate

def monotone_increasing_digits(n: int) -> int:
    s = str(n)

    for i in range(1, len(s)):
        if s[i - 1] > s[i]:
            break
    else:
        return n

    for j in range(i - 1, 0, -1):
        if s[j - 1] < s[j] and s[j] > '0':
            break
    else:
        j = 0

    return int(s[:j] + chr(ord(s[j]) - 1) + '9' * (len(s) - j - 1))
```

## 290. 单词规律

[:link: 来源](https://leetcode-cn.com/problems/word-pattern/)

### 题目

给定一种规律 `pattern` 和一个字符串 `str`, 判断 `str` 是否遵循相同的规律。

这里的**遵循**指完全匹配，例如，`pattern` 里的每个字母和字符串 `str` 中的每个非空单词之间存在着双向连接的对应规律。

#### 示例

```raw
输入：pattern = "abba", str = "dog cat cat dog"
输出：true
```

```raw
输入：pattern = "abba", str = "dog cat cat fish"
输出：false
```

```raw
输入：pattern = "aaaa", str = "dog cat cat dog"
输出：false
```

```raw
输入：pattern = "abba", str = "dog dog dog dog"
输出：false
```

#### 说明

你可以假设 `pattern` 只包含小写字母, `str` 包含了由单个空格分隔的小写字母。

### 题解

两个字典记录双射。

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        return wrod_pattern(pattern, s)

def wrod_pattern(pattern: str, s: str) -> bool:
    words = s.split()
    if len(words) != len(pattern):
        return False

    p2w, w2p = {}, {}
    for p, w in zip(pattern, words):
        if (p in p2w and p2w[p] != w) or (w in w2p and w2p[w] != p):
            return False
        if p not in p2w:
            p2w[p], w2p[w] = w, p

    return True
```

## 1663. 具有给定数值的最小字符串

[:link: 来源](https://leetcode-cn.com/problems/smallest-string-with-a-given-numeric-value/)

### 题目

**小写字符**的**数值**是它在字母表中的位置（从 `1` 开始），因此 `'a'` 的数值为 `1`, `'b'` 的数值为 `2`, `'c'` 的数值为 `3`, 以此类推。

字符串由若干小写字符组成，**字符串的数值**为各字符的数值之和。例如，字符串 `"abe"` 的数值等于 `1 + 2 + 5 = 8`.

给你两个整数 `n` 和 `k`. 返回**长度**等于 `n` 且**数值**等于 `k` 的**字典序最小**的字符串。

注意，如果字符串 `x` 在字典排序中位于 `y` 之前，就认为 `x` 字典序比 `y` 小，有以下两种情况：

- `x` 是 `y` 的一个前缀；
- 如果 `i` 是 `x[i] != y[i]` 的第一个位置，且 `x[i]` 在字母表中的位置比 `y[i]` 靠前。

#### 示例

```raw
输入：n = 3, k = 27
输出："aay"
解释：字符串的数值为 1 + 1 + 25 = 27, 它是数值满足要求且长度等于 3 字典序最小的字符串。
```

```raw
输入：n = 5, k = 73
输出："aaszz"
```

#### 提示

- `1 <= n <= 1e5`;
- `n <= k <= 26 * n`.

### 题解

贪心。构造形如 `AmZ` 的串，`A` 由 `'a'` 组成，`Z` 由 `'z'` 组成，`m` 是某个小写字母。

```python
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        return get_smallest_string(n, k)

def get_smallest_string(n: int, k: int) -> str:
    k -= n
    z, m = divmod(k, 25)
    if n <= z:
        return 'z' * n

    a = n - z - 1
    return 'a' * a + chr(m + ord('a')) + 'z' * z
```

## 714. 买卖股票的最佳时机含手续费

[:link: 来源](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

### 题目

给定一个整数数组 `prices`, 其中第 `i` 个元素代表了第 `i` 天的股票价格；非负整数 `fee` 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

#### 示例

```raw
输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
输出：8
解释：能够达到的最大利润：
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润：((8 - 1) - 2) + ((9 - 4) - 2) = 8.
```

- `0 < len(prices) <= 50000`;
- `0 < prices[i] < 50000`;
- `0 <= fee < 50000`.

### 题解

动态规划。`cl`, `op` 分别记录平仓或持仓的情况下迄今的最大盈利。

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        return max_profit(prices, fee)

def max_profit(prices: List[int], fee: int) -> int:
    cl, op = 0, -prices[0]
    for price in prices[1:]:
        cl, op = max(cl, op + price - fee), max(op, cl - price)
    return cl
```

## 137. 只出现一次的数字 II

[:link: 来源](https://leetcode-cn.com/problems/single-number-ii/)

### 题目

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

#### 说明

你的算法应该具有线性时间复杂度。你可以不使用额外空间来实现吗？

#### 示例

```raw
输入：[2, 2, 3, 2]
输出：3
```

```raw
输入：[0, 1, 0, 1, 0, 1, 99]
输出：99
```

### 题解

#### 计数

最通用。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

from collections import Counter

def single_number(nums: List[int]) -> int:
    return next(n for n, c in Counter(nums).items() if c == 1)
```

#### 求和

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

def single_number(nums: List[int]) -> int:
    return (sum(set(nums)) * 3 - sum(nums)) // 2
```

#### 位运算

`once`, `twice` 记录了出现次数为模三余一和二的二进制位。空间复杂度为常数级。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

def single_number(nums: List[int]) -> int:
    once = twice = 0
    for n in nums:
        once = ~twice & (once ^ n)
        twice = ~once & (twice ^ n)
    return once
```

## 136. 只出现一次的数字

[:link: 来源](https://leetcode-cn.com/problems/single-number/)

### 题目

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

#### 说明

你的算法应该具有线性时间复杂度。你可以不使用额外空间来实现吗？

#### 示例

```raw
输入：[2, 2, 1]
输出：1
```

```raw
输入：[4, 1, 2, 1, 2]
输出：4
```

### 题解

#### 计数

最通用。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

from collections import Counter

def single_number(nums: List[int]) -> int:
    return next(n for n, c in Counter(nums).items() if c == 1)
```

#### 求和

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

def single_number(nums: List[int]) -> int:
    return sum(set(nums)) * 2 - sum(nums)
```

#### 位运算

利用按位异或的对合性。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return single_number(nums)

from operator import xor

def single_number(nums: List[int]) -> int:
    return accumulate(xor, nums)
```

## 389. 找不同

[:link: 来源](https://leetcode-cn.com/problems/find-the-difference/)

### 题目

给定两个字符串 `s` 和 `t`, 它们只包含小写字母。

字符串 `t` 由字符串 `s` 随机重排，然后在随机位置添加一个字母。

请找出在 `t` 中被添加的字母。

#### 示例

```raw
输入：s = "abcd", t = "abcde"
输出："e"
解释：'e' 是那个被添加的字母。
```

```raw
输入：s = "", t = "y"
输出："y"
```

```raw
输入：s = "a", t = "aa"
输出："a"
```

```raw
输入：s = "ae", t = "aea"
输出："a"
```

#### 提示

- `0 <= len(s) <= 1000`;
- `len(t) == len(s) + 1`;
- `s` 和 `t` 只包含小写字母。

### 题解

#### 计数

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return find_the_difference(s, t)

from collections import Counter

def find_the_difference(s: str, t: str) -> str:
    for k in Counter(t) - Counter(s):
        return k
```

#### 求和

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return find_the_difference(s, t)

def find_the_difference(s: str, t: str) -> str:
    return chr(sum(map(ord, t)) - sum(map(ord, s)))
```

## 383. 赎金信

[:link: 来源](https://leetcode-cn.com/problems/ransom-note/)

### 题目

给定一个赎金信 `ransom` 字符串和一个杂志 `magazine` 字符串，判断第一个字符串 `ransom` 能不能由第二个字符串 `magazine` 里面的字符构成。如果可以构成，返回 `true`; 否则返回 `false`.

#### 说明

为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。

#### 注意

你可以假设两个字符串均只含有小写字母。

#### 示例

```raw
canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true
```

### 题解

#### 计数

```python
class Solution:
    def canConstruct(self, ransom: str, magazine: str) -> bool:
        return can_construct(ransom, magazine)

from collections import Counter

def can_construct(ransom: str, magazine: str) -> bool:
    return not (Counter(ransom) - Counter(magazine))
```

## 540. 有序数组中的单一元素

[:link: 来源](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

### 题目

给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

#### 示例

```raw
输入：[1, 1, 2, 3, 3, 4, 4, 8, 8]
输出：2
```

```raw
输入：[3, 3, 7, 7, 10, 11, 11]
输出：10
```

#### 注意

您的方案应该在 $\mathrm{O}(\log n)$ 时间复杂度和 $\mathrm{O}(1)$ 空间复杂度中运行。

### 题解

#### 二分查找

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        return single_non_duplicate(nums)

def single_non_duplicate(nums: List[int]) -> int:
    low, high = 0, len(nums)
    while low + 1 < high:
        mid = (low + high) // 2
        mid -= mid % 2
        if nums[mid] == nums[mid + 1]:
            low = mid + 2
        else:
            high = mid + 1
    return nums[low]
```

## 48. 旋转图像

[:link: 来源](https://leetcode-cn.com/problems/rotate-image/)

### 题目

给定一个 $n\times n$ 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

#### 说明

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

#### 示例

```raw
输入：matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

输出：matrix = [
    [7, 4, 1],
    [8, 5, 2],
    [9, 6, 3],
]
```

```raw
输入：matrix = [
    [ 5,  1,  9, 11],
    [ 2,  4,  8, 10],
    [13,  3,  6,  7],
    [15, 14, 12, 16],
]

输出：matrix = [
    [15, 13,  2,  5],
    [14,  3,  4,  1],
    [12,  6,  8,  9],
    [16,  7, 10, 11],
]
```

### 题解

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        return rotate(matrix)

def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    for i in range(n // 2):
        for j in range((n + 1) // 2):
            matrix[        i][        j], matrix[        j][n - i - 1],  \
            matrix[n - i - 1][n - j - 1], matrix[n - j - 1][        i] = \
            matrix[n - j - 1][        i], matrix[        i][        j],  \
            matrix[        j][n - i - 1], matrix[n - i - 1][n - j - 1]
```

## 1209. 删除字符串中的所有相邻重复项 II

[:link: 来源](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string-ii/)

### 题目

给你一个字符串 `s`,「k 倍重复项删除操作」将会从 `s` 中选择 `k` 个相邻且相等的字母，并删除它们，使被删去的字符串的左侧和右侧连在一起。

你需要对 `s` 重复进行无限次这样的删除操作，直到无法继续为止。

在执行完所有删除操作后，返回最终得到的字符串。

本题答案保证唯一。

#### 示例

```raw
输入：s = "abcd", k = 2
输出："abcd"
解释：没有要删除的内容。
```

```raw
输入：s = "deeedbbcccbdaa", k = 3
输出："aa"
解释：
先删除 "eee" 和 "ccc", 得到 "ddbbbdaa"
再删除 "bbb", 得到 "dddaa"
最后删除 "ddd", 得到 "aa"
```

```raw
输入：s = "pbbcggttciiippooaais", k = 2
输出："ps"
```

#### 提示

- `1 <= len(s) <= 1e5`;
- `2 <= k <= 1e4`;
- `s` 中只含有小写英文字母。

### 题解

栈。每一字符与栈顶消重。是{% post_link leetcode-1047 %}的推广。

```python
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        return remove_duplicates(s, k)

from operator import mul

def remove_duplicates(s: str, k: int) -> str:
    r, cnt = [], []
    for c in s:
        if not r or r[-1] != c:
            r.append(c)
            cnt.append(1)
        elif cnt[-1] < k - 1:
            cnt[-1] += 1
        else:
            r.pop()
            cnt.pop()
    return ''.join(map(mul, r, cnt))
```

## 917. 仅仅反转字母

[:link: 来源](https://leetcode-cn.com/problems/reverse-only-letters/)

### 题目

给定一个字符串 `S`, 返回 “反转后的” 字符串，其中不是字母的字符都保留在原地，而所有字母的位置发生反转。

#### 示例

```raw
输入："ab-cd"
输出："dc-ba"
```

```raw
输入："a-bC-dEf-ghIj"
输出："j-Ih-gfE-dCba"
```

```raw
输入："Test1ng-Leet=code-Q!"
输出："Qedo1ct-eeLg=ntse-T!"
```

#### 提示

- `len(S) <= 100`;
- `33 <= ord(S[i]) <= 122`;
- `S` 中不包含 `'\'` or `'"'`.

### 题解

双指针。

```python
class Solution:
    def reverseOnlyLetters(self, S: str) -> str:
        return reverse_only_letters(S)

def reverse_only_letters(s: str) -> str:
    s = list(s)
    i, j = 0, len(s) - 1
    while i < j:
        if s[i].isalpha() and s[j].isalpha():
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
        elif not s[i].isalpha():
            i += 1
        elif not s[j].isalpha():
            j -= 1
    return ''.join(s)
```

## 316. 去除重复字母

[:link: 来源](https://leetcode-cn.com/problems/remove-duplicate-letters/)

### 题目

给你一个字符串 `s`, 请你去除字符串中重复的字母，使得每个字母只出现一次。需保证**返回结果的字典序最小**（要求不能打乱其他字符的相对位置）。

#### 注意

该题与 [1081](https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters/) 相同。

#### 示例

```raw
输入：s = "bcabc"
输出："abc"
```

```raw
输入：s = "cbacdcbc"
输出："acdb"
```

#### 提示

- `1 <= len(s) <= 1e4`;
- `s` 由小写英文字母组成。

### 题解

贪心，单调栈。集合 `included` 记录已经加入到栈 `stack` 中的字符。如果栈顶字符 `top` 与当前字符 `c` 出现逆序 `top >= c`, 且未来还有机会遇到栈顶字符，即 `remaining[top] > 0`, 则不断出栈。

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        return remove_duplicate_letters(s)

from collections import Counter

def remove_duplicate_letters(s: str) -> str:
    stack, included, remaining = [], set(), Counter(s)
    for c in s:
        if c not in included:
            while stack and (top := stack[-1]) >= c and remaining[top]:
                stack.pop()
                included.remove(top)
            stack.append(c)
            included.add(c)
        remaining[c] -= 1
    return ''.join(stack)
```

## 746. 使用最小花费爬楼梯

[:link: 来源](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

### 题目

数组的每个索引作为一个阶梯，第 `i` 个阶梯对应着一个非负数的体力花费值 `cost[i]`（索引从 0 开始）。

每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。

您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 `0` 或 `1` 的元素作为初始阶梯。

#### 示例

```raw
输入：cost = [10, 15, 20]
输出：15
解释：最低花费是从 cost[1] 开始，然后走两步即可到阶梯顶，一共花费 15.
```

```raw
输入：cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出：6
解释：最低花费方式是从 cost[0] 开始，逐个经过那些 1, 跳过 cost[3], 一共花费 6.
```

#### 注意

- `cost` 的长度将会在 `[2, 1000]`;
- 每一个 `cost[i]` 将会是一个 `int` 类型，范围为 `[0, 999]`.

### 题解

动态规划。

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        return min_cost_climbing_stairs(cost)

def min_cost_climbing_stairs(cost: List[int]) -> int:
    r1 = r2 = 0
    for c in cost:
        r1, r2 = min(r1, r2) + c, r1
    return min(r1, r2)
```

## 387. 字符串中的第一个唯一字符

[:link: 来源](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

### 题目

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 `-1`.

#### 示例

```raw
输入：s = "leetcode"
输出：0
```

```raw
输入：s = "loveleetcode"
输出：2
``` 

#### 提示

你可以假定该字符串只包含小写字母。

### 题解

计数，迭代。

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        return first_unique_character(s)

from collections import Counter

def first_unique_character(s: str) -> int:
    counter = Counter(s)
    return next((i for i, c in enumerate(s) if counter[c] == 1), -1)
```

## 135. 分发糖果

[:link: 来源](https://leetcode-cn.com/problems/candy/)

### 题目

老师想给孩子们分发糖果，有 `N` 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

- 每个孩子至少分配到 `1` 个糖果；
- 相邻的孩子中，评分高的孩子必须获得更多的糖果。

那么这样下来，老师至少需要准备多少颗糖果呢？

#### 示例

```raw
输入：[1, 0, 2]
输出：5
解释：你可以分别给这三个孩子分发 2、1、2 颗糖果。
```

```raw
输入：[1, 2, 2]
输出：4
解释：你可以分别给这三个孩子分发 1、2、1 颗糖果。第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
```

### 题解

#### 朴素贪心

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        return candy(ratings)

def candy(ratings: List[int]) -> int:
    n = len(ratings)

    l = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            l[i] = l[i - 1] + 1

    r = [1] * n
    for i in range(n - 1, 0, -1):
        if ratings[i - 1] > ratings[i]:
            r[i - 1] = r[i] + 1

    return sum(map(max, l, r))
```

#### 优化贪心

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        return candy(ratings)

from operator import sub

def candy(ratings: List[int]) -> int:
    r, inc, dec, pre = 1, 1, 0, 1
    for d in map(sub, ratings[1:], ratings):
        if d >= 0:
            dec = 0
            pre = 1 if d == 0 else pre + 1
            r += pre
            inc = pre
        else:
            dec += 1
            if dec == inc:
                dec += 1
            r += dec
            pre = 1
    return r
```

## 455. 分发饼干

[:link: 来源](https://leetcode-cn.com/problems/assign-cookies/)

### 题目

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 `i`, 都有一个胃口值 `g[i]`, 这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 `j`, 都有一个尺寸 `s[j]`. 如果 `s[j] >= g[i]`, 我们可以将这个饼干 `j` 分配给孩子 `i`, 这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

#### 示例

```raw
输入：g = [1, 2, 3], s = [1, 1]
输出：1
解释：
你有三个孩子和两块小饼干，3 个孩子的胃口值分别是 1, 2, 3.
虽然你有两块小饼干，由于他们的尺寸都是 1, 你只能让胃口值是 1 的孩子满足。
所以你应该输出 1.
```

```raw
输入：g = [1, 2], s = [1, 2, 3]
输出：2
解释：
你有两个孩子和三块小饼干，2 个孩子的胃口值分别是 1, 2.
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出 2.
```

#### 提示

- `1 <= len(g) <= 3e4`;
- `0 <= len(s) <= 3e4`;
- `1 <= g[i], s[j] <= 2 ** 31 - 1`.

### 题解

排序，贪心。使用最小的代价满足每个胃口小的孩子。

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        return find_content_children(g, s)

def find_content_children(g: List[int], s: List[int]) -> int:
    g, s, i = sorted(g), sorted(s), 0
    for t in s:
        if i >= len(g):
            break
        if g[i] <= t:
            i += 1
    return i
```

## 84. 柱状图中最大的矩形

[:link: 来源](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

### 题目

给定 `n` 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 `1`.

求在该柱状图中，能够勾勒出来的矩形的最大面积。

#### 示例

```raw
输入：[2, 1, 5, 6, 2, 3]
输出：10
```

### 题解

单调栈。每次出栈都对应找到了一组 `(li, mh, ri)`, 代表了一个高度为 `mh` 的柱，左右最近矮于它的柱的索引分别是 `li` 和 `ri`. 从而确定了一个局部最大矩形，高度为 `mh`, 宽度为左右矮柱所夹部分 `ri - li - 1`.

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        return largest_rectangle_area(heights)

from itertools import chain

def largest_rectangle_area(heights: List[int]) -> int:
    r, s = 0, [(-1, -1)]
    for ri, rh in enumerate(chain(heights, [0])):
        while s and rh <= (mh := s[-1][1]):
            s.pop()
            li = s[-1][0]
            r = max(r, (ri - li - 1) * mh)
        s.append((ri, rh))
    return r
```

## 85. 最大矩形

[:link: 来源](https://leetcode-cn.com/problems/maximal-rectangle/)

### 题目

给定一个仅包含 `'0'` 和 `'1'`, 大小为 $rows\times cols$ 的二维二进制矩阵，找出只包含 `'1'` 的最大矩形，并返回其面积。

#### 示例

```raw
输入：matrix = [
    ['1', '0', '1', '0', '0'],
    ['1', '0', '1', '1', '1'],
    ['1', '1', '1', '1', '1'],
    ['1', '0', '0', '1', '0']
]
输出：6
```

```raw
输入：matrix = []
输出：0
```

```raw
输入：matrix = [['0']]
输出：0
```

```raw
输入：matrix = [['1']]
输出：1
```

```raw
输入：matrix = [['0', '0']]
输出：0
```

#### 提示

- `rows == len(matrix)`;
- `cols == len(matrix[0])`;
- `0 <= row, cols <= 200`;
- `matrix[i][j]` 为 `'0'` 或 `'1'`.

### 题解

将矩阵逐行转化为柱状图，再利用{% post_link leetcode-84 %}的方法求解。

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        return maximal_rectangle(matrix)

from itertools import chain

def maximal_rectangle(matrix: List[List[str]]) -> int:
    if not matrix:
        return 0

    r, hs = 0, [0] * len(matrix[0])
    for row in matrix:
        hs = [h + 1 if int(e) else 0 for h, e in zip(hs, row)]
        r = max(r, maximal_rectangle_histogram(hs))
    return r

def maximal_rectangle_histogram(heights: List[int]) -> int:
    r, s = 0, [(-1, -1)]
    for ri, rh in enumerate(chain(heights, [0])):
        while s and rh <= (mh := s[-1][1]):
            s.pop()
            li = s[-1][0]
            r = max(r, (ri - li - 1) * mh)
        s.append((ri, rh))
    return r
```

## 205. 同构字符串

[:link: 来源](https://leetcode-cn.com/problems/isomorphic-strings/)

### 题目

给定两个字符串 `s` 和 `t`, 判断它们是否是同构的。

如果 `s` 中的字符可以被替换得到 `t`, 那么这两个字符串是同构的。

所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

#### 示例

```raw
输入：s = "egg", t = "add"
输出：true
```

```raw
输入：s = "foo", t = "bar"
输出：false
```

```raw
输入：s = "paper", t = "title"
输出：true
```

#### 说明

你可以假设 `s` 和 `t` 具有相同的长度。

### 题解

#### 高效

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        return is_isomorphix(s, t)

def is_isomorphix(s: str, t: str) -> bool:
    s2t, tcs = {}, set()
    for sc, tc in zip(s, t):
        if sc in s2t:
            if s2t[sc] != tc:
                return False
        else:
            if tc in tcs:
                return False
            s2t[sc] = tc
            tcs.add(tc)
    return True
```

#### 简洁

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        return is_isomorphix(s, t)

def is_isomorphix(s: str, t: str) -> bool:
    return len(set(s)) == len(set(t)) == len(set(zip(s, t)))
```

## 856. 括号的分数

[:link: 来源](https://leetcode-cn.com/problems/score-of-parentheses/)

### 题目

给定一个平衡括号字符串 `S`, 按下述规则计算该字符串的分数：

- `'()'` 得 `1` 分；
- `A + B` 得 `A`, `B` 的分数之和分，其中 `A` 和 `B` 是平衡括号字符串；
- `'(' + A + ')'` 得 `A` 的分数的二倍分，其中 `A` 是平衡括号字符串。

#### 示例

```raw
输入："()"
输出：1
```

```raw
输入："(())"
输出：2
```

```raw
输入："()()"
输出：2
```

```raw
输入："(()(()))"
输出：6
```

#### 提示

- `S` 是平衡括号字符串，且只含有 `'('` 和 `')'`;
- `2 <= len(S) <= 50`.

### 题解

```python
class Solution:
    def scoreOfParentheses(self, S: str) -> int:
        return score_of_parentheses(S)

def score_of_parentheses(s: str) -> int:
    stack = [0]
    for c in s:
        if c == '(':
            stack.append(0)
        else:
            t = stack.pop()
            stack[-1] += t * 2 if t else 1
    return stack.pop()
```

## 188. 买卖股票的最佳时机 IV

[:link: 来源](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

### 题目

给定一个整数数组 `prices`, 它的第 `i` 个元素 `prices[i]` 是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 `k` 笔交易。

#### 注意

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

#### 示例

```raw
输入：k = 2, prices = [2, 4, 1]
输出：2
解释：
在第 1 天（股票价格 = 2）的时候买入，在第 2 天（股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4 - 2 = 2.
```

```raw
输入：k = 2, prices = [3, 2, 6, 5, 0, 3]
输出：7
解释：
在第 2 天（股票价格 = 2）的时候买入，在第 3 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 2 = 4.
随后，在第 5 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出, 这笔交易所能获得利润 = 3 - 0 = 3.
```

#### 提示

- `0 <= k <= 1e9`;
- `0 <= len(prices) <= 1000`;
- `0 <= prices[i] <= 1000`.

### 题解

- 动态规划。仿照{% post_link leetcode-714 %}，其中 `p[j * 2]` 表示最多进行 `j` 次交易且当前为空仓的利润，`p[j * 2 + 1]` 表示最多进行 `j` 次交易且当前为开仓的利润，这也暗示 `p[:(j + 1) * 2]` 可以完整地表示 `j` 次交易的最大利润；
- 最多进行 `len(prices) // 2` 次有效的交易（买卖股票），故而可缩小总共需申请的状态空间。考虑第 `i` 支股票时，最多进行 `(i + 1) // 2` 次有效的交易，故而可缩小每次迭代需更新的状态空间。

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        return max_profit(k, prices)

def max_profit(k: int, prices: List[int]) -> int:
    p = [0] + [float('-inf')] * (min(k, len(prices) // 2) * 2)
    for i, price in enumerate(prices):
        for j in range(1, min(len(p), ((i + 1) // 2 + 1) * 2)):
            p[j] = max(p[j], p[j - 1] + price * (-1 if j % 2 else 1))
    return p[-1]
```

## 330. 按要求补齐数组

[:link: 来源](https://leetcode-cn.com/problems/patching-array/)

### 题目

给定一个已排序的正整数数组 `nums`, 和一个正整数 `n`. 从 `[1, n]` 区间内选取任意个数字补充到 `nums` 中，使得 `[1, n]` 区间内的任何数字都可以用 `nums` 中某几个数字的和来表示。请输出满足上述要求的最少需要补充的数字个数。

#### 示例

```raw
输入：nums = [1, 3], n = 6
输出：1
解释：
根据 nums 里现有的组合 [1], [3], [1, 3]，可以得出 1, 3, 4.
现在如果我们将 2 添加到 nums 中， 组合变为: [1], [2], [3], [1, 3], [2, 3], [1, 2, 3].
其和可以表示数字 1, 2, 3, 4, 5, 6, 能够覆盖 [1, 6] 区间里所有的数。
所以我们最少需要添加一个数字。
```

```raw
输入：nums = [1, 5, 10], n = 20
输出：2
解释：我们需要添加 [2, 4].
```

```raw
输入：nums = [1, 2, 2], n = 5
输出：0
```

### 题解

贪心。当区间 `[1, m)` 中的整数可以被表示时，可以利用 `k <= m` 的数进行扩张，从而表示区间 `[1, m + k)` 中的整数。当 `nums` 中没有合适的 `k` 时，最节省的扩张是直接补充 `m`, 这样可以表示区间 `[1, 2 * m)` 中的整数，此时需要补充的个数 `r += 1`.

```python
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        return min_patches(nums, n)

def min_patches(nums: List[int], n: int) -> int:
    m, i, r = 1, 0, 0
    while m <= n:
        if i < len(nums) and (k := nums[i]) <= m:
            m += k
            i += 1
        else:
            m *= 2
            r += 1
    return r
```

## 1046. 最后一块石头的重量

[:link: 来源](https://leetcode-cn.com/problems/last-stone-weight/)

### 题目

有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块**最重的**石头，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`, 且 `x <= y`. 那么粉碎的可能结果如下：

- 如果 `x == y`, 那么两块石头都会被完全粉碎；
- 如果 `x != y`, 那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y - x`.

最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 `0`.

#### 示例

```raw
输入：[2, 7, 4, 1, 8, 1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2, 4, 1, 1, 1];
再选出 2 和 4，得到 2，所以数组转换为 [2, 1, 1, 1];
接着是 2 和 1，得到 1，所以数组转换为 [1, 1, 1];
最后选出 1 和 1，得到 0，最终数组转换为 [1], 这就是最后剩下那块石头的重量。
```

#### 提示

- `1 <= len(stones) <= 30`;
- `1 <= stones[i] <= 1000`.

### 题解

#### 排序

模拟。排序，二分插入。

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        return last_stone_weight(stones)

from bisect import insort

def last_stone_weight(stones: List[int]) -> int:
    stones = sorted(stones)
    while len(stones) >= 2:
        x = stones.pop()
        y = stones.pop()
        if x > y:
            insort(stones, x - y)
    return stones.pop() if stones else 0
```

#### 堆

模拟。最小堆。通过取负数转化成最小堆。

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        return last_stone_weight(stones)

from heapq import heapify, heappop, heappush

def last_stone_weight(stones: List[int]) -> int:
    stones = [-s for s in stones]
    heapify(stones)
    while len(stones) >= 2:
        x = heappop(stones)
        y = heappop(stones)
        if x < y:
            heappush(stones, x - y)
    return -heappop(stones) if stones else 0
```

## 561. 数组拆分 I

[:link: 来源](https://leetcode-cn.com/problems/array-partition-i/)

### 题目

给定长度为 $2n$ 的整数数组 `nums`, 你的任务是将这些数分成 $n$ 对, 例如 $(a_1,b_1),(a_2,b_2),\dots,(a_n,b_n)$, 使得 $\sum_{i=1}^n\min(a_i,b_i)$ 最大。

返回该**最大总和**。

#### 示例

```raw
输入：nums = [1, 4, 3, 2]
输出：4
解释：所有可能的分法（忽略元素顺序）为：
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3;
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3;
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4;
所以最大总和为 4.
```

```raw
输入：nums = [6, 2, 6, 5, 1, 2]
输出：9
解释：最优的分法为 (2, 1), (2, 5), (6, 6). min(2, 1) + min(2, 5) + min(6, 6) = 1 + 2 + 6 = 9.
```

#### 提示

- `1 <= n <= 1e4`;
- `len(nums) == 2 * n`;
- `-1e4 <= nums[i] <= 1e4`.

### 题解

贪心。排序后即 $[a_1,b_1,a_2,b_2,\dots,a_n,b_n]$ 且 $a_i<b_i$, 于是求 $\sum_{i=1}^na_i$ 即可。

```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        return array_pair_sum(nums)

def array_pair_sum(nums: List[int]) -> int:
    return sum(sorted(nums)[::2])
```

## 435. 无重叠区间

[:link: 来源](https://leetcode-cn.com/problems/non-overlapping-intervals/)

### 题目

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

#### 注意

- 可以认为区间的终点总是大于它的起点；
- 区间 `[1, 2]` 和 `[2, 3]` 的边界相互“接触”，但没有相互重叠。

#### 示例

```raw
输入：[[1, 2], [2, 3], [3, 4], [1, 3]]
输出：1
解释：移除 [1, 3] 后，剩下的区间没有重叠。
```

```raw
输入：[[1, 2], [1, 2], [1, 2]]
输出：2
解释：你需要移除两个 [1, 2] 来使剩下的区间没有重叠。
```

```raw
输入：[[1, 2], [2, 3]]
输出：0
解释：你不需要移除任何区间，因为它们已经是无重叠的了。
```

### 题解

贪心。从 `intervals` 逐一选出能保留下来的区间，对于当前所有可以保留的区间，选择右端点最小的。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        return erase_overlap_intervals(intervals)

from operator import itemgetter

def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    intervals = sorted(intervals, key=itemgetter(1))
    last, reserved = float('-inf'), 0
    for l, r in intervals:
        if l >= last:
            last = r
            reserved += 1
    return len(intervals) - reserved
```

## 1010. 总持续时间可被 60 整除的歌曲

[:link: 来源](https://leetcode-cn.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/)

### 题目

在歌曲列表中，第 `i` 首歌曲的持续时间为 `time[i]` 秒。

返回其总持续时间（以秒为单位）可被 `60` 整除的歌曲对的数量。形式上，我们希望索引的数字 `i` 和 `j` 满足，`i < j` 且有 `(time[i] + time[j]) % 60 == 0`.

#### 示例

```raw
输入：[30, 20, 150, 100, 40]
输出：3
解释：这三对的总持续时间可被 60 整除，
(time[0] = 30, time[2] = 150) 总持续时间 180;
(time[1] = 20, time[3] = 100) 总持续时间 120;
(time[1] = 20, time[4] = 40) 总持续时间 60.
```

```raw
输入：[60, 60, 60]
输出：3
解释：所有三对的总持续时间都是 120, 可以被 60 整除。
```

#### 提示

- `1 <= len(time) <= 60000`;
- `1 <= time[i] <= 500`.

### 题解

计数。

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        return num_pairs_divisible_by_60(time)

from collections import Counter

def num_pairs_divisible_by_60(time: List[int]) -> int:
    c = Counter(t % 60 for t in time)
    return sum(c[t] * c[60 - t] for t in c if t < 30) \
        + (c[0] * (c[0] - 1) + c[30] * (c[30] - 1)) // 2
```
