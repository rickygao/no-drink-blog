---
title: LeetCode 月报 202011
date: 2020-11-24 22:00:00
tags: [LeetCode]
mathjax: true
---

梦开始的地方，并不完整的月报完成了 19 道题目。

<!-- more -->

## 222. 完全二叉树的节点个数{#leetcode-222}

[:link: 来源](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

### 题目

给出一个完全二叉树，求出该树的节点个数。

#### 说明

完全二叉树的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 $h$ 层，则该层包含 $[1, 2^h]$ 个节点。

#### 示例

```raw
输入：

    1
   / \
  2   3
 / \  /
4  5 6

输出：6
```

### 题解

- 利用完全二叉树的性质，左子树和右子树至少存在一棵满二叉树，可以由高度直接计算节点个数，而另外一棵则是完全二叉树，可以通过递归计算；
- 复用高度计算的结果。

#### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def countNodes(self, root: TreeNode) -> int:
        return count_nodes(root)

def count_nodes(root: TreeNode, lheight: int = -1) -> int:
    if not root:
        return 0

    if lheight == -1:
        lheight = tree_height(root.left)
    rheight = tree_height(root.right)
    if lheight == rheight:
        return 2 ** lheight + count_nodes(root.right, rheight - 1)
    else:
        return 2 ** rheight + count_nodes(root.left, lheight - 1)

def tree_height(root: TreeNode) -> int:
    if not root:
        return 0

    height = 1
    while root.left:
        root = root.left
        height += 1
    return height
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
    def countNodes(self, root: TreeNode) -> int:
        return count_nodes(root)

def count_nodes(root: TreeNode) -> int:
    if not root:
        return 0

    count = 0
    lheight = tree_height(root.left)
    while root:
        rheight = tree_height(root.right)
        if lheight == rheight:
            count += 2 ** lheight
            lheight = rheight - 1
            root = root.right
        else:
            count += 2 ** rheight
            lheight = lheight - 1
            root = root.left
    return count

def tree_height(root: TreeNode) -> int:
    height = 0
    while root:
        root = root.left
        height += 1
    return height
```

## 1370. 上升下降字符串{#leetcode-1370}

[:link: 来源](https://leetcode-cn.com/problems/increasing-decreasing-string/)

### 题目

给你一个字符串 `s`, 请你根据下面的算法重新构造字符串：

1. 从 `s` 中选出**最小**的字符，将它**接在**结果字符串的后面；
2. 从 `s` 剩余字符中选出**最小**的字符，且该字符比上一个添加的字符大，将它**接在**结果字符串后面；
3. 重复步骤 2, 直到你没法从 `s` 中选择字符；
4. 从 `s` 中选出**最大**的字符，将它**接在**结果字符串的后面；
5. 从 `s` 剩余字符中选出**最大**的字符，且该字符比上一个添加的字符小，将它**接在**结果字符串后面；
6. 重复步骤 5, 直到你没法从 `s` 中选择字符；
7. 重复步骤 1 到 6, 直到 `s` 中所有字符都已经被选过。

在任何一步中，如果最小或者最大字符不止一个，你可以选择其中任意一个，并将其添加到结果字符串。

请你返回将 `s` 中字符重新排序后的**结果字符串**。

#### 示例

```raw
输入：s = "aaaabbbbcccc"
输出："abccbaabccba"
解释：第一轮的步骤 1, 2, 3 后，结果字符串为 result = "abc".
第一轮的步骤 4, 5, 6 后，结果字符串为 result = "abccba".
第一轮结束，现在 s = "aabbcc", 我们再次回到步骤 1.
第二轮的步骤 1, 2, 3 后，结果字符串为 result = "abccbaabc".
第二轮的步骤 4, 5, 6 后，结果字符串为 result = "abccbaabccba".
```

```raw
输入：s = "rat"
输出："art"
解释：单词 "rat" 在上述算法重排序以后变成 "art".
```

```raw
输入：s = "leetcode"
输出："cdelotee"
```

```raw
输入：s = "ggggggg"
输出："ggggggg"
```

```raw
输入：s = "spo"
输出："ops"
```

#### 提示

- `1 <= len(s) <= 500`;
- `s` 只包含小写英文字母。

### 题解

桶计数，往复遍历。

```python
class Solution:
    def sortString(self, s: str) -> str:
        return sort_string(s)

def sort_string(s: str) -> str:
    ord_a = ord('a')

    counter = [0] * 26
    for c in s:
        counter[ord(c) - ord_a] += 1

    turn, result = 0, []
    desc = False
    for _ in range(max(counter)):
        it = range(len(counter))
        if desc:
            it = reversed(it)

        for i in it:
            if counter[i] > turn:
                result.append(chr(i + ord_a))

        turn += 1
        desc = not desc

    return ''.join(result)
```

## 1. 两数之和{#leetcode-1}

[:link: 来源](https://leetcode-cn.com/problems/two-sum/)

### 题目

给定一个整数数组 `nums` 和一个目标值 `target`, 请你在该数组中找出和为目标值的那**两个**整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

#### 示例

```raw
给定 nums = [2, 7, 11, 15], target = 9,
因为 nums[0] + nums[1] = 2 + 7 = 9,
所以返回 [0, 1].
```

### 题解

索引反查字典。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        return two_sum(nums, target)

def two_sum(nums: List[int], target: int) -> List[int]:
    seen = dict()
    for i, m in enumerate(nums):
        n = target - m
        if n in seen:
            return [i, seen[n]]
        seen[m] = i
    raise ValueError
```

## 2. 两数相加{#leetcode-2}

[:link: 来源](https://leetcode-cn.com/problems/add-two-numbers/)

### 题目

给出两个**非空**的链表用来表示两个非负的整数。其中，它们各自的位数是按照**逆序**的方式存储的，并且它们的每个节点只能存储**一位**数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 `0` 之外，这两个数都不会以 `0` 开头。

#### 示例

```raw
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

### 题解

给出了一个通用的解，可以传入任意多个数。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        return add_numbers(l1, l2)

def add_numbers(*heads: List[ListNode]) -> ListNode:
    nodes, current = list(heads), 0
    node = dummy = ListNode()
    while any(nodes) or current:
        current += sum(n.val for n in nodes)
        current, r = divmod(current, 10)
        nodes = [n.next for n in nodes if n.next]
        node.next = node = ListNode(val=r)
    return dummy.next
```

## 3. 无重复字符的最长子串{#leetcode-3}

[:link: 来源](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

### 题目

给定一个字符串，请你找出其中不含有重复字符的**最长子串**的长度。

#### 示例

```raw
输入：s = "abcabcbb"
输出：3 
解释：因为无重复字符的最长子串是 "abc", 所以其长度为 3.
```

```raw
输入：s = "bbbbb"
输出：1
解释：因为无重复字符的最长子串是 "b", 所以其长度为 1.
```

```raw
输入：s = "pwwkew"
输出：3
解释：因为无重复字符的最长子串是 "wke", 所以其长度为 3.
     请注意，你的答案必须是子串的长度，"pwke" 是一个子序列，不是子串.
```

```raw
输入：s = ""
输出：0
```

#### 提示

- `0 <= len(s) <= 5e4`;
- `s` 由英文字母、数字、符号和空格组成。

### 题解

- `[i, j]` 滑动窗口；
- 利用字典进行 `i` 跳跃。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        return length_of_longest_substring(s)

def length_of_longest_substring(s: str) -> int:
    c2p = dict()
    i = mlen = 0
    for j, c in enumerate(s):
        p = c2p.get(c, -1)
        if p >= i:
            mlen = max(j - i, mlen)
            i = p + 1
        c2p[c] = j
    mlen = max(len(s) - i, mlen)
    return mlen
```

## 164. 最大间距{#leetcode-164}

[:link: 来源](https://leetcode-cn.com/problems/maximum-gap/)

### 题目

给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。

如果数组元素个数小于 `2`, 则返回 `0`.

#### 示例

```raw
输入：[3, 6, 9, 1]
输出：3
解释：排序后的数组是 [1, 3, 6, 9], 其中相邻元素 (3, 6) 和 (6, 9) 之间都存在最大差值 3.
```

```raw
输入：[10]
输出：0
解释：数组元素个数小于 2, 因此返回 0.
```

#### 说明

- 你可以假设数组中所有元素都是非负整数，且数值在 32 位有符号整数范围内；
- 请尝试在线性时间复杂度和空间复杂度的条件下解决此问题。

### 题解

#### 排序

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        return maximum_gap(nums)

from operator import sub

def maximum_gap(nums: List[int]) -> int:
    nums = sorted(nums)
    return max(map(sub, nums[1:], nums[:-1]), default=0)
```

#### 桶

记整数列表的长度为 $l$, 最小值和最大值分别为 $m$, $n$. 则桶的大小为 $s=\frac{n-m}{l-1}$, 第 $i$ 个桶容纳 $[m+is,m+(i+1)s)$, 可保证最大间距不会出现在桶内。

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        return maximum_gap(nums)

def maximum_gap(nums: List[int]) -> int:
    if len(nums) < 2:
        return 0

    mi, ma = min(nums), max(nums)
    size_bucket = max((ma - mi) // (len(nums) - 1), 1)
    num_buckets = (ma - mi) // size_bucket + 1
    min_buckets = [...] * num_buckets
    max_buckets = [...] * num_buckets

    for n in nums:
        i_bucket = (n - mi) // size_bucket
        if min_buckets[i_bucket] is ... or n < min_buckets[i_bucket]:
            min_buckets[i_bucket] = n
        if max_buckets[i_bucket] is ... or n > max_buckets[i_bucket]:
            max_buckets[i_bucket] = n

    r, prev = 0, ...
    for i in range(num_buckets):
        # check if an empty bucket
        if min_buckets[i] is ...:
            continue
        if prev is ...:
            r = max(r, min_buckets[i] - max_buckets[prev])
        prev = i
    return r
```

## 905. 按奇偶排序数组{#leetcode-905}

[:link: 来源](https://leetcode-cn.com/problems/sort-array-by-parity/)

### 题目

给定一个非负整数数组 `A`, 返回一个数组，在该数组中，`A` 的所有偶数元素之后跟着所有奇数元素。

你可以返回满足此条件的任何数组作为答案。

#### 示例

```raw
输入：[3, 1, 2, 4]
输出：[2, 4, 3, 1]
输出 [4, 2, 3, 1], [2, 4, 1, 3] 和 [4, 2, 1, 3] 也会被接受。
```

#### 提示

- `1 <= len(A) <= 5000`;
- `0 <= A[i] <= 5000`.

### 题解

双指针，一个向后遍历，一个向前遍历，找到每一个需要交换的数对。

```python
class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        return sort_array_by_parity(A)

def sort_array_by_parity(l: List[int]) -> List[int]:
    i = 0
    j = len(l) - 1
    while i < j:
        while l[i] % 2 == 0 and i < j:
            i += 1
        while l[j] % 2 == 1 and i < j:
            j -= 1
        if i < j:
            tmp = l[i]
            l[i] = l[j]
            l[j] = tmp
    return l
```

## 922. 按奇偶排序数组 II{#leetcode-922}

[:link: 来源](https://leetcode-cn.com/problems/sort-array-by-parity-ii/)

### 题目

给定一个非负整数数组 `A`, `A` 中一半整数是奇数，一半整数是偶数。

对数组进行排序，以便当 `A[i]` 为奇数时，`i` 也是奇数；当 `A[i]` 为偶数时, `i` 也是偶数。

你可以返回任何满足上述条件的数组作为答案。

#### 示例

```raw
输入：[4, 2, 5, 7]
输出：[4, 5, 2, 7]
解释：[4, 7, 2, 5], [2, 5, 4, 7], [2, 7, 4, 5] 也会被接受。
```

#### 提示

- `2 <= len(A) <= 20000`;
- `len(A) % 2 == 0`;
- `0 <= A[i] <= `.

### 题解

双指针，一个遍历偶数位置，一个遍历奇数位置，找到每一个需要交换的数对。

```python
class Solution:
    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        return sort_array_by_parity_ii(A)

def sort_array_by_parity_ii(l: List[int]) -> List[int]:
    i = 0
    j = 1
    while i < len(l) and j < len(l):
        while i < len(l) and l[i] % 2 == 0:
            i += 2
        while j < len(l) and l[j] % 2 == 1:
            j += 2
        if i < len(l) and j < len(l):
            tmp = l[i]
            l[i] = l[j]
            l[j] = tmp
    return l
```

## 1344. 时钟指针的夹角{#leetcode-1344}

[:link: 来源](https://leetcode-cn.com/problems/angle-between-hands-of-a-clock/)

### 题目

给你两个数 `hour` 和 `minutes`. 请你返回在时钟上，由给定时间的时针和分针组成的较小角的角度（60 单位制）。

#### 示例

```raw
输入：hour = 12, minutes = 30
输出：165
```

```raw
输入：hour = 3, minutes = 30
输出；75
```

```raw
输入：hour = 3, minutes = 15
输出：7.5
```

```raw
输入：hour = 4, minutes = 50
输出：155
```

```raw
输入：hour = 12, minutes = 0
输出：0
```

#### 提示

- `1 <= hour <= 12`;
- `0 <= minutes <= 59`;
- 与标准答案误差在 ${10}^{-5}$ 以内的结果都被视为正确结果。

### 题解

数学题。

```python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        return angle_clock(hour, minutes)

def angle_clock(hour: int, minutes: int) -> float:
    angle_hour = (hour + minutes / 60) / 12
    angle_minutes = minutes / 60
    delta_angle = abs(angle_hour - angle_minutes)
    if delta_angle > .5:
        delta_angle = 1 - delta_angle
    return delta_angle * 360
```

## 697. 数组的度{#leetcode-697}

[:link: 来源](https://leetcode-cn.com/problems/degree-of-an-array/)

### 题目

给定一个非空且只包含非负数的整数数组 `nums`, 数组的度的定义是指数组里任一元素出现频数的最大值。

你的任务是找到与 `nums` 拥有相同大小的度的最短连续子数组，返回其长度。

#### 示例

```raw
输入：[1, 2, 2, 3, 1]
输出：2
解释：输入数组的度是 2, 因为元素 1 和 2 的出现频数最大，均为 2.
连续子数组里面拥有相同度的有如下所示：
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2].
最短连续子数组 [2, 2] 的长度为 2, 所以返回 2.
```

```raw
输入：[1, 2, 2, 3, 1, 4, 2]
输出：6
```

#### 注意

- `1 <= len(nums) <= 50000`;
- `0 <= nums[i] <= 49999`.

### 题解

字典计数，记录出现区间。频次最大的数的出现区间中长度最短的即为所求。

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        return find_shortest_sub_array(nums)

def find_shortest_sub_array(nums: List[int]) -> int:
    counter = dict()
    starts = dict()
    stops = dict()

    for i, n in enumerate(nums):
        count = counter.get(n, 0)
        counter[n] = count + 1
        if count == 0:
            starts[n] = i
        stops[n] = i

    max_count = max(counter.values())
    min_slice = min((
        stops[n] - starts[n] + 1
        for n, count in counter.items()
        if count == max_count
    ), default=0)

    return min_slice
```

## 454. 四数相加 II{#leetcode-454}

[:link: 来源](https://leetcode-cn.com/problems/4sum-ii/)

### 题目

给定四个包含整数的数组列表 `A`, `B`, `C`, `D`, 计算有多少个元组 `(i, j, k, l)`, 使得 `A[i] + B[j] + C[k] + D[l] = 0`.

为了使问题简单化，所有的 `A`, `B`, `C`, `D` 具有相同的长度 $N$, 且 $0 \le N \le 500$. 所有整数的范围在 $-2^{28}$ 到 $2^{28}-1$ 之间，最终结果不会超过 $2^{31}-1$.

#### 示例

```raw
输入：
A = [ 1,  2]
B = [-2, -1]
C = [-1,  2]
D = [ 0,  2]

输出：
2

解释：
两个元组如下：
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
```

### 题解

二分计数查找。

```python
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        return sum_count(A, B, C, D)

from itertools import product
from collections import Counter

def sum_count(A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
    AB = Counter(a + b for a, b in product(A, B))
    return sum(AB[-(c + d)] for c, d in product(C, D))
```

## 57. 插入区间{#leetcode-57}

[:link: 来源](https://leetcode-cn.com/problems/insert-interval/)

### 题目

给出一个无重叠的，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

#### 示例

```raw
输入：intervals = [[1, 3], [6, 9]], newInterval = [2, 5]
输出：[[1, 5], [6, 9]]
```

```raw
输入：intervals = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], newInterval = [4, 8]
输出：[[1, 2], [3, 10], [12, 16]]
解释：这是因为新的区间 [4, 8] 与 [3, 5], [6, 7], [8, 10] 重叠。
```

### 题解

- 处理边界条件；
- 查找重叠区间窗口 `[i, j)`, 并处理窗口边界的区间合并。

#### 线性查找

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        return insert(intervals, newInterval)

def insert(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    if not intervals:
        return [new_interval]

    if intervals[0][0] > new_interval[1]:
        return [new_interval, *intervals]

    if intervals[-1][1] < new_interval[0]:
        return [*intervals, new_interval]

    i = 0
    while intervals[i][1] < new_interval[0]:
        i += 1

    j = i
    while j < len(intervals) and intervals[j][0] <= new_interval[1]:
        j += 1

    new_interval = [min(intervals[i][0], new_interval[0]), max(intervals[j - 1][1], new_interval[1])]
    return [*intervals[:i], new_interval, *intervals[j:]]
```

#### 二分查找

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        return insert(intervals, newInterval)

def insert(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    if not intervals:
        return [new_interval]

    if intervals[0][0] > new_interval[1]:
        return [new_interval, *intervals]

    if intervals[-1][1] < new_interval[0]:
        return [*intervals, new_interval]

    low, high = 0, len(intervals)
    while low < high:
        mid = (low + high) // 2
        if intervals[mid][1] < new_interval[0]:
            low = mid + 1
        else:
            high = mid
    i = low

    low, high= i, len(intervals)
    while low < high:
        mid = (low + high) // 2
        if intervals[mid][0] <= new_interval[1]:
            low = mid + 1
        else:
            high = mid
    j = low

    new_interval = [min(intervals[i][0], new_interval[0]), max(intervals[j - 1][1], new_interval[1])]
    return [*intervals[:i], new_interval, *intervals[j:]]
```

#### 二分查找（原地）

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        return insert(intervals, newInterval)

def insert(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    if not intervals:
        intervals.append(new_interval)

    elif intervals[0][0] > new_interval[1]:
        intervals.insert(0, new_interval)

    elif intervals[-1][1] < new_interval[0]:
        intervals.append(new_interval)

    else:
        low, high = 0, len(intervals)
        while low < high:
            mid = (low + high) // 2
            if intervals[mid][1] < new_interval[0]:
                low = mid + 1
            else:
                high = mid
        i = low

        low, high = i, len(intervals)
        while low < high:
            mid = (low + high) // 2
            if intervals[mid][0] <= new_interval[1]:
                low = mid + 1
            else:
                high = mid
        j = low

        new_interval[0] = min(intervals[i][0], new_interval[0])
        new_interval[1] = max(intervals[j - 1][1], new_interval[1])
        intervals[i: j] = [new_interval]

    return intervals
```

## 493. 翻转对{#leetcode-493}

[:link: 来源](https://leetcode-cn.com/problems/reverse-pairs/)

### 题目

给定一个数组 `nums`, 如果 `i < j` 且 `nums[i] > 2 * nums[j]` 我们就将 `(i, j)` 称作一个重要翻转对。

你需要返回给定数组中的重要翻转对的数量。

#### 示例

```raw
输入：[1, 3, 2, 3, 1]
输出：2
```

```raw
输入：[2, 4, 3, 5, 1]
输出：3
```

### 题解

#### 二分查找

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        return reverse_pairs(nums)

import bisect

def reverse_pairs(nums: List[int]) -> int:
    r, nums2 = 0, []
    for n in reversed(nums):
        r += bisect.bisect_left(nums2, n)
        bisect.insort_left(nums2, n * 2)
    return r
```

#### 树状数组

- 离散化；
- 树状数组。

```python
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        return reverse_pairs(nums)

from bisect import bisect_left

def reverse_pairs(nums: List[int]) -> int:
    # discretize
    nums12 = nums + [n * 2 for n in nums]
    nums12.sort()
    mapping = [(
        bisect_left(sorted_nums, n),
        bisect_left(sorted_nums, n * 2)
    ) for n in nums]

    r, bit= 0, [0] * (len(nums) * 2)
    for n1, n2 in reversed(mapping):
        # query
        while n1 > 0:
            r += bit[n1 - 1]
            n1 -= n1 & (-n1)

        # update
        n2 += 1
        while 0 < n2 <= len(bit):
            bit[n2 - 1] += 1
            n2 += n2 & (-n2)

    return r
```

## 1626. 无矛盾的最佳球队{#leetcode-1626}

[:link: 来源](https://leetcode-cn.com/problems/best-team-with-no-conflicts/)

### 题目

假设你是球队的经理。对于即将到来的锦标赛，你想组合一支总体得分最高的球队。球队的得分是球队中所有球员的分数**总和**。

然而，球队中的矛盾会限制球员的发挥，所以必须选出一支**没有矛盾**的球队。如果一名年龄较小球员的分数**严格大于**一名年龄较大的球员，则存在矛盾。同龄球员之间不会发生矛盾。

给你两个列表 `scores` 和 `ages`, 其中每组 `scores[i]` 和 `ages[i]` 表示第 `i` 名球员的分数和年龄。请你返回**所有可能的无矛盾球队中得分最高那支的分数**。

#### 示例

```raw
输入：scores = [1, 3, 5, 10, 15], ages = [1, 2, 3, 4, 5]
输出：34
解释：你可以选中所有球员。
```

```raw
输入：scores = [4, 5, 6, 5], ages = [2, 1, 2, 1]
输出：16
解释：最佳的选择是后 3 名球员。注意，你可以选中多个同龄球员。
```

```raw
输入：scores = [1, 2, 3, 5], ages = [8, 9, 10, 1]
输出：6
解释：最佳的选择是前 3 名球员。
```

#### 提示

- `1 <= len(scores) == len(ages) <= 1e3`;
- `1 <= scores[i] <= 1e6`, `1 <= ages[i] <= 1e3`.

### 题解

排序，动态规划。`m[j]` 表示前 `j` 个球员进行组合且第 `j` 个球员出场的最高得分。

```python
class Solution:
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        return best_team_score(scores, ages)

def best_team_score(scores: List[int], ages: List[int]) -> int:
    players, m = sorted(zip(ages, scores)), []
    for i, (ai, si) in enumerate(players):
        m.append(max(si, max((
            m[j] + si
            for j, (aj, sj) in enumerate(players[:i])
            if not (ai > aj and si < sj)
        ), default=0)))
    return max(m, default=0)
```

## 976. 三角形的最大周长{#leetcode-976}

[:link: 来源](https://leetcode-cn.com/problems/largest-perimeter-triangle/)

### 题目

给定由一些正数（代表长度）组成的数组 `A`, 返回由其中三个长度组成的、**面积不为零**的三角形的最大周长。

如果不能形成任何面积不为零的三角形，返回 `0`.

#### 示例

```raw
输入：[2, 1, 2]
输出：5
```

```raw
输入：[1, 2, 1]
输出：0
```

```raw
输入：[3, 2, 3, 4]
输出：10
```

```raw
输入：[3, 6, 2, 3]
输出：8
```

#### 提示

- `3 <= len(A) <= 1e4`;
- `1 <= A[i] <= 1e6`.

### 题解

排序，贪心。对于每条边 `a`, 如果以该边为最长边，则最有可能形成合法三角形且周长最大的是 `b`, `c` 选取策略是一致的，即选取次长的两条边。

```python
class Solution:
    def largestPerimeter(self, A: List[int]) -> int:
        return largest_perimeter(A)

def largest_perimeter(sides: List[int]) -> int:
    sides.sort(reverse=True)
    return max((
        a + b + c
        for a, b, c in zip(sides, sides[1:], sides[2:])
        if b + c > a
    ), default=0)
```

## 100. 相同的树{#leetcode-100}

[:link: 来源](https://leetcode-cn.com/problems/same-tree/)

### 题目

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

#### 示例

```raw
输入：
  1     1
 / \   / \
2   3 2   3
输出：true
```

```raw
输入：
  1     1
 /       \
2         2
输出：false
```

```raw
输入：
  1     1
 / \   / \
2   1 1   2
输出：false
```

### 题解

#### 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        return is_same_tree(p, q)

def is_same_tree(p: TreeNode, q: TreeNode) -> bool:
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
```

#### 迭代

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        return is_same_tree(p, q)

def is_same_tree(t1: TreeNode, t2: TreeNode) -> bool:
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
        s2 += [n2.left, n2.right]
    return True
```

## 767. 重构字符串{#leetcode-767}

[:link: 来源](https://leetcode-cn.com/problems/reorganize-string/)

### 题目

给定一个字符串 `S`, 检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

#### 示例

```raw
输入：S = "aab"
输出："aba"
```

```raw
输入：S = "aaab"
输出：""
```

#### 注意

`S` 只包含小写字母并且长度在 $\left[1,500\right]$ 区间内。

### 题解

- 不可行的充要条件是最大出现次数大于长度的一半向上取整。
- 贪心。先填充奇数位置，再填充偶数位置。

```python
class Solution:
    def reorganizeString(self, S: str) -> str:
        return reorganize_string(S)

from collections import Counter

def reorganize_string(s: str) -> str:
    counter = Counter(s)
    if max(counter.values(), default=0) > (len(s) + 1) // 2:
        return ''
    result = [None] * len(s)
    even, odd, half = 0, 1, len(s) // 2
    for element, count in counter.items():
        if count <= half:
            while count > 0 and odd < len(result):
                result[odd] = element
                count -= 1
                odd += 2
        while count > 0:
            result[even] = element
            count -= 1
            even += 2
    return ''.join(result)
```

## 1566. 重复至少 K 次且长度为 M 的模式{#leetcode-1566}

[:link: 来源](https://leetcode-cn.com/problems/detect-pattern-of-length-m-repeated-k-or-more-times/)

### 题目

给你一个正整数数组 `arr`, 请你找出一个长度为 `m` 且在数组中至少重复 `k` 次的模式。

**模式**是由一个或多个值组成的子数组（连续的子序列），**连续**重复多次但**不重叠**。模式由其长度和重复次数定义。

如果数组中存在至少重复 `k` 次且长度为 `m` 的模式，则返回 `true`, 否则返回 `false`.

#### 示例

```raw
输入：arr = [1, 2, 4, 4, 4, 4], m = 1, k = 3
输出：true
解释：模式 (4) 的长度为 1, 且连续重复 4 次。注意，模式可以重复 k 次或更多次，但不能少于 k 次。
```

```raw
输入：arr = [1, 2, 1, 2, 1, 1, 1, 3], m = 2, k = 2
输出：true
解释：模式 (1, 2) 长度为 2, 且连续重复 2 次。另一个符合题意的模式是 (2, 1), 同样重复 2 次。
```

```raw
输入：arr = [1, 2, 1, 2, 1, 3], m = 2, k = 3
输出：false
解释：模式 (1, 2) 长度为 2, 但是只连续重复 2 次。不存在长度为 2 且至少重复 3 次的模式。
```

```raw
输入：arr = [1, 2, 3, 1, 2], m = 2, k = 2
输出：false
解释：模式 (1, 2) 出现 2 次但并不连续，所以不能算作连续重复 2 次。
```

```raw
输入：arr = [2, 2, 2, 2], m = 2, k = 3
输出：false
解释：长度为 2 的模式只有 (2, 2), 但是只连续重复 2 次。注意，不能计算重叠的重复次数。
```

#### 提示

- `2 <= len(arr) <= 100`;
- `1 <= arr[i] <= 100`;
- `1 <= m <= 100`;
- `2 <= k <= 100`.

### 题解

#### 暴力

```python
class Solution:
    def containsPattern(self, arr: List[int], m: int, k: int) -> bool:
        return contains_pattern(arr, m, k)

def contains_pattern(arr: List[int], m: int, k: int) -> bool:
    return any(arr[i:i + m] * k == arr[i:i + m * k] for i in range(len(arr) - m * k + 1))
```

#### 优化

```python
class Solution:
    def containsPattern(self, arr: List[int], m: int, k: int) -> bool:
        return contains_pattern(arr, m, k)

def contains_pattern(arr: List[int], m: int, k: int) -> bool:
    count, total = 0, m * (k - 1)
    for e1, e2 in zip(arr, arr[m:]):
        if e1 == e2:
            count += 1
            if count == total:
                return True
        else:
            count = 0
    return False
```

## 17. 电话号码的字母组合{#leetcode-17}

[:link: 来源](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

### 题目

给定一个仅包含数字 `2` 到 `9` 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 `1` 不对应任何字母。

{% asset_img telephone-keypad.png 200 181 "'手机键盘' '手机键盘'" %}

#### 示例

```raw
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

#### 说明

尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。

### 题解

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        return letter_combinations(digits)

from itertools import product

MAPPING = {
    '2': 'abc', '3': 'def',
    '4': 'ghi', '5': 'jkl', '6': 'mno',
    '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
}

def letter_combinations(digits: str) -> List[str]:
    if digits == '':
        return []

    mappings = (MAPPING[digit] for digit in digits)
    return [''.join(c) for c in product(*mappings)]
```
