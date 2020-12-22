---
title: LeetCode 2. 两数相加
date: 2020-11-25 22:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/add-two-numbers/)

## 题目

给出两个**非空**的链表用来表示两个非负的整数。其中，它们各自的位数是按照**逆序**的方式存储的，并且它们的每个节点只能存储**一位**数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 `0` 之外，这两个数都不会以 `0` 开头。

### 示例

```raw
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

<!-- more -->

## 题解

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

def add_numbers(*l: List[ListNode]) -> ListNode:
    l = list(l)
    head = node = ListNode()
    current = 0
    while any(l) or current > 0:
        for i in range(len(l)):
            if l[i]:
                current += l[i].val
                l[i] = l[i].next
        current, r = divmod(current, 10)
        node.next = ListNode(val=r)
        node = node.next
    return head.next
```
