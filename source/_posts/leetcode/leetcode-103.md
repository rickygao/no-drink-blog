---
title: LeetCode 103. 二叉树的锯齿形层次遍历
date: 2020-12-11 17:15:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

## 题目

给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

### 示例

```raw
输入： 
  3
 / \
9  20
  /  \
 15   7
输出：[[3], [20, 9], [15, 7]]
```

<!-- more -->

## 题解

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
