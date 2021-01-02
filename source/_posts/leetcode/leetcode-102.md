---
title: LeetCode 102. 二叉树的层序遍历
date: 2020-12-11 17:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

## 题目

给你一个二叉树，请你返回其按**层序遍历**得到的节点值。（即逐层地，从左到右访问所有节点）。

### 示例

```raw
输入：
  3
 / \
9  20
  /  \
 15   7
输出：[[3], [9, 20], [15, 7]]
```

<!-- more -->

## 题解

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
