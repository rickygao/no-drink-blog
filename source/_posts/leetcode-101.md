---
title: LeetCode 101. 对称二叉树
date: 2020-12-03 23:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/symmetric-tree/)

## 题目

给定一个二叉树，检查它是否是镜像对称的。

### 示例

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

### 进阶

你可以运用递归和迭代两种方法解决这个问题吗？

<!-- more -->

## 题解

### 递归

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

### 迭代

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
