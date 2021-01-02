---
title: LeetCode 100. 相同的树
date: 2020-11-29 19:00:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/same-tree/)

## 题目

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

### 示例

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

<!-- more -->

## 题解

### 递归

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

### 迭代

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
