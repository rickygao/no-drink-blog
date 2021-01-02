---
title: LeetCode 222. 完全二叉树的节点个数
date: 2020-11-24 22:00:00
tags: [LeetCode]
mathjax: true
---

[:link: 来源](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

## 题目

给出一个完全二叉树，求出该树的节点个数。

### 说明

完全二叉树的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 $h$ 层，则该层包含 $[1, 2^h]$ 个节点。

### 示例

```raw
输入：

    1
   / \
  2   3
 / \  /
4  5 6

输出：6
```

<!-- more -->

## 题解

- 利用完全二叉树的性质，左子树和右子树至少存在一棵满二叉树，可以由高度直接计算节点个数，而另外一棵则是完全二叉树，可以通过递归计算；
- 复用高度计算的结果。

### 递归

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

### 迭代

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
