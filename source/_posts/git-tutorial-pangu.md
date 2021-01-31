---
title: Git 教程之盘古开天地
date: 2020-11-30 21:30:00
tags: [Git, Tutorial, CLI]
---

## 准备

1. 一个 `shell`；
2. 还有 `git`；
3. [GitHub](https://github.com) Account；
4. [Git Book](https://git-scm.com/book/zh/v2/)；
5. 最重要的是随时能够动的手。

盘古会在 `shell` 中使用 `git --version` 命令来检视 `git` 的版本号。遇到不理解或者感兴趣的问题还会主动去查阅文档和书籍。

<!-- more -->

## 基础

进入项目目录。

```shell
mkdir git-tutorial && cd git-tutorial
```

初始化一个 `git` 仓库。

```shell
$ git init
Initialized empty Git repository in /Users/rickygao/Projects/git-tutorial/.git/

$ ls -la
total 0
drwxr-xr-x   3 rickygao  staff   96 11 29 12:56 .
drwxr-xr-x  17 rickygao  staff  544 11 29 12:56 ..
drwxr-xr-x   9 rickygao  staff  288 11 29 12:56 .git
```

`git` 将维护 `.git` 达成各种魔法​​。相信大家已经[很了解](https://www.runoob.com/git/git-workspace-index-repo.html)什么是**工作区**、**暂存区**和**版本库**了。

盘古决定种一棵树！

```shell
$ touch tree
$ ls -l
total 0
-rw-r--r--  1 rickygao  staff  0 11 29 13:09 tree
```

这里已经有一棵树了，盘古对于创世很谨慎，它还没有被提交到了版本库中，因为：

```shell
$ git status
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
  tree

nothing added to commit but untracked files present (use "git add" to track)
```

这告诉我们，盘古还没有任何提交，同时有一个未被追踪的文件。`git` 很贴心地提示了使用 `git add` 去追踪这个文件。盘古决定试一下：

```shell
$ git add
Nothing specified, nothing added.
Maybe you wanted to say 'git add .'?
```

`git` 居然是个大骗子，可恶啊。（其实是没看到需要加 `<file>` 参数）还好盘古会英语，发现是没有指定文件，于是他决定把当前目录所有的文件添加进去。

```shell
$ git add .
$ git status
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
  new file:   tree
```

盘古对这棵树很满意，决定提交到版本库。

```shell
$ git commit -m 'plant a tree'
[master (root-commit) a88008d] plant a tree
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 tree
```

大意了啊，`0 insertions(+)` 忘记了添加内容了。

```shell
$ echo 'Here is a tree, a beautiful tree.' > tree
$ cat tree
Here is a tree, a beautiful tree.

$ git status
On branch master
Changes not staged for commit:
 (use "git add <file>..." to update what will be committed)
 (use "git restore <file>..." to discard changes in working directory)
  modified:   tree

no changes added to commit (use "git add" and/or "git commit -a")
```

`git` 发现了 `tree` 被修改了，这真是太强了，这下盘古修改了什么就一览无余了！同时，`git` 还告诉盘古可以试试：

```shell
$ git commit -a -m 'tree is beautiful'
[master f476c1c] tree is beautiful
 1 file changed, 1 insertion(+)
```

干了这么多活儿，盘古决定回顾一下之前都做了什么：

```shell
$ git log
commit f476c1c57eeb29c1e0eff31d737c97d0dd903dc6 (HEAD -> master)
Author: gaoruijun <rckgao@gmail.com>
Date:   Sun Nov 29 13:26:06 2020 +0800

    tree is beautiful

commit a88008d64e110bad1ed8dd4442e0e20cb08bb0ba
Author: gaoruijun <rckgao@gmail.com>
Date:   Sun Nov 29 13:19:34 2020 +0800

    plant a tree
(END)
```

为了更好地完成开天地的工作，盘古决定去了解[重制](https://git-scm.com/book/zh/v2/Git-工具-重置揭密)、[分支](https://git-scm.com/book/zh/v2/Git-分支-分支简介)等魔法。

## 交流

盘古登录了[盘古交流平台](https://github.com)，阅读了[创建新仓库](https://docs.github.com/cn/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/creating-a-new-repository)并创建了一个远程仓库。

热爱学习的盘古还阅读了[使用命令行添加现有项目到 GitHub](https://docs.github.com/cn/free-pro-team@latest/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line)，决定试一下。

```shell
$ git remote add origin https://github.com/rickygao/git-tutorial.git
$ git branch -M main
$ git push -u origin main
Username for 'https://github.com': rckgao@gmail.com
Password for 'https://rckgao@gmail.com@github.com':
Enumerating objects: 6, done.
Counting objects: 100% (6/6), done.
Delta compression using up to 4 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (6/6), 437 bytes | 218.00 KiB/s, done.
Total 6 (delta 0), reused 0 (delta 0)
To https://github.com/rickygao/git-tutorial.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

芜湖，这样其他盘古也能看到这个仓库了！

盘古很孤独，但是又想体验协作开发，于是他为了模拟别人向远程仓库提交内容，点击了仓库页面上的 `Add a README` 并 `Commit new file`。这样一来本地的版本库就比远程仓库慢了一个提交。他会使用：

```shell
$ git pull
remote: Enumerating objects: 4, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), done.
From https://github.com/rickygao/git-tutorial
   f476c1c..05d0fe3  main       -> origin/main
Updating f476c1c..05d0fe3
Fast-forward
 README.md | 1 +
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
```

并发现本地已经取得了远程仓库的最新版本，也就是 `git log` 可以看到 `README.md` 的提交了。

之后盘古将使用 `git push` 向远程仓库「推」，而不需要指定 `origin main` 参数，这是因为刚刚他使用了 `-u` 选项，已经将本地的 `main` 分支和远程仓库 `origin` 的 `main` 分支对应上了。

盘古现在可以愉快地使用线性版本进行版本控制啦！如果他需要更复杂的版本控制特性，就需要去了解分支了。
