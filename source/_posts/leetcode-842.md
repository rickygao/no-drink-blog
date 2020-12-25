---
title: LeetCode 842. 将数组拆分成斐波那契序列
date: 2020-12-08 13:30:00
tags: [LeetCode]
---

[:link: 来源](https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/)

## 题目

给定一个数字字符串 `S`, 比如 `S = "123456579"`, 我们可以将它分成斐波那契式的序列 `[123, 456, 579]`.

形式上，斐波那契式序列是一个非负整数列表 `F` 且满足：

- `0 <= F[i] <= 2 ** 31 - 1`（也就是说，每个整数都符合 32 位有符号整数类型）；
- `len(F) >= 3`;
- 对于所有的 `0 <= i < len(F) - 2`, 都有 `F[i] + F[i + 1] = F[i + 2]` 成立。

另外，请注意，将字符串拆分成小块时，每个块的数字一定不要以零开头，除非这个块是数字 `0` 本身。

返回从 `S` 拆分出来的任意一组斐波那契式的序列块，如果不能拆分则返回 `[]`.

### 示例

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

### 提示

- `1 <= len(S) <= 200`;
- 字符串 `S` 中只含有数字。

<!-- more -->

## 题解

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
