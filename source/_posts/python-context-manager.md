---
title: Python 的上下文管理器
date: 2020-12-05 13:30:00
tags: [Python, Context Manager]
---

写过 Python 的朋友对如下代码一定不陌生：

```python
with open('somefile', 'w') as f:
    f.write('foobar')
```

打开文件 somefile 作为 `f`, 并向 `f` 写入字符串 foobar, 并且在结束时帮我们自动关闭了文件 `f`. 这几乎等同于

```python
f = open('somefile', 'w')
try:
    f.write('foobar')
finally:
    f.close()
```

是什么魔法帮我们调用了 `f.close` 呢？是 [with](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with) 语句使用了上下文管理器，即 [Context Manager](https://docs.python.org/zh-cn/3/library/stdtypes.html#typecontextmanager). `f` 不仅是一个文件描述符，更是一个上下文管理器，而作为一个上下文管理器，`f` 定义了进入和退出上下文时的行为。

<!-- more -->

## 使用魔术方法实现上下文管理器

Python 中有一类方法的方法名会由双下划线 `__` 包裹起来，比如 `__init__`, `_len_` 等，它们叫做魔术方法。而使用魔术方法来实现上下文管理器重点在于实现 `__enter__` 和 `__exit__` 方法。顾名思义，它们分别于进入和退出上下文时被调用。除此之外，`Python 3.5` 还引入了 `__aenter__` 和 `__aexit__`, 用于实现异步上下文管理器，它们被 `async with` 使用，这里不加考虑。如下我们可以实现一个自己的上下文管理器：

```python
class MyContextManager:
    def __enter__(self):
        print('Enter my context.')
        return 'a message from __enter__'

    def __exit__(self, except_type, except_value, traceback):
        print('Exit my context.')
```

`__enter__` 的返回值可以被调用者使用，`__exit__` 如果返回 `True` 则会压制上下文中抛出的错误。于是，我们可以这样来使用它

```python
with MyContextManager() as message:
    print(f'Wow, my context manager sends "{message}" to me.')
```

我们会看到输出

```raw
Enter my context.
Wow, my context manager sends "a message from __enter__" to me.
Exit my context.
```

同时，`contextlib` 为我们提供了[抽象类 `AbstractContextManager`](https://docs.python.org/zh-cn/3/library/contextlib.html#contextlib.AbstractContextManager), 我们可以选择继承它，即

```python
from contextlib import AbstractContextManager

class MyContextManager(AbstractContextManager):
    def __enter__(self):
        print('Enter my context.')
        return 'a message from __enter__'

    def __exit__(self, except_type, except_value, traceback):
        print('Exit my context.')
```

它会在子类被实例化时检查是否实现了实现魔术方法 `__enter__` 和 `__exit__`, 如没有将抛出错误。

## 使用装饰器实现上下文管理器

Python 的装饰器是强大的。`contextlib` 为我们提供了一种方式，可以利用[装饰器 `contextmanager`](https://docs.python.org/zh-cn/3/library/contextlib.html#contextlib.contextmanager) 和函数（而不是一个完整的类的定义）来实现一个上下文管理器。实际上，一个简单的上下文管理器就应该以简单的方式来实现。例如，要实现与上文功能相同的上下文管理器，只需要

```python
from contextlib import contextmanager

@contextmanager
def my_context_manager():
    print('Enter my context.')
    yield 'a message yeilded from my_context'
    print('Exit my context.')

with my_context_manager() as message:
    print(f'Wow, my context manager sends "{message}" to me.')
```

同样地会得到

```raw
Enter my context.
Wow, my context manager sends "a message yeilded from my_context" to me.
Exit my context.
```

除此之外，自 `Python 3.7` 起，`asynccontextmanager` 也被提供，作为 `contextmanager` 的异步版本。

## 常见的上下文管理器

有了以上的背景知识，本文开始时提出的问题迎刃而解。我们可以做如下实验：

```python
f = open('somefile', 'w')
print(f.__enter__)
print(f.__exit__)
f.write('foobar')
f.close()
```

会得到如下输出：

```raw
<built-in method __enter__ of _io.TextIOWrapper object at 0x7ffb19b84d40>
<built-in method __exit__ of _io.TextIOWrapper object at 0x7ffb19b84d40>
```

原来 `open` 返回的对象是 `_io.TextIOWrapper` 类型，并实现了 `__enter__` 和 `__exit__` 魔术方法。那么，除此之外还有哪些常见的上下文管理器呢？

### `contextlib.closing`

顾名思义，`contextlib.closing` 接收一个可关闭（实现了 `close` 方法）的对象 `thing` 作为参数，在进入上下文时会将 `thing` 直接返回，在退出上下文时会帮我们自动地关闭该可关闭的对象 `thing`. 它几乎相当于

```python
from contextlib import contextmanager

@contextmanager
def closing(thing):
    yield thing
    thing.close()
```

这为一个非上下文管理器的可关闭的对象提供了一个转变为上下文管理器的包装，减轻了使用此类对象时的心智负担。

### `contextlib.nullcontext`

`nullcontext` 是上下文管理器的空对象模式，当我们希望统一各个分支的代码逻辑，而又不希望该上下文管理器发挥实质作用时即可派上用场。它接收一个参数，并在进入上下文时直接返回它。官方文档给出以下例子：

```python
from contextlib import nullcontext

def process_file(file_or_path):
    if isinstance(file_or_path, str):
        # If string, open file
        cm = open(file_or_path)
    else:
        # Caller is responsible for closing file
        cm = nullcontext(file_or_path)

    with cm as file:
        # Perform processing on the file
        pass
```

### `contextlib.suppress`

顾名思义，该上下文管理器将抑制上下文中抛出的错误。如果给定了一系列错误类型作为参数，则只抑制指定类型的错误。用例如下所示：

```python
import os
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove('somefile')
```

### `contextlib.{redirect_stdin, redirect_stdout, redirect_stderr}`

这三个上下文管理器将重定向 `{sys.stdin, sys.stdout, sys.stderr}` 到指定文件，并在退出上下文时恢复。这可以帮助我们非侵入式地截获标准输入、标准输出和标准错误。

```python
from io import StringIO
from contextlib import redirect_stdout

f = StringIO()
with redirect_stdout(f):
    print('I am printed to standard out stream, am I?')
print(f"No, you don't. I got yours.")
print(f'You said "{f.getvalue().strip()}"')
```

将会输出

```raw
No, you don't. I got yours.
You said "I am printed to standard out stream, am I?"
```

### `threading.{Lock, RLock}`

锁或可重入锁的实例也是上下文管理器，于是你可以

```python
from threading import Lock

lock = Lock()

with lock:
    print('something critical')
```

那么在进入和退出上下文时，则会自动加锁和解锁。

### `torch.{no_grad, enable_grad}`

[`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html) 和 [`torch.enable_grad`](https://pytorch.org/docs/stable/generated/torch.enable_grad.html) 为 `PyTorch` 中自动求导提供开关控制。它们既可以作为上下文管理器来使用，也可以作为装饰器使用。

## 与上下文管理器有关的工具

### `contextlib.ContextDecorator`

该类是一个 `mixin`, 我们可以为一个上下文管理器混入它，来获得将原上下文管理器作为装饰器使用的能力。也就是说它基本上只是一个语法糖，一旦 `cm` 混入该类，则允许你将

```python
def f():
    with cm():
        pass
```

改写为

```python
@cm()
def f():
    pass
```

这就如同 `torch.no_grad` 和 `torch.enable_grad` 一样。
