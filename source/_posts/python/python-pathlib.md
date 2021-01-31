---
title: Python 的路径操作 pathlib
date: 2020-12-19 01:30:00
tags: [Python]
---

不会吧？不会吧？不会有人还在用 `os.path` 等模块来操作文件系统路径吧？难写又难读，看着就头痛。

今天我们将介绍 [`pathlib` 模块](https://docs.python.org/zh-cn/3/library/pathlib.html)。它同 [`os.path` 模块](https://docs.python.org/zh-cn/3/library/zh-cn/os.path.html)一样也是 Python 标准库的一员。它提供了面向对象的路径操作，简洁易用、语义明确，让你真正体验到「人生苦短，我用 Python」的路径操作。

## 模块概览

`pathlib` 模块中最重要的内容就是 `Path` 类，我们可以简单地使用一个路径字符串构造它，例如 `Path('~/Downloads')`。同时它提供了许多易于理解和使用的方法。这里将不逐一说明，使用时可在官方文档检索方法的详细说明。以下提供了它们与命令式风格的模块提供的函数的简单对照，但功能并不完全一致，可供参考。

<!-- more -->

### 构造操作

以下操作可以用于构造 `Path` 实例。

| legacy                    | pathlib                                                |
| ------------------------- | ------------------------------------------------------ |
| os.path.join()            | PurePath.joinpath(), PurePath.\_\_truediv\_\_()        |
| os.getcwd()               | Path.cwd()                                             |
| os.path.expanduser()      | Path.expanduser(), Path.home()                         |
| os.path.abspath()         | Path.resolve()                                         |
| os.readlink()             | Path.readlink()                                        |
| os.listdir()              | Path.iterdir()                                         |
| glob.glob(), glob.iglob() | Path.glob(), Path.rglob()                              |
| N/A                       | Path.with_name(), Path.with_stem(), Path.with_suffix() |

### 查询操作

以下操作可以用于查询 `Path` 实例对应的文件或目录的相关属性。

| legacy             | pathlib                                               |
| ------------------ | ----------------------------------------------------- |
| os.path.basename() | PurePath.name, PurePath.stem                          |
| os.path.dirname()  | PurePath.parent, PurePath.parents                     |
| os.path.splitext() | PurePath.suffix, PurePath.suffixes                    |
| os.path.isabs()    | PurePath.is_absolute()                                |
| os.path.isdir()    | Path.is_dir()                                         |
| os.path.isfile()   | Path.is_file()                                        |
| os.path.islink()   | Path.is_symlink()                                     |
| os.path.samefile() | Path.samefile()                                       |
| os.path.exists()   | Path.exists()                                         |
| os.stat()          | Path.stat(), Path.lstat(), Path.owner(), Path.group() |

### 更改操作

以下操作可以用于更改（包括创建、修改和删除）`Path` 实例对应的文件或目录。

| legacy                    | pathlib                     |
| ------------------------- | --------------------------- |
| os.mkdir(), os.makedirs() | Path.mkdir()                |
| os.link()                 | Path.link_to()              |
| os.symlink()              | Path.symlink_to()           |
| os.rename()               | Path.rename()               |
| os.replace()              | Path.replace()              |
| os.chmod()                | Path.chmod(), Path.lchmod() |
| os.rmdir()                | Path.rmdir()                |
| os.remove(), os.unlink()  | Path.unlink()               |
| N/A                       | Path.touch()                |

### 输入输出操作

以下操作可以用于对（包括创建、修改和删除）`Path` 实例对应的文件进行输入输出。

| legacy         | pathlib                               |
| -------------- | ------------------------------------- |
| open()         | Path.open()                           |
| open().read()  | Path.read_bytes(), Path.read_text()   |
| open().write() | Path.write_bytes(), Path.write_text() |

## 实战演练

在这一部分，我们将在现实使用场景中使用 `pathlib` 做一些有实际意义的事情，以展示其强大的功能。

### 收集图片

```python collect_images.py
from itertools import chain
from pathlib import Path

def collect_files_to_lines(source_folder, target_file, formats):
    source_folder = Path(source_folder)
    file_paths = (source_folder.rglob(f'*.{f}') for f in formats)
    file_paths = sorted(chain.from_iterable(file_paths))
    Path(target_file).write_text('\n'.join(map(str, file_paths)))
    return len(file_paths)

if __name__ == '__main__':
    num_images = collect_files_to_lines(
        'images', 'images.txt', ['png', 'jpg', 'gif', 'bmp'])
    print(f'{num_images = }')
```

`collect_images.py` 脚本将收集 `images` 目录下的所有以 `png`、`jpg`、`gif` 或 `bmp` 为后缀的图片文件，按字典序排序后写入 `images.txt` 文件中。使用了 `rglob` 和 `write_text` 方法。

### 拷贝文件

```python copy_files.py
from itertools import chain
from pathlib import Path
import shutil

def copy_files_to_folder(source_folder, target_folder, formats):
    source_folder, target_folder = Path(source_folder), Path(target_folder)
    file_paths = (source_folder.rglob(f'*.{f}') for f in formats)
    file_paths = sorted(chain.from_iterable(file_paths))
    for source_path in file_paths:
        relative_path = source_path.relative_to(source_folder)
        target_path = target_folder.joinpath(relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, target_path)
    return len(file_paths)

if __name__ == '__main__':
    num_copied = copy_files_to_folder('source', 'target', ['txt'])
    print(f'{num_copied = }')
```

`copy_files.py` 脚本将收集 `source` 目录下的所有以 `txt` 为后缀的文件，在保持原来的目录结构的情况下拷贝到 `target` 目录下。

脚本中使用了 `relative_to` 方法来获取相对路径以及 `mkdir` 方法的 `parents` 和 `exist_ok` 参数来方便地创建目录结构。另外，`target_folder.joinpath(relative_path)` 也可以等价地写成 `target_folder / relative_path`。

随着 Python 版本的迭代，很多如 `shutil.copy` 这样接受路径字符串的函数，甚至包括一些第三方包（如 NumPy 和 Pillow）中的某些函数（如 `numpy.save`、`numpy.load` 和 `PIL.Image.open`），现在也可以接受路径对象了。
