---
title: C语言里的一些小知识(随手记)
date: 2024-10-15 21:47:21
tags:
 - C语言
categories:
 - C语言
description: 记录自己有时候看到学到的一些C语言里的小知识，以便日后查阅参考。
---


# 宏相关

## `#include`与`#define`

宏其实就是粘贴代码。宏定义的好处是可以减少代码重复，提高代码的可读性和可维护性。

比如：

```c
#include <stdio.h>

#define NAMES(X) X(TOM) X(JERRY) X(MIKE)
#define PRINT(X) puts(#X" hello!");

int main()
{
    NAMES(PRINT);
    return 0;
}
```

上面的代码定义了两个宏：`NAMES` 和 `PRINT`。`NAMES` 接受一个参数 `X`，然后将 `X(TOM)`、`X(JERRY)`、`X(MIKE)` 展开为三个 `printf` 语句。`PRINT` 接受一个参数 `X`，然后将 `#X` 展开为字符串 `"X = %d\n"`，并用 `X` 作为参数。

除此之外，`#include`是用来包含头文件的宏，`#include <stdio.h>`的作用就是把头文件`stdio.h`里的内容全部复制粘贴到这个源文件中。

宏定义的展开是在预编译的阶段进行的，可以通过`gcc -E file.c`来查看预编译后的代码。上面代码预编译后的部分结果：

```c
extern char *ctermid (char *__s) __attribute__ ((__nothrow__ , __leaf__))
  __attribute__ ((__access__ (__write_only__, 1)));
# 867 "/usr/include/stdio.h" 3 4
extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));

extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) ;

extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));
# 885 "/usr/include/stdio.h" 3 4
extern int __uflow (FILE *);
extern int __overflow (FILE *, int);
# 902 "/usr/include/stdio.h" 3 4

# 10 "new6.c" 2

# 14 "new6.c"
int main()
{
    puts("TOM"" hello!"); puts("JERRY"" hello!"); puts("MIKE"" hello!");;
    return 0;
}

```

前面那一部分是`stdio.h`中的内容，在`main`主函数中可以看到，相关的宏都已经被展开了。

## `include <header.h>`和`#include "header.h"`

`#include <header.h>`是系统头文件，位于系统目录下，一般是由操作系统提供的。 

`#include "header.h"`是用户头文件，位于当前目录下，一般是由程序员自己编写的。

可以通过`gcc --verbose src.c`来查看头文件搜索路径。以上面的代码为例，`gcc --verbose new6.c`的关于头文件的搜索路径的输出如下：

```c
#include "..." search starts here:
#include <...> search starts here:
 /usr/lib/gcc/x86_64-linux-gnu/11/include
 /usr/local/include
 /usr/include/x86_64-linux-gnu
 /usr/include
End of search list.
```
可以看到，系统头文件搜索路径在 `/usr/include` 目录下，用户头文件搜索路径在当前目录下。

## 下划线

```c
#define FORALL_REGS(_)  _(X) _(Y)
```

上面的宏定义了一个宏 `FORALL_REGS`，它接受一个参数 `_`，然后将 `X` 和 `Y` 作为参数展开。举个例子：`FORALL_REGS(PRINT)`, 展开为 `PRINT(X) PRINT(Y)`。这个其实就是将之前的字母换成了下划线。

## ##运算符

在宏定义中，`##` 是一个预处理器运算符，称为“标记拼接运算符”。它用于将两个标记（tokens）连接成一个单一的标记。

如：

```c
#define DEFIEN(X)       static int X, X##1;
```

当你使用 `DEFIEN(foo)` 这个宏时，预处理器会将其展开为：

```c
static int foo, foo1;
```

这里，`X` 被替换为 `foo`，而 `X##1` 会将 `X` 和 `1` 拼接成 `foo1`。这样可以动态生成变量名，非常有用。

# 输入输出

## 打印字符串

用`printf("%s", str)`打印字符串时,str是这个字符串首个字符的地址,而不是字符串本身,当打印到结束符`\0`停止打印。

```c
int main()
{
    char str1[] = "hello world";
    char str2[] = {'h', 'e', 'l', 'l', 'o','\0', 'w', 'o', 'r', 'l', 'd', '\0'};
    char str3[] = {'h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd', '\0'};
    char str4[] = {'h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd'};
    printf("str1:   %s\n", str1);
    printf("str2:   %s\n", str2);
    printf("str3:   %s\n", str3);
    printf("str4:   %s\n", str4);
    printf("&str1[0]: %s\n", &str1[0]);
    printf("&str2[0]: %s\n", &str2[0]);
    printf("&str3[0]: %s\n", &str3[0]);
    printf("&str4[0]: %s\n", &str4[0]);
    printf("p str1:   %p\n", str1);
    printf("p str2:   %p\n", str2);
    printf("p str3:   %p\n", str3);
    printf("p str4:   %p\n", str4);
    return 0;
}
```

输出：

```
str1:   hello world
str2:   hello
str3:   helloworld
str4:   helloworldhelloworld
&str1[0]: hello world
&str2[0]: hello
&str3[0]: helloworld
&str4[0]: helloworldhelloworld
p str1:   0x7ffdf492f950
p str2:   0x7ffdf492f95c
p str3:   0x7ffdf492f945
p str4:   0x7ffdf492f93b
```

像`char str1[] = "hello world";`这种定义方式，编译器会自动在末尾添加结束符。如果像`str4`那样定义，结尾没有`\0`，那么会一直打印直到`\0`。

我这个虚拟机是小端存储，那么在内存中，这些字符串应该是这样的：

|address|memory| 
| ---   |  --- |
| f93b  |  h   |
| ···   |      |
| f944  | d    |
| f945  | h    |
| ···   |      |  
| f94f  | \0   |

刚好，`str4`一直打印，直到遇见`str3`的`\0`才停止，所以才是上面的结果。
