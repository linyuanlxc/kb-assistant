---
title: FreeRTOS-2
tags:
  - FreeRTOS
  - STM32
  - C语言
categories:
  - STM32
description: FreeRTOS任务操作以及一些命名规范
date: 2023-09-02 13:58:53
---


# 任务操作相关的API函数

| API函数             | 描述             |
| ------------------- | ---------------- |
| xTaskCreate()       | 动态方式创建任务 |
| xTaskCreateStatic() | 静态方式创建任务 |
| xTaskDelete()       | 删除任务         |

由xTaskCreate()创建的任务，在创建成功后立马进入就绪态。被删除的任务将从所有的列表删除。动态创建方式是由FreeRTOS进行内存管理，静态则是由人工进行管理。静态创建比较麻烦，因此大多使用动态创建。


若使用静态创建，需要将宏`configSUPPORT_STATIC_ALLOCATION`置为1。同样若为动态创建，则需将宏`configSUPPORT_DYNAMIC_ALLOCATION`置为1。

## xTaskCreate()函数

函数原型

<div align=center>
<img src="FreeRTOS-2-1.png" height = '360'>
</div>

形参描述

| 形参          | 描述                                                                   |
| ------------- | ---------------------------------------------------------------------- |
| pxTaskCode    | 指向任务函数的指针，即任务函数的名字                                   |
| pcName        | 任务函数的名字                                                         |
| usStackDepth  | 任务堆栈大小，单位：字                                                 |
| pvParameters  | 传递给任务函数的参数                                                   |
| uxPriority    | 任务函数优先级                                                         |
| pxCreatedTask | 任务句柄，任务成功创建后，会返回任务句柄。任务句柄就是任务的任务控制块 |

返回值

| 返回值                                | 描述                   |
| ------------------------------------- | ---------------------- |
| pdPASS                                | 任务创建成功           |
| errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY | 内存不足，任务创建失败 |

## xTaskCreateStatic()函数

函数原型

<div align=center>
<img src="FreeRTOS-2-2.png" height = '360'>
</div>

形参描述

| 形参           | 描述                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------- |
| pxTaskCode     | 指向任务函数的指针，即任务函数的名字                                                                |
| pcName         | 任务函数的名字                                                                                      |
| ulStackDepth   | 任务堆栈大小，单位：字                                                                              |
| pvParameters   | 传递给任务函数的参数                                                                                |
| uxPriority     | 任务函数优先级                                                                                      |
| puxStackBuffer | 任务栈指针，内存由用户分配提供,就是定义一个数组，数组的名字就是这个指针，数组的大小就是任务堆栈大小 |
| pxTaskBuffer   | 任务控制块指针，内存由用户分配提供                                                                  |
| pxCreatedTask  | 任务句柄，任务成功创建后，会返回任务句柄。任务句柄就是任务的任务控制块                              |

返回值

| 返回值 | 描述                                 |
| ------ | ------------------------------------ |
| NULL   | 用户没有提供相应的内存，任务创建失败 |
| 其他值 | 任务句柄，创建成功                   |

## vTaskDelete()函数

| 形参          | 描述               |
| ------------- | ------------------ |
| vTaskToDelete | 待删除的任务的句柄 |

当传入的实参为`NULL`时，代表删除任务自身

该函数无返回值


# 命名规范

- u: unsigned
- s: short
- l: long
- c: char
- x: 用户自定义的数据类型，如结构体，队列等。表示的类型为BaseType_t。如函数`xTaskCreate()`，其返回值就是BaseType_t类型
- e: 枚举
- p: 指针
- prv: static函数
- v: void函数，无返回值，如`vTaskDelete()`

函数名包含了该函数的返回值、函数所在的文件、函数功能。

如`xTaskCreateStatic()`，其返回值类型为BaseType_t，在task.c文件中，功能是静态创建任务。

参考：`https://blog.csdn.net/freestep96/article/details/126692753`