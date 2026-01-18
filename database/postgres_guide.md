SQL 事务与并发控制完全指南
> **最后更新**：2026-01-15

## 目录

1. [PostgreSQL 事务基础](#postgresql-事务基础)
2. [MVCC 多版本并发控制](#mvcc-多版本并发控制)
3. [事务隔离级别](#事务隔离级别)
4. [PostgreSQL 如何实现隔离级别](#postgresql-如何实现隔离级别)
5. [锁机制详解](#锁机制详解)
6. [高并发下的可见性问题](#高并发下的可见性问题)
7. [SSI 可串行化快照隔离](#ssi-可串行化快照隔离)
8. [PostgreSQL vs MySQL 对比](#postgresql-vs-mysql-对比)
9. [实战与最佳实践](#实战与最佳实践)
10. [常见问题解答](#常见问题解答)

---

## PostgreSQL 事务基础

### 事务概述

PostgreSQL 是一个完全支持 ACID 特性的关系型数据库，其事务实现具有以下特点：

```
┌─────────────────────────────────────────────────────┐
│            PostgreSQL 事务特性                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ✓ 完全支持 ACID                                    │
│  ✓ 默认自动提交（autocommit）                       │
│  ✓ 支持保存点（SAVEPOINT）                          │
│  ✓ 支持两阶段提交（2PC）                            │
│  ✓ 基于 MVCC 的并发控制                             │
│  ✓ 支持 SSI（可串行化快照隔离）                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 事务的基本操作

```sql
-- 开始事务
BEGIN;
-- 或者
START TRANSACTION;

-- 执行操作
UPDATE accounts SET balance = balance - 1000 WHERE id = 'A';
UPDATE accounts SET balance = balance + 1000 WHERE id = 'B';

-- 提交事务
COMMIT;

-- 或者回滚事务
ROLLBACK;

-- 使用保存点
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
SAVEPOINT sp1;
UPDATE accounts SET balance = balance + 100 WHERE id = 'B';
-- 如果出错，可以回滚到保存点
ROLLBACK TO SAVEPOINT sp1;
COMMIT;
```

### 事务 ID（XID）

```
PostgreSQL 使用事务 ID（XID）来追踪每个事务：

┌─────────────────────────────────────────────────────┐
│                                                      │
│  XID 特性：                                          │
│  ├─ 32位无符号整数                                  │
│  ├─ 单调递增                                        │
│  ├─ 每个写事务分配一个唯一 XID                      │
│  ├─ 只读事务不分配 XID（优化）                      │
│  └─ 存在回卷问题（约 21 亿次后需要 VACUUM）         │
│                                                      │
│  特殊 XID：                                          │
│  ├─ 0 = InvalidTransactionId                        │
│  ├─ 1 = BootstrapTransactionId                      │
│  └─ 2 = FrozenTransactionId（已冻结，永远可见）     │
│                                                      │
└─────────────────────────────────────────────────────┘

查看当前事务 ID：
SELECT txid_current();
```

### ACID 实现机制

| 特性 | PostgreSQL 实现机制 | 说明 |
|------|---------------------|------|
| **原子性** | WAL + CLOG | 通过事务日志和提交日志保证 |
| **一致性** | 约束 + 触发器 + 其他三个特性 | 由数据库和应用共同保证 |
| **隔离性** | MVCC + 锁 + SSI | 多版本并发控制和锁机制 |
| **持久性** | WAL（Write-Ahead Log） | 先写日志，后写数据 |

---

## MVCC 多版本并发控制

### PostgreSQL MVCC 核心思想

```
PostgreSQL 的 MVCC 实现与 MySQL 有本质区别：

┌─────────────────────────────────────────────────────┐
│                                                      │
│  MySQL MVCC:                                         │
│  └─ 旧版本存储在 Undo Log 中                        │
│  └─ 数据页只保留最新版本                            │
│  └─ 通过回滚指针形成版本链                          │
│                                                      │
│  PostgreSQL MVCC:                                    │
│  └─ 所有版本直接存储在数据表中（堆表）              │
│  └─ 新旧版本共存于同一张表                          │
│  └─ 通过 VACUUM 清理死元组                          │
│  └─ 不需要回滚段（Undo Log）                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 元组（Tuple）结构

```
PostgreSQL 中每行数据称为"元组"（Tuple），包含隐藏的系统列：

┌─────────────────────────────────────────────────────────────────┐
│                     元组头部（23 字节）                          │
├────────────┬────────────┬────────────┬────────────┬─────────────┤
│   xmin     │   xmax     │   cmin     │   cmax     │  ctid       │
│ (4 bytes)  │ (4 bytes)  │ (4 bytes)  │ (4 bytes)  │ (6 bytes)   │
├────────────┴────────────┴────────────┴────────────┴─────────────┤
│                      用户数据列                                  │
│              (id, name, balance, ...)                           │
└─────────────────────────────────────────────────────────────────┘

字段说明：
├─ xmin: 创建该元组的事务 ID（插入或更新）
├─ xmax: 删除该元组的事务 ID（删除或更新）
│        └─ 0 表示该元组是活跃的（未被删除）
├─ cmin: 插入该元组的命令 ID（事务内的操作序号）
├─ cmax: 删除该元组的命令 ID
└─ ctid: 元组的物理位置（页号, 偏移量）

查看系统列：
SELECT xmin, xmax, ctid, * FROM your_table;
```

### 更新操作的实现

```
PostgreSQL 的 UPDATE 实际上是 DELETE + INSERT：

原始状态（id=1, name='Tom'）：
┌─────────────────────────────────────────┐
│ xmin=100 │ xmax=0 │ id=1 │ name='Tom'   │  ← 活跃元组
└─────────────────────────────────────────┘

事务 200 执行 UPDATE SET name='Jerry'：
┌─────────────────────────────────────────┐
│ xmin=100 │ xmax=200 │ id=1 │ name='Tom' │  ← 旧版本（死元组）
├─────────────────────────────────────────┤
│ xmin=200 │ xmax=0 │ id=1 │ name='Jerry' │  ← 新版本（活跃）
└─────────────────────────────────────────┘

这种设计的影响：
├─ 优点：回滚速度快（只需标记事务为中止）
├─ 优点：读操作不阻塞写操作
├─ 缺点：表会膨胀（需要 VACUUM 清理）
└─ 缺点：频繁更新会产生大量死元组
```

### 删除操作的实现

```
DELETE 不是物理删除，而是标记 xmax：

执行前：
┌─────────────────────────────────────────┐
│ xmin=100 │ xmax=0 │ id=1 │ name='Tom'   │  ← 活跃元组
└─────────────────────────────────────────┘

事务 300 执行 DELETE：
┌─────────────────────────────────────────┐
│ xmin=100 │ xmax=300 │ id=1 │ name='Tom' │  ← 死元组
└─────────────────────────────────────────┘

元组仍然存在，等待 VACUUM 清理
```

### 可见性判断规则

```
PostgreSQL 根据元组的 xmin 和 xmax 判断对当前事务是否可见：

可见性判断流程：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  1. 检查 xmin（创建事务）                           │
│     ├─ xmin 是当前事务 → 可见（自己创建的）        │
│     ├─ xmin 未提交 → 不可见                         │
│     ├─ xmin 已中止 → 不可见                         │
│     └─ xmin 已提交且早于快照 → 继续检查 xmax       │
│                                                      │
│  2. 检查 xmax（删除事务）                           │
│     ├─ xmax = 0 → 可见（未被删除）                 │
│     ├─ xmax 是当前事务 → 不可见（自己删除的）      │
│     ├─ xmax 未提交 → 可见（删除未生效）            │
│     ├─ xmax 已中止 → 可见（删除已回滚）            │
│     └─ xmax 已提交且早于快照 → 不可见              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 事务快照（Snapshot）

```
快照定义了事务能看到的数据版本范围：

快照组成：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  xmin: 最小活跃事务 ID                              │
│       └─ 小于此值的事务一定已完成                   │
│                                                      │
│  xmax: 下一个待分配的事务 ID                        │
│       └─ 大于等于此值的事务一定还未开始             │
│                                                      │
│  xip_list: 快照创建时活跃的事务 ID 列表            │
│       └─ 这些事务在 [xmin, xmax) 范围内但未完成    │
│                                                      │
└─────────────────────────────────────────────────────┘

查看当前快照：
SELECT txid_current_snapshot();
-- 返回格式：xmin:xmax:xip_list
-- 例如：100:105:102,103 表示：
--   xmin=100, xmax=105, 活跃事务为 102 和 103
```

### VACUUM 清理机制

```
VACUUM 是 PostgreSQL 特有的维护操作，用于回收死元组空间：

为什么需要 VACUUM？
├─ PostgreSQL 的 MVCC 会产生死元组
├─ 死元组占用磁盘空间
├─ 死元组会影响查询性能
└─ 需要定期清理以维护性能

VACUUM 类型：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  VACUUM（普通）:                                     │
│  ├─ 标记死元组空间可重用                            │
│  ├─ 不会释放空间给操作系统                          │
│  ├─ 不会阻塞读写操作                                │
│  └─ 适合日常维护                                    │
│                                                      │
│  VACUUM FULL:                                        │
│  ├─ 完全重写表，释放空间给操作系统                  │
│  ├─ 会加排他锁，阻塞所有操作                        │
│  ├─ 需要额外磁盘空间                                │
│  └─ 只在必要时使用                                  │
│                                                      │
│  Autovacuum:                                         │
│  ├─ 自动后台运行                                    │
│  ├─ 根据配置的阈值触发                              │
│  └─ 推荐保持开启                                    │
│                                                      │
└─────────────────────────────────────────────────────┘

VACUUM 相关命令：
-- 普通 VACUUM
VACUUM table_name;

-- 带统计信息更新
VACUUM ANALYZE table_name;

-- 完全重写
VACUUM FULL table_name;

-- 查看表的死元组情况
SELECT relname, n_dead_tup, n_live_tup
FROM pg_stat_user_tables
WHERE relname = 'your_table';
```

### HOT（Heap-Only Tuple）优化

```
HOT 是 PostgreSQL 对频繁更新场景的优化：

触发条件：
├─ 更新不涉及索引列
├─ 新元组可以放在同一数据页
└─ 满足条件时不需要更新索引

优势：
├─ 减少索引膨胀
├─ 减少 I/O 操作
└─ 提高更新性能

示例：
┌─────────────────────────────────────────────────────┐
│  Page N                                              │
├─────────────────────────────────────────────────────┤
│  Tuple1 (xmin=100) ──────┐                          │
│  Tuple2 (xmin=200) ←─────┘ ctid 指向               │
│                                                      │
│  索引仍然指向 Tuple1                                │
│  通过 ctid 链找到最新版本 Tuple2                    │
└─────────────────────────────────────────────────────┘
```

---

## 事务隔离级别

### PostgreSQL 支持的隔离级别

```
PostgreSQL 实际支持三种隔离级别（READ UNCOMMITTED 被映射为 READ COMMITTED）：

┌─────────────────────────────────────────────────────────────────┐
│ 级别              │ 脏读 │ 不可重复读 │ 幻读 │ 串行化异常 │ 实际行为 │
├───────────────────┼──────┼────────────┼──────┼────────────┼──────────┤
│ READ UNCOMMITTED  │  ✓   │     ✗      │  ✗   │     ✗      │ 同 RC    │
│ (读未提交)        │      │            │      │            │          │
├───────────────────┼──────┼────────────┼──────┼────────────┼──────────┤
│ READ COMMITTED    │  ✓   │     ✗      │  ✗   │     ✗      │ 默认级别 │
│ (读已提交)        │      │            │      │            │          │
├───────────────────┼──────┼────────────┼──────┼────────────┼──────────┤
│ REPEATABLE READ   │  ✓   │     ✓      │  ✓*  │     ✗      │ 快照隔离 │
│ (可重复读)        │      │            │      │            │          │
├───────────────────┼──────┼────────────┼──────┼────────────┼──────────┤
│ SERIALIZABLE      │  ✓   │     ✓      │  ✓   │     ✓      │ SSI      │
│ (串行化)          │      │            │      │            │          │
└─────────────────────────────────────────────────────────────────┘

✓ = 已解决  ✗ = 可能存在
✓* = PostgreSQL RR 通过快照完全防止幻读（只读场景）

重要：PostgreSQL 默认隔离级别是 READ COMMITTED
      MySQL InnoDB 默认隔离级别是 REPEATABLE READ
```

### READ COMMITTED（读已提交）

```
特点：
├─ PostgreSQL 的默认隔离级别
├─ 每条 SQL 语句开始时获取新快照
├─ 只能看到语句执行前已提交的数据
├─ 同一事务中不同语句可能看到不同数据
└─ 解决脏读，存在不可重复读和幻读

示例：
┌─────────────────────────────────────────────────────────────────┐
│   时间   │     事务 A              │     事务 B                 │
├──────────┼─────────────────────────┼────────────────────────────┤
│   T1     │ BEGIN;                  │                            │
│   T2     │ SELECT balance          │                            │
│          │ WHERE id=1;             │                            │
│          │ → 返回 1000             │                            │
│   T3     │                         │ BEGIN;                     │
│   T4     │                         │ UPDATE accounts            │
│          │                         │ SET balance=500            │
│          │                         │ WHERE id=1;                │
│   T5     │                         │ COMMIT;                    │
│   T6     │ SELECT balance          │                            │
│          │ WHERE id=1;             │                            │
│          │ → 返回 500 (不可重复读) │                            │
│   T7     │ COMMIT;                 │                            │
└─────────────────────────────────────────────────────────────────┘

每个 SELECT 使用新快照，所以能看到 B 的已提交修改
```

### REPEATABLE READ（可重复读）

```
特点：
├─ 事务开始时（第一条语句）获取快照
├─ 整个事务期间使用同一快照
├─ 保证同一查询返回相同结果
├─ 完全防止幻读（快照隔离）
└─ 可能出现串行化异常（写偏斜）

与 MySQL RR 的重要区别：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  MySQL RR:                                           │
│  ├─ 使用间隙锁防止幻读                              │
│  ├─ 当前读（FOR UPDATE）可能触发幻读                │
│  └─ 锁机制可能导致死锁                              │
│                                                      │
│  PostgreSQL RR:                                      │
│  ├─ 纯快照隔离，不使用间隙锁                        │
│  ├─ 任何读取都使用同一快照                          │
│  ├─ 如果检测到写冲突，事务会被中止                  │
│  └─ 需要应用层处理重试                              │
│                                                      │
└─────────────────────────────────────────────────────┘

示例：
┌─────────────────────────────────────────────────────────────────┐
│   时间   │     事务 A              │     事务 B                 │
├──────────┼─────────────────────────┼────────────────────────────┤
│   T1     │ BEGIN ISOLATION LEVEL   │                            │
│          │ REPEATABLE READ;        │                            │
│   T2     │ SELECT balance          │                            │
│          │ WHERE id=1;             │                            │
│          │ → 返回 1000             │                            │
│          │ (创建快照)              │                            │
│   T3     │                         │ BEGIN;                     │
│   T4     │                         │ UPDATE accounts            │
│          │                         │ SET balance=500            │
│          │                         │ WHERE id=1;                │
│   T5     │                         │ COMMIT;                    │
│   T6     │ SELECT balance          │                            │
│          │ WHERE id=1;             │                            │
│          │ → 仍然返回 1000         │                            │
│          │ (使用原快照)            │                            │
│   T7     │ COMMIT;                 │                            │
└─────────────────────────────────────────────────────────────────┘

重要：即使 B 已提交，A 仍然看到旧值（快照隔离）
```

### SERIALIZABLE（串行化）

```
特点：
├─ 最高隔离级别
├─ 使用 SSI（可串行化快照隔离）实现
├─ 基于快照隔离 + 冲突检测
├─ 不使用传统的锁（除了写锁）
├─ 检测到串行化冲突时会中止事务
└─ 性能比传统串行化好很多

PostgreSQL 的 SERIALIZABLE 特殊之处：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  传统数据库 SERIALIZABLE:                           │
│  ├─ 使用两阶段锁（2PL）                             │
│  ├─ 读操作加共享锁                                  │
│  ├─ 范围查询加范围锁                                │
│  └─ 性能很差，容易死锁                              │
│                                                      │
│  PostgreSQL SERIALIZABLE (SSI):                     │
│  ├─ 基于快照隔离                                    │
│  ├─ 读操作不加锁                                    │
│  ├─ 使用谓词锁（轻量级）检测冲突                    │
│  ├─ 乐观并发控制                                    │
│  └─ 冲突时回滚而不是等待                            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 设置隔离级别

```sql
-- 设置当前事务的隔离级别
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- 或
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- 设置会话默认隔离级别
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 设置全局默认隔离级别（postgresql.conf）
default_transaction_isolation = 'read committed'

-- 查看当前隔离级别
SHOW transaction_isolation;
-- 或
SELECT current_setting('transaction_isolation');
```

---

## PostgreSQL 如何实现隔离级别

### 各隔离级别实现机制

```
┌─────────────────────────────────────────────────────────────────┐
│  隔离级别          │        PostgreSQL 实现机制                  │
├────────────────────┼────────────────────────────────────────────┤
│  READ UNCOMMITTED  │ 与 READ COMMITTED 相同（不允许脏读）       │
│                    │ PostgreSQL 设计上不支持脏读                 │
├────────────────────┼────────────────────────────────────────────┤
│  READ COMMITTED    │ 每条语句开始时创建新快照                   │
│                    │ 只能看到语句开始前已提交的数据             │
│                    │ MVCC + 语句级快照                          │
├────────────────────┼────────────────────────────────────────────┤
│  REPEATABLE READ   │ 事务开始时创建快照，全程使用               │
│                    │ 快照隔离（Snapshot Isolation）             │
│                    │ 写冲突检测（First-Updater-Wins）          │
├────────────────────┼────────────────────────────────────────────┤
│  SERIALIZABLE      │ SSI（可串行化快照隔离）                    │
│                    │ 快照隔离 + 谓词锁 + 依赖检测               │
│                    │ 检测 rw-antidependency 循环                │
└─────────────────────────────────────────────────────────────────┘
```

### 快照隔离详解

```
快照隔离（Snapshot Isolation）是 PostgreSQL RR 的核心机制：

工作原理：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  1. 事务开始时记录当前快照状态                       │
│     ├─ 最小活跃事务 ID                              │
│     ├─ 最大事务 ID                                  │
│     └─ 活跃事务列表                                 │
│                                                      │
│  2. 所有读操作使用这个快照                          │
│     └─ 只能看到快照时刻已提交的数据                 │
│                                                      │
│  3. 写操作检查冲突                                  │
│     ├─ 如果要更新的行在快照后被其他事务修改        │
│     └─ 当前事务会被中止（serialization failure）   │
│                                                      │
└─────────────────────────────────────────────────────┘

写冲突检测（First-Updater-Wins）：
┌─────────────────────────────────────────────────────────────────┐
│   时间   │     事务 A              │     事务 B                 │
├──────────┼─────────────────────────┼────────────────────────────┤
│   T1     │ BEGIN ISOLATION LEVEL   │ BEGIN ISOLATION LEVEL      │
│          │ REPEATABLE READ;        │ REPEATABLE READ;           │
│   T2     │ SELECT * FROM t         │ SELECT * FROM t            │
│          │ WHERE id=1;             │ WHERE id=1;                │
│   T3     │ UPDATE t SET v=1        │                            │
│          │ WHERE id=1;             │                            │
│   T4     │ COMMIT;                 │                            │
│   T5     │                         │ UPDATE t SET v=2           │
│          │                         │ WHERE id=1;                │
│          │                         │ ERROR: could not serialize │
│          │                         │ access due to concurrent   │
│          │                         │ update                     │
└─────────────────────────────────────────────────────────────────┘

事务 B 必须重试！
```

### CLOG（Commit Log）

```
CLOG 记录每个事务的提交状态：

┌─────────────────────────────────────────────────────┐
│                                                      │
│  事务状态（2 bits per transaction）：               │
│  ├─ 00 = IN_PROGRESS（进行中）                     │
│  ├─ 01 = COMMITTED（已提交）                       │
│  ├─ 02 = ABORTED（已中止）                         │
│  └─ 03 = SUB_COMMITTED（子事务已提交）             │
│                                                      │
│  存储位置：                                          │
│  └─ pg_xact/ 目录下的文件                          │
│                                                      │
│  作用：                                              │
│  ├─ 快速判断事务是否已提交                          │
│  └─ 用于可见性判断                                  │
│                                                      │
└─────────────────────────────────────────────────────┘

可见性判断时的 CLOG 查询：
1. 检查元组的 xmin 对应的事务状态
2. 如果已提交，再检查 xmax
3. 根据快照判断是否可见
```

---

## 锁机制详解

### PostgreSQL 锁分类

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│  表级锁（Table-Level Locks）：                       │
│  ├─ ACCESS SHARE                                    │
│  ├─ ROW SHARE                                       │
│  ├─ ROW EXCLUSIVE                                   │
│  ├─ SHARE UPDATE EXCLUSIVE                          │
│  ├─ SHARE                                           │
│  ├─ SHARE ROW EXCLUSIVE                             │
│  ├─ EXCLUSIVE                                       │
│  └─ ACCESS EXCLUSIVE                                │
│                                                      │
│  行级锁（Row-Level Locks）：                         │
│  ├─ FOR UPDATE                                      │
│  ├─ FOR NO KEY UPDATE                               │
│  ├─ FOR SHARE                                       │
│  └─ FOR KEY SHARE                                   │
│                                                      │
│  其他锁：                                            │
│  ├─ 咨询锁（Advisory Locks）                        │
│  └─ 谓词锁（Predicate Locks）- SSI 使用            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 表级锁详解

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 锁模式                 │ 触发操作                │ 冲突锁               │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ ACCESS SHARE           │ SELECT                  │ ACCESS EXCLUSIVE     │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ ROW SHARE              │ SELECT FOR UPDATE/SHARE │ EXCLUSIVE,           │
│                        │                         │ ACCESS EXCLUSIVE     │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ ROW EXCLUSIVE          │ INSERT, UPDATE, DELETE  │ SHARE, SHARE ROW     │
│                        │                         │ EXCL, EXCL, ACC EXCL │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ SHARE UPDATE EXCLUSIVE │ VACUUM, CREATE INDEX    │ SHARE UPDATE EXCL,   │
│                        │ CONCURRENTLY            │ SHARE, SHARE ROW     │
│                        │                         │ EXCL, EXCL, ACC EXCL │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ SHARE                  │ CREATE INDEX            │ ROW EXCL, SHARE      │
│                        │ (非 CONCURRENTLY)       │ UPDATE EXCL, SHARE   │
│                        │                         │ ROW EXCL, EXCL,      │
│                        │                         │ ACC EXCL             │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ SHARE ROW EXCLUSIVE    │ 触发器等                │ ROW EXCL以上全部     │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ EXCLUSIVE              │ REFRESH MAT VIEW CONCUR │ ROW SHARE以上全部    │
├────────────────────────┼─────────────────────────┼──────────────────────┤
│ ACCESS EXCLUSIVE       │ ALTER TABLE, DROP,      │ 与所有锁冲突         │
│                        │ TRUNCATE, VACUUM FULL   │                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 行级锁详解

```
PostgreSQL 行级锁的特点：
├─ 不存储在内存锁表中，直接存储在元组头部
├─ 不会产生锁表溢出问题
└─ 支持更细粒度的锁控制

行级锁类型：
┌─────────────────────────────────────────────────────────────────┐
│  锁模式              │ 说明                                     │
├──────────────────────┼──────────────────────────────────────────┤
│  FOR UPDATE          │ 最强的行锁，阻止其他事务的任何修改       │
│                      │ SELECT ... FOR UPDATE                    │
├──────────────────────┼──────────────────────────────────────────┤
│  FOR NO KEY UPDATE   │ 类似 FOR UPDATE，但允许其他事务获取      │
│                      │ FOR KEY SHARE（用于外键检查）            │
│                      │ UPDATE 默认使用此锁                       │
├──────────────────────┼──────────────────────────────────────────┤
│  FOR SHARE           │ 共享锁，阻止 UPDATE/DELETE                │
│                      │ SELECT ... FOR SHARE                     │
├──────────────────────┼──────────────────────────────────────────┤
│  FOR KEY SHARE       │ 最弱的行锁，只阻止 DELETE 和主键修改     │
│                      │ 用于外键检查                              │
└─────────────────────────────────────────────────────────────────┘

兼容性矩阵：
┌────────────────────┬──────────┬────────────────┬───────────┬───────────────┐
│                    │ FOR KEY  │ FOR SHARE      │ FOR NO KEY│ FOR UPDATE    │
│                    │ SHARE    │                │ UPDATE    │               │
├────────────────────┼──────────┼────────────────┼───────────┼───────────────┤
│ FOR KEY SHARE      │    ✓     │       ✓        │     ✓     │      ✗        │
├────────────────────┼──────────┼────────────────┼───────────┼───────────────┤
│ FOR SHARE          │    ✓     │       ✓        │     ✗     │      ✗        │
├────────────────────┼──────────┼────────────────┼───────────┼───────────────┤
│ FOR NO KEY UPDATE  │    ✓     │       ✗        │     ✗     │      ✗        │
├────────────────────┼──────────┼────────────────┼───────────┼───────────────┤
│ FOR UPDATE         │    ✗     │       ✗        │     ✗     │      ✗        │
└────────────────────┴──────────┴────────────────┴───────────┴───────────────┘
```

### PostgreSQL 与 MySQL 锁的区别

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL                │           MySQL InnoDB       │
├──────────────────────────────────────────────┼──────────────────────────────┤
│  无间隙锁（Gap Lock）                        │  有间隙锁                    │
│  └─ RR 下不锁间隙                            │  └─ RR 下锁索引间隙          │
├──────────────────────────────────────────────┼──────────────────────────────┤
│  写冲突时回滚重试                            │  写冲突时等待                 │
│  └─ 乐观方式                                 │  └─ 悲观方式                  │
├──────────────────────────────────────────────┼──────────────────────────────┤
│  行锁存储在元组头部                          │  行锁存储在内存锁表           │
│  └─ 无锁表溢出问题                           │  └─ 可能锁表溢出              │
├──────────────────────────────────────────────┼──────────────────────────────┤
│  更细粒度的行锁                              │  只有共享锁和排他锁           │
│  └─ 4 种行锁模式                             │  └─ 2 种行锁模式              │
├──────────────────────────────────────────────┼──────────────────────────────┤
│  死锁检测较少触发                            │  间隙锁容易导致死锁           │
│  └─ 无间隙锁冲突                             │  └─ 间隙锁之间可能死锁        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 显式加锁

```sql
-- 行级锁
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
SELECT * FROM accounts WHERE id = 1 FOR SHARE;
SELECT * FROM accounts WHERE id = 1 FOR NO KEY UPDATE;
SELECT * FROM accounts WHERE id = 1 FOR KEY SHARE;

-- 跳过锁定的行
SELECT * FROM accounts WHERE id = 1 FOR UPDATE SKIP LOCKED;

-- 不等待，立即返回错误
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;

-- 表级锁
LOCK TABLE accounts IN SHARE MODE;
LOCK TABLE accounts IN EXCLUSIVE MODE;
LOCK TABLE accounts IN ACCESS EXCLUSIVE MODE;

-- 咨询锁（应用级锁）
SELECT pg_advisory_lock(123);      -- 获取锁
SELECT pg_advisory_unlock(123);    -- 释放锁
SELECT pg_try_advisory_lock(123);  -- 尝试获取（不等待）
```

### 死锁处理

```
PostgreSQL 死锁检测：
├─ 默认每秒检测一次（deadlock_timeout = 1s）
├─ 检测到死锁时，选择一个事务中止
├─ 被中止的事务收到错误：ERROR: deadlock detected
└─ 应用需要捕获错误并重试

查看和诊断死锁：
-- 查看当前锁等待
SELECT
    blocked.pid AS blocked_pid,
    blocked.usename AS blocked_user,
    blocking.pid AS blocking_pid,
    blocking.usename AS blocking_user,
    blocked.query AS blocked_query
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked
    ON blocked.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocked_locks.locktype = blocking_locks.locktype
    AND blocked_locks.database IS NOT DISTINCT FROM blocking_locks.database
    AND blocked_locks.relation IS NOT DISTINCT FROM blocking_locks.relation
    AND blocked_locks.page IS NOT DISTINCT FROM blocking_locks.page
    AND blocked_locks.tuple IS NOT DISTINCT FROM blocking_locks.tuple
    AND blocked_locks.virtualxid IS NOT DISTINCT FROM blocking_locks.virtualxid
    AND blocked_locks.transactionid IS NOT DISTINCT FROM blocking_locks.transactionid
    AND blocked_locks.classid IS NOT DISTINCT FROM blocking_locks.classid
    AND blocked_locks.objid IS NOT DISTINCT FROM blocking_locks.objid
    AND blocked_locks.objsubid IS NOT DISTINCT FROM blocking_locks.objsubid
    AND blocked_locks.pid != blocking_locks.pid
JOIN pg_catalog.pg_stat_activity blocking
    ON blocking.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- 死锁相关配置
deadlock_timeout = '1s'          -- 死锁检测延迟
log_lock_waits = on              -- 记录锁等待日志
lock_timeout = '10s'             -- 锁等待超时
```

---

## 高并发下的可见性问题

### 问题分类与 PostgreSQL 解决方案

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  问题类型      │  PostgreSQL 解决方案                                       │
├────────────────┼────────────────────────────────────────────────────────────┤
│  脏读          │  不支持（即使 READ UNCOMMITTED 也不允许）                  │
├────────────────┼────────────────────────────────────────────────────────────┤
│  不可重复读    │  REPEATABLE READ 级别（快照隔离）                          │
├────────────────┼────────────────────────────────────────────────────────────┤
│  幻读          │  REPEATABLE READ 级别（快照完全防止）                      │
│                │  注意：写操作可能触发串行化异常                            │
├────────────────┼────────────────────────────────────────────────────────────┤
│  丢失更新      │  RR 级别写冲突检测 / SELECT FOR UPDATE                    │
├────────────────┼────────────────────────────────────────────────────────────┤
│  写偏斜        │  SERIALIZABLE 级别（SSI）                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 丢失更新问题

```
场景：两个事务同时读取并更新同一数据

┌─────────────────────────────────────────────────────────────────┐
│   时间   │     事务 A              │     事务 B                 │
├──────────┼─────────────────────────┼────────────────────────────┤
│   T1     │ BEGIN;                  │ BEGIN;                     │
│   T2     │ SELECT stock FROM p     │                            │
│          │ WHERE id=1;             │                            │
│          │ → 读到 10               │                            │
│   T3     │                         │ SELECT stock FROM p        │
│          │                         │ WHERE id=1;                │
│          │                         │ → 读到 10                  │
│   T4     │ UPDATE p SET stock=9    │                            │
│          │ WHERE id=1;             │                            │
│   T5     │ COMMIT;                 │                            │
│   T6     │                         │ UPDATE p SET stock=9       │
│          │                         │ WHERE id=1;                │
│   T7     │                         │ COMMIT;                    │
└─────────────────────────────────────────────────────────────────┘

结果：stock=9，丢失了一次扣减
```

#### PostgreSQL 解决方案

```sql
-- 方案1：悲观锁（SELECT FOR UPDATE）
BEGIN;
SELECT stock FROM products WHERE id = 1 FOR UPDATE;  -- 加行锁
IF (stock >= 1) THEN
    UPDATE products SET stock = stock - 1 WHERE id = 1;
END IF;
COMMIT;

-- 方案2：原子更新（推荐）
UPDATE products
SET stock = stock - 1
WHERE id = 1 AND stock >= 1;

IF (ROW_COUNT = 0) THEN
    -- 库存不足
END IF;

-- 方案3：乐观锁（版本号）
UPDATE products
SET stock = stock - 1, version = version + 1
WHERE id = 1 AND version = 5;

IF (ROW_COUNT = 0) THEN
    -- 版本冲突，需重试
END IF;

-- 方案4：REPEATABLE READ（PostgreSQL 特有）
-- PostgreSQL RR 会检测写冲突并中止后来的事务
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT stock FROM products WHERE id = 1;
-- 如果其他事务已修改该行并提交，下面的 UPDATE 会失败
UPDATE products SET stock = stock - 1 WHERE id = 1;
COMMIT;
-- 应用需要捕获 serialization failure 并重试
```

### 写偏斜问题

```
场景：两个事务基于相同的快照做出决策，但组合结果违反约束

示例 - 值班约束（至少一人值班）：
┌─────────────────────────────────────────────────────────────────┐
│   时间   │     事务 A (Alice)      │     事务 B (Bob)           │
├──────────┼─────────────────────────┼────────────────────────────┤
│   T1     │ BEGIN ISOLATION LEVEL   │ BEGIN ISOLATION LEVEL      │
│          │ REPEATABLE READ;        │ REPEATABLE READ;           │
│   T2     │ SELECT COUNT(*)         │                            │
│          │ FROM doctors            │                            │
│          │ WHERE on_call = true;   │                            │
│          │ → 返回 2 (≥1, 可离开)  │                            │
│   T3     │                         │ SELECT COUNT(*)            │
│          │                         │ FROM doctors               │
│          │                         │ WHERE on_call = true;      │
│          │                         │ → 返回 2 (≥1, 可离开)     │
│   T4     │ UPDATE doctors          │                            │
│          │ SET on_call = false     │                            │
│          │ WHERE name = 'Alice';   │                            │
│   T5     │                         │ UPDATE doctors             │
│          │                         │ SET on_call = false        │
│          │                         │ WHERE name = 'Bob';        │
│   T6     │ COMMIT;                 │                            │
│   T7     │                         │ COMMIT;                    │
└─────────────────────────────────────────────────────────────────┘

结果：没有人值班！RR 级别无法检测这种冲突。
原因：两个事务修改的是不同的行，没有写-写冲突。

这就是为什么需要 SERIALIZABLE 级别！
```

#### 使用 SERIALIZABLE 解决写偏斜

```sql
-- SERIALIZABLE 级别会检测读-写依赖冲突
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

SELECT COUNT(*) FROM doctors WHERE on_call = true;
-- SSI 记录：事务读取了 on_call = true 的行

UPDATE doctors SET on_call = false WHERE name = 'Alice';
-- SSI 记录：事务修改了 doctors 表

COMMIT;
-- SSI 检查是否存在依赖循环

-- 如果存在写偏斜，其中一个事务会收到：
-- ERROR: could not serialize access due to read/write dependencies
-- among transactions
-- HINT: The transaction might succeed if retried.
```

---

## SSI 可串行化快照隔离

### SSI 原理

```
SSI（Serializable Snapshot Isolation）是 PostgreSQL 9.1 引入的创新技术：

传统 SERIALIZABLE 问题：
├─ 使用两阶段锁（2PL）
├─ 读操作加共享锁
├─ 性能差，死锁多
└─ 不适合高并发场景

SSI 创新：
├─ 基于快照隔离
├─ 读操作不加锁
├─ 使用轻量级谓词锁追踪读取
├─ 检测 rw-antidependency 循环
└─ 冲突时回滚而不是等待

工作原理：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  1. 所有事务在快照隔离基础上运行                    │
│                                                      │
│  2. 追踪读写依赖关系：                              │
│     ├─ rw-conflict: T1 读 → T2 写（后提交）        │
│     └─ ww-conflict: T1 写 → T2 写                   │
│                                                      │
│  3. 检测危险结构（dangerous structure）：           │
│     └─ T1 --rw→ T2 --rw→ T3                        │
│        且 T3 在 T1 之前提交                         │
│                                                      │
│  4. 发现危险结构时中止相关事务                      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 谓词锁（SIREAD Lock）

```
SSI 使用谓词锁（SIREAD Lock）追踪读取操作：

特点：
├─ 不阻塞其他事务（只用于追踪）
├─ 记录事务读取了哪些数据
├─ 用于检测 rw-antidependency
└─ 事务提交后仍保留一段时间

锁粒度（从细到粗）：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  1. 元组级（Tuple-level）                           │
│     └─ 读取单行时                                   │
│                                                      │
│  2. 页级（Page-level）                              │
│     └─ 元组级锁太多时升级                           │
│                                                      │
│  3. 关系级（Relation-level）                        │
│     └─ 页级锁太多时升级                             │
│     └─ 顺序扫描时                                   │
│                                                      │
└─────────────────────────────────────────────────────┘

查看谓词锁：
SELECT * FROM pg_locks WHERE locktype = 'tuple';
```

### SSI 示例

```
写偏斜场景在 SERIALIZABLE 级别下的处理：

┌─────────────────────────────────────────────────────────────────┐
│   时间   │     事务 A              │     事务 B                 │
├──────────┼─────────────────────────┼────────────────────────────┤
│   T1     │ BEGIN ISOLATION LEVEL   │ BEGIN ISOLATION LEVEL      │
│          │ SERIALIZABLE;           │ SERIALIZABLE;              │
│   T2     │ SELECT * FROM doctors   │                            │
│          │ WHERE on_call = true;   │                            │
│          │ (获取 SIREAD 锁)        │                            │
│   T3     │                         │ SELECT * FROM doctors      │
│          │                         │ WHERE on_call = true;      │
│          │                         │ (获取 SIREAD 锁)           │
│   T4     │ UPDATE doctors          │                            │
│          │ SET on_call = false     │                            │
│          │ WHERE name = 'Alice';   │                            │
│          │ (检测到 rw-conflict)    │                            │
│   T5     │                         │ UPDATE doctors             │
│          │                         │ SET on_call = false        │
│          │                         │ WHERE name = 'Bob';        │
│          │                         │ (检测到 rw-conflict)       │
│   T6     │ COMMIT;                 │                            │
│          │ (成功)                  │                            │
│   T7     │                         │ COMMIT;                    │
│          │                         │ ERROR: could not serialize │
│          │                         │ access due to read/write   │
│          │                         │ dependencies               │
└─────────────────────────────────────────────────────────────────┘

SSI 检测到：
├─ 事务 A 读取了事务 B 修改的数据（rw-conflict）
├─ 事务 B 读取了事务 A 修改的数据（rw-conflict）
└─ 形成循环依赖，必须中止一个事务
```

### SSI 配置参数

```sql
-- 最大谓词锁数量（内存限制）
max_pred_locks_per_transaction = 64    -- 每事务
max_pred_locks_per_relation = -2       -- 每关系（负数表示比例）
max_pred_locks_per_page = 2            -- 每页（超过则升级）

-- 查看 SSI 相关统计
SELECT * FROM pg_stat_database
WHERE datname = current_database();
```

### SSI vs MySQL SERIALIZABLE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PostgreSQL SSI              │    MySQL SERIALIZABLE         │
├─────────────────────────────────────────────┼───────────────────────────────┤
│  基于快照隔离                               │  基于两阶段锁（2PL）          │
├─────────────────────────────────────────────┼───────────────────────────────┤
│  读操作不加锁                               │  读操作加共享锁               │
├─────────────────────────────────────────────┼───────────────────────────────┤
│  使用谓词锁追踪读取                         │  使用间隙锁防止幻读           │
├─────────────────────────────────────────────┼───────────────────────────────┤
│  冲突时回滚重试                             │  冲突时等待锁                 │
├─────────────────────────────────────────────┼───────────────────────────────┤
│  并发性能好                                 │  并发性能差                   │
├─────────────────────────────────────────────┼───────────────────────────────┤
│  可能出现假阳性回滚                         │  锁等待明确                   │
├─────────────────────────────────────────────┼───────────────────────────────┤
│  需要应用层处理重试                         │  应用层相对简单               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PostgreSQL vs MySQL 对比

### MVCC 实现对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          特性             │     PostgreSQL         │     MySQL InnoDB        │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  旧版本存储位置           │  堆表中（同一表）      │  Undo Log（回滚段）     │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  版本链                   │  通过 ctid 链接        │  通过 roll_ptr 链接     │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  元组标记                 │  xmin/xmax            │  DB_TRX_ID/DB_ROLL_PTR │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  版本清理                 │  VACUUM（显式）        │  Purge 线程（后台）     │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  表膨胀                   │  会膨胀，需 VACUUM     │  不膨胀                  │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  回滚速度                 │  快（只需标记中止）    │  慢（需应用 Undo Log）  │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  HOT 更新                 │  支持                  │  不支持                  │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  索引更新                 │  UPDATE 可能需要       │  UPDATE 需要            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 隔离级别对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          特性             │     PostgreSQL         │     MySQL InnoDB        │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  默认隔离级别             │  READ COMMITTED        │  REPEATABLE READ        │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  是否支持脏读             │  不支持（设计决定）    │  READ UNCOMMITTED 支持  │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  RR 级别快照时机          │  第一条语句时          │  第一条语句时           │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  RR 级别幻读              │  完全防止（快照）      │  部分防止（间隙锁）     │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  RR 级别写冲突            │  中止后来的事务        │  等待锁                  │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  SERIALIZABLE 实现        │  SSI（乐观）           │  2PL（悲观）            │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  SERIALIZABLE 性能        │  较好                  │  较差                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RR 级别行为差异详解

```
重要差异：PostgreSQL RR vs MySQL RR 处理写冲突的方式完全不同！

场景：两个事务同时更新同一行

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL RR              │        MySQL RR             │
├───────────────────────────────────────────────┼─────────────────────────────┤
│  BEGIN;                                       │  BEGIN;                     │
│  SELECT * FROM t WHERE id=1;                  │  SELECT * FROM t WHERE id=1;│
│  -- 创建快照                                  │  -- 创建快照                │
│                                               │                             │
│  -- 另一事务提交了对 id=1 的更新 --          │  -- 同左 --                 │
│                                               │                             │
│  UPDATE t SET v=1 WHERE id=1;                │  UPDATE t SET v=1 WHERE id=1;│
│  -- ERROR: could not serialize access        │  -- 等待另一事务的锁        │
│  -- 事务被中止，需要重试                      │  -- 或者获得锁后执行        │
└─────────────────────────────────────────────────────────────────────────────┘

PostgreSQL：First-Updater-Wins
├─ 第一个提交的事务获胜
├─ 后来的事务被中止
└─ 应用必须处理重试

MySQL：等待锁
├─ 后来的事务等待锁
├─ 获得锁后执行更新
└─ 读取的是新值（当前读）
```

### 锁机制对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          特性             │     PostgreSQL         │     MySQL InnoDB        │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  间隙锁（Gap Lock）       │  不支持                │  支持（RR 级别）        │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  临键锁（Next-Key Lock）  │  不支持                │  支持（默认行锁）       │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  行锁存储                 │  元组头部              │  内存锁表               │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  行锁粒度                 │  4 种模式              │  2 种模式               │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  表级锁                   │  8 种模式              │  较简单                  │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  谓词锁                   │  支持（SSI）           │  不支持                  │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  咨询锁                   │  支持                  │  支持（GET_LOCK）       │
├───────────────────────────┼────────────────────────┼─────────────────────────┤
│  锁升级                   │  支持                  │  自动升级到表锁         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 幻读处理对比

```
幻读场景：事务中两次范围查询返回不同行数

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL                 │        MySQL InnoDB         │
├───────────────────────────────────────────────┼─────────────────────────────┤
│  RC 级别：                                    │  RC 级别：                  │
│  └─ 会出现幻读                               │  └─ 会出现幻读              │
├───────────────────────────────────────────────┼─────────────────────────────┤
│  RR 级别：                                    │  RR 级别：                  │
│  └─ 快照隔离完全防止幻读（只读场景）         │  └─ 间隙锁防止大部分幻读    │
│  └─ 写操作可能触发串行化异常                 │  └─ 当前读可能出现幻读      │
├───────────────────────────────────────────────┼─────────────────────────────┤
│  SERIALIZABLE：                               │  SERIALIZABLE：             │
│  └─ SSI 完全防止                             │  └─ 全部加锁完全防止        │
└─────────────────────────────────────────────────────────────────────────────┘

PostgreSQL RR 下幻读完全被防止的原因：
├─ 所有读取使用事务开始时的快照
├─ 其他事务的插入/删除对当前事务不可见
└─ 但如果要更新/删除新插入的行，会失败或跳过

MySQL RR 下仍可能出现幻读的情况：
├─ 先 SELECT（快照读）
├─ 再 SELECT FOR UPDATE（当前读）
└─ 两次结果可能不同
```

### 总结对比表

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PostgreSQL vs MySQL 核心差异                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. MVCC 实现：                                                              │
│     PG: 多版本存于堆表，需要 VACUUM                                          │
│     MySQL: 旧版本存于 Undo Log，自动清理                                     │
│                                                                              │
│  2. 默认隔离级别：                                                           │
│     PG: READ COMMITTED                                                       │
│     MySQL: REPEATABLE READ                                                   │
│                                                                              │
│  3. 写冲突处理：                                                             │
│     PG: 回滚重试（乐观）                                                     │
│     MySQL: 等待锁（悲观）                                                    │
│                                                                              │
│  4. 幻读防止：                                                               │
│     PG: 快照隔离                                                             │
│     MySQL: 间隙锁                                                            │
│                                                                              │
│  5. 真正的可串行化：                                                         │
│     PG: SSI（性能好）                                                        │
│     MySQL: 2PL（性能差）                                                     │
│                                                                              │
│  6. 死锁倾向：                                                               │
│     PG: 较少（无间隙锁）                                                     │
│     MySQL: 较多（间隙锁冲突）                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 实战与最佳实践

### 选择合适的隔离级别

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  场景                          │   PostgreSQL 推荐                          │
├────────────────────────────────┼────────────────────────────────────────────┤
│  一般 OLTP 应用                │  READ COMMITTED（默认）                    │
├────────────────────────────────┼────────────────────────────────────────────┤
│  需要一致性快照的报表          │  REPEATABLE READ                           │
├────────────────────────────────┼────────────────────────────────────────────┤
│  库存扣减等并发更新            │  RC + SELECT FOR UPDATE                    │
│                                │  或 RC + 乐观锁                            │
├────────────────────────────────┼────────────────────────────────────────────┤
│  需要防止写偏斜                │  SERIALIZABLE                              │
├────────────────────────────────┼────────────────────────────────────────────┤
│  财务系统关键操作              │  SERIALIZABLE + 重试逻辑                   │
├────────────────────────────────┼────────────────────────────────────────────┤
│  高并发读多写少                │  READ COMMITTED                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 处理串行化失败

```sql
-- PostgreSQL RR/SERIALIZABLE 可能返回的错误：
-- ERROR: could not serialize access due to concurrent update
-- ERROR: could not serialize access due to read/write dependencies

-- 应用层重试模式
DO $$
DECLARE
    retry_count INT := 0;
    max_retries INT := 3;
BEGIN
    LOOP
        BEGIN
            -- 事务逻辑
            BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

            -- 你的业务操作
            UPDATE accounts SET balance = balance - 100 WHERE id = 1;
            UPDATE accounts SET balance = balance + 100 WHERE id = 2;

            COMMIT;
            EXIT; -- 成功，退出循环

        EXCEPTION
            WHEN serialization_failure OR deadlock_detected THEN
                -- 串行化失败或死锁，重试
                retry_count := retry_count + 1;
                IF retry_count >= max_retries THEN
                    RAISE EXCEPTION 'Max retries exceeded';
                END IF;
                -- 随机延迟避免活锁
                PERFORM pg_sleep(random() * 0.1);
        END;
    END LOOP;
END $$;
```

### 库存扣减场景

```sql
-- 方案1：悲观锁（推荐高竞争场景）
BEGIN;
SELECT stock FROM products WHERE id = 1 FOR UPDATE;
-- 如果 stock >= 1
UPDATE products SET stock = stock - 1 WHERE id = 1;
INSERT INTO orders (...) VALUES (...);
COMMIT;

-- 方案2：乐观锁/原子更新（推荐低竞争场景）
UPDATE products
SET stock = stock - 1
WHERE id = 1 AND stock >= 1;
-- 检查影响行数

-- 方案3：SKIP LOCKED（队列处理场景）
BEGIN;
SELECT * FROM products
WHERE id = 1 AND stock > 0
FOR UPDATE SKIP LOCKED;  -- 跳过已锁定的行
-- 处理逻辑
COMMIT;

-- 方案4：NOWAIT（快速失败）
BEGIN;
SELECT * FROM products WHERE id = 1 FOR UPDATE NOWAIT;
-- 如果锁定失败，立即返回错误
-- ERROR: could not obtain lock on row
COMMIT;
```

### VACUUM 最佳实践

```sql
-- 检查表的膨胀情况
SELECT
    schemaname,
    relname,
    n_live_tup,
    n_dead_tup,
    round(n_dead_tup * 100.0 / nullif(n_live_tup + n_dead_tup, 0), 2) as dead_pct,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- Autovacuum 关键配置
autovacuum = on                           -- 开启自动清理
autovacuum_vacuum_threshold = 50          -- 触发阈值（行数）
autovacuum_vacuum_scale_factor = 0.2      -- 触发比例
autovacuum_analyze_threshold = 50         -- ANALYZE 阈值
autovacuum_analyze_scale_factor = 0.1     -- ANALYZE 比例

-- 对高更新表调整 autovacuum 参数
ALTER TABLE hot_table SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

-- 手动执行 VACUUM（维护窗口）
VACUUM ANALYZE table_name;      -- 普通清理+更新统计
VACUUM FULL table_name;         -- 完全重写（需要排他锁）
```

### 监控查询

```sql
-- 查看当前活跃事务
SELECT
    pid,
    usename,
    application_name,
    state,
    query_start,
    now() - query_start as duration,
    wait_event_type,
    wait_event,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- 查看锁等待
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_query,
    blocking_activity.query AS blocking_query
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- 查看长事务
SELECT
    pid,
    now() - xact_start as xact_duration,
    state,
    query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
ORDER BY xact_start;
```

---

## 常见问题解答

### Q1: PostgreSQL 为什么默认使用 READ COMMITTED 而不是 REPEATABLE READ？

```
原因分析：

1. 历史和哲学
   ├─ PostgreSQL 认为应用应该处理并发问题
   ├─ RC 的行为更直观、更可预测
   └─ 符合 SQL 标准（标准默认也是 RC）

2. 性能考虑
   ├─ RC 每条语句获取新快照，无需长期持有
   ├─ RR 可能导致长事务持有快照，影响 VACUUM
   └─ RC 的内存开销更小

3. 实际需求
   ├─ 大多数 OLTP 应用不需要 RR
   ├─ 需要时可以显式使用 SELECT FOR UPDATE
   └─ 真正需要一致性时可以用 SERIALIZABLE

4. 与 MySQL 的对比
   └─ MySQL 默认 RR 是为了主从复制的 binlog 一致性
   └─ PostgreSQL 的逻辑复制不依赖隔离级别
```

### Q2: PostgreSQL REPEATABLE READ 和 MySQL REPEATABLE READ 有什么本质区别？

```
关键区别：

1. 写冲突处理
   ┌────────────────────────────────────────────────────────┐
   │  PostgreSQL RR:                                        │
   │  ├─ 检测到写冲突时中止后来的事务                      │
   │  ├─ 报错：could not serialize access                  │
   │  └─ 应用必须重试                                      │
   │                                                        │
   │  MySQL RR:                                             │
   │  ├─ 写冲突时等待锁                                    │
   │  ├─ 获得锁后继续执行                                  │
   │  └─ 可能覆盖其他事务的修改                            │
   └────────────────────────────────────────────────────────┘

2. 幻读防止
   ┌────────────────────────────────────────────────────────┐
   │  PostgreSQL RR:                                        │
   │  ├─ 纯快照隔离，完全防止幻读（只读场景）             │
   │  └─ 不使用间隙锁                                      │
   │                                                        │
   │  MySQL RR:                                             │
   │  ├─ 使用间隙锁防止幻读                                │
   │  ├─ 快照读和当前读结果可能不同                        │
   │  └─ 间隙锁可能导致死锁                                │
   └────────────────────────────────────────────────────────┘

3. 实际行为示例
   场景：两个事务同时更新同一行

   PostgreSQL: 第二个事务会失败，需要重试
   MySQL: 第二个事务等待，然后可能成功

结论：
├─ PostgreSQL RR 更"纯粹"，是真正的快照隔离
├─ MySQL RR 是快照隔离 + 锁的混合体
└─ 开发时需要注意这个重要区别
```

### Q3: 什么时候应该使用 SERIALIZABLE 隔离级别？

```
适用场景：

1. 需要防止写偏斜
   └─ 多个事务基于相同条件做出决策，可能违反约束

2. 复杂的业务规则
   └─ 跨多行的一致性约束

3. 对正确性要求极高
   └─ 金融核心系统
   └─ 计费系统

使用注意：

1. 必须实现重试逻辑
   ├─ SSI 可能中止事务
   └─ 应用必须能够处理重试

2. 性能影响
   ├─ 比传统 2PL SERIALIZABLE 好很多
   ├─ 但仍有额外开销
   └─ 可能出现假阳性回滚

3. 示例代码（伪代码）
   while True:
       try:
           begin_transaction(SERIALIZABLE)
           do_business_logic()
           commit()
           break
       except SerializationFailure:
           rollback()
           sleep(random_backoff)
           continue
```

### Q4: 为什么 PostgreSQL 不支持间隙锁？

```
设计选择：

1. MVCC 哲学不同
   ├─ PostgreSQL 强调乐观并发控制
   ├─ 冲突时回滚而不是等待
   └─ 与间隙锁的悲观方式相悖

2. SSI 更优雅
   ├─ 通过谓词锁检测冲突
   ├─ 不需要锁定不存在的行
   └─ 性能通常更好

3. 避免复杂性
   ├─ 间隙锁容易导致死锁
   ├─ 锁的范围难以预测
   └─ 对开发者不友好

替代方案：

如果确实需要锁定范围：
├─ 使用 SELECT FOR UPDATE 锁定已有行
├─ 使用唯一约束防止重复插入
├─ 使用 SERIALIZABLE 级别检测冲突
└─ 使用咨询锁（Advisory Lock）实现应用级锁
```

### Q5: 如何处理 PostgreSQL 的表膨胀问题？

```
预防措施：

1. 保持 autovacuum 开启
   ├─ 确保 autovacuum = on
   ├─ 合理配置阈值和频率
   └─ 监控 autovacuum 执行情况

2. 避免长事务
   ├─ 长事务会阻止 VACUUM 清理
   ├─ 设置 idle_in_transaction_session_timeout
   └─ 监控并终止长事务

3. 合理设计更新模式
   ├─ 利用 HOT 更新（不更新索引列）
   ├─ 避免频繁更新大表
   └─ 考虑分区表

处理已膨胀的表：

1. 检查膨胀程度
   SELECT
       relname,
       pg_size_pretty(pg_relation_size(relid)) as size,
       n_dead_tup
   FROM pg_stat_user_tables
   ORDER BY n_dead_tup DESC;

2. 普通 VACUUM（不锁表）
   VACUUM VERBOSE table_name;

3. VACUUM FULL（锁表，完全重写）
   -- 需要在维护窗口执行
   VACUUM FULL table_name;

4. 使用 pg_repack（在线重写，推荐）
   -- 第三方工具，几乎无锁重写表
   pg_repack -t table_name database_name
```

---

## 附录：核心概念速查表

### 隔离级别速查

| 级别 | 脏读 | 不可重复读 | 幻读 | 写偏斜 | 实现机制 |
|------|------|-----------|------|--------|---------|
| READ UNCOMMITTED | ✓ | ✗ | ✗ | ✗ | 同 RC（PG 不支持脏读）|
| READ COMMITTED | ✓ | ✗ | ✗ | ✗ | 语句级快照 |
| REPEATABLE READ | ✓ | ✓ | ✓ | ✗ | 事务级快照 |
| SERIALIZABLE | ✓ | ✓ | ✓ | ✓ | SSI |

### MVCC 核心概念

| 概念 | 说明 |
|------|------|
| xmin | 创建元组的事务 ID |
| xmax | 删除元组的事务 ID（0 表示活跃）|
| ctid | 元组物理位置（页号, 偏移）|
| 快照 | 事务可见的数据版本范围 |
| CLOG | 记录事务提交状态 |
| VACUUM | 清理死元组，回收空间 |

### PostgreSQL vs MySQL 速查

| 特性 | PostgreSQL | MySQL InnoDB |
|------|------------|--------------|
| 默认隔离级别 | RC | RR |
| 脏读支持 | 不支持 | 支持 |
| MVCC 存储 | 堆表 | Undo Log |
| 写冲突处理 | 回滚重试 | 等待锁 |
| 间隙锁 | 无 | 有 |
| SERIALIZABLE | SSI | 2PL |
| 版本清理 | VACUUM | Purge 线程 |

---

*本文档涵盖了 PostgreSQL 事务隔离级别和并发控制的核心知识，以及与 MySQL 的详细对比。如有错误或建议，欢迎指正。*
