---
name: git-session-commit
description: 在 WBR 仓库中执行安全的 git 提交流程。用于用户要求“提交一下”“帮我 commit”“做 git 提交”这类任务时，先清空暂存区，再只提交当前 session 内由当前 agent 涉及的文件，避免把用户自己的改动或其他未审查文件带进提交，并使用中文提交信息概括修改内容。
---

# Git Session Commit

## 概述

把提交动作收敛成固定流程：先清空暂存区，再只暂存当前 session 明确涉及的文件，最后用中文总结修改内容并提交。不要使用 `git add .`、`git commit -a` 或其他会把无关改动一起带走的方式。

## 提交流程

1. 先查看当前状态：

```powershell
git status --short
```

2. 清空暂存区：

```powershell
git restore --staged .
```

3. 明确“当前 session 涉及的文件”集合。

只把这几类文件算进来：

- 当前 agent 在这次 session 里新建、修改或删除的文件。
- 为完成这次任务而同步更新、且已经确认属于同一变更集的配套文件，例如被规则要求同步维护的 `AGENTS.md`。

不要把这些文件算进来：

- 用户在本地原本就有的未提交改动。
- 当前任务无关、只是同一仓库里碰巧处于脏状态的文件。
- 没有亲自检查过 diff 的文件。

4. 只按显式路径暂存本次 session 文件。

普通新增或修改：

```powershell
git add -- path\\to\\file1 path\\to\\file2
```

如果包含删除，使用显式路径的 `-A`：

```powershell
git add -A -- path\\to\\file1 path\\to\\deleted-file path\\to\\dir
```

5. 复核暂存内容，只允许出现本次 session 文件：

```powershell
git diff --cached --name-status
git diff --cached
```

6. 用中文提交，提交信息要概括这次修改内容，避免空泛标题。

示例：

```powershell
git commit -m "新增 git 提交流程 skill"
git commit -m "补充单因子回测与相关性分析 skill"
git commit -m "修复 benchmark 日期过滤并更新说明"
```

7. 在最终回复里报告提交结果，并用中文总结修改内容。

## 判断规则

- 如果无法可靠地区分哪些文件属于当前 session，不要猜测，也不要扩大暂存范围；先缩小到确定属于本次工作的文件，必要时再向用户确认。
- 如果仓库里已有其他 staged 文件，仍然先执行 `git restore --staged .`，再重新按本 skill 的文件集合暂存。
- 如果只改了一个文件，也照样先清空暂存区，再按显式路径暂存，不要省略这一步。

## 输出要求

- 提交信息使用中文。
- 给用户的总结也使用中文。
- 总结聚焦“这次改了什么”，不要把未提交的其他脏文件混进描述。
