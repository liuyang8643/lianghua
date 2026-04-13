---
name: git-session-commit
description: 在 WBR 仓库中执行安全的 git 提交流程。用于用户要求“提交一下”“帮我 commit”“做 git 提交”这类任务时，先清空暂存区，再只提交当前 session 内由当前 agent 涉及的文件，避免把用户自己的改动或其他未审查文件带进提交；如果当前分支跟踪远程分支，则先 fetch 上游，并在非主干整合场景下优先 rebase 到最新上游，而在需要把工作合入主干时使用 merge；冲突时优先做可证明正确的合并，遇到双方都合理的同块修改时交给用户裁决。
---

# Git Session Commit

## 概述

把提交动作收敛成固定流程：先清空暂存区，再只暂存当前 session 明确涉及的文件；如果当前分支存在上游分支，则先 fetch 上游。默认在非主干整合场景下优先 rebase 到最新上游；只有当当前任务明确是在把工作合入主干时，才使用 merge。冲突时优先解决机械或意图单一的差异；如果同一块内容两边修改都合理且需要业务判断，则暂停并交给用户裁决。不要使用 `git add .`、`git commit -a` 或其他会把无关改动一起带走的方式。

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

6. 检查当前分支是否存在上游分支。

```powershell
git rev-parse --abbrev-ref --symbolic-full-name "@{u}"
```

- 如果命令失败，说明当前分支没有配置上游分支，后续跳过 fetch / rebase。
- 如果命令成功，记下上游分支名，例如 `origin/main` 或 `origin/feature/foo`，并先 fetch 对应远程：

```powershell
git fetch --prune origin
```

这里的 `origin` 只是示例，应替换为上游分支实际对应的 remote 名称。

7. 用中文提交，提交信息要概括这次修改内容，避免空泛标题。

示例：

```powershell
git commit -m "新增 git 提交流程 skill"
git commit -m "补充单因子回测与相关性分析 skill"
git commit -m "修复 benchmark 日期过滤并更新说明"
```

8. 如果第 6 步检测到了上游分支，先判断当前任务是在“同步当前分支”还是“把当前工作合入主干”。

- 如果只是让当前分支跟上上游，或者没有明确要求合入主干，默认走 rebase。
- 如果用户明确要求把当前工作合入 `main`、`master`、`trunk` 或仓库约定的主干分支，走 merge，不要把这一步改写成 rebase。

9. 在执行 rebase 或 merge 之前，先确认提交后工作树是否干净。

```powershell
git status --short
```

- 如果工作树已经干净，且第 8 步判定为 rebase，执行：

```powershell
git rebase origin/main
```

这里的 `origin/main` 只是示例，应替换为第 6 步检测到的真实上游分支名。
- 如果工作树已经干净，且第 8 步判定为“合入主干”，先更新目标主干分支，再使用 merge 把当前工作合进去。命令形态示例：

```powershell
git fetch --prune origin
git checkout main
git merge --no-ff feature/my-branch
```

这里的 `main` 和 `feature/my-branch` 都只是示例，应替换为实际主干分支和当前工作分支名。是否保留 `--no-ff` 取决于仓库既有约定；如果仓库已有明确 merge 策略，遵循仓库约定。
- 如果工作树里仍有未纳入本次提交的本地改动，不要为了 rebase 自动 stash 或挪动这些未审查改动；先向用户说明“本地提交已完成，但由于工作树不干净，未自动 rebase 到上游”。

10. 如果 rebase 或 merge 发生冲突，先分析并尽量自行解决。

- 先查看冲突文件和状态：

```powershell
git status --short
git diff --name-only --diff-filter=U
```

- 对于一眼能证明该怎么合并的情况，例如一侧只是格式化、重命名后的等价改动、导入顺序调整、注释或样板更新，而另一侧保留了实质逻辑，应直接编辑冲突文件解决，随后：

```powershell
git add -- path\\to\\resolved-file
git rebase --continue
```

如果当前是在 merge 流程里，则改为完成冲突解决后执行：

```powershell
git add -- path\\to\\resolved-file
git commit
```

- 如果同一块内容两边都做了实质修改，而且两边看起来都合理，需要产品、策略或业务取舍，停止自动解决并交给用户裁决。此时要明确指出冲突文件、冲突点、两边各自主张了什么，不要靠猜测选边。

11. 在最终回复里报告提交结果，并用中文总结修改内容，同时说明是否执行了 fetch / rebase / merge，以及是否存在等待用户裁决的冲突。

## 判断规则

- 如果无法可靠地区分哪些文件属于当前 session，不要猜测，也不要扩大暂存范围；先缩小到确定属于本次工作的文件，必要时再向用户确认。
- 如果仓库里已有其他 staged 文件，仍然先执行 `git restore --staged .`，再重新按本 skill 的文件集合暂存。
- 如果只改了一个文件，也照样先清空暂存区，再按显式路径暂存，不要省略这一步。
- 如果当前分支没有上游分支，不要猜测远程目标；直接跳过 fetch / rebase。
- 如果任务目标是把工作合入主干，优先使用 merge；不要为了线性历史把“合入主干”改写成 rebase。
- 如果任务目标不是合入主干，优先使用 rebase 而不是 merge 来同步上游更新。
- 如果提交后工作树仍有未纳入本次提交的改动，不要为了完成 rebase 自动 stash、搬运或重写这些改动，除非用户明确要求。
- 遇到 rebase 冲突时，只有在合并结果可以基于代码事实自证正确时才自行解决；如果冲突需要取舍判断，就交给用户。

## 输出要求

- 提交信息使用中文。
- 给用户的总结也使用中文。
- 总结聚焦“这次改了什么”，不要把未提交的其他脏文件混进描述。
- 说明本次是否检测到上游分支、是否执行了 fetch / rebase / merge，以及是否因为冲突或工作树不干净而停在中间状态。
