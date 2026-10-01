# inverted-index

把文档分词后建立倒排表，支持多词项合取检索与文档删除，索引可落盘后重载。

## 依赖

仅标准库（Python 3.10+）。

## 安装与运行

无需安装，直接以模块方式运行（目录参数统一为 `--root`）：

```bash
python3 -m inverted_index --root ./state init
```

子命令：`init`、`add <doc_id> <text>`、`get <doc_id>`、`update <doc_id> <text>`、`delete <doc_id>`、`apply <operations>`、`query <term> [term ...]`、`rank <term> [term ...]`、`bm25 <term> [term ...] [--filter <expression>] [--k1 <float>] [--b <float>]`、`phrase <text>`、`near <left> <right> <max_gap>`、`search <expression>`、`expand <pattern>`、`terms`、`stats`、`reload`、`report`。`apply` 的参数是一个 JSON 数组文本，成功时标准输出一个 JSON 数组；`expand` 成功时把匹配词项列表以 JSON 数组写到标准输出。

## 公开接口

`inverted_index.InvertedIndex(root)`：

- `init() -> None` 建立空索引。
- `add(doc_id, text) -> int` 分词后写入文档，返回文档总数。
- `get(doc_id) -> str | None` 取回原文档文本。
- `update(doc_id, text) -> int` 替换已有文档的原文并重建其倒排项、词频与位置，文档总数不变，返回替换后的总数；目标文档不存在抛出 `KeyError`，`doc_id`/`text` 为空或非字符串抛出 `ValueError`。
- `delete(doc_id) -> bool` 删除文档并清掉它的倒排项。
- `apply(operations) -> list[int | bool]` 原子批量写入：`operations` 是非空 JSON 风格对象数组，按数组顺序在同一快照上执行，整批只落盘一次；后项可见前项结果（如先 delete 再 add、先 add 再 update 同一 id）。每个对象 `op` 只能是 `add`、`update`、`delete`，`doc_id` 必须是非空字符串；`add`/`update` 必须且只能再提供非空字符串 `text`，`delete` 不提供 `text`，其余字段一律拒绝。返回值逐元素对应：`add`/`update` 返回执行到该步时的文档总数，`delete` 返回该 id 当时是否存在（不存在为 `false`，与单文档 `delete` 语义一致）。空数组、元素不是对象、`op` 未知、字段缺失或多余、`doc_id`/`text` 类型错误，以及批次中 `add` 已存在或 `update` 不存在，均抛出 `ValueError`（单独调用 `update` 不存在仍抛 `KeyError`）；任一操作失败时已提交状态保持不变。
- `query(terms) -> list[dict]` 返回同时包含全部词项的文档，按 `(命中词项数, doc_id)` 稳定排序。
- `rank(terms) -> list[dict]` 按 TF-IDF 求和打分，返回 `{id, matched, score}`，按 `(score 降序, id 升序)` 排序。
- `bm25(terms, expression=None, k1=1.2, b=0.75) -> list[dict]` BM25 打分：词项只转小写并按首次出现去重，不拆词、不展开通配符；`expression` 沿用 `search` 的完整语法且为可选过滤（`None` 表示不过滤），过滤只约束返回文档，统计量（N、dl、avgdl、df）始终取整个索引，无词项文档也计入 N 与 avgdl。返回至少命中一个词项且符合过滤条件的文档，每项为 `{id, matched, score}`，matched 为命中的不同词项数，每个命中词贡献 `ln(1+(N-df+0.5)/(df+0.5))*tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))`，分数按 `round(score, 6)` 输出并按 `(score 降序, id 升序)` 排列。空索引、全部文档无词项、无命中或过滤后无候选均返回空数组。`terms` 必须是非空 list 且元素均为非空字符串；`k1` 必须是有限正数，`b` 必须是 [0,1] 内的有限数字，二者只接受 int/float 且拒绝布尔值；`expression` 非字符串且非 None、表达式不合法抛出 `ValueError`；即使无候选也会先校验全部输入。只读快照，不改写 `index.json`。
- `phrase(text) -> list[dict]` 精确短语检索：返回 `{id, positions}`，按文档 id 升序；positions 是短语首个词项在文档词项序列中的从 0 开始位置，升序，重复出现保留多个位置。
- `near(left, right, max_gap) -> list[dict]` 相邻短语近邻检索：两片段按现有 `tokenize` 语义分词，`max_gap` 为非负整数，限定两片段间严格夹着的词项数上限；两片段可任意先后出现，各自连续且不重叠。返回 `{id, occurrences}`，按文档 id 升序；occurrences 为 `[left_start, right_start]` 对（right 先出现时字段仍按左右对应），按 `(left_start, right_start)` 升序，重复位置组合全部保留，未命中文档不出现。任一片段分词后为空，或 `max_gap` 为布尔值、负数、非整数时抛出 `ValueError`。
- `search(expression) -> list[str]` 组合检索：表达式是单个字符串，由普通词项（连续 ASCII 字母、数字、下划线，按现有规则转小写）、通配词项（未被双引号包裹且含 `*` 或 `?` 的连续片段，`*` 匹配零个或多个字符、`?` 匹配恰好一个字符，必须含至少一个字面字符，命中至少一个展开词项所在的文档）、双引号短语（沿用 `phrase` 的连续词序语义，短语内仍用 `tokenize`，星号问号不作通配符）、括号和大小写敏感的 `AND`、`OR`、`NOT` 组成；`NOT` 优先于 `AND`，`AND` 优先于 `OR`，同层从左到右结合。词项命中包含该词项的文档，短语命中包含该连续词序的文档，`AND`/`OR` 取两侧集合的交/并，`NOT` 以当前索引全部文档为全集取补。返回按文档 id 升序的字符串列表，无命中返回空列表；查询只读快照，不改写 `index.json`。空表达式、括号不配对、缺少操作数、操作数之间缺少 `AND`/`OR`、`NOT` 后无操作数、连续运算符、未闭合或空的双引号短语、短语分词后为空、通配词项没有字面字符以及不支持的字符均抛出 `ValueError`。
- `expand(pattern) -> list[str]` 通配展开：返回索引中所有匹配 `pattern` 的词项，按 Unicode 码点升序且去重，无匹配返回空列表；`pattern` 按现有词项规则转小写，只允许 ASCII 字母、数字、下划线和通配符 `*`、`?`。空模式、非字符串模式或含非法字符抛出 `ValueError`；只读快照，不改写 `index.json`。
- `terms() -> list[str]` 升序返回全部词项。
- `stats() -> dict` 返回文档数、词项数与倒排项数。
- `reload() -> None` 从落盘的索引快照重新载入。

## 约定

- 所有写操作立即持久化；进程被杀死后 `recover`/`init` 之外的重开不得丢失已确认的写。
- 非法输入抛出 `ValueError`，未知标识抛出 `KeyError`。
- 退出码：0 成功，1 存储或校验错误，2 用法错误。

## 限制

- 分词只按非字母数字切分并转小写，没有词干化或停用词。
- 排序为简单 TF-IDF 求和（`rank`）与 BM25（`bm25`），不含字段权重。
- 没有段合并与增量落盘，全量重写。

## 语料

`corpus.md` 是本项目对应的技术标签语料（GitHub 热门技术标签）。`report` 子命令输出本域声明覆盖的分类与标签。
