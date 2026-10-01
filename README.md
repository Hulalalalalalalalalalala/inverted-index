# inverted-index

把文档分词后建立倒排表，支持多词项合取检索与文档删除，索引可落盘后重载。

## 依赖

仅标准库（Python 3.10+）。

## 安装与运行

无需安装，直接以模块方式运行（目录参数统一为 `--root`）：

```bash
python3 -m inverted_index --root ./state init
```

子命令：`init`、`add <doc_id> <text>`、`get <doc_id>`、`update <doc_id> <text>`、`delete <doc_id>`、`apply <operations>`、`query <term> [term ...]`、`rank <term> [term ...]`、`phrase <text>`、`near <left> <right> <max_gap>`、`search <expression>`、`expand <pattern>`、`terms`、`stats`、`reload`、`report`。`apply` 的参数是一个 JSON 数组文本，成功时标准输出一个 JSON 数组。

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
- `phrase(text) -> list[dict]` 精确短语检索：返回 `{id, positions}`，按文档 id 升序；positions 是短语首个词项在文档词项序列中的从 0 开始位置，升序，重复出现保留多个位置。
- `near(left, right, max_gap) -> list[dict]` 相邻短语近邻检索：两片段按现有 `tokenize` 语义分词，`max_gap` 为非负整数，限定两片段间严格夹着的词项数上限；两片段可任意先后出现，各自连续且不重叠。返回 `{id, occurrences}`，按文档 id 升序；occurrences 为 `[left_start, right_start]` 对（right 先出现时字段仍按左右对应），按 `(left_start, right_start)` 升序，重复位置组合全部保留，未命中文档不出现。任一片段分词后为空，或 `max_gap` 为布尔值、负数、非整数时抛出 `ValueError`。
- `search(expression) -> list[str]` 组合检索：表达式是单个字符串，由普通词项（连续 ASCII 字母、数字、下划线，按现有规则转小写）、通配词项（未被双引号包裹、含 `*` 或 `?` 的连续片段；`*` 匹配零个或多个字符，`?` 匹配恰好一个字符，至少含一个字面字符）、双引号短语（沿用 `phrase` 的连续词序语义，短语内仍用 `tokenize`，不把 `*`/`?` 当通配符）、括号和大小写敏感的 `AND`、`OR`、`NOT` 组成；`NOT` 优先于 `AND`，`AND` 优先于 `OR`，同层从左到右结合。词项命中包含该词项的文档，通配词项命中其任一展开词项所在的文档（无展开词项为空集合），短语命中包含该连续词序的文档，`AND`/`OR` 取两侧集合的交/并，`NOT` 以当前索引全部文档为全集取补。返回按文档 id 升序的字符串列表，无命中返回空列表；查询只读快照，不改写 `index.json`。空表达式、括号不配对、缺少操作数、操作数之间缺少 `AND`/`OR`、`NOT` 后无操作数、连续运算符、未闭合或空的双引号短语、短语分词后为空、通配词项没有字面字符以及不支持的字符均抛出 `ValueError`。
- `expand(pattern) -> list[str]` 词项通配展开：`pattern` 是单个字符串，按词项规则转小写，`*` 匹配零个或多个字符，`?` 匹配恰好一个字符；返回索引中所有匹配词项，按 Unicode 码点顺序升序且不重复，无匹配返回空列表。`pattern` 为空、不是字符串或包含空白、通配符与 ASCII 字母数字下划线之外的字符时抛出 `ValueError`。
- `terms() -> list[str]` 升序返回全部词项。
- `stats() -> dict` 返回文档数、词项数与倒排项数。
- `reload() -> None` 从落盘的索引快照重新载入。

## 约定

- 所有写操作立即持久化；进程被杀死后 `recover`/`init` 之外的重开不得丢失已确认的写。
- 非法输入抛出 `ValueError`，未知标识抛出 `KeyError`。
- 退出码：0 成功，1 存储或校验错误，2 用法错误。

## 限制

- 分词只按非字母数字切分并转小写，没有词干化或停用词。
- 排序为简单 TF-IDF 求和（`rank`），不含 BM25 或字段权重。
- 没有段合并与增量落盘，全量重写。

## 语料

`corpus.md` 是本项目对应的技术标签语料（GitHub 热门技术标签）。`report` 子命令输出本域声明覆盖的分类与标签。
