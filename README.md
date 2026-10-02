# inverted-index

把文档分词后建立倒排表，支持多词项合取检索与文档删除，索引可落盘后重载。

## 依赖

仅标准库（Python 3.10+）。

## 安装与运行

无需安装，直接以模块方式运行（目录参数统一为 `--root`）：

```bash
python3 -m inverted_index --root ./state init
```

子命令：`init`、`add <doc_id> <text>`、`get <doc_id>`、`update <doc_id> <text>`、`delete <doc_id>`、`apply <operations>`、`query <term> [term ...]`、`rank <term> [term ...]`、`bm25 <term> [term ...] [--filter <expression>] [--k1 <float>] [--b <float>]`、`phrase <text>`、`sloppy-phrase <text> [--slop <int>]`、`window <text> [--max-gap <int>]`、`near <left> <right> <max_gap>`、`search <expression>`、`highlight <expression>`、`snippets <expression> [--context <int>] [--max-fragments <int>]`、`expand <pattern>`、`fuzzy <term> [--max-distance <int>]`、`terms`、`stats`、`reload`、`report`。`apply` 的参数是一个 JSON 数组文本，成功时标准输出一个 JSON 数组；`expand` 成功时把匹配词项列表以 JSON 数组写到标准输出；`fuzzy` 成功时把 `{term, distance}` 对象数组以 JSON 写到标准输出。

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
- `bm25(terms, expression=None, k1=1.2, b=0.75) -> list[dict]` BM25 打分：词项只转小写并按首次出现去重，不拆词、不展开通配符；`expression` 沿用 `search` 的完整语法且为可选过滤（`None` 表示不过滤），过滤只约束返回文档，统计量（N、dl、avgdl、df）始终取整个索引，无词项文档也计入 N 与 avgdl。返回至少命中一个词项且符合过滤条件的文档，每项为 `{id, matched, score}`，matched 为命中的不同词项数，每个命中词贡献 `ln(1+(N-df+0.5)/(df+0.5))*tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))`，分数按 `round(score, 6)` 输出并按 `(score 降序, id 升序)` 排列。空索引、全部文档无词项、无命中或过滤后无候选均返回空数组。`terms` 必须是非空 list 且元素均为非空字符串；`k1` 必须是有限正数，`b` 必须是 [0,1] 内的有限数字，二者只接受 int/float 且拒绝布尔值；任意大小的 Python 正整数 `k1` 均合法（含 `10**1000`，按其趋于无穷的极限值参与评分），越界的超大整数 `b` 抛 `ValueError`；每个返回 score 必为有限数字（内部对超大 `k1` 做尺度归一，不会溢出为 inf），极小正浮点 `k1` 照常评分；非有限浮点、非正 `k1`、越界 `b`、错误类型、非法词项或表达式统一抛 `ValueError`，不会泄漏 `OverflowError` 或 `ZeroDivisionError`；`expression` 非字符串且非 None、表达式不合法同样抛出；即使无候选也会先校验全部输入。只读快照，不改写 `index.json`。
- `phrase(text) -> list[dict]` 精确短语检索：返回 `{id, positions}`，按文档 id 升序；positions 是短语首个词项在文档词项序列中的从 0 开始位置，升序，重复出现保留多个位置。
- `sloppy_phrase(text, slop=0) -> list[dict]` 有序宽松短语检索：查询文本按现有 `tokenize` 规则分词（不展开通配符），查询词按原次序匹配到文档词项序列中严格递增、从 0 开始的位置，重复词占用不同位置；相邻匹配位置之间夹着的词项总数不超过 `slop`。返回 `{id, occurrences}`，按文档 id 升序；occurrences 保存全部匹配的位置数组，去重后按字典序升序，同起点的不同组合与重叠组合均保留；单词查询返回每次出现的单元素数组；`slop` 为 0 时命中文档与 `phrase` 一致。`slop` 只接受 0 到 2147483647 的整数并拒绝布尔值；文本非字符串或分词后为空、`slop` 类型或范围错误均抛 `ValueError`，先校验全部输入再读快照；空索引或无命中返回空数组；只读快照，不改写 `index.json`。
- `window(text, max_gap=0) -> list[dict]` 无序最小覆盖窗口检索：查询文本按现有 `tokenize` 规则分词（不展开通配符），窗口是文档词项序列的连续区间，按多重集覆盖查询各词及其重复次数，同一位置不能重复使用；仅当不存在仍能覆盖的严格子区间（等价地，去掉首词或末词后均不再覆盖）且窗口词项数减去查询词项数不超过 `max_gap` 时保留。返回 `{id, windows}`，按文档 id 的 Unicode 码点升序；windows 为半开 `[start, end)` 词项位置二元数组，从 0 开始、终点不包含，去重后按起点、终点升序，保留重叠窗口；单词查询返回每次出现的单词窗口。`max_gap` 只接受 0 到 2147483647 的整数并拒绝布尔值；文本非字符串或分词后为空、`max_gap` 类型或范围错误均抛 `ValueError`，先校验全部输入再读快照；空索引或无命中返回空数组；只读快照、直接使用旧快照，不改写 `index.json`。
- `near(left, right, max_gap) -> list[dict]` 相邻短语近邻检索：两片段按现有 `tokenize` 语义分词，`max_gap` 为非负整数，限定两片段间严格夹着的词项数上限；两片段可任意先后出现，各自连续且不重叠。返回 `{id, occurrences}`，按文档 id 升序；occurrences 为 `[left_start, right_start]` 对（right 先出现时字段仍按左右对应），按 `(left_start, right_start)` 升序，重复位置组合全部保留，未命中文档不出现。任一片段分词后为空，或 `max_gap` 为布尔值、负数、非整数时抛出 `ValueError`。
- `search(expression) -> list[str]` 组合检索：表达式是单个字符串，由普通词项（连续 ASCII 字母、数字、下划线，按现有规则转小写）、通配词项（未被双引号包裹且含 `*` 或 `?` 的连续片段，`*` 匹配零个或多个字符、`?` 匹配恰好一个字符，必须含至少一个字面字符，命中至少一个展开词项所在的文档）、双引号短语（沿用 `phrase` 的连续词序语义，短语内仍用 `tokenize`，星号问号不作通配符）、近邻原子条件 `NEAR("left","right",distance)`（两片段必须各自以双引号包裹且分词后非空，沿用 `near` 的分词与距离口径：各自连续出现且互不重叠，允许任意先后，严格夹在两段之间的词项数不大于 distance 即命中；distance 为 0 到 2147483647 的 ASCII 十进制整数，允许前导零；片段内逗号与括号属于文本，双引号不可嵌入或转义，星号问号不展开；函数名只接受大写 `NEAR`，名称与 `(` 之间、括号内参数周围与逗号两侧允许空白；不带括号的独立 `NEAR` 仍按普通词项处理）、容错原子条件 `FUZZY("term",distance)`（恰好两个参数：词项必须双引号包裹且遵守 `fuzzy` 的词项规则——非空 ASCII 字母、数字、下划线，转小写比较，通配符不特殊；distance 为 0 到 2 的 ASCII 十进制整数，允许前导零；命中包含任一展开词项的文档；函数名只接受大写 `FUZZY`，名称与 `(` 之间及参数周围允许空白；不带括号的独立 `FUZZY` 仍按普通词项处理）、有序宽松短语原子条件 `SLOP("text",slop)`（恰好两个参数：文本必须双引号包裹，引号内按 `tokenize` 分词且分词后非空，星号问号不展开，双引号不可嵌入或转义；slop 为 0 到 2147483647 的 ASCII 十进制整数，允许前导零；命中语义同 `sloppy_phrase`——查询词按原次序落在严格递增的位置且相邻匹配位置间夹着的词项总数不超过 slop；函数名只接受大写 `SLOP`，名称与 `(` 之间及参数周围允许空白；不带括号的独立 `SLOP` 仍按普通词项处理）、无序最小覆盖窗口原子条件 `WINDOW("text",gap)`（恰好两个参数：文本必须双引号包裹，引号内按 `tokenize` 分词且分词后非空，星号问号不展开，双引号不可嵌入或转义；窗口是文档词项序列的连续区间，按多重集覆盖查询各词及其重复次数且同一位置不可重复使用，仅当不存在仍能覆盖的严格子区间、且窗口词项数减去查询词项数不大于 gap 时命中，命中文档与 `window` 接口一致；gap 为 0 到 2147483647 的 ASCII 十进制整数，允许前导零；函数名只接受大写 `WINDOW`，名称与 `(` 之间及参数、逗号周围允许空白；不带括号的独立 `WINDOW` 仍按普通词项处理）、括号和大小写敏感的 `AND`、`OR`、`NOT` 组成；`NOT` 优先于 `AND`，`AND` 优先于 `OR`，同层从左到右结合。词项命中包含该词项的文档，短语命中包含该连续词序的文档，`AND`/`OR` 取两侧集合的交/并，`NOT` 以当前索引全部文档（含无词项文档）为全集取补。返回按文档 id 升序的字符串列表（同一文档即使多处命中也只出现一次），无命中返回空列表；查询只读快照，不改写 `index.json`。非字符串、NEAR 片段未加双引号或分词后为空、参数数目错误、距离不符合 0..2147483647 的 ASCII 十进制整数规则、分隔符缺失、括号或引号未闭合、空表达式、括号不配对、缺少操作数、操作数之间缺少 `AND`/`OR`、`NOT` 后无操作数、连续运算符、空的双引号短语、短语分词后为空、通配词项没有字面字符以及不支持的字符均抛出 `ValueError`；即使索引为空也先校验表达式。
- `highlight(expression) -> list[dict]` 原文高亮定位：语法、筛选语义与命中文档集合与 `search` 完全一致（同样先校验表达式再读快照，空索引也先校验），按文档 id 升序返回，每项只含 `id`、`text`、`spans`：`text` 为保存的完整原文（不改大小写、空白与标点），`spans` 为半开区间 `[start, end)` 对数组，位置按原文 Unicode 码点从 0 计数（非 UTF-8 字节、非 UTF-16 单元）。普通词项标出每次出现的完整词项；通配条件标出每个展开词项的全部出现位置；短语标出每次连续匹配从首词开头到末词结尾的整个区间（保留中间原文的标点与空白）；`NEAR` 条件标出所有满足现有距离与非重叠规则的片段组合中的左右片段，每个片段各占一个区间，不把片段间隔纳入高亮；`FUZZY` 条件标出编辑距离内每个展开词项的全部出现位置；`SLOP` 条件把每次有序匹配标为从首个命中词开头到末个命中词结尾的一个完整原文区间（包含夹词、标点与空白）；`WINDOW` 条件把每个保留窗口标为从首词开头到末词结尾的一个完整原文区间（包含夹词与标点）。`AND` 整体成立时收集两侧区间；`OR` 只收集在该文档中成立的分支；`NOT` 只参与筛选、其内部不产生区间，双重否定也不恢复高亮——因此仅靠否定条件命中的文档返回原文但 `spans` 为空。收集结果先去重，再把重叠或首尾相接（`end == next_start`）的区间合并，最终按起点升序输出。非字符串、空白表达式或其他非法语法统一抛出 `ValueError`；合法表达式无命中返回空数组；读快照时存储缺失抛 `FileNotFoundError`，快照 JSON 语法、结构或一致性错误抛 `SnapshotError`，底层读文件失败透传 `OSError`；只读不改写 `index.json`，旧快照可直接使用。
- `snippets(expression, context=20, max_fragments=3) -> list[dict]` 检索摘要：语法、筛选语义、命中文档集合与高亮区间和 `highlight` 完全一致（同样先校验全部参数再读快照，空索引也先校验），按文档 id 升序返回，每项只含 `id`、`fragments`。每个高亮区间向左右各扩展最多 `context` 个 Unicode 码点（越界截到原文边界），扩展窗口重叠或首尾相接时传递性合并，按起点升序保留前 `max_fragments` 个窗口。每个片段只含 `start`、`end`、`text`、`spans`：`start`/`end` 是窗口在原文中的半开区间（按码点计数），`text` 是对应原文切片（保留原文大小写、标点、空白与换行，不插入省略号或 HTML），`spans` 是所有与窗口相交的高亮区间截到窗口边界后换算成相对片段起点的半开区间数组，沿用 `highlight` 的去重合并规则。仅靠否定条件命中（无高亮区间）的文档返回原文开头最多 `2*context+1` 个码点的单个片段且 `spans` 为空；原文为空则 `fragments` 为空数组。合法表达式无命中返回空数组；同次调用的全部结果来自同一快照。`context` 只接受非负整数，`max_fragments` 只接受正整数，均拒绝布尔值；参数类型或范围错误、非字符串或空白表达式及非法语法统一抛出 `ValueError`；存储缺失抛 `FileNotFoundError`，快照错误抛 `SnapshotError`，底层读文件失败透传 `OSError`；只读不改写 `index.json`，旧快照可直接使用。
- `expand(pattern) -> list[str]` 通配展开：返回索引中所有匹配 `pattern` 的词项，按 Unicode 码点升序且去重，无匹配返回空列表；`pattern` 按现有词项规则转小写，只允许 ASCII 字母、数字、下划线和通配符 `*`、`?`。空模式、非字符串模式或含非法字符抛出 `ValueError`；只读快照，不改写 `index.json`。
- `fuzzy(term, max_distance=1) -> list[dict]` 编辑距离匹配：返回词典中与 `term` 的 Levenshtein 距离不超过 `max_distance` 的词项，每项只含 `term` 与 `distance`，按 `(distance, term 码点)` 升序、去重，精确匹配以距离 0 包含在内；插入、删除、替换单个字符各计一次操作，相邻字符交换不算一次操作。`term` 只接受非空的 ASCII 字母、数字、下划线字符串，转小写后比较，不分词、不去空白、不解释通配符；`max_distance` 只接受 0 到 2 的整数并拒绝布尔值。空索引或无候选返回空数组；所有参数在读取快照前校验；只读快照，不改写 `index.json`。
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
