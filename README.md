# inverted-index

把文档分词后建立倒排表，支持多词项合取检索与文档删除，索引可落盘后重载。

## 依赖

仅标准库（Python 3.10+）。

## 安装与运行

无需安装，直接以模块方式运行（目录参数统一为 `--root`）：

```bash
python3 -m inverted_index --root ./state init
```

子命令：`init`、`add <doc_id> <text>`、`get <doc_id>`、`update <doc_id> <text>`、`delete <doc_id>`、`query <term> [term ...]`、`rank <term> [term ...]`、`phrase <text>`、`terms`、`stats`、`reload`、`report`。

## 公开接口

`inverted_index.InvertedIndex(root)`：

- `init() -> None` 建立空索引。
- `add(doc_id, text) -> int` 分词后写入文档，返回文档总数。
- `get(doc_id) -> str | None` 取回原文档文本。
- `update(doc_id, text) -> int` 替换已有文档的原文并按当前分词规则重建其倒排项，文档总数不变，返回替换后的总数；文档不存在抛出 `KeyError`。
- `delete(doc_id) -> bool` 删除文档并清掉它的倒排项。
- `query(terms) -> list[dict]` 返回同时包含全部词项的文档，按 `(命中词项数, doc_id)` 稳定排序。
- `rank(terms) -> list[dict]` 按 TF-IDF 求和打分，返回 `{id, matched, score}`，按 `(score 降序, id 升序)` 排序。
- `phrase(text) -> list[dict]` 精确短语检索：返回 `{id, positions}`，按文档 id 升序；positions 是短语首个词项在文档词项序列中的从 0 开始位置，升序，重复出现保留多个位置。
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
