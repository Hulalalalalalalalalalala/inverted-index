"""inverted_index: 倒排索引与词项检索."""

__version__ = "0.1.0"

#: The technical domain this package belongs to.
DOMAIN = "information-retrieval"

#: Category headings in corpus.md whose tags this domain claims.
SOURCE_CATEGORIES = ("🔍 搜索 / 内容管理 / 富文本", "🧾 元数据 / Catalog / 搜索")

from .core import InvertedIndex  # noqa: E402  (re-exported after the constants above)

__all__ = ["InvertedIndex", "DOMAIN", "SOURCE_CATEGORIES", "__version__"]
