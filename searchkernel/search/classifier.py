import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

RelationshipDirection = Literal["incoming", "outgoing", "both"]


class QueryType(Enum):
    FACTUAL = auto()
    NAVIGATIONAL = auto()
    EXPLORATORY = auto()


_CAMEL_CASE_PATTERN = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")
_SNAKE_CASE_PATTERN = re.compile(r"\b[a-z]+_[a-z_]+\b")
_BACKTICK_PATTERN = re.compile(r"`[^`]+`")
_VERSION_PATTERN = re.compile(r"\b[vV]?\d+\.\d+(?:\.\d+)?(?:-\w+)?\b")
_QUOTED_PHRASE_PATTERN = re.compile(r'"[^"]+"|\'[^\']+\'')
_ARTIFACT_PATTERN = re.compile(
    r"\b[A-Za-z0-9]+(?:[_:./\\-][A-Za-z0-9]+)+\b"
)
_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:call(?:er|ee)?s?|depend(?:s|ency|encies)?|"
    r"import(?:s|ed)?|referenc(?:e|es|ed)|relat(?:ed|es)|"
    r"link(?:ed|s)?|connect(?:ed|s)?|neighbor(?:s|ing)?|"
    r"adjacent|upstream|downstream|inbound|outbound|"
    r"embed(?:s|ded|ding)?|transclud(?:e|es|ed|ing)?|"
    r"similar|point(?:s|ed)?\s+to)\b",
    re.IGNORECASE,
)
_INBOUND_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:inbound|upstream)\b|"
    r"^\s*(?:which|what)\s+(?:pages|documents|notes)\s+"
    r"(?:link(?:s)?\s+to|are\s+linked\s+from|point(?:s)?\s+to|"
    r"embed(?:s|ded|ding)?|transclud(?:e|es|ed|ing)?)\b|"
    r"\b(?:pages|documents|notes)\s+that\s+"
    r"(?:embed(?:s|ded|ding)?|transclud(?:e|es|ed|ing)?)\b",
    re.IGNORECASE,
)
_OUTBOUND_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:outbound|downstream)\b|"
    r"\b(?:does|do)\b.*\b(?:link(?:s|ed)?\s+to|"
    r"embed(?:s|ded|ding)?|transclud(?:e|es|ed|ing)?)\b",
    re.IGNORECASE,
)
_RELATIONSHIP_TARGET_PATTERNS = (
    (re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes|files)\s+"
        r"are\s+neighbors\s+of\s+(?P<target>.+?)\s*[?.!]?$",
        re.IGNORECASE,
    ), "both"),
    (re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes|files)\s+"
        r"(?:link(?:s)?\s+to(?:\s+or\s+depend(?:s)?\s+on)?|"
        r"depend(?:s)?\s+on(?:\s+or\s+link(?:s)?\s+to)?|"
        r"are\s+linked\s+from|point(?:s)?\s+to|"
        r"embed(?:s|ded|ding)?|transclud(?:e|es|ed|ing)?)\s+"
        r"(?P<target>.+?)\s*[?.!]?$",
        re.IGNORECASE,
    ), "incoming"),
    (re.compile(
        r"^show\s+me\s+(?:pages|documents|notes|files)\s+that\s+"
        r"(?:link(?:s)?\s+to|depend(?:s)?\s+on|embed(?:s|ded|ding)?|"
        r"transclud(?:e|es|ed|ing)?)\s+(?P<target>.+?)\s*[?.!]?$",
        re.IGNORECASE,
    ), "incoming"),
    (re.compile(
        r"^what\s+links\s+to\s+(?P<target>.+?)\s*[?.!]?$",
        re.IGNORECASE,
    ), "incoming"),
    (re.compile(
        r"^(?:which|what)\s+(?:pages|documents|notes|files)\s+does\s+"
        r"(?P<target>.+?)\s+(?:link(?:s)?\s+to|depend(?:s)?\s+on|"
        r"point(?:s)?\s+to|embed(?:s|ded|ding)?|"
        r"transclud(?:e|es|ed|ing)?)\s*[?.!]?$",
        re.IGNORECASE,
    ), "outgoing"),
)

_NAVIGATIONAL_KEYWORDS = frozenset(
    [
        "section",
        "chapter",
        "guide",
        "tutorial",
        "documentation",
        "doc",
        "docs",
        "page",
        "article",
        "overview",
    ]
)
_NAVIGATIONAL_PHRASES = [
    re.compile(r"\bin\s+the\s+\w+", re.IGNORECASE),
    re.compile(r"\[\[.+?\]\]"),
]

_QUESTION_WORDS = frozenset(
    [
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "explain",
        "describe",
        "tell",
    ]
)


@dataclass(frozen=True, slots=True)
class QuerySignals:
    """Stable lexical signals used by the deterministic query router."""

    artifact: bool
    quoted: bool
    navigational: bool
    exploratory: bool
    relationship: bool
    graph_direction: RelationshipDirection = "outgoing"
    relationship_intent: "RelationshipIntent | None" = None

    @property
    def names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.artifact:
            names.append("artifact")
        if self.quoted:
            names.append("quoted")
        if self.navigational:
            names.append("navigational")
        if self.exploratory:
            names.append("exploratory")
        if self.relationship:
            names.append("relationship")
        return tuple(names)


@dataclass(frozen=True, slots=True)
class RelationshipIntent:
    target: str
    direction: RelationshipDirection


def parse_relationship_intent(query: str) -> RelationshipIntent | None:
    normalized = " ".join(query.strip().split()).strip(" .?!")
    normalized = re.sub(
        r"\s+and\s+what\s+do\s+their\s+neighbors?\s+explain$",
        "",
        normalized,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" .?!")
    for pattern, direction in _RELATIONSHIP_TARGET_PATTERNS:
        match = pattern.match(normalized)
        if match is not None:
            target = re.sub(
                r"^(?:the|a|an)\s+", "", match.group("target").strip(),
                count=1, flags=re.IGNORECASE,
            ).strip(" .?!")
            if target:
                return RelationshipIntent(target, direction)
    return None


def _has_factual_signals(query: str) -> bool:
    if "[[" in query and "]]" in query:
        return False
    if _CAMEL_CASE_PATTERN.search(query):
        return True
    if _SNAKE_CASE_PATTERN.search(query):
        return True
    if _BACKTICK_PATTERN.search(query):
        return True
    if _VERSION_PATTERN.search(query):
        return True
    if _ARTIFACT_PATTERN.search(query):
        return True
    return bool(_QUOTED_PHRASE_PATTERN.search(query))


def _has_navigational_signals(query: str) -> bool:
    words = set(query.lower().split())
    if words & _NAVIGATIONAL_KEYWORDS:
        return True
    for pattern in _NAVIGATIONAL_PHRASES:
        if pattern.search(query):
            return True
    return False


def _has_exploratory_signals(query: str) -> bool:
    words = query.lower().split()
    if not words:
        return False
    if words[0] in _QUESTION_WORDS:
        return True
    return bool(query.strip().endswith("?"))


def analyze_query(query: str) -> QuerySignals:
    """Extract routing signals without calling a generative model."""
    relationship_intent = parse_relationship_intent(query)
    return QuerySignals(
        artifact=_has_factual_signals(query),
        quoted=bool(_QUOTED_PHRASE_PATTERN.search(query)),
        navigational=_has_navigational_signals(query),
        exploratory=_has_exploratory_signals(query),
        relationship=bool(_RELATIONSHIP_PATTERN.search(query)),
        graph_direction=(
            relationship_intent.direction
            if relationship_intent is not None
            else _relationship_direction(query)
        ),
        relationship_intent=relationship_intent,
    )


def _relationship_direction(query: str) -> RelationshipDirection:
    if _INBOUND_RELATIONSHIP_PATTERN.search(query) and not (
        _OUTBOUND_RELATIONSHIP_PATTERN.search(query)
    ):
        return "incoming"
    return "outgoing"


def classify_query(query: str) -> QueryType:
    if _has_factual_signals(query):
        return QueryType.FACTUAL
    if _has_navigational_signals(query):
        return QueryType.NAVIGATIONAL
    if _has_exploratory_signals(query):
        return QueryType.EXPLORATORY
    return QueryType.EXPLORATORY


def get_adaptive_weights(
    query_type: QueryType,
    base_semantic: float,
    base_keyword: float,
    base_graph: float,
) -> tuple[float, float, float]:
    if query_type == QueryType.FACTUAL:
        return (base_semantic, base_keyword * 1.5, base_graph)
    if query_type == QueryType.NAVIGATIONAL:
        return (base_semantic, base_keyword, base_graph * 1.5)
    return (base_semantic * 1.3, base_keyword, base_graph)
