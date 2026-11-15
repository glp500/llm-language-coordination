"""Freeze HM prose translations before gameplay; Python expressions stay intact."""

import ast
import copy
import hashlib
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "vendor/hypothetical-minds"
GAMES = {
    "cc": ("collaborative_cooking__asymmetric", "cc_hypothetical_minds"),
    "pd": ("prisoners_dilemma_in_the_matrix__repeated", "pd_hypothetical_minds"),
    "rws": ("running_with_scissors_in_the_matrix__repeated", "rws_hypothetical_minds"),
    "arena": ("running_with_scissors_in_the_matrix__arena", "rws_arena_hm"),
}
LANGUAGES = {"en": "English", "nl": "Dutch", "fr": "French", "es": "Spanish", "ar": "Arabic", "zh": "Chinese"}
RESPONSE_INSTRUCTIONS = {
    "en": "Write all explanations, strategies, hypotheses and feedback in English. Keep object IDs, function names and Python/JSON keys unchanged.",
    "nl": "Schrijf alle uitleg, strategieën, hypothesen en feedback in het Nederlands. Behoud object-ID's, functienamen en Python/JSON-sleutels ongewijzigd.",
    "fr": "Rédige toutes les explications, stratégies, hypothèses et tous les retours en français. Conserve les identifiants d'objets, les noms de fonctions et les clés Python/JSON.",
    "es": "Escribe todas las explicaciones, estrategias, hipótesis y comentarios en español. Mantén sin cambios los identificadores de objetos, los nombres de funciones y las claves Python/JSON.",
    "ar": "اكتب جميع الشروح والاستراتيجيات والفرضيات والتعليقات باللغة العربية. أبقِ معرّفات الكائنات وأسماء الدوال ومفاتيح Python/JSON دون تغيير.",
    "zh": "请用中文撰写所有解释、策略、假设和反馈。保持对象标识符、函数名和 Python/JSON 键不变。",
}
SLOTS = re.compile(r"⟦(\d+)⟧")


def sources():
    """Only the four selected HM agents and their original episode runners."""
    return [
        p
        for game, agent in GAMES.values()
        for p in (UPSTREAM / "llm_plan/agent" / game / f"{agent}.py", UPSTREAM / "environments" / f"{game}.py")
    ]


def validate(source, translated):
    """Translation must preserve expressions, numbers and quoted protocol identifiers."""
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Translation must contain text")
    if Counter(SLOTS.findall(source)) != Counter(SLOTS.findall(translated)):
        raise ValueError("Translation changed a placeholder")
    numbers = lambda text: Counter(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?", SLOTS.sub("", text)))
    if numbers(source) != numbers(translated):
        raise ValueError("Translation changed a number")
    for token in re.findall(r"['\"]([a-z][a-z_]+)['\"]", source):
        if token not in translated:
            raise ValueError(f"Translation changed protocol identifier: {token}")


class Text(ast.NodeTransformer):
    """Replace prose literals, preserving parsing strings, keys and executable expressions."""

    def __init__(self, translations=None, protected=()):
        self.translations, self.texts = translations, {}
        self.protected = protected

    def replacement(self, text):
        if not re.search(r"[A-Za-z]{2}", text) or not re.search(r"\s", text):
            return text
        if text.strip() in {"```python", "```json"} or re.fullmatch(r"(?:move_to|fire_at|interact|wait)\(.*\)", text):
            return text
        if any(
            text == token or (re.search(r"\s", token.strip()) and text.startswith(token))
            for token in self.protected
            if token.strip()
        ):
            return text
        key = hashlib.sha256(text.encode()).hexdigest()
        self.texts[key] = text
        if self.translations is None:
            return text
        if key not in self.translations:
            raise ValueError(f"Missing frozen translation: {key}")
        translated = self.translations[key]
        validate(text, translated)
        return translated

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node  # Docstrings are not agent inputs.
        return self.generic_visit(node)

    def visit_Compare(self, node):
        return node  # Comparisons define executable behaviour.

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)
        return node  # Dictionary keys are the machine protocol.

    def visit_Dict(self, node):
        node.values = [self.visit(value) for value in node.values]
        return node

    def visit_Call(self, node):
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name in {"print", "print_and_save", "write", "split", "rsplit", "find", "index", "startswith", "endswith", "replace"}:
            return node
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.replacement(node.value)
        return node

    def visit_JoinedStr(self, node):
        expressions, parts = [], []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                value.value = self.visit(value.value)
                parts.append(f"⟦{len(expressions)}⟧")
                expressions.append(value)
            else:
                parts.append(value.value)
        source = "".join(parts)
        translated = self.replacement(source)
        if translated == source:
            return node
        values = [
            copy.deepcopy(expressions[int(part)]) if index % 2 else ast.Constant(value=part)
            for index, part in enumerate(SLOTS.split(translated))
            if part
        ]
        return ast.copy_location(ast.JoinedStr(values=values), node)


def inventory(source):
    """A stable content-keyed inventory; duplicate wording is translated only once."""
    tree = ast.parse(source)
    visitor = Text(protected=control_strings(tree))
    visitor.visit(tree)
    return visitor.texts


def source_texts():
    """Combine all selected templates, with an explicit error for an uninitialized submodule."""
    if not (UPSTREAM / "main.py").is_file():
        raise ValueError("Run git submodule update --init --recursive first")
    return {key: value for path in sources() for key, value in inventory(path.read_text()).items()}


def localize(source, translations):
    """Compile-time text replacement; no translation calls occur during gameplay."""
    tree = ast.parse(source)
    return ast.fix_missing_locations(Text(translations, control_strings(tree)).visit(tree))


def control_strings(tree):
    """A tag used by comparisons or parsing must remain identical wherever it is constructed."""
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    nodes += [
        argument
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") in {"find", "split", "rsplit", "replace", "startswith", "endswith", "index", "rfind"}
        for argument in node.args
    ]
    return {
        item.value for node in nodes for item in ast.walk(node) if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def load_module(name, path, translations=None, perception=False):
    """Load pinned upstream code; replace its Ray-only environment import with our dm_env adapter."""
    source = path.read_text()
    if path.parent.name == "environments" and path.name == "prisoners_dilemma_in_the_matrix__repeated.py":
        if source.count("while step < 1200:") != 1:
            raise ValueError("Upstream PD stop condition changed; review its termination contract")
        source = source.replace("while step < 1200:", "while not done:")
    tree = localize(source, translations) if translations is not None else ast.parse(source)
    if perception:
        matches = [
            alias
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "llm_plan.env.utils"
        ]
        if len(matches) != 1:
            raise ValueError("Upstream environment import changed")
        matches[0].name = "hm"
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 -- Pinned project source; translations add no expressions.
    return module


async def prepare(directory, controller):
    """Translate static templates once, checkpointing every response for resumable preparation."""
    texts = source_texts()
    directory.mkdir(parents=True, exist_ok=True)
    for code, name in LANGUAGES.items():
        path = directory / f"{code}.json"
        corpus = json.loads(path.read_text()) if path.exists() else {"language": code, "reviewed": code == "en", "texts": {}}
        corpus["response_instruction"] = RESPONSE_INSTRUCTIONS[code]
        if set(corpus["texts"]) - set(texts):
            raise ValueError(f"Corpus contains obsolete templates: {code}")
        for key, source in texts.items():
            if key not in corpus["texts"]:
                if code == "en":
                    corpus["texts"][key] = source
                else:
                    instruction = (
                        f"Translate the supplied game-agent text into {name}. Preserve every ⟦number⟧ marker, "
                        "all numerical values and signs, quoted identifiers, Python/JSON examples, object IDs and "
                        "function names (move_to, interact, wait, fire_at). Translate prose only. "
                        'Return a JSON object with exactly one key, "text".'
                    )
                    response = await controller.complete(instruction, source, json_output=True)
                    corpus["texts"][key] = json.loads(response)["text"]
            validate(source, corpus["texts"][key])
            path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
        if set(corpus["texts"]) != set(texts):
            raise ValueError(f"Corpus contains obsolete templates: {code}")
