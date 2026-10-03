"""Parser discovery for the content viewer.

Every ``*_parser.py`` in this directory is imported and each :class:`BaseParser` subclass
in it is registered, highest priority first.
"""

import os
import inspect
import importlib

from ... import log

from .base_parser import BaseParser

logger = log.get_logger("viewer.parsers")

_parsers = []
_loaded = False


def load_parsers():
    """Load every parser class in this directory."""
    global _parsers, _loaded

    if _loaded:
        return _parsers

    parsers_dir = os.path.dirname(__file__)
    package_name = __name__

    # Load parsers from this directory (installed parsers)
    for filename in os.listdir(parsers_dir):
        if not filename.endswith("_parser.py") or filename == "base_parser.py":
            continue

        module_name = filename[:-3]
        full_module_name = f"{package_name}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj is BaseParser:
                    continue
                if not issubclass(obj, BaseParser):
                    continue

                parser_info = {
                    "name": obj.PARSER_NAME,
                    "view": getattr(obj, "PARSER_VIEW", None) or obj.PARSER_NAME,
                    "priority": obj.PARSER_PRIORITY,
                    "class": obj,
                    "detect_input": obj.detect_input,
                    "handle_input": obj.handle_input,
                    "detect_output": obj.detect_output,
                    "parse_output": obj.parse_output,
                }

                _parsers.append(parser_info)
                logger.info(
                    f"[Parsers] Loaded parser: {obj.PARSER_NAME} (priority {obj.PARSER_PRIORITY})"
                )

        except Exception as e:
            logger.error(f"[Parsers] Failed to load {filename}: {e}")

    _parsers.sort(key=lambda p: p["priority"], reverse=True)
    _loaded = True

    return _parsers


def get_parsers():
    """Get all loaded parsers, loading them if necessary."""
    if not _loaded:
        load_parsers()
    return _parsers


def get_parser_by_name(name: str):
    """Get a specific parser by name."""
    for parser in get_parsers():
        if parser["name"] == name:
            return parser
    return None


def get_all_parser_names():
    """Get list of all loaded parser names."""
    return [p["name"] for p in get_parsers()]


def find_input_handler(content):
    """Find the first parser that can handle this input content."""
    for parser in get_parsers():
        if not parser["detect_input"]:
            continue
        try:
            if parser["detect_input"](content):
                return parser
        except Exception as e:
            logger.error(f"[Parsers] Error in {parser['name']}.detect_input(): {e}")
    return None


def find_output_parser(content: str):
    """Find the first parser that can parse this output content."""
    for parser in get_parsers():
        if not parser["detect_output"]:
            continue
        try:
            if parser["detect_output"](content):
                return parser
        except Exception as e:
            logger.error(f"[Parsers] Error in {parser['name']}.detect_output(): {e}")
    return None


def handle_input(content, logger=None):
    """
    Try to handle input content using available parsers.

    Returns:
        dict with keys: display_content, output_values, content_hash
        or None if no parser matched
    """
    parser = find_input_handler(content)
    if parser is None:
        return None

    try:
        result = parser["handle_input"](content, logger)
        if result:
            result["parser_name"] = parser["name"]
        return result
    except Exception as e:
        if logger:
            logger.error(f"[Parsers] Error in {parser['name']}.handle_input(): {e}")
        return None


MULTIVIEW_MARKER = "$WAS_MULTIVIEW$"


def find_all_input_handlers(content):
    """Find ALL parsers that can handle this input content."""
    handlers = []
    for parser in get_parsers():
        if not parser["detect_input"]:
            continue
        try:
            if parser["detect_input"](content):
                handlers.append(parser)
        except Exception as e:
            logger.error(f"[Parsers] Error in {parser['name']}.detect_input(): {e}")
    return handlers


def handle_all_inputs(content, logger=None):
    """
    Try to handle input content using ALL matching parsers.
    Returns multi-view payload if multiple parsers match.

    Returns:
        dict with keys:
            - If single match: display_content, output_values, content_hash, parser_name
            - If multi match: display_content (with MULTIVIEW_MARKER), output_values,
                             content_hash, views (list of view data)
        or None if no parser matched
    """
    import json

    handlers = find_all_input_handlers(content)

    if not handlers:
        return None

    # Single handler - return as before
    if len(handlers) == 1:
        parser = handlers[0]
        try:
            result = parser["handle_input"](content, logger)
            if result:
                result["parser_name"] = parser["name"]
            return result
        except Exception as e:
            if logger:
                logger.error(f"[Parsers] Error in {parser['name']}.handle_input(): {e}")
            return None

    # Multiple handlers - create multi-view payload
    views = []
    output_values = None

    for parser in handlers:
        try:
            result = parser["handle_input"](content, logger)
            if result:
                view_data = {
                    "name": parser.get("view") or parser["name"],
                    "priority": parser["priority"],
                    "display_content": result.get("display_content", ""),
                    "content_hash": result.get("content_hash", ""),
                }
                views.append(view_data)

                # Use output_values from highest priority parser
                if output_values is None:
                    output_values = result.get("output_values", [])
        except Exception as e:
            if logger:
                logger.error(f"[Parsers] Error in {parser['name']}.handle_input(): {e}")

    if not views:
        return None

    # Sort by priority (highest first) - first view is default
    views.sort(key=lambda v: v["priority"], reverse=True)

    # Create multi-view payload
    multiview_data = {
        "type": "multiview",
        "default_view": views[0]["name"],
        "views": views,
    }

    if logger:
        view_names = [v["name"] for v in views]
        logger.info(f"[Parsers] Multi-view content detected: {view_names}")

    return {
        "display_content": MULTIVIEW_MARKER + json.dumps(multiview_data),
        "output_values": output_values,
        "content_hash": f"multiview_{len(views)}_{views[0]['content_hash']}",
        "parser_name": "multiview",
        "is_multiview": True,
        "available_views": [v["name"] for v in views],
    }


def parse_output(content: str, logger=None):
    """
    Try to parse output content using available parsers.

    Returns:
        dict with keys: output_values, display_text, content_hash
        or None if no parser matched
    """
    parser = find_output_parser(content)
    if parser is None:
        return None

    try:
        result = parser["parse_output"](content, logger)
        if result:
            result["parser_name"] = parser["name"]
        return result
    except Exception as e:
        if logger:
            logger.error(f"[Parsers] Error in {parser['name']}.parse_output(): {e}")
        return None


def find_state_parser(state_data: dict):
    """Find the first parser that can handle this state data."""
    if not isinstance(state_data, dict):
        return None
    for parser in get_parsers():
        parser_class = parser["class"]
        if not hasattr(parser_class, "detect_state"):
            continue
        try:
            if parser_class.detect_state(state_data):
                return parser
        except Exception as e:
            logger.error(f"[Parsers] Error in {parser['name']}.detect_state(): {e}")
    return None


def parse_state(state_data, logger=None):
    """
    Try to parse state data using available parsers.

    Args:
        state_data: dict or JSON string of state data
        logger: Optional logger

    Returns:
        dict with parser-specific keys, or None if no parser matched
    """
    import json

    if isinstance(state_data, str):
        if not state_data or state_data == "{}":
            return None
        try:
            state_data = json.loads(state_data)
        except json.JSONDecodeError:
            return None

    if isinstance(state_data, list):
        state_data = state_data[0] if state_data else None
        if isinstance(state_data, str):
            try:
                state_data = json.loads(state_data)
            except json.JSONDecodeError:
                return None

    if not isinstance(state_data, dict):
        return None

    parser = find_state_parser(state_data)
    if parser is None:
        return None

    try:
        result = parser["class"].parse_state(state_data, logger)
        if result:
            result["parser_name"] = parser["name"]
        return result
    except Exception as e:
        if logger:
            logger.error(f"[Parsers] Error in {parser['name']}.parse_state(): {e}")
        return None


def find_display_handler(content):
    """Find the first parser that can prepare display content for this input."""
    for parser in get_parsers():
        parser_class = parser["class"]
        if not hasattr(parser_class, "detect_display_content"):
            continue
        try:
            if parser_class.detect_display_content(content):
                return parser
        except Exception as e:
            logger.error(
                f"[Parsers] Error in {parser['name']}.detect_display_content(): {e}"
            )
    return None


def prepare_display(content, logger=None):
    """
    Try to prepare display content using available parsers.

    Returns:
        dict with keys: display_content, content_hash, count (optional)
        or None if no parser matched
    """
    parser = find_display_handler(content)
    if parser is None:
        return None

    try:
        result = parser["class"].prepare_display(content, logger)
        if result:
            result["parser_name"] = parser["name"]
        return result
    except Exception as e:
        if logger:
            logger.error(f"[Parsers] Error in {parser['name']}.prepare_display(): {e}")
        return None


def get_default_outputs(content, output_types: list, logger=None):
    """
    Try to get default outputs using available parsers.

    Args:
        content: Input content
        output_types: List of expected output type names (e.g., ["IMAGE", "MASK"])
        logger: Optional logger

    Returns:
        tuple of default values, or None if no parser matched
    """
    for parser in get_parsers():
        parser_class = parser["class"]
        if not hasattr(parser_class, "get_default_outputs"):
            continue
        try:
            result = parser_class.get_default_outputs(content, output_types, logger)
            if result is not None:
                return result
        except Exception as e:
            if logger:
                logger.error(
                    f"[Parsers] Error in {parser['name']}.get_default_outputs(): {e}"
                )
    return None
