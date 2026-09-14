"""Token substitution for filenames, paths and text.

:class:`TextTokens` resolves ``[time]``, ``[hostname]``, ``[user]``, ``[cuda_device]``,
``[cuda_name]`` and ``[time(%Y-%m-%d)]``. Custom tokens live in the ``custom_tokens``
category of the settings database.
"""

from __future__ import annotations

import os
import re
import socket
import time

from .database import WASDatabase, get_settings_db

__all__ = ["ESCAPED", "TextTokens"]

#: A bracketed run a backslash marks as text. The backslash is left in place, and
#: :func:`modules.compat.tokens.expand` drops it after the tokens around it are replaced.
ESCAPED = re.compile(r"\\\[[^\]]*\]")

#: What a marked run is held as while the tokens around it are replaced. The character
#: cannot occur in a widget value, which carries text a person typed.
HOLD = "\x00was-escape-{0}\x00"

#: Longest a value may grow to while its tokens are replaced. A token whose value names
#: another token is replaced again by every later token in the pass, so a short value can
#: grow by a multiple on each one.
MAX_EXPANDED = 1 << 20


class TextTokens:
    """The token table for one execution.

    Attributes:
        WDB: The settings database the custom tokens are stored in.
        custom_tokens: The ``custom_tokens`` category, read at construction.
        tokens: The built-in tokens, resolved at construction.
    """

    def __init__(self, database: WASDatabase | None = None):
        """Resolve the built-in tokens and attach the custom-token category.

        Args:
            database: The database custom tokens are read from and written to. Defaults
                to the shared settings database, which is what every node passes.
        """
        # comfy.model_management pulls in torch and only exists inside ComfyUI. Importing
        # it here keeps the modules/ tree importable outside a running ComfyUI.
        import comfy.model_management

        self.WDB = get_settings_db() if database is None else database
        if not self.WDB.catExists("custom_tokens"):
            self.WDB.insertCat("custom_tokens")
        self.custom_tokens = self.WDB.getDict("custom_tokens")

        self.tokens = {
            "[time]": str(time.time()).replace(".", "_"),
            "[hostname]": socket.gethostname(),
            "[cuda_device]": str(comfy.model_management.get_torch_device()),
            "[cuda_name]": str(comfy.model_management.get_torch_device_name(device=comfy.model_management.get_torch_device())),
        }

        if "." in self.tokens["[time]"]:
            self.tokens["[time]"] = self.tokens["[time]"].split(".")[0]

        try:
            self.tokens["[user]"] = os.getlogin() if os.getlogin() else "null"
        except Exception:
            self.tokens["[user]"] = "null"

    def addToken(self, name: str, value: str) -> None:
        """Add or replace a custom token and store it."""
        self.custom_tokens[name] = value
        self.WDB.insert("custom_tokens", name, value)

    def removeToken(self, name: str) -> None:
        """Drop a custom token and remove it from the store.

        Raises:
            KeyError: No such token.
        """
        self.custom_tokens.pop(name)
        self.WDB.delete("custom_tokens", name)

    def format_time(self, format_code: str) -> str:
        """Local time rendered through a ``time.strftime`` format code."""
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text: str) -> str:
        """Replace every token in ``text`` with its value, custom tokens overriding built-ins.

        A run a backslash marks is carried through untouched, backslash included.

        Args:
            text: The string to expand.

        Returns:
            ``text`` with every known token replaced.
        """
        held: list[str] = []

        def hold(match: "re.Match") -> str:
            held.append(match.group(0))
            return HOLD.format(len(held) - 1)

        text = ESCAPED.sub(hold, text)

        tokens = self.tokens.copy()
        if self.custom_tokens:
            tokens.update(self.custom_tokens)

        # Update time
        tokens["[time]"] = str(time.time())
        if "." in tokens["[time]"]:
            tokens["[time]"] = tokens["[time]"].split(".")[0]

        for token, value in tokens.items():
            if token.startswith("[time("):
                continue
            pattern = re.compile(re.escape(token))
            # The replacement is a callable, which re.sub inserts verbatim. A string
            # replacement is a template instead: a value holding a Windows path is a bad
            # \U escape and a value holding \1 is a group reference, and the value cannot
            # be escaped the way the pattern is, being the text to insert.
            text = pattern.sub(lambda match, replacement=value: replacement, text)
            if len(text) > MAX_EXPANDED:
                raise ValueError(
                    f"expanding tokens grew the text past {MAX_EXPANDED} characters at "
                    f"{token}. A token whose value names another token grows on every "
                    f"replacement; give one of them a plain value."
                )

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r"\[time\((.*?)\)\]", replace_custom_time, text)

        for index, literal in enumerate(held):
            text = text.replace(HOLD.format(index), literal)

        return text
