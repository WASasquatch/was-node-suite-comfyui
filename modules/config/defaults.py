"""The configuration schema, as built-in default values.

:data:`SUPERSEDED_FEATURE_DEFAULTS` holds the ``features.*`` defaults each earlier
:data:`VERSION` shipped, keyed by version.
"""

from __future__ import annotations

VERSION = 15

#: Feature defaults as each earlier version shipped them, newest first. A config written by
#: an older build carries that build's answers, and nothing in the file records whether a
#: value was chosen by the user or written for them. Comparing key by key against the
#: defaults of the version that wrote the file is what stands in for that: a key still
#: holding that version's default is brought forward to this version's, and a key holding
#: anything else is the user's answer and is kept. A choice that agrees with the default it
#: was given is indistinguishable from no choice at all, and is moved with the rest, which
#: is why every move is named in the log. Keyed by the version the entry describes.
SUPERSEDED_FEATURE_DEFAULTS = {
    14: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": False,
        "pssr": False,
        "preprocessors": True,
        "extras": True,
        "viewer": True,
        "threejs": True,
    },
    13: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "preprocessors": True,
        "extras": True,
        "viewer": True,
        "threejs": False,
    },
    12: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "preprocessors": True,
        "extras": True,
        "viewer": True,
    },
    11: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "preprocessors": True,
        "extras": True,
        "viewer": True,
    },
    10: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "preprocessors": True,
        "extras": True,
        "viewer": True,
    },
    9: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "extras": True,
        "viewer": True,
    },
    8: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "extras": True,
        "viewer": True,
    },
    7: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "extras": True,
        "viewer": True,
    },
    6: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "extras": True,
        "viewer": True,
    },
    5: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": True,
        "pssr": False,
        "extras": True,
        "viewer": True,
    },
    4: {
        "blip": True,
        "clipseg": True,
        "sam": True,
        "midas": True,
        "diffusers": True,
        "network": False,
        "yunet": True,
        "document_export": False,
        "pssr": False,
        "extras": True,
        "viewer": True,
    },
    3: {
        "blip": False,
        "clipseg": False,
        "sam": False,
        "midas": False,
        "diffusers": False,
        "network": False,
        "pssr": False,
        "extras": True,
        "viewer": True,
    },
}

FEATURE_GROUPS = (
    "blip",
    "clipseg",
    "sam",
    "midas",
    "diffusers",
    "network",
    "yunet",
    "document_export",
    "pssr",
    "preprocessors",
    "extras",
    "viewer",
    "threejs",
    "photoshop",
)

#: Groups that start on. ``extras`` and ``viewer`` are the two other packs this one
#: absorbed: each pair registers the same node ids, ComfyUI keeps whichever it loads last,
#: and one setting per pack decides which of the two provides them. Nothing on disk is
#: touched either way. The rest need nothing installed that ComfyUI does not already
#: require: blip, clipseg, sam and midas run on transformers, and diffusers uses ComfyUI's
#: own loader. They gate model weights and VRAM rather than dependencies, so a user who
#: never runs one pays nothing for having them listed. ``yunet`` is on and costs nothing
#: either: its detector runs in torch on weights that ship with the pack.
#: ``preprocessors`` is on and costs nothing either: every answer it gives runs in torch on
#: weights the pack publishes. ``photoshop`` is on and costs nothing either: the PSD and
#: TIFF layouts are written and read here, with no package behind them.
#:
#: Off, deliberately: network is a consent decision rather than a dependency one, since it
#: permits outbound requests. ``document_export`` is the only group whose packages are not
#: ComfyUI requirements, so it starts off and a fresh install brings in nothing. It gates
#: no node at all: off refuses the three Save DOC formats written through a library, and
#: the other seven still write.
FEATURE_DEFAULTS = {
    **dict.fromkeys(FEATURE_GROUPS, False),
    "extras": True,
    "viewer": True,
    "blip": True,
    "clipseg": True,
    "sam": True,
    "midas": True,
    "diffusers": True,
    "yunet": True,
    "preprocessors": True,
    "threejs": True,
    "photoshop": True,
}

LEGACY_GROUPS = (
    "loaders",
    "switches",
    "text_type",
    "core_dupes",
    "dupes",
    "sampling",
    "cache",
    "debug",
    "superseded",
)

#: Legacy groups that load anyway. These are the retired nodes a v2 workflow is likely to
#: name, loader wrappers, the mask and seed nodes a core node now covers, the text-type
#: shims and KSampler (WAS), and every one of them is a thin wrapper needing no package,
#: no model weights and no network. A workflow that names one would otherwise open with a
#: missing node and nothing to connect it to a setting.
#:
#: The rest stay off: ``cache`` and ``debug`` are tools rather than graph nodes,
#: ``dupes`` is CLIPSEG2, which needs CLIPSeg's weights like the node it duplicates, and
#: ``superseded`` holds nodes a better tool in this pack has replaced.
LEGACY_DEFAULTS = {
    **dict.fromkeys(LEGACY_GROUPS, False),
    "loaders": True,
    "switches": True,
    "text_type": True,
    "core_dupes": True,
    "sampling": True,
}

DEFAULTS = {
    "version": VERSION,
    "logging": {
        "level": "info",
        "rich": True,
        "quotes": False,
    },
    "paths": {
        "wildcards": None,
        "styles": None,
        "luts": None,
        "allow_read": [],
        "allow_write": [],
    },
    "text": {
        "strip_comments": True,
    },
    #: ``clean_html`` removes script and frame markup from the HTML a document node emits.
    #: On strips that markup; off emits the document exactly as written.
    "document": {
        "clean_html": True,
    },
    "history": {
        "display_limit": 36,
    },
    #: ``preview_max_edge`` reduces the picture a node publishes to its own interface to
    #: this many pixels on its longest edge. 1024 is a picture a panel inside a node can
    #: still be zoomed into, and it is where the cost stops mattering: a 3840 by 2160 frame
    #: costs 7.9 MB and 315 ms to hold and encode, and 1.4 MB and 136 ms at 1024, including
    #: the resize. 0 holds the image at the size the node received it, up to the channel's
    #: own ceiling of 8 MB a frame, which keeps every pixel of detail and pays for it in
    #: held memory and in PNG encoding on every run of every node with a panel open. A panel
    #: measuring the picture reports its own fidelity from the source size the channel sends
    #: beside it, so a reduced preview is labelled rather than silently trusted.
    "interface": {
        "preview_max_edge": 1024,
    },
    "video": {
        "extra_codecs": {},
    },
    #: ``install_extensions`` unpacks ``.zip`` view extensions for the content viewer at
    #: startup and pip-installs anything they require. Off leaves a view extension to be
    #: installed by copying two files. Read straight off disk at startup, before ComfyUI
    #: imports any custom node.
    "viewer": {
        "install_extensions": False,
    },
    #: ``install_missing`` installs the requirements file of a feature group that is on but
    #: has nothing installed for it, at startup, before ComfyUI imports any custom node. Off
    #: by default, so nothing is installed for a user who did not ask: the group's packages
    #: are named in the log with the command that installs them. On, it only ever adds: pip
    #: is asked what it would do first, and a plan that would replace a version already
    #: installed is refused and printed for the user to run themselves. Read straight off
    #: disk at startup.
    "dependencies": {
        "install_missing": False,
    },
    "features": dict(FEATURE_DEFAULTS),
    "legacy": dict(LEGACY_DEFAULTS),
    "nodes": {
        "enable": [],
        "disable": [],
    },
}
