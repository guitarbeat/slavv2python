"""Central page registry and navigation helpers for the Streamlit workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class PageDefinition:
    """Stable metadata for one application route."""

    page_id: str
    title: str
    icon: str
    url_path: str
    group: str
    caption: str


PAGE_DEFINITIONS = (
    PageDefinition(
        "dashboard",
        "Dashboard",
        ":material/space_dashboard:",
        "",
        "Workspace",
        "Run readiness and next actions",
    ),
    PageDefinition(
        "workspaces",
        "Workspaces",
        ":material/folder_open:",
        "workspaces",
        "Workspace",
        "Inspect and reopen saved runs",
    ),
    PageDefinition(
        "processing",
        "Processing",
        ":material/tune:",
        "processing",
        "Workspace",
        "Create or resume pipeline results",
    ),
    PageDefinition(
        "curation",
        "Curation",
        ":material/edit_note:",
        "curation",
        "Workspace",
        "Review vertices and edges",
    ),
    PageDefinition(
        "visualization",
        "Visualization",
        ":material/view_in_ar:",
        "visualization",
        "Results",
        "Inspect the current network",
    ),
    PageDefinition(
        "analysis",
        "Analysis",
        ":material/analytics:",
        "analysis",
        "Results",
        "Quantify topology and morphology",
    ),
    PageDefinition(
        "about",
        "About",
        ":material/info:",
        "about",
        "Resources",
        "Method and implementation references",
    ),
)

_REGISTERED_PAGES: dict[str, Any] = {}


def register_pages(
    handlers: Mapping[str, str | Callable[[], None]],
) -> dict[str, list[Any]]:
    """Build and retain the pages registered with ``st.navigation``."""
    grouped: dict[str, list[Any]] = {}
    _REGISTERED_PAGES.clear()
    for definition in PAGE_DEFINITIONS:
        page = st.Page(
            handlers[definition.page_id],
            title=definition.title,
            icon=definition.icon,
            url_path=definition.url_path or None,
            default=definition.page_id == "dashboard",
        )
        _REGISTERED_PAGES[definition.page_id] = page
        grouped.setdefault(definition.group, []).append(page)
    return grouped


def switch_to(page_id: str) -> None:
    """Navigate to a registered application page by stable identifier."""
    page = _REGISTERED_PAGES.get(page_id)
    if page is None:
        raise KeyError(f"Unknown or unregistered application page: {page_id}")
    st.switch_page(page)


def page_definition(page_id: str) -> PageDefinition:
    """Return route metadata by stable identifier."""
    for definition in PAGE_DEFINITIONS:
        if definition.page_id == page_id:
            return definition
    raise KeyError(page_id)


__all__ = ["PAGE_DEFINITIONS", "PageDefinition", "page_definition", "register_pages", "switch_to"]
