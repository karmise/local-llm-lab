"""Optional Allure integration with static titles and explicit attachments."""

from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from playwright.sync_api import Page

P = ParamSpec("P")
T = TypeVar("T")


def _backend() -> Any:
    try:
        return import_module("allure")
    except ModuleNotFoundError as error:
        if error.name != "allure":
            raise
        return None


def title(text: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Use Allure's native title decorator when the optional backend is installed."""
    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        backend = _backend()
        return backend.title(text)(function) if backend is not None else function

    return decorate


def step(title: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Report a synchronous operation without recording its arguments."""
    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            backend = _backend()
            with backend.step(title) if backend is not None else nullcontext():
                return function(*args, **kwargs)

        return wrapped

    return decorate


def set_metadata(*, feature: str, story: str) -> None:
    backend = _backend()
    if backend is not None:
        backend.dynamic.feature(feature)
        backend.dynamic.story(story)


def attach_text(text: str, *, name: str) -> None:
    backend = _backend()
    if backend is not None:
        backend.attach(text, name=name, attachment_type="text/plain")


def attach_file(path: Path, *, name: str, media_type: str, extension: str) -> None:
    backend = _backend()
    if backend is not None:
        backend.attach.file(str(path), name=name, attachment_type=media_type, extension=extension)


def attach_screenshot(page: "Page", *, name: str) -> None:
    backend = _backend()
    if backend is not None:
        backend.attach(
                page.screenshot(full_page=True, animations="disabled"), name=name, attachment_type="image/png",
                extension="png")


def attach_browser_artifacts(directory: Path) -> None:
    for artifact in sorted(directory.glob("*")):
        if artifact.suffix == ".png":
            attach_file(artifact, name=artifact.name, media_type="image/png", extension="png")
        elif artifact.suffix == ".zip":
            attach_file(artifact, name=artifact.name, media_type="application/zip", extension="zip")


def traceability_labels(requirements: list[str], phases: list[str]) -> None:
    backend = _backend()
    if backend is not None:
        for identifier in requirements:
            backend.dynamic.label("requirement", identifier)
        for phase in phases:
            backend.dynamic.label("qualification_phase", phase)
