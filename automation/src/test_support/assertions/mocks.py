"""Mock call expectations stay out of executable scenario bodies."""

from typing import Any
from unittest.mock import Mock


def called_once(mock: Mock) -> None:
    mock.assert_called_once()


def called_once_with(mock: Mock, *args: Any, **kwargs: Any) -> None:
    mock.assert_called_once_with(*args, **kwargs)


def not_called(mock: Mock) -> None:
    mock.assert_not_called()
