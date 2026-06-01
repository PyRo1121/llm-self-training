"""Suppress known-noise warnings/logs during Unsloth/TRL train and preflight."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings

_UNSLOTH_IMPORT_NOISE = (
    "Will patch your computer",
    "patch everything to make training faster",
    "Flash Attention 2 installation seems to be broken",
    "inline_inbuilt_nn_modules",
)


class _FilteredStream:
    def __init__(self, stream: object) -> None:
        self._stream = stream

    def write(self, data: str) -> int:
        if any(n in data for n in _UNSLOTH_IMPORT_NOISE):
            return len(data)
        return self._stream.write(data)  # type: ignore[union-attr]

    def flush(self) -> None:
        self._stream.flush()  # type: ignore[union-attr]

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)


@contextlib.contextmanager
def suppress_unsloth_import_noise():
    """Hide Unsloth bootstrap print() spam during `import unsloth`."""
    out, err = sys.stdout, sys.stderr
    sys.stdout = _FilteredStream(out)
    sys.stderr = _FilteredStream(err)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err


class _MessageFilter(logging.Filter):
    def __init__(self, *needles: str) -> None:
        super().__init__()
        self._needles = needles

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(n in msg for n in self._needles)


def apply_train_quiet(*, before_unsloth: bool = True) -> None:
    """Call before `import unsloth` in CLI entrypoints."""
    if before_unsloth:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\._dynamo")
    warnings.filterwarnings(
        "ignore",
        message=r".*inline_inbuilt_nn_modules.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*warmup_ratio is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*keep_end.*truncation mode is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Unsloth should be imported before.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Flash Attention 2 installation seems to be broken.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*use_return_dict is deprecated.*",
        category=FutureWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message=r".*AttentionMaskConverter.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*attention mask API under.*modeling_attn_mask_utils.*",
        category=FutureWarning,
    )

    for name in (
        "torch",
        "torch.utils._pytree",
        "transformers.configuration_utils",
        "transformers",
    ):
        log = logging.getLogger(name)
        log.addFilter(
            _MessageFilter(
                "register_constant()",
                "KernelPreference",
                "ScaleCalculationMode",
                "inline_inbuilt_nn_modules",
            )
        )
        if name == "transformers.configuration_utils":
            log.setLevel(logging.ERROR)

    # PyTorch 2.12 enum deprecation uses WARNING on submodule loggers.
    logging.getLogger().addFilter(
        _MessageFilter("register_constant()", "KernelPreference", "ScaleCalculationMode")
    )
