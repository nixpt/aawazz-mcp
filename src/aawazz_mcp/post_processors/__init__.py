"""Built-in post-processors — DSP voice profiles, gain control, VAD trim.

Phase 5 of v1.3. Importing this package registers every built-in
post-processor as a side effect of importing the sibling modules.
Third-party post-processors register via the
``aawazz.post_processors`` entry-point group (see SPEC_v1.3 §2.2).
"""

from __future__ import annotations

from aawazz_mcp.post_processors import dsp  # noqa: F401
from aawazz_mcp.post_processors import gain  # noqa: F401
from aawazz_mcp.post_processors import vad  # noqa: F401
