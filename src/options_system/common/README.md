# common/

**Shared plumbing used by every other module.** This is the small foundation
layer: the `loguru`-based logger factory (`get_logger()`), loading and exposing
the typed `pydantic-settings` configuration, and any shared data types /
enums / constants (e.g. instrument identifiers, order-side enums, common
exceptions) that more than one module needs. It contains no trading logic and
must not depend on any other `options_system` subpackage — everything else
depends on `common`, not the other way around. Keep it thin and boring.
