# Project agent configuration

Verified installed `codex-cli 0.164.0` and official [Subagents documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents), accessed 2026-11-04. Current documentation defines standalone `.codex/agents/*.toml` with required name, description, developer_instructions and `[agents] max_concurrent_threads_per_session` in `.codex/config.toml`. These are project-only; no model, permissions or global settings are overridden.

`codex --strict-config doctor --summary` reports configuration loaded successfully. Its unrelated network/runtime checks can fail under restricted network; that is not agent-config validation failure. Native subagents actually ran in this execution. The coordinator additionally enforces the user limit of four total agents including itself.

No recurring automation is installed. Future executions must read saved status, inspect processes and select the next unblocked task; instructions are not a persistent background worker. Role definitions are in autonomy/roles.
