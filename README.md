# Supercharged Memory for Claude Code

Give your Claude Code agent **persistent memory across sessions** with a 7-layer memory stack.

Out of the box, Claude Code has zero memory between sessions. Every conversation starts from scratch. This system fixes that.

## What You Get

- **Auto Memory** - Permanent notes Claude always sees (MEMORY.md)
- **Session Bootstrap** - Auto-loads recent context on startup via hook
- **Working Memory** - Active session state (goals, scratchpad, references)
- **Episodic Memory** - Search past Claude Code conversations
- **Hybrid Search** - Vector + keyword + graph fusion search
- **Knowledge Graph** - Entities + relationships, queryable
- **RLM-Graph** - Recursive queries for large contexts

## Guides

| Guide | What It Covers |
|-------|---------------|
| [Architecture Overview](docs/01-ARCHITECTURE-OVERVIEW.md) | How the 7 layers work together, design principles |
| [Implementation Guide](docs/02-IMPLEMENTATION-GUIDE.md) | Step-by-step code for every layer, copy-paste ready |
| [Quick Start](docs/03-QUICK-START.md) | Get running in 30 minutes, common commands, troubleshooting |

## Quick Start

```bash
# Layer 1: Auto Memory (5 minutes)
PROJECT_KEY="-Users-$(whoami)-$(basename $(pwd))"
mkdir -p ~/.claude/projects/${PROJECT_KEY}/memory/
echo "# Project Memory" > ~/.claude/projects/${PROJECT_KEY}/memory/MEMORY.md

# Layer 2-3: Bootstrap + Working Memory (15 minutes)
mkdir -p scripts/ .planning/{handoffs,working-memory} memory/
# Copy scripts from Implementation Guide, wire up SessionStart hook

# Layer 4-6: Search + Knowledge Graph (30 minutes)
pip install networkx
# Install episodic-memory plugin, seed knowledge graph
```

See the [Quick Start guide](docs/03-QUICK-START.md) for full instructions.

## Requirements

- Claude Code CLI
- Python 3.10+
- `networkx` (for Knowledge Graph)

## How It Works

```
SESSION START
    |
    v
[SessionStart Hook] --> context_loader.py
    |                        |
    |   Scans:               |
    |   - memory/*.md        |
    |   - .planning/handoffs |
    |   - working-memory     |
    |                        |
    v                        v
[MEMORY.md loaded]     [Context injected]
    |                        |
    +--------+---------------+
             |
             v
    Claude Code Session
    (has: time, goals, recent work, notes)
             |
             |--- Search past decisions --> episodic-memory
             |--- Query relationships   --> knowledge_graph.py
             |--- Complex queries       --> rlm_graph.py
             |--- Learn something new   --> MEMORY.md / knowledge graph
             |
             v
    SESSION END
    (working memory archived or cleaned up)
```

## License

MIT
