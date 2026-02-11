# Supercharged Memory for Claude Code

### Give your AI agent the same memory architecture that makes human cognition work.

---

## The Problem Nobody Talks About

Claude Code is powerful. But every time you start a new session, it has **total amnesia**. It doesn't remember what you built yesterday, what decisions you made last week, who your teammates are, or what your project even does. You spend the first 5 minutes of every session re-explaining context that should already be known.

This isn't a minor inconvenience. It's the single biggest bottleneck in AI-assisted development.

You wouldn't hire a developer who forgot everything at the end of each workday. So why accept that from your AI agent?

---

## How Humans Remember (And Why It Matters)

Neuroscience has identified distinct memory systems in the human brain, each serving a different purpose. AI agents need the same architecture to function effectively:

| Human Memory | What It Does | AI Equivalent | This System's Layer |
|-------------|-------------|---------------|-------------------|
| **Long-term memory** | Facts you've committed to permanent storage | Permanent notes always available | **Auto Memory** (MEMORY.md) |
| **Prospective memory** | Remembering to do things in the future | Session bootstrap that loads pending work | **Session Bootstrap** |
| **Working memory** | Holding active thoughts while problem-solving | Goals, scratchpad, state tracking | **Working Memory** |
| **Episodic memory** | Recalling specific past experiences and conversations | Searchable index of all past sessions | **Episodic Memory** |
| **Semantic memory** | General knowledge and relationships between concepts | Entity-relationship graph | **Knowledge Graph** |
| **Associative recall** | Finding connections between related memories | Multi-source search fusion | **Hybrid Search** |
| **Chunking** | Breaking complex info into manageable pieces | Graph-based context partitioning | **RLM-Graph** |

Humans don't have one type of memory. They have an interconnected system where different memory types reinforce each other. Your AI agent needs the same thing.

---

## What This System Actually Does

### Before: Every Session Starts Cold
```
You: "Continue working on the auth module"
Claude: "I don't have context about an auth module. Can you tell me about your project?"
You: *sighs, spends 10 minutes re-explaining everything*
```

### After: Every Session Picks Up Where You Left Off
```
Claude (automatically, before you say anything):
  ✓ TIME VERIFIED: 4:46 PM on Wednesday
  ✓ Active Goal: Debug auth module [high priority]
  ✓ Last note: "Found JWT validation bug in token rotation"
  ✓ 3 incomplete items from yesterday
  ✓ Session handoff loaded from last night's work

You: "Continue working on the auth module"
Claude: "Picking up from where we left off - the JWT validation bug in token
         rotation. Last session we identified the issue was in the refresh
         logic. Let me check if the fix from yesterday's handoff resolved it..."
```

That's the difference. Zero re-explanation. Instant continuity.

---

## The 7-Layer Memory Stack

Each layer solves a different memory problem. They work independently (implement any one without the others) and compound when combined:

```
Layer 7: RLM-Graph ──────── When context is too large, partition intelligently
Layer 6: Knowledge Graph ── Who knows who? What connects to what?
Layer 5: Hybrid Search ──── Find anything across all memory sources
Layer 4: Episodic Memory ── "What did I decide about X last month?"
Layer 3: Working Memory ─── Track goals, notes, and state mid-session
Layer 2: Session Bootstrap ─ Auto-load recent context on startup
Layer 1: Auto Memory ────── Permanent notes Claude always sees
```

### Layer 1: Auto Memory
Claude's **long-term memory**. A markdown file (MEMORY.md) that's always loaded into the system prompt. Confirmed patterns, key decisions, architecture notes. Survives forever.

### Layer 2: Session Bootstrap
Claude's **prospective memory**. A Python script that runs automatically via a SessionStart hook. Scans the last 48 hours of work, finds incomplete items, loads handoffs from previous sessions, and injects the actual current time (so Claude never guesses the date wrong).

### Layer 3: Working Memory
Claude's **mental scratchpad**. Tracks active goals, timestamped observations, key-value state, and reference pointers. When Claude's context window compresses (hits token limits), working memory persists to disk and reloads. Goals and discoveries survive context loss.

### Layer 4: Episodic Memory
Claude's **autobiographical memory**. A searchable index of all past Claude Code conversations. Supports semantic search (find by meaning), keyword search (find exact terms), and multi-concept AND matching. Ask "What did I decide about the auth approach?" and get ranked results with similarity scores.

### Layer 5: Hybrid Search
Claude's **associative recall**. Combines keyword matching, knowledge graph traversal, and (optionally) vector embeddings into a single ranked result set. Prevents blind spots that any single search method would have.

### Layer 6: Knowledge Graph
Claude's **semantic memory**. A NetworkX graph of entities (people, projects, companies, systems) and their relationships. Makes implicit connections explicit and queryable. "How is Alice connected to ProjectAlpha?" returns the actual graph path with context at every hop.

### Layer 7: RLM-Graph
Claude's **chunking ability**. When a query involves too many entities and relationships to fit in a single context window, RLM-Graph uses the knowledge graph's topology to create meaningful partitions, processes each one, then merges the results. Complex questions that would normally be truncated randomly are instead decomposed intelligently.

---

## Why This Works Better Than Alternatives

**vs. Just using CLAUDE.md:** CLAUDE.md is static instructions. This system gives Claude dynamic, searchable, self-updating memory that grows over time.

**vs. RAG pipelines:** RAG retrieves documents. This system retrieves *context* - goals, relationships, decisions, and state - from multiple complementary memory types simultaneously.

**vs. Conversation summaries:** Summaries lose detail. Episodic memory preserves the full reasoning chain. You can read back the exact conversation where a decision was made.

**vs. Vector databases alone:** Vector search finds similar content but misses structured relationships. The knowledge graph captures explicit connections (Alice *manages* ProjectAlpha, ProjectAlpha *depends on* ServiceX) that semantic similarity can't represent.

---

## Get Started

| Time | What You Get |
|------|-------------|
| **5 minutes** | Auto Memory - permanent notes across sessions |
| **15 minutes** | + Session Bootstrap - automatic context loading on startup |
| **30 minutes** | + Working Memory, Episodic Memory, Knowledge Graph |
| **Half day** | + Hybrid Search, RLM-Graph for complex queries |

Each layer is independent. Start with Layer 1 and add more as needed.

## Guides

| Guide | What It Covers |
|-------|---------------|
| [Architecture Overview](docs/01-ARCHITECTURE-OVERVIEW.md) | How the 7 layers work together, design principles, flow diagrams |
| [Implementation Guide](docs/02-IMPLEMENTATION-GUIDE.md) | Step-by-step code for every layer, copy-paste ready |
| [Quick Start](docs/03-QUICK-START.md) | Get running in 30 minutes, common commands, troubleshooting |

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
    |   Scans last 48h:      |
    |   - memory files       |
    |   - session handoffs   |
    |   - working memory     |
    |   - incomplete items   |
    |                        |
    v                        v
[MEMORY.md loaded]     [Context injected]
    |                        |
    +--------+---------------+
             |
             v
    Claude Code Session
    (knows: time, goals, recent work, pending items, relationships)
             |
             |--- Recall past decisions --> episodic memory search
             |--- Query relationships   --> knowledge graph
             |--- Complex questions     --> RLM-Graph (recursive)
             |--- Learn something new   --> writes to MEMORY.md + graph
             |
             v
    SESSION END
    (working memory persisted, handoff created if needed)
```

## Key Design Principles

1. **Mirrors human cognition** - Different memory types for different needs, just like the brain
2. **Layers are independent** - Implement any layer without the others
3. **Graceful degradation** - If one layer fails, the rest keep working
4. **Zero manual effort** - Session bootstrap requires no user intervention
5. **Grows over time** - The knowledge graph and episodic memory get more valuable with every session
6. **Token-conscious** - MEMORY.md has a ~200 line limit; RLM-Graph manages context overflow
7. **Disk-backed** - Everything persists to files and SQLite, survives restarts and crashes

## License

MIT
