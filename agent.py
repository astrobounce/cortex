"""
Week 1 — Orchestrator: a hand-written plan -> act -> observe loop.

Usage:
    python agent.py "How many customers live in Texas?"
    python agent.py            # interactive mode
"""

import json
import os
import sys

import anthropic
import psycopg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:arv@123@localhost:5432/pagila"
)
MAX_STEPS = 10   # hard stop so a confused agent can't loop forever
MAX_ROWS = 50    # never flood the context window with huge results

SYSTEM_PROMPT = """You are a careful PostgreSQL assistant.
You answer questions about the connected database by using tools.

Rules:
- Never guess table or column names. Use list_tables and describe_table first.
- Use run_select only for SELECT / WITH queries.
- If a query fails, read the error, fix the query, and try again.
- When you have the answer, reply in plain language and show the final SQL you used.
"""

# ---------------------------------------------------------------------------
# Tools: what the model is ALLOWED to do (schemas) ...
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "list_tables",
        "description": "List all user tables in the database (schema.table).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "describe_table",
        "description": "Show the columns and data types of one table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name, e.g. 'customer'"}
            },
            "required": ["table"],
        },
    },
    {
        "name": "run_select",
        "description": f"Run a read-only SELECT query. Returns at most {MAX_ROWS} rows.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT query"}},
            "required": ["sql"],
        },
    },
]


# ---------------------------------------------------------------------------
# ... and how YOUR code actually does it (implementations)
# ---------------------------------------------------------------------------
def get_conn():
    # Read-only + 5s timeout enforced by Postgres itself, not by trusting the model.
    return psycopg.connect(
        DB_URL,
        autocommit=True,
        options="-c default_transaction_read_only=on -c statement_timeout=5000",
    )


def list_tables(_args):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT table_schema || '.' || table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY 1
            """
        ).fetchall()
    return json.dumps([r[0] for r in rows])


def describe_table(args):
    table = args["table"].split(".")[-1]
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Table '{table}' not found. Call list_tables first.")
    return json.dumps([{"column": c, "type": t} for c, t in rows])


def run_select(args):
    sql = args["sql"].strip().rstrip(";")
    if not sql.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT/WITH queries are allowed.")
    with get_conn() as conn:
        cur = conn.execute(sql)
        cols = [d.name for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS)
    return json.dumps(
        {"columns": cols, "rows": [list(r) for r in rows], "truncated": len(rows) == MAX_ROWS},
        default=str,
    )


TOOL_IMPLS = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "run_select": run_select,
}


def execute_tool(name, args):
    """Run a tool. Errors are returned to the model (not raised) so it can recover."""
    try:
        return TOOL_IMPLS[name](args), False
    except Exception as e:  # noqa: BLE001 — the model needs to SEE the error
        return f"ERROR: {type(e).__name__}: {e}", True


# ---------------------------------------------------------------------------
# The orchestrator loop
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def run_agent(question, verbose=True):
    messages = [{"role": "user", "content": question}]

    for step in range(1, MAX_STEPS + 1):
        # 1. PLAN — the model decides what to do next
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # Done? The model answered instead of asking for a tool.
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # 2. ACT — run every tool the model asked for
        tool_results = []
        for block in response.content:
            if block.type == "text" and verbose and block.text.strip():
                print(f"\n[step {step}] 💭 {block.text.strip()}")
            if block.type == "tool_use":
                if verbose:
                    print(f"[step {step}] 🔧 {block.name}({json.dumps(block.input)})")
                output, is_error = execute_tool(block.name, block.input)
                if verbose:
                    preview = output if len(output) < 200 else output[:200] + "…"
                    print(f"[step {step}] {'❌' if is_error else '👀'} {preview}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                        "is_error": is_error,
                    }
                )

        # 3. OBSERVE — feed results back; the loop repeats
        messages.append({"role": "user", "content": tool_results})

    return f"Stopped: hit the {MAX_STEPS}-step limit without a final answer."


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("\n✅ " + run_agent(" ".join(sys.argv[1:])))
    else:
        print("Postgres agent — type a question (Ctrl+C to quit)")
        while True:
            try:
                q = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q:
                print("\n✅ " + run_agent(q))
