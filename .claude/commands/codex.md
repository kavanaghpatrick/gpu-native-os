# Codex CLI - Autonomous Task Execution

Use OpenAI's Codex CLI for long-running autonomous tasks (7-24 hours) and parallel agent workflows.

## When to Use

- Long autonomous tasks that don't need immediate results
- Parallel work across multiple branches
- Tasks requiring extensive exploration
- When you want to "fire and forget"

## Version

Current: v0.77.0

## Basic Commands

### Interactive Mode
```bash
codex
```

### Autonomous Execution
```bash
codex exec --full-auto "task description"
```

### Safe Code Review (Read-Only)
```bash
codex exec --sandbox read-only "Review this codebase for security issues"
```

## Execution Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Interactive | `codex` | Guided development |
| Full Auto | `codex exec --full-auto` | Autonomous execution |
| Sandbox Read-Only | `codex exec --sandbox read-only` | Safe analysis |

## Parallel Agent Pattern

Use git worktrees for true parallel execution:

```bash
# Create worktrees for parallel work
git worktree add ../project-feature1 -b feature1
git worktree add ../project-feature2 -b feature2
git worktree add ../project-feature3 -b feature3

# Launch Codex instances in parallel
cd ../project-feature1 && codex exec --full-auto "Implement feature 1" &
cd ../project-feature2 && codex exec --full-auto "Implement feature 2" &
cd ../project-feature3 && codex exec --full-auto "Implement feature 3" &

# Wait for all to complete
wait

# Merge results
cd ../project
git merge feature1 feature2 feature3
```

## Sandbox Limitations (v0.77.0)

Known constraints in sandbox mode:
- Cannot create git branches (`.git/refs` access blocked)
- Cannot install npm packages (network blocked)
- Shell profile errors appear but don't block execution

**Workaround**: Create branches manually before launching Codex, or run outside sandbox.

## Task Prompting

Be specific about what Codex should do:

```bash
# Good: Specific, measurable
codex exec --full-auto "Add unit tests for src/auth.rs covering:
1. Valid login with correct credentials
2. Invalid login with wrong password
3. Rate limiting after 5 failed attempts
Run tests and fix any failures."

# Bad: Vague
codex exec --full-auto "Improve the auth code"
```

## Long-Running Tasks

For tasks taking hours:

```bash
# Start in background with nohup
nohup codex exec --full-auto "Refactor entire codebase to use new error handling pattern" > codex_output.log 2>&1 &

# Check progress
tail -f codex_output.log

# Get process ID for later
echo $! > codex.pid
```

## Integration Pattern

From CLAUDE.md workflow - use Codex for long tasks while continuing other work:

```bash
# Fire off Codex for background task
codex exec --full-auto "Add comprehensive test coverage to all untested modules" &
CODEX_PID=$!

# Continue with other work...
# Later, check if done:
if kill -0 $CODEX_PID 2>/dev/null; then
    echo "Codex still running"
else
    echo "Codex completed"
fi
```

## Best Practices

1. **Use worktrees** - Avoid conflicts with your current work
2. **Be explicit** - Codex works better with clear, detailed instructions
3. **Set expectations** - Long tasks may take hours
4. **Check output** - Review Codex changes before merging
5. **Use read-only for review** - Sandbox mode prevents accidental changes

## Error Recovery

If Codex fails mid-task:
```bash
# Check what was done
git status
git diff

# Reset if needed
git checkout .

# Or commit partial progress
git add -p  # Interactive staging
git commit -m "Partial progress from Codex"
```
