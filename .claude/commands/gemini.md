# Gemini CLI - Expert Analysis

Use Google's Gemini CLI for architecture analysis, security reviews, and complex debugging. Supports up to 1M tokens context.

## When to Use

- Architecture analysis and design review
- Security vulnerability assessment
- Complex debugging requiring large context
- When you need to analyze many files at once (1M token context)

## Basic Usage

```bash
gemini -p "$ARGUMENTS"
```

## Code Review Pattern

```bash
gemini -p "Review this code for critical issues (crashes, security, memory safety):

$(cat PATH_TO_FILE)"
```

## Multi-File Analysis

Gemini's 1M token context makes it ideal for analyzing multiple files:

```bash
gemini -p "Analyze these related files for consistency issues:

=== File 1: src/module.rs ===
$(cat src/module.rs)

=== File 2: src/module.metal ===
$(cat src/module.metal)

Look for:
1. Buffer binding mismatches
2. Struct alignment issues
3. Missing error handling"
```

## Architecture Review

```bash
gemini -p "Review this codebase architecture:

$(find src -name '*.rs' -exec echo '=== {} ===' \; -exec head -50 {} \;)

Evaluate:
1. Module organization
2. Dependency flow
3. Separation of concerns
4. Potential coupling issues"
```

## Security Audit

```bash
gemini -p "Security audit for this code:

$(cat src/security_critical.rs)

Check for:
1. Input validation gaps
2. Buffer overflows
3. Race conditions
4. Information leakage"
```

## Comparative Analysis

```bash
gemini -p "Compare these two implementations and identify which is safer:

=== Implementation A ===
$(cat impl_a.rs)

=== Implementation B ===
$(cat impl_b.rs)"
```

## Tips

1. **Use for large context** - Gemini handles 1M tokens, ideal for multi-file analysis
2. **Be specific** - Clear prompts get better results
3. **Pipe output** - `gemini -p "..." | tee analysis.md` to save results
4. **Quota limits** - Free tier has daily limits; may hit "quota exhausted" errors

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `TerminalQuotaError` | Daily quota exhausted | Wait for reset (shows time) |
| `Authentication error` | Not logged in | Run `gemini auth login` |
| `Request too large` | Exceeded context | Split into smaller requests |

## Integration with Claude Code

When stuck on a problem for >5 minutes, launch parallel Gemini + Grok analysis:

```bash
# Terminal 1: Gemini for "why"
gemini -p "Why might this code cause [problem]? $(cat file.rs)"

# Terminal 2: Grok for "fix"
# (use /project:grok skill)
```
