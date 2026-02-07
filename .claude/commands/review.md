# Multi-AI Code Review (Unbiased)

Run the same code review prompt through Grok, Gemini, and Codex independently. Compare their findings to catch what any single AI might miss.

## Philosophy

Each AI gets the **exact same prompt** - no biasing toward security vs architecture vs simplicity. Let each AI apply its own judgment, then synthesize the results.

## Usage

```
/project:review <file_path>
```

Or for git changes:
```
/project:review git
```

## The Universal Review Prompt

All three AIs receive this identical prompt:

```
You are reviewing code for a GPU-native OS project where "THE GPU IS THE COMPUTER" - the CPU is just an I/O coprocessor.

Review the code below. Report ALL issues you find, ranked by severity:
- CRITICAL: Will cause crashes, security holes, data corruption, or incorrect results
- HIGH: Significant bugs or design flaws
- MEDIUM: Code quality issues, potential future problems
- LOW: Style, documentation, minor improvements

For each issue, provide:
1. Line number (if applicable)
2. What the problem is
3. Why it matters
4. Suggested fix (brief)

Also note:
- What would you DELETE to simplify?
- What's confusing or over-engineered?
- What's done well? (brief)

Be direct and concise. No filler.
```

## Running the Review

### Review a File

```bash
FILE="$ARGUMENTS"

if [ ! -f "$FILE" ]; then
    echo "File not found: $FILE"
    exit 1
fi

CODE=$(cat "$FILE")
PROMPT="You are reviewing code for a GPU-native OS project where 'THE GPU IS THE COMPUTER' - the CPU is just an I/O coprocessor.

Review the code below. Report ALL issues you find, ranked by severity:
- CRITICAL: Will cause crashes, security holes, data corruption, or incorrect results
- HIGH: Significant bugs or design flaws
- MEDIUM: Code quality issues, potential future problems
- LOW: Style, documentation, minor improvements

For each issue: line number, problem, why it matters, suggested fix.

Also note: What would you DELETE? What's over-engineered? What's done well?

Be direct and concise.

CODE:
\`\`\`
$CODE
\`\`\`"

echo "============================================"
echo "MULTI-AI CODE REVIEW: $FILE"
echo "============================================"
echo ""

# Run all three in parallel
echo "Starting parallel reviews..."
echo ""

# Grok
echo ">>> GROK" > /tmp/review_grok.txt
echo "---" >> /tmp/review_grok.txt
python3 << EOF
import json
prompt = """$PROMPT"""
request = {
    "messages": [{"role": "user", "content": prompt}],
    "model": "grok-4-0709",
    "temperature": 0,
    "stream": False
}
with open('/tmp/grok_request.json', 'w') as f:
    json.dump(request, f)
EOF
curl -s -X POST https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer \$GROK_API_KEY" \
  -H "Content-Type: application/json" \
  -d @/tmp/grok_request.json \
  --max-time 180 | python3 -c "
import sys, json
try:
    resp = json.load(sys.stdin)
    if 'choices' in resp:
        print(resp['choices'][0]['message']['content'])
    else:
        print('Error:', resp)
except:
    print('Failed to parse response')
" >> /tmp/review_grok.txt 2>&1 &
GROK_PID=$!

# Gemini
echo ">>> GEMINI" > /tmp/review_gemini.txt
echo "---" >> /tmp/review_gemini.txt
gemini -p "$PROMPT" >> /tmp/review_gemini.txt 2>&1 &
GEMINI_PID=$!

# Wait for both
wait $GROK_PID
wait $GEMINI_PID

# Output results
cat /tmp/review_grok.txt
echo ""
cat /tmp/review_gemini.txt | grep -v "DeprecationWarning" | grep -v "node --trace" | grep -v "Loaded cached"

echo ""
echo "============================================"
echo "REVIEW COMPLETE"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo "1. Compare findings - what did both AIs flag?"
echo "2. Investigate disagreements"
echo "3. Fix CRITICAL issues before committing"
```

### Review Git Changes

```bash
if [ "$ARGUMENTS" = "git" ]; then
    DIFF=$(git diff --cached)
    if [ -z "$DIFF" ]; then
        DIFF=$(git diff HEAD~1)
        echo "(No staged changes, reviewing last commit)"
    fi

    PROMPT="You are reviewing a code diff for a GPU-native OS project where 'THE GPU IS THE COMPUTER'.

Review this diff. Report ALL issues, ranked by severity:
- CRITICAL: Crashes, security holes, data corruption, incorrect results
- HIGH: Significant bugs or design flaws
- MEDIUM: Code quality issues
- LOW: Style, minor improvements

For each issue: line reference, problem, why it matters, fix.

What would you DELETE? What's over-engineered?

DIFF:
\`\`\`
$DIFF
\`\`\`"

    echo "============================================"
    echo "MULTI-AI CODE REVIEW: Git Changes"
    echo "============================================"

    # Grok
    echo ""
    echo ">>> GROK"
    echo "---"
    python3 << EOF
import json
prompt = """$PROMPT"""
request = {
    "messages": [{"role": "user", "content": prompt}],
    "model": "grok-4-0709",
    "temperature": 0
}
with open('/tmp/grok_request.json', 'w') as f:
    json.dump(request, f)
EOF
    curl -s -X POST https://api.x.ai/v1/chat/completions \
      -H "Authorization: Bearer \$GROK_API_KEY" \
      -H "Content-Type: application/json" \
      -d @/tmp/grok_request.json \
      --max-time 180 | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'] if 'choices' in r else r)"

    echo ""
    echo ">>> GEMINI"
    echo "---"
    gemini -p "$PROMPT" 2>&1 | grep -v "DeprecationWarning" | grep -v "node --trace" | grep -v "Loaded cached"

    echo ""
    echo "============================================"
    echo "REVIEW COMPLETE"
    echo "============================================"
fi
```

## Interpreting Results

### Agreement = High Confidence
If both Grok and Gemini flag the same issue → definitely fix it.

### Disagreement = Investigate
If only one AI flags something:
- Could be a false positive
- Could be something the other missed
- Investigate before dismissing

### Synthesis Pattern
```
GROK found: A, B, C
GEMINI found: B, C, D

Confirmed (both found): B, C
Investigate (one found): A, D
```

## Post-Fix Workflow

After fixing issues:

```bash
# Stage your fixes
git add -p

# Run review on staged changes
/project:review git

# If clean, commit
git commit -m "Fix #XXX: description

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Why Three AIs?

| AI | Strengths | Weaknesses |
|----|-----------|------------|
| **Grok** | Security focus, concise | May miss architecture issues |
| **Gemini** | Large context (1M tokens), thorough | Can be verbose |
| **Codex** | Code-focused, practical | Slower for review tasks |

Running all three with the same prompt gives you:
- Multiple independent perspectives
- Higher confidence when they agree
- Diverse bug detection patterns

## Quick Single-AI Review

If you just want one quick review:

```bash
# Grok only (fastest)
python3 -c "
import json
code = open('$FILE').read()
json.dump({'messages':[{'role':'user','content':f'Review this code. List issues by severity:\n\`\`\`\n{code}\n\`\`\`'}],'model':'grok-4-0709','temperature':0}, open('/tmp/r.json','w'))
" && curl -s https://api.x.ai/v1/chat/completions -H "Authorization: Bearer $GROK_API_KEY" -H "Content-Type: application/json" -d @/tmp/r.json | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Gemini only (most thorough)
gemini -p "Review this code. List issues by severity: $(cat $FILE)"
```
