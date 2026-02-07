# Grok API - Code Review and Analysis

Use the Grok API (xAI) to get an external AI perspective on code, focusing on CRITICAL issues (crashes, security, data loss).

## When to Use

- Code reviews focusing on critical bugs
- Security vulnerability analysis
- Getting a second opinion on complex logic
- When you need a different AI's perspective

## API Call Pattern

**IMPORTANT**: Always use Python for JSON escaping to avoid shell quoting issues.

```bash
python3 << 'EOF'
import json
import sys

# Read the code or prompt
code_or_prompt = """
$ARGUMENTS
"""

# Build request
request = {
    "messages": [
        {
            "role": "system",
            "content": "You are an expert code reviewer. Focus on CRITICAL issues only: crashes, security vulnerabilities, data loss, race conditions. Ignore style issues."
        },
        {
            "role": "user",
            "content": code_or_prompt
        }
    ],
    "model": "grok-4-0709",
    "temperature": 0,
    "stream": False
}

# Write to temp file
with open('/tmp/grok_request.json', 'w') as f:
    json.dump(request, f)

print("Request written to /tmp/grok_request.json")
EOF

curl -s -X POST https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $GROK_API_KEY" \
  -H "Content-Type: application/json" \
  -d @/tmp/grok_request.json \
  --max-time 300 | python3 -c "
import sys, json
try:
    resp = json.load(sys.stdin)
    if 'choices' in resp:
        print(resp['choices'][0]['message']['content'])
    elif 'error' in resp:
        print(f\"Error: {resp['error']}\", file=sys.stderr)
    else:
        print(json.dumps(resp, indent=2))
except Exception as e:
    print(f'Parse error: {e}', file=sys.stderr)
"
```

## For File Review

To review a specific file:

```bash
python3 << 'EOF'
import json

# Read the file
with open('PATH_TO_FILE', 'r') as f:
    code = f.read()

request = {
    "messages": [
        {
            "role": "system",
            "content": "You are an expert code reviewer. Focus on CRITICAL issues: crashes, security, data loss, race conditions, memory safety. Be concise."
        },
        {
            "role": "user",
            "content": f"Review this code for critical issues:\n\n```\n{code}\n```"
        }
    ],
    "model": "grok-4-0709",
    "temperature": 0,
    "stream": False
}

with open('/tmp/grok_request.json', 'w') as f:
    json.dump(request, f)
EOF

curl -s -X POST https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $GROK_API_KEY" \
  -H "Content-Type: application/json" \
  -d @/tmp/grok_request.json \
  --max-time 300 | python3 -c "
import sys, json
resp = json.load(sys.stdin)
if 'choices' in resp:
    print(resp['choices'][0]['message']['content'])
else:
    print(json.dumps(resp, indent=2))
"
```

## Available Models

Check your available models:
```bash
curl -s https://api.x.ai/v1/models -H "Authorization: Bearer $GROK_API_KEY" | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[])]"
```

Common models (availability depends on API key permissions):

| Model | Best For | Speed |
|-------|----------|-------|
| `grok-4-0709` | Deep reasoning, security audits, code review | Medium |
| `grok-2-image-1212` | Image analysis | Medium |

**Note**: Your API key may only have access to specific models. Check console.x.ai to manage permissions.

## Timeout

Set `--max-time 300` (5 minutes) for complex requests. The CLAUDE.md specifies 300000ms timeout.

## Error Handling

Common errors:
- `401`: Check `$GROK_API_KEY` is set in ~/.zshrc
- `429`: Rate limited - wait and retry
- `timeout`: Increase --max-time or simplify request
