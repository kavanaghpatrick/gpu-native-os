#!/bin/bash
# Benchmark all gpu_ripgrep modes vs vanilla ripgrep
# Usage: ./benchmark_modes.sh <pattern> <directory>

PATTERN="${1:-fn}"
DIR="${2:-.}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}GPU Ripgrep Benchmark: All Modes vs Vanilla Ripgrep${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "Pattern: ${YELLOW}\"$PATTERN\"${RESET}"
echo -e "Directory: ${YELLOW}$DIR${RESET}"
echo ""

# Count files
FILE_COUNT=$(find "$DIR" -type f \( -name "*.rs" -o -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.go" -o -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.md" -o -name "*.txt" -o -name "*.json" \) 2>/dev/null | wc -l | tr -d ' ')
echo -e "Searchable files: ${CYAN}$FILE_COUNT${RESET}"
echo ""

# Build release version
echo -e "${YELLOW}Building release...${RESET}"
cargo build --release --example gpu_ripgrep 2>/dev/null
echo ""

# Clear caches between runs
clear_cache() {
    sync
    sudo purge 2>/dev/null || true
}

# Run benchmark
run_benchmark() {
    local name="$1"
    local cmd="$2"
    local runs=3
    local times=()

    for i in $(seq 1 $runs); do
        # Get timing, suppress output
        local start=$(python3 -c "import time; print(time.time())")
        eval "$cmd" > /dev/null 2>&1
        local end=$(python3 -c "import time; print(time.time())")
        local elapsed=$(python3 -c "print(($end - $start) * 1000)")
        times+=("$elapsed")
    done

    # Calculate average
    local avg=$(python3 -c "print(sum([$( IFS=,; echo "${times[*]}" )]) / $runs)")
    printf "  %-20s %7.1f ms\n" "$name:" "$avg"
}

echo -e "${BOLD}Benchmark Results (average of 3 runs):${RESET}"
echo -e "${CYAN}──────────────────────────────────────────${RESET}"

# Ripgrep baseline (warm cache)
echo -e "\n${YELLOW}Warming caches...${RESET}"
rg "$PATTERN" "$DIR" > /dev/null 2>&1
cargo run --release --example gpu_ripgrep -- "$PATTERN" "$DIR" > /dev/null 2>&1

echo -e "\n${BOLD}Warm Cache Results:${RESET}"
run_benchmark "ripgrep" "rg '$PATTERN' '$DIR'"
run_benchmark "gpu-batch" "cargo run --release --example gpu_ripgrep -- '$PATTERN' '$DIR' --batch"
run_benchmark "gpu-streaming" "cargo run --release --example gpu_ripgrep -- '$PATTERN' '$DIR' --streaming"
run_benchmark "gpu-persistent" "cargo run --release --example gpu_ripgrep -- '$PATTERN' '$DIR' --persistent"
run_benchmark "gpu-mmap" "cargo run --release --example gpu_ripgrep -- '$PATTERN' '$DIR' --mmap"

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}Detailed Run (streaming mode):${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
cargo run --release --example gpu_ripgrep -- "$PATTERN" "$DIR" --streaming -m 5
