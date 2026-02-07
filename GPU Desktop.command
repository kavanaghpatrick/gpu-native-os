#!/bin/bash
# GPU Desktop Launcher
# Double-click this file to launch the GPU Desktop Environment

cd "$(dirname "$0")"

echo "╔═══════════════════════════════════════╗"
echo "║     GPU Desktop Environment           ║"
echo "╠═══════════════════════════════════════╣"
echo "║  Controls:                            ║"
echo "║  • Drag title bars to move windows    ║"
echo "║  • Drag edges to resize               ║"
echo "║  • Arrow keys to navigate apps        ║"
echo "║  • Type in Terminal or Editor         ║"
echo "║  • ESC to quit                        ║"
echo "╚═══════════════════════════════════════╝"
echo ""

cargo run --release --example gpu_desktop

echo ""
echo "GPU Desktop closed."
