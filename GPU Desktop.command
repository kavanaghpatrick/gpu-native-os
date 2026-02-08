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
echo "Building... (this may take a moment)"

# Build first, then run the binary directly
# This gives better focus behavior than cargo run
cargo build --release --example visual_wasm_apps 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "Launching GPU Desktop..."
    echo "(Click on the window if keyboard focus is not active)"
    echo ""
    # Run the binary directly - slightly better focus behavior
    exec ./target/release/examples/visual_wasm_apps
else
    echo ""
    echo "Build failed. Check the error messages above."
    read -p "Press Enter to close..."
fi
