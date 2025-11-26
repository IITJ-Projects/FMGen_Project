#!/bin/bash

# Script to generate architecture diagram image from Mermaid
# This script requires one of the following:
# 1. mermaid-cli (mmdc) - npm install -g @mermaid-js/mermaid-cli
# 2. Or manually: Open mermaid_diagram.html in browser and take screenshot

echo "Generating architecture diagram..."

# Check if mermaid-cli is available
if command -v mmdc &> /dev/null; then
    echo "Using mermaid-cli to generate diagram..."
    mmdc -i mermaid_diagram.mmd -o architecture_diagram.png -w 1200 -H 800 -b white
    echo "Diagram generated: architecture_diagram.png"
else
    echo "mermaid-cli not found. Please use one of these options:"
    echo ""
    echo "Option 1: Install mermaid-cli"
    echo "  npm install -g @mermaid-js/mermaid-cli"
    echo "  Then run this script again"
    echo ""
    echo "Option 2: Manual conversion"
    echo "  1. Open mermaid_diagram.html in your browser"
    echo "  2. Take a screenshot of the rendered diagram"
    echo "  3. Save it as architecture_diagram.png in this directory"
    echo ""
    echo "Option 3: Use online Mermaid Live Editor"
    echo "  1. Go to https://mermaid.live"
    echo "  2. Paste the Mermaid code from mermaid_diagram.mmd"
    echo "  3. Click 'Actions' -> 'Download PNG' or 'Download SVG'"
    echo "  4. Save as architecture_diagram.png"
fi


