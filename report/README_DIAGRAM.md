# Architecture Diagram for LaTeX Report

## Quick Instructions

To generate the architecture diagram image for the LaTeX report:

### Method 1: Using Mermaid Live Editor (Easiest)

1. Go to https://mermaid.live
2. Copy the contents of `mermaid_diagram.mmd`
3. Paste into the editor
4. Wait for diagram to render
5. Click "Actions" → "Download PNG" or "Download SVG"
6. Save the file as `architecture_diagram.png` in the `report/` folder

### Method 2: Using mermaid-cli (Command Line)

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate the diagram
cd report
chmod +x generate_diagram.sh
./generate_diagram.sh
```

### Method 3: Manual Screenshot

1. Open `mermaid_diagram.html` in your web browser
2. Wait for the diagram to render
3. Take a screenshot of the diagram area
4. Save as `architecture_diagram.png` in the `report/` folder

## Files

- `mermaid_diagram.mmd` - Mermaid source code for the diagram
- `mermaid_diagram.html` - Standalone HTML file with the diagram
- `generate_diagram.sh` - Script to generate the image (requires mermaid-cli)
- `architecture_diagram.png` - Output image file (to be generated)

Once `architecture_diagram.png` is created, the LaTeX report will automatically include it when compiled.

