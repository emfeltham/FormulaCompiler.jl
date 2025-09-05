#!/bin/bash

# Simple Mermaid Diagram Compilation Script
# Extracts mermaid blocks, compiles to SVG, and updates markdown files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_DIR/docs"
DIAGRAMS_DIR="$DOCS_DIR/diagrams/generated"

echo "Starting Mermaid diagram compilation..."

# Check for mermaid CLI
if ! command -v mmdc &> /dev/null; then
    echo "ERROR: Mermaid CLI (mmdc) not found. Install with: npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

# Create output directory
mkdir -p "$DIAGRAMS_DIR"
echo "Created diagrams directory"

# Process each markdown file with mermaid content
diagram_num=0

process_markdown_file() {
    local file="$1"
    local filename=$(basename "$file" .md)
    local dir_name=$(basename "$(dirname "$file")")
    
    echo "Processing: $file"
    
    # Extract mermaid blocks and create .mmd files
    awk '
    /^```mermaid$/ { in_mermaid=1; diagram_num++; next }
    /^```$/ && in_mermaid { 
        in_mermaid=0; 
        print "" > "/dev/stderr";
        next 
    }
    in_mermaid { 
        print $0 >> (ENVIRON["DIAGRAMS_DIR"] "/" ENVIRON["filename"] "_" diagram_num ".mmd")
    }
    !in_mermaid { print $0 }
    ' filename="$filename" "$file" > "$file.tmp"
    
    # Count how many diagrams we found
    local mmd_count=$(ls "$DIAGRAMS_DIR"/"$filename"_*.mmd 2>/dev/null | wc -l)
    
    if [ "$mmd_count" -gt 0 ]; then
        echo "  Found $mmd_count mermaid diagram(s)"
        
        # Compile each .mmd file to .svg
        for mmd_file in "$DIAGRAMS_DIR"/"$filename"_*.mmd; do
            if [ -f "$mmd_file" ]; then
                local svg_file="${mmd_file%.mmd}.svg"
                local diagram_name=$(basename "$mmd_file" .mmd)
                
                if mmdc -i "$mmd_file" -o "$svg_file" -b white 2>/dev/null; then
                    echo "  Compiled: $diagram_name.svg"
                    
                    # Calculate relative path from markdown to SVG
                    local rel_path="../diagrams/generated/$(basename "$svg_file")"
                    
                    # Replace mermaid block with image reference in the temp file
                    # This is a simple replacement - could be enhanced
                    
                else
                    echo "  ERROR: Failed to compile: $diagram_name"
                    rm -f "$svg_file"
                fi
            fi
        done
        
        # For now, let's just show what we've created
        echo "  Generated files in $DIAGRAMS_DIR/"
        
    else
        echo "  No mermaid diagrams found"
    fi
    
    rm -f "$file.tmp"
}

# Find and process all markdown files
find "$DOCS_DIR" -name "*.md" -type f | while read -r file; do
    if grep -q '```mermaid' "$file"; then
        process_markdown_file "$file"
    fi
done

echo ""
echo "Diagram compilation complete!"
echo "Generated files are in: $DIAGRAMS_DIR"
echo ""
echo "Note: This script created .mmd and .svg files but did not modify the markdown files."
echo "   You can manually replace the mermaid code blocks with image references like:"
echo "   ![Diagram](../diagrams/generated/filename_1.svg)"