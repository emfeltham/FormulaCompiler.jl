#!/bin/bash

# Mermaid Diagram Compilation Script
#
# This script extracts Mermaid diagrams from markdown files, saves them as .mmd files,
# compiles them to SVG using the Mermaid CLI, and updates the markdown files to reference
# the SVG files instead of inline Mermaid code blocks.
#
# Usage: ./scripts/compile_diagrams.sh
#
# Requirements:
#   - Mermaid CLI: npm install -g @mermaid-js/mermaid-cli

set -e

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="${SCRIPT_DIR}/../docs"
DIAGRAMS_DIR="${DOCS_DIR}/diagrams/generated"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Starting Mermaid diagram compilation...${NC}"

# Check if mermaid CLI is available
if ! command -v mmdc &> /dev/null; then
    echo -e "${RED}❌ Mermaid CLI (mmdc) not found. Please install with:${NC}"
    echo "npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

# Create diagrams directory
mkdir -p "$DIAGRAMS_DIR"
echo -e "${GREEN}✓ Created diagrams directory: $DIAGRAMS_DIR${NC}"

# Counter for naming diagrams
diagram_counter=0

# Function to extract and process mermaid blocks from a file
process_file() {
    local file="$1"
    local rel_path=$(python3 -c "import os; print(os.path.relpath('$file', '$DOCS_DIR'))")
    
    echo -e "${BLUE}Processing: $rel_path${NC}"
    
    # Check if file contains mermaid blocks
    if ! grep -q '```mermaid' "$file"; then
        echo "  No mermaid diagrams found"
        return
    fi
    
    # Create a temporary file for the updated content
    local temp_file=$(mktemp)
    local mermaid_found=false
    local in_mermaid=false
    local current_diagram=""
    local diagram_name=""
    
    # Read file line by line
    while IFS= read -r line; do
        if [[ "$line" == '```mermaid' ]]; then
            in_mermaid=true
            mermaid_found=true
            diagram_counter=$((diagram_counter + 1))
            
            # Generate diagram name based on file path
            local base_name=$(echo "$rel_path" | sed 's/\//_/g' | sed 's/\.md$//')
            diagram_name="${base_name}_diagram_${diagram_counter}"
            
            current_diagram=""
            continue
        elif [[ "$line" == '```' ]] && [[ "$in_mermaid" == true ]]; then
            # End of mermaid block - process it
            in_mermaid=false
            
            # Save mermaid content to .mmd file
            local mmd_file="${DIAGRAMS_DIR}/${diagram_name}.mmd"
            echo "$current_diagram" > "$mmd_file"
            echo -e "  ${GREEN}✓ Saved: $(basename "$mmd_file")${NC}"
            
            # Compile to SVG
            local svg_file="${DIAGRAMS_DIR}/${diagram_name}.svg"
            if mmdc -i "$mmd_file" -o "$svg_file" -b white 2>/dev/null; then
                echo -e "  ${GREEN}✓ Compiled: $(basename "$svg_file")${NC}"
                
                # Generate relative path from markdown file to SVG (macOS compatible)
                local file_dir=$(dirname "$file")
                local svg_rel_path=$(python3 -c "import os; print(os.path.relpath('$svg_file', '$file_dir'))")
                
                # Add SVG reference to updated content
                echo "![Diagram]($svg_rel_path)" >> "$temp_file"
            else
                echo -e "  ${RED}❌ Failed to compile: $diagram_name${NC}"
                # Fall back to original mermaid block
                echo '```mermaid' >> "$temp_file"
                echo "$current_diagram" >> "$temp_file"
                echo '```' >> "$temp_file"
            fi
            
            current_diagram=""
            continue
        fi
        
        if [[ "$in_mermaid" == true ]]; then
            # Accumulate mermaid content
            if [[ -n "$current_diagram" ]]; then
                current_diagram="$current_diagram"$'\n'"$line"
            else
                current_diagram="$line"
            fi
        else
            # Regular content - copy as-is
            echo "$line" >> "$temp_file"
        fi
    done < "$file"
    
    # Replace original file if we found mermaid diagrams
    if [[ "$mermaid_found" == true ]]; then
        mv "$temp_file" "$file"
        echo -e "  ${GREEN}✓ Updated markdown file${NC}"
    else
        rm "$temp_file"
    fi
}

# Find and process all markdown files
echo -e "${BLUE}📋 Finding markdown files...${NC}"

find "$DOCS_DIR" -name "*.md" -type f | while read -r file; do
    process_file "$file"
    echo
done

echo -e "${GREEN}✅ Diagram compilation complete!${NC}"
echo -e "${BLUE}📁 Generated files are in: $DIAGRAMS_DIR${NC}"