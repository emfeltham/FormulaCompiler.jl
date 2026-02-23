// Load Mermaid from CDN
(function() {
    // Create script element for Mermaid
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
    script.onload = function() {
        // Initialize Mermaid once loaded
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            themeVariables: {
                fontFamily: 'inherit',
                fontSize: '16px'
            },
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            },
            sequence: {
                diagramMarginX: 50,
                diagramMarginY: 10,
                actorMargin: 50,
                width: 150,
                height: 65,
                boxMargin: 10,
                boxTextMargin: 5,
                noteMargin: 10,
                messageMargin: 35
            }
        });
        
        // Process existing code blocks
        document.querySelectorAll('pre code.language-mermaid').forEach(function(block) {
            var pre = block.parentNode;
            var container = document.createElement('div');
            container.className = 'mermaid';
            container.textContent = block.textContent;
            pre.parentNode.replaceChild(container, pre);
        });
        
        // Re-run Mermaid on the new elements
        mermaid.init();
    };
    
    // Add to head
    document.head.appendChild(script);
    
    // Also add some CSS for better styling
    var style = document.createElement('style');
    style.textContent = `
        .mermaid {
            text-align: center;
            margin: 1em 0;
        }
        .mermaid svg {
            max-width: 100%;
            height: auto;
        }
    `;
    document.head.appendChild(style);
})();