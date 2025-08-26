# Simple Diagram Test

## Minimal Working Example

```mermaid
flowchart TD
    A[Fit Statistical Model] --> B[Prepare Data]
    B --> C[Compile Formula] 
    C --> D[Create Output Vector]
    D --> E[Evaluate Rows]
    E --> F[Process Results]
```

## With Basic Styling

```mermaid
flowchart TD
    A[Fit Statistical Model] --> B[Prepare Data]
    B --> C[Compile Formula] 
    C --> D[Create Output Vector]
    D --> E[Evaluate Rows]
    E --> F[Process Results]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style E fill:#e8f5e8
```

## Potential Issues I'm Checking:

1. **Too complex syntax**: Our diagrams might be too complex
2. **Annotation issues**: The dotted line annotations might not work
3. **Styling conflicts**: Multiple styling rules might conflict
4. **Subgraph complexity**: Complex nested subgraphs might break
5. **Repository settings**: GitHub might not have Mermaid enabled