# Diagram Rendering Test

Let's test if Mermaid diagrams render properly in this repository.

## Test 1: Simple Flowchart

```mermaid
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
```

## Test 2: Basic User Workflow

```mermaid
flowchart TD
    A[Fit Model] --> B[Compile Formula] 
    B --> C[Evaluate Rows]
    
    style A fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style B fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style C fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
```

## Test 3: Sequence Diagram

```mermaid
sequenceDiagram
    participant A as User
    participant B as System
    A->>B: Request
    B-->>A: Response
```

If you can see these diagrams rendered (not just the code), then Mermaid is working correctly in this repository.