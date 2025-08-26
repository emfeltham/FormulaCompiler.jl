# GitHub Mermaid Test

Testing if diagrams render on GitHub.

## Simple User Workflow

```mermaid
flowchart TD
    A["Fit Statistical Model<br>GLM.lm, MixedModels.fit"] --> B["Prepare Data<br>Tables.columntable"]
    B --> C["Compile Formula<br>compile_formula"] 
    C --> D["Create Output Vector<br>Vector{Float64}"]
    D --> E["Evaluate Rows<br>compiled(output, data, idx)"]
    E --> F["Process Results<br>~50ns per row, 0 allocations"]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5  
    style E fill:#e8f5e8
    style F fill:#fff3e0
```

## Simple System Architecture

```mermaid
graph TB
    subgraph "External Ecosystem"
        GLM[GLM.jl<br/>Linear Models]
        MM[MixedModels.jl<br/>Mixed Effects]  
        Data[Tables.jl<br/>Data Sources]
    end
    
    subgraph "FormulaCompiler Core"
        Comp[Compilation System<br/>Position Mapping]
        Exec[Runtime System<br/>Zero Allocation]
        Scen[Scenario System<br/>Memory Efficient]
    end
    
    GLM --> Comp
    MM --> Comp
    Data --> Comp
    Comp --> Exec
    Data --> Scen
    Scen --> Exec
    
    style Comp fill:#f3e5f5
    style Exec fill:#e8f5e8  
    style Scen fill:#e1f5fe
```

## Basic Sequence

```mermaid
sequenceDiagram
    participant User
    participant Compiled
    participant Data
    
    User->>Compiled: compiled(output, data, idx)
    Compiled->>Data: Extract values
    Data-->>Compiled: Column values  
    Compiled-->>User: Populated output
```

If these render as diagrams (not code blocks), then GitHub Mermaid is working!