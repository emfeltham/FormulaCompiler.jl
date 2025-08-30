# FormulaCompiler.jl Visual Guide

This document contains the key diagrams illustrating FormulaCompiler.jl's design and usage patterns.

## User Workflow

### Basic Usage Flow
How to go from a fitted statistical model to fast, zero-allocation evaluation:

```mermaid
flowchart TD
    A["Fit Statistical Model<br>GLM.lm, MixedModels.fit"] --> B["Prepare Data<br>Tables.columntable"]
    B --> C["Compile Formula<br>compile_formula"] 
    C --> D["Create Output Vector<br>Vector{Float64}"]
    D --> E["Evaluate Rows<br>compiled(output, data, idx)"]
    E --> F["Process Results<br>Fast evaluation, 0 allocations"]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5  
    style E fill:#e8f5e8
    style F fill:#fff3e0
```

**Example Code:**
```julia
# Fit model
model = lm(@formula(y ~ x * group), df)

# Compile for fast evaluation  
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Zero-allocation evaluation
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, 1)      # First row
compiled(output, data, 500)    # 500th row
```

### Scenario Analysis Workflow
How to use the override system for counterfactual analysis:

```mermaid
flowchart TD
    A["Compiled Formula<br>from Basic Workflow"] --> B{Analysis Type?}
    B -->|Single Override| C["create_scenario"]
    B -->|Parameter Grid| D["create_scenario_grid"]
    
    C --> E["Override Variables<br>x = 2.0, group = Treatment"]
    D --> F["Multiple Combinations<br>x: [1,2,3] × group: [A,B]"]
    
    E --> G["Evaluate Scenario<br>compiled(output, scenario.data, idx)"]
    F --> H["Batch Evaluation<br>6 scenarios (3×2)"]
    
    G --> I["Policy Insights<br>>99% memory savings"]
    H --> I
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style G fill:#e8f5e8
    style H fill:#e8f5e8
    style I fill:#fff3e0
```

**Example Code:**
```julia
# Single scenario
scenario = create_scenario("treatment", data; x = 2.0, group = "Treatment")
compiled(output, scenario.data, 1)

# Parameter grid
grid = create_scenario_grid("sensitivity", data, Dict(
    :x => [1.0, 2.0, 3.0],
    :group => ["Control", "Treatment"]
))
# Automatically creates 6 scenarios with all combinations
```

## System Architecture

### High-Level Component Overview

```mermaid
graph TB
    subgraph "External Ecosystem"
        GLM["GLM.jl<br>Linear Models"]
        MM["MixedModels.jl<br>Mixed Effects"]
        Data["Tables.jl<br>Data Sources"]
        Cat["CategoricalArrays.jl<br>Categorical Data"]
    end
    
    subgraph "FormulaCompiler Core"
        Comp["Compilation System<br>Position Mapping"]
        Exec["Runtime System<br>Zero Allocation Execution"]
        Scen["Scenario System<br>Memory Efficient Overrides"]
        Utils["Core Utilities<br>OverrideVector, helpers"]
    end
    
    subgraph "Performance Results"
        Perf1["~50ns per row"]
        Perf2["Zero allocations"] 
        Perf3[">99% memory savings"]
    end
    
    GLM --> Comp
    MM --> Comp
    Data --> Comp
    Cat --> Comp
    Comp --> Exec
    Data --> Scen
    Utils --> Scen
    Scen --> Exec
    
    Exec --> Perf1
    Exec --> Perf2
    Scen --> Perf3
    
    style Comp fill:#f3e5f5
    style Exec fill:#e8f5e8  
    style Scen fill:#e1f5fe
    style Perf1 fill:#e8f5e8
    style Perf2 fill:#e8f5e8
    style Perf3 fill:#e8f5e8
```

## Technical Implementation

### Compilation Pipeline
How statistical formulas are transformed into zero-allocation evaluators:

```mermaid
flowchart TD
    A["Statistical Model<br>GLM/MixedModel with fitted formula"] --> B["Extract Formula<br>model.mf.f"]
    B --> C["Decompose Terms<br>identify and classify components"]
    
    C --> D{Term Classification}
    D -->|Constants| E["Constant Operation<br>Fixed values in output"]
    D -->|Continuous| F["Continuous Operation<br>Direct column access"]
    D -->|Categorical| G["Categorical Operation<br>Contrast matrix application"]
    D -->|Functions| H["Function Operation<br>log, exp, sqrt, etc."]
    D -->|Interactions| I["Interaction Operation<br>Kronecker product patterns"]
    
    E --> J["Position Analysis<br>Determine output locations"]
    F --> J
    G --> J
    H --> J
    I --> J
    
    J --> K["Memory Layout Planning<br>Scratch space allocation"]
    K --> L["Type Specialization<br>Embed positions in operation types"]
    L --> M["Code Generation<br>Create type-stable evaluator"]
    M --> N["Compiled Evaluator<br>Zero-allocation callable object"]
    
    style A fill:#e1f5fe
    style N fill:#e8f5e8
    style J fill:#f3e5f5
    style L fill:#fff3e0
```

### Runtime Execution Flow
Step-by-step process during evaluation:

```mermaid
sequenceDiagram
    participant User
    participant Compiled as Compiled Evaluator
    participant Data as Data Table
    participant Output as Output Vector
    
    Note over User,Output: Single Row Evaluation (~50ns total)
    
    User->>Compiled: compiled(output, data, row_idx)
    Note over Compiled: Type-stable dispatch<br>All positions embedded in types
    
    loop For each operation in chain
        Compiled->>Data: Access column value
        Note over Data: Val{Column} dispatch<br>Compile-time column resolution
        Data-->>Compiled: Type-stable value
        
        Note over Compiled: Apply operation<br>Position-mapped<br>Zero allocations
        Compiled->>Output: Store result at position
        Note over Output: Direct memory write
    end
    
    Output-->>User: Populated vector
    Note over User: Ready for analysis
```

## Categorical Variable Handling

### Complete Categorical System
How categorical variables and interactions are processed:

```mermaid
graph TD
    subgraph "Compilation Phase"
        A["CategoricalTerm<br>from fitted model"] --> B["Extract Contrast Matrix<br>term.contrasts.matrix"]
        B --> C["Determine Levels<br>term.contrasts.levels"]
        C --> D["Map to Output Positions<br>position specialization"]
    end
    
    subgraph "Runtime Phase"  
        E["Data Row Access"] --> F["Extract Level Code<br>levelcode(categorical_data[idx])"]
        F --> G["Apply Contrast Row<br>contrast_matrix[level_code, :]"]
        G --> H["Store in Output Positions<br>zero allocations"]
    end
    
    subgraph "Interaction Handling"
        I["Component 1<br>group1: 4 levels → 3 contrast cols"]
        J["Component 2<br>group2: 3 levels → 2 contrast cols"]
        
        I --> K["Kronecker Product<br>3 × 2 = 6 interaction columns"]
        J --> K
        
        K --> L["StatsModels Compatible Ordering<br>First component varies fast"]
    end
    
    subgraph "Scenario Integration"
        M["Categorical Override"] --> N["Validate Level Exists<br>Error if invalid"]
        N --> O["Create OverrideVector<br>Constant categorical level"]
        O --> P["Same Performance<br>As original data"]
    end
    
    D --> E
    H --> Output["Final Output Vector<br>Ready for analysis"]
    L --> Output
    P --> F
    
    style B fill:#f3e5f5
    style F fill:#e8f5e8
    style K fill:#fff3e0
    style O fill:#f3e5f5
    style Output fill:#e8f5e8
```

**Key Features:**
- **Perfect StatsModels compatibility**: Uses exact contrast matrices from fitted models
- **All contrast types supported**: DummyCoding, EffectsCoding, HelmertCoding, custom contrasts  
- **Efficient interactions**: Proper Kronecker product implementation with correct ordering
- **Scenario integration**: Categorical overrides work seamlessly with zero-allocation performance

## Performance Summary

FormulaCompiler.jl achieves exceptional performance through:

- **Position mapping**: All output locations determined at compile time
- **Type specialization**: Runtime decisions embedded in compile-time types  
- **Zero allocations**: No intermediate memory allocation during evaluation
- **Memory efficiency**: >99% savings for scenario analysis via OverrideVector
- **Ecosystem integration**: Perfect compatibility with Julia statistical packages

**Benchmark Results**: ~50ns per row, 0 allocations, 10-100x faster than `modelmatrix()[row, :]`