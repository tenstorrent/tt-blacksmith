# Coding Style Guidelines

## Introduction
This document outlines the coding style guidelines for the tt-blacksmith project. Every contributor is expected to adhere to these standards to ensure code consistency and maintainability.

## General Guidelines

### Early Returns
Always prefer early returns to reduce nesting and improve readability. This means checking for conditions that would lead to an early exit from a function at the beginning of the function body.

Example:
```python
def process_data(data):
    if not data:
        return None
    # Continue processing data
```

### Keyword Arguments
When calling functions with multiple parameters, especially when some parameters have default values, use keyword arguments for clarity.

Example:
```python
def create_tensor(shape, dtype=float, device='tt'):
    pass

tensor = create_tensor(shape=(3, 4), device='cpu')
```

### Class Design
- Only use inheritance when there is a clear "is-a" relationship. Prefer composition over inheritance to promote code reuse and flexibility.
- Use abstract base classes (ABCs) to define interfaces when necessary, but avoid overusing them. Only create an ABC when there is a clear need for multiple implementations of the same interface.
- Favor data classes for simple data containers to reduce boilerplate code.
- Always implement the `__repr__` method for classes to provide a clear string representation, which aids in debugging.


## Logging
Use the provided logging utilities from `blacksmith.tools.logging` for all logging purposes. Avoid using print statements for debugging or information output.


## Naming Conventions
Follow the PEP 8 naming conventions with the following specifics:
- Classes: Use PascalCase (e.g., DataManager).
- Functions & Variables: Use snake_case (e.g., calculate_offset).
- Constants: Use SCREAMING_SNAKE_CASE (e.g., MAX_RETRIES = 5).
- Private Members: Prefix with a single underscore for internal package/class use (e.g., _internal_method).


## Type Annotations
All functions and methods **must** include type annotations for parameters and return types, except in cases where it is extremely obvious (e.g., main function returns None) or impractical. Avoid using the `Any` type unless absolutely necessary, and prefer more specific types whenever possible.

## Docstrings
Use the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings.

Docstring should be divided into sections with clear headings for parameters, return values, and exceptions (if applicable). Given that we are using type annotations, the type information in the docstring should, for the most part, be omitted.

Example:
```python
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.
    
    Returns:
        The sum of the two numbers.
    """
    return a + b
```

## Comments and TODOs

Comments should explain why the code exists or non-obvious choices. 

Bad examples:
```python
# Stating the obvious
x = torch.randn(32, 3, 224, 224)  # Create a tensor

# Redundant with code
model.train()  # Set model to training mode

# Outdated or misleading
# TODO: Fix this later (written 2 years ago, never fixed)
loss = criterion(output, target)

# Commented-out code
# model = ResNet50()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model = EfficientNet()

# Vague and unhelpful
# Do the thing
hidden = self.transformer(x)
```

Good examples:
```python
# Explains WHY, not what
# Use mixed precision to fit larger batch sizes in memory
with torch.cuda.amp.autocast():
    output = model(input)

# Documents non-obvious behavior
# Detach to prevent gradients flowing back through the target network
# This is critical for stable Q-learning
target_q = target_net(next_state).detach()

#  Explains math or algorithms
# Apply temperature scaling before softmax to control prediction confidence
# Higher T → more uniform distribution, lower T → sharper peaks
logits = logits / temperature

# Warning about gotchas
# Note: PyTorch's CrossEntropyLoss expects raw logits, not probabilities
# Don't apply softmax before passing to this loss function
loss = F.cross_entropy(logits, labels)

# Documents parameter choices
# Warmup for 10% of training stabilizes learning with large batch sizes
warmup_steps = int(0.1 * total_steps)
```

Keep comments up to date. Outdated comments are worse than none.
Write comments as full sentences when possible.

Use a consistent TODO format that is searchable and includes your GitHub username as well as the issue for accountability: 

```python
# TODO(pglusac): Support sharded training checkpoints. See https://github.com/tenstorrent/tt-blacksmith/issues/... 
```



If there is a corresponding GitHub issue or ticket, reference it.

## Packaging
All code must be organized into packages and modules. If it is not a package/module, it should not be part of the codebase. Use the standard Python packaging structure, with an `__init__.py` file in each package directory.

## Imports
Use absolute imports whenever possible. This improves readability and avoids potential issues with relative imports.

Example:
```python
from blacksmith.module import MyClass
from blacksmith.utils.helpers import my_function
```

Make sure to import only what is necessary to keep the namespace clean.

Avoid:
```python
from blacksmith.module import *
import blacksmith.module
```

