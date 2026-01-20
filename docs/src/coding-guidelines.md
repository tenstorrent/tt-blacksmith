# Coding Guidelines

## Introduction
This document outlines the coding style guidelines for the TT-Blacksmith project. Every contributor is expected to adhere to these standards to ensure code consistency and maintainability.

## General Guidelines

### Line Length
Keep all lines (including imports and comments) to a maximum of **88 characters** (the Black default). This prevents horizontal scrolling and improves code review readability.

### Formatting
To maintain a consistent style across the codebase, we use the following tools:
- Black: The uncompromising code formatter that ensures a uniform layout.
- isort: Automatically sorts and categorizes your imports.

### Pre-commit Hooks
We use pre-commit to automate these checks. This ensures that every commit meets our quality standards before it even leaves your local machine.
To set this up, run the following in your terminal:
1. `pip install pre-commit`
2. `pre-commit install`

Once installed, these tools will run automatically every time you execute git commit. If a tool finds an issue, it will either fix it automatically or block the commit until you address the error.

### Early Returns
Always prefer early returns to reduce nesting and improve readability. This means checking for conditions that would lead to an early exit from a function at the beginning of the function body.

Example:
```python
def process_data(data):
    if not data:
        return None
    # Continue processing data
```

### Magic Numbers
Avoid using magic numbers directly in the code. Instead, define them as named constants at the top of the module or in a separate configuration file. This improves code readability and makes it easier to update values in the future.

Bad example:
```python
def do_compute(x):
    return 14 * x + 52.33
```

### Complex Conditionals
For complex conditional statements, break them down into smaller, well-named boolean variables.

Bad example:
```python
if (tensor.shape[0] > 0 and tensor.dtype == torch.float32 and tensor.device.type == 'tt'):
    pass
```

Good example:
```python
is_non_empty = tensor.shape[0] > 0
is_float32 = tensor.dtype == torch.float32
is_tt = tensor.device.type == 'tt'
if is_non_empty and is_float32 and is_tt:
    pass
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

Example:
```python
from abc import ABC, abstractmethod
from typing import override

class Model(ABC):
    @abstractmethod
    def predict(self, x):
        pass

class MyModel(Model):
    @override
    def predict(self, x):
        return x * 2
```

### String Interpolation
Use f-strings where possible for string interpolation for better readability over other methods like `str.format()` or concatenation. When using f-strings, ensure that the expressions inside the curly braces are simple and do not contain complex logic.

Example:
```python
value = 42
message = f"The answer is {value}."
```

Bad example:
```python
message = f"The answer is {calculate_answer()}."  # Avoid complex expressions

message = f"The answer is {value * 2}."  # Avoid using expressions inside f-strings
```

### Library Standards

#### Path Manipulation
Use `pathlib` over `os.path` for all path manipulations. The `os` module treats paths as strings, which can lead to errors and less readable code. `pathlib` provides an object-oriented approach to handle filesystem paths.

Example:
```python
from pathlib import Path
data_dir = Path('./data/')
file_path = data_dir / 'file.csv'
```

#### Parallelism
Use `joblib` for parallel processing tasks. It provides a simple and efficient way to run tasks in parallel, with easy-to-use abstractions.

Example:
```python
from joblib import Parallel, delayed
def process_item(item):
    pass
results = Parallel(n_jobs=4)(delayed(process_item)(item) for item in items)
```

#### Data Modeling
Use `pydantic` models for data validation and configurations. Use `dataclasses` for simple data containers that do not require validation.

Example:
```python
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class DataPoint:
    x: float
    y: float

class ConfigModel(BaseModel):
    learning_rate: float
    batch_size: int
```

#### Type Hints
Use modern type hinting features available in Python 3.9 and above, such as built-in generic types (e.g., `list[int]` instead of `List[int]` from `typing`). Also leverage `Self`, `TypeAlias`, and union types using the `|` operator. Strongly refrain from using the `Any` type.

Example:
```python
def process_data(data: list[int] | None) -> dict[str, float]:
    pass
```

## Logging
Use the provided logging utilities from `blacksmith.tools.logging` for all logging purposes. Avoid using print statements for debugging or information output.


## Naming Conventions
Follow the PEP 8 naming conventions with the following specifics:
- Classes: Use PascalCase (e.g., `DataManager`).
- Functions & Variables: Use snake_case (e.g., `calculate_offset`).
- Constants: Use SCREAMING_SNAKE_CASE (e.g., `MAX_RETRIES = 5`).
- Private Members: Prefix with a single underscore for internal package/class use (e.g., `_internal_method`).

Bad examples:
```python
# vague and non-descriptive names
def func1(x):
    pass

a = 10
b = 20

# unclear class names
class Data:
    pass

# overly long names
def calculate_the_total_sum_of_all_elements(input_list):
    pass

# overly short names
def lv(num):
    if num > 10:
        print("Value is greater than ten")


# names that explain the logic
def log_if_value_is_greater_than_ten(num):
    if num > 10:
        print("Value is greater than ten")
```


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

    Raises:
        ValueError: If `param2` is equal to `param1`

    """
    return a + b
```

## Comments and TODOs

Comments should explain why the code exists or non-obvious choices. Always use full sentences and proper punctuation. Avoid stating the obvious or redundant comments that do not add value.

**Bad** examples:
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
---
**Good** examples:
```python
# Explains WHY, not what.
# Use mixed precision to fit larger batch sizes in memory.
with torch.cuda.amp.autocast():
    output = model(input)

# Documents non-obvious behavior.
# Detach to prevent gradients flowing back through the target network.
# This is critical for stable Q-learning.
target_q = target_net(next_state).detach()

#  Explains math or algorithms.
# Apply temperature scaling before softmax to control prediction confidence.
# Higher T → more uniform distribution, lower T → sharper peaks.
logits = logits / temperature

# Warning about gotchas.
# Note: PyTorch's CrossEntropyLoss expects raw logits, not probabilities.
# Don't apply softmax before passing to this loss function.
loss = F.cross_entropy(logits, labels)

# Documents parameter choices
# Warmup for 10% of training stabilizes learning with large batch sizes.
warmup_steps = int(0.1 * total_steps)
```

Keep comments up to date. Outdated comments are worse than none.
Write comments as full sentences when possible.

Use a consistent TODO format that is searchable and includes your GitHub username as well as the issue for accountability:

```python
# TODO(pglusacTT): Support sharded training checkpoints. See https://github.com/tenstorrent/tt-blacksmith/issues/...
```

If there is a corresponding GitHub issue or ticket, reference it. If not, create one.

Hacks or temporary solutions should be clearly marked with a `# HACK` comment, along with an explanation and a reference to a GitHub issue.

```python
# HACK(pglusacTT): Fix by doing something ugly. See https://github.com/tenstorrent/tt-blacksmith/issues/...
```


## Packaging
All code must be organized into packages and modules. If it is not a package/module, it should not be part of the codebase. Use the standard Python packaging structure, with an `__init__.py` file in each package directory.

## Imports
When importing, make sure to adhere to the following:
- Use absolute imports whenever possible. This improves readability and avoids potential issues with relative imports.
- Import length:
    - **If importing ≤ 5 items**:
    Import them directly.
    ```python
    from torch import Tensor, nn, optim
    ```
    - **If importing > 5 items**:
    Import the parent module instead to keep the namespace clean.
    ```python
    import torch.nn.functional as F
    ```
- Avoid wildcards and import only what is necessary.

    Bad example:
    ```python
    from module import *
    import blacksmith.module
    ```

Example:
```python
from blacksmith.module import MyClass
from blacksmith.utils.helpers import my_function, another_function
```
