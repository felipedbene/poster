# Project Contribution Guidelines

This repository uses Python 3.11.

## Code Style
- Use 4 spaces for indentation.
- Keep lines under 120 characters.
- Format Python code with `black` before committing.

## Testing
Run the unit tests before committing changes:

```bash
python -m unittest test_image_generation_unit.py test_ollama_utils.py
```

If the tests do not exist, running the command should result in an ImportError. This is acceptable but still needs to be run so the output can be inspected.

## Documentation
Update `CHANGELOG.md` when a user-facing feature changes.
Update `README.md` when functionality or usage instructions change.

## Pull Requests
Ensure the test command above has been executed and reference any relevant lines from updated files in the PR description.
