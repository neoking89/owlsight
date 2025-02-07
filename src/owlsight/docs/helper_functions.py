import ast
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

def parse_init_file(init_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse __init__.py file to get imports and their original paths.
    """
    with open(init_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    import_map = {}
    all_list = []
    
    # Get the base package name from the init path
    pkg_path = Path(init_path).parent
    base_package = pkg_path.name
    
    for node in ast.walk(tree):
        # Get __all__ list
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.List):
                        all_list = [
                            element.value
                            for element in node.value.elts 
                            if isinstance(element, ast.Constant)
                        ]
        
        # Get imports
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for name in node.names:
                import_name = name.asname or name.name
                # Handle relative imports
                if node.level > 0:
                    # Convert relative import to absolute
                    if module:
                        import_path = f"{base_package}.{module}.{name.name}"
                    else:
                        import_path = f"{base_package}.{name.name}"
                else:
                    import_path = f"{module}.{name.name}"
                import_map[import_name] = import_path
    
    return all_list, import_map

def get_object_docstring(import_path: str) -> Dict[str, str]:
    """
    Get docstring for an object and its methods if it's a class.
    """
    try:
        module_path, object_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, object_name)
        
        result = {}
        
        # Get object's own docstring and signature
        if doc := inspect.getdoc(obj):
            result['docstring'] = doc.strip()
        
        try:
            result['signature'] = str(inspect.signature(obj))
        except (ValueError, TypeError):
            result['signature'] = ""
        
        # If it's a class, get method docstrings and signatures
        if inspect.isclass(obj):
            result['methods'] = {}
            for name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if not name.startswith('_'):  # Skip private methods
                    method_info = {}
                    if method_doc := inspect.getdoc(method):
                        method_info['docstring'] = method_doc.strip()
                    try:
                        method_info['signature'] = str(inspect.signature(method))
                    except (ValueError, TypeError):
                        method_info['signature'] = ""
                    result['methods'][name] = method_info
        
        return result
    except Exception as e:
        print(f"Error processing {import_path}: {str(e)}")
        return {}

def get_init_docstrings(init_path: str) -> Dict[str, Dict[str, str]]:
    """
    Get docstrings for all objects in __all__ list.
    """
    # Parse the init file
    all_list, import_map = parse_init_file(init_path)
    
    # Get docstrings for each object
    docstrings = {}
    for obj_name in all_list:
        if obj_name in import_map:
            import_path = import_map[obj_name]
            docstrings[obj_name] = get_object_docstring(import_path)
    
    return docstrings

def format_docstrings(docstrings: Dict[str, Dict[str, str]]) -> str:
    """Format docstrings into a readable string with proper indentation."""
    output = []
    
    # Group by type
    classes = {}
    functions = {}
    for obj_name, docs in docstrings.items():
        if 'methods' in docs:
            classes[obj_name] = docs
        else:
            functions[obj_name] = docs
    
    # Format classes
    if classes:
        output.append("\n### Classes\n")
        for class_name, docs in classes.items():
            # Class header with signature
            sig = docs.get('signature', '')
            output.append(f"#### {class_name}\n")
            if sig:
                output.append("```python")
                output.append(f"class {class_name}{sig}")
                output.append("```\n")
            
            # Class description and handle Examples section
            if 'docstring' in docs and docs['docstring'].strip():
                docstring = docs['docstring'].strip()
                sections = docstring.split('\n\n')
                
                for section in sections:
                    if section.startswith('Examples'):
                        # Format the Examples section
                        output.append("**Examples:**\n")
                        output.append("```python")
                        # Remove 'Examples' header and any leading/trailing whitespace
                        example_code = section.replace('Examples', '').strip()
                        # Remove common indentation from example code
                        example_lines = example_code.split('\n')
                        if example_lines:
                            min_indent = min(len(line) - len(line.lstrip()) 
                                          for line in example_lines if line.strip())
                            example_code = '\n'.join(line[min_indent:] if line.strip() else line 
                                                   for line in example_lines)
                        output.append(example_code)
                        output.append("```\n")
                    else:
                        output.append(section + "\n")
            
            # Methods
            if 'methods' in docs and docs['methods']:
                output.append("**Methods:**\n")
                for method_name, method_info in docs['methods'].items():
                    sig = method_info.get('signature', '')
                    doc = method_info.get('docstring', '').strip()
                    
                    # Only show public methods
                    if not method_name.startswith('_'):
                        output.append(f"- `{method_name}{sig}`")
                        if doc:
                            # Add first line of docstring only
                            doc_first_line = doc.split('\n')[0]
                            output.append(f"  - {doc_first_line}")
                output.append("")  # Add spacing between classes
    
    # Format functions with similar Examples handling
    if functions:
        output.append("\n### Functions\n")
        for func_name, docs in functions.items():
            # Function header with signature
            sig = docs.get('signature', '')
            output.append(f"#### {func_name}\n")
            if sig:
                output.append("```python")
                output.append(f"def {func_name}{sig}")
                output.append("```\n")
            
            # Function description and handle Examples section
            if 'docstring' in docs and docs['docstring'].strip():
                docstring = docs['docstring'].strip()
                sections = docstring.split('\n\n')
                
                for section in sections:
                    if section.startswith('Examples'):
                        # Format the Examples section
                        output.append("**Examples:**\n")
                        output.append("```python")
                        # Remove 'Examples' header and any leading/trailing whitespace
                        example_code = section.replace('Examples', '').strip()
                        # Remove common indentation from example code
                        example_lines = example_code.split('\n')
                        if example_lines:
                            min_indent = min(len(line) - len(line.lstrip()) 
                                          for line in example_lines if line.strip())
                            example_code = '\n'.join(line[min_indent:] if line.strip() else line 
                                                   for line in example_lines)
                        output.append(example_code)
                        output.append("```\n")
                    else:
                        output.append(section + "\n")
    
    return '\n'.join(output)
