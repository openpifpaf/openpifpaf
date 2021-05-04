import nbformat
from glob import glob

# Collect a list of all notebooks in the content folder
notebooks = glob("./**/*.ipynb", recursive=True)

# Text to look for in adding tags
text_search_dict = {
    "# HIDDEN": "remove-cell",  # Remove the whole cell
    "# NO CODE": "remove-input",  # Remove only the input
    "# HIDE CODE": "hide-input",  # Hide the input w/ a button to show
    "import openpifpaf": "remove-cell",
}

# Search through each notebook and look for the text, add a tag if necessary
for i_path in notebooks:
    notebook = nbformat.read(i_path, nbformat.NO_CONVERT)

    for cell in notebook.cells:
        cell_tags = cell.get('metadata', {}).get('tags', [])

        # remove all tags that were previously set
        for tag in set(text_search_dict.values()):
            if tag in cell_tags:
                cell_tags.remove(tag)

        for key, val in text_search_dict.items():
            if key in cell['source']:
                if val not in cell_tags:
                    cell_tags.append(val)

        if cell_tags:
            cell['metadata']['tags'] = cell_tags
        elif 'tags' in cell['metadata']:
            del cell['metadata']['tags']

    nbformat.write(notebook, i_path)
