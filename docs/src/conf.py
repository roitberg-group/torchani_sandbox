import torchani

project = "TorchANI"
copyright = "2024, Roitberg Group"
author = "TorchANI developers"

version = torchani.__version__
release = torchani.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]
autodoc_typehints_format = "short"  # Avoid qualified names in return types
autodoc_typehints = "description"  # Write types in description, not in signature
napoleon_google_docstring = True
napoleon_numpy_docstring = False
python_use_unqualifierd_type_names = True  # Not sure if needed

templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
pygments_style = "sphinx"
html_theme = "pydata_sphinx_theme"
htmlhelp_basename = "TorchANIdoc"
html_static_path = ["_static"]
html_css_files = ["style.css"]

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "examples_autogen",
    "filename_pattern": r".*\.py",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

latex_documents = [
    (
        master_doc,
        "TorchANI.tex",
        "TorchANI Documentation",
        "TorchANI developers",
        "manual",
    ),
]

man_pages = [(master_doc, "torchani", "TorchANI Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "TorchANI",
        "TorchANI Documentation",
        author,
        "TorchANI",
        "One line description of project.",
        "Miscellaneous",
    ),
]

html_additional_pages = {}
html_use_modindex = True
html_domain_indices = False
html_copy_source = False
html_file_suffix = '.html'

html_sidebars = {
    "index": ["search-button-field"],
    "**": ["search-button-field", "sidebar-nav-bs"]
}

html_theme_options = {
    "github_url": "https://github.com/aiqm/torchani",
    "header_links_before_dropdown": 6,
    "icon_links": [],
    "logo": {
        "text": "TorchANI",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "show_version_warning_banner": True,
    "secondary_sidebar_items": ["page-toc"],
}
