import torchani

# General variables used in |substitutions|
project = "TorchANI"
copyright = "2024, Roitberg Group"
author = "TorchANI developers"
version = ".".join(torchani.__version__.split(".")[:2])
if ".dev." in torchani.__version__:
    version = f'{version}" (development)"'
release = torchani.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Automatically generate docs for all submodules
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # For google-style docstr
    "sphinx_gallery.gen_gallery",  # For rendering user guide
    'sphinx_design',  # For grid directive
]
# Extensions config
python_use_unqualifierd_type_names = True  # Not sure if needed
# autodoc
autodoc_typehints_format = "short"  # Avoid qualified names in return types
autodoc_typehints = "description"  # Write types in description, not in signature
autodoc_default_options = {
    "member-order": "bysource",  # Document in the same order as python source code
}
# napoleon
napoleon_google_docstring = True  # Use google-style docstrings only
napoleon_numpy_docstring = False
# sphinx-gallery
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "examples_autogen",
    "filename_pattern": r".*\.py",
}
# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}


# General sphinx config
nitpicky = True  # Fail if refs can't be resolved
master_doc = "index"  # Main toctree
default_role = "py:obj"  # Behavior of `inline-backticks`
source_suffix = {".rst": "restructuredtext"}  # Suffix of files
pygments_style = "sphinx"  # Code render style
templates_path = ["_templates"]

# HTML config
html_title = f"{project} v{version} Manual"
html_static_path = ["_static"]  # Static html resources
html_css_files = ["style.css"]  # Overrides for theme style sheet
html_theme = "pydata_sphinx_theme"
html_use_modindex = True
html_domain_indices = False
html_copy_source = False
html_file_suffix = '.html'
htmlhelp_basename = "torchani-docs"

# TODO
# html_logo = '_static/logo.svg'
# html_favicon = '_static/favicon.ico'

# PyData Theme config
# Primary HTML sidebar (left)
html_sidebars = {
    "index": [],
    "installing": [],
    "user-guide": ["sidebar-nav-bs"],
    "api": ["page-toc"],
    # "api": ["sidebar-nav-bs"],
    "publications": [],
}
html_theme_options = {
    # "show_nav_level": 1, (TODO: what does this do?)
    "show_toc_level": 1,  # default is 2?
    # "navigation_depth": 4,  Default (TODO what does this do?)
    "primary_sidebar_end": [],
    # navbar (Top bar)
    # "navbar_align": "content", Default
    # "navbar_start": ["navbar-logo"],  Default
    "navbar_center": ["navbar-nav"],
    # "navbar_persistent": ["search-button"], Default
    # "navbar_end": ["theme-switcher", "navbar-icon-links"], Default
    # "header_links_before_dropdown": 5,  # Default, Headers before collapse to "more v"
    # Secondary HTML sidebar (right)
    "secondary_sidebar_items": [],  # TODO: Here for debugging, could be []
    # Misc
    "github_url": "https://github.com/aiqm/torchani",
    "icon_links": [],
    "logo": {"text": "TorchANI"},
    "show_version_warning_banner": True,
}

# Other: info, tex, man
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
