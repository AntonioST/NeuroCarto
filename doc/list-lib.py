"""
::

    # environment
    pip                            23.3.1                        pypi_0     pypi
    python                         3.10.13                   h955ad1f_0
    # required
    bokeh                          3.3.2                         pypi_0     pypi
    matplotlib                     3.7.1                py310h06a4308_1
    numpy                          1.22.3               py310h4f1e569_0
    scipy                          1.7.3                py310hac523dd_2
    typing_extensions              4.6.3                py310h06a4308_0
    # optional
    bg-atlasapi                    1.0.2                         pypi_0     pypi
    pillow                         9.4.0                py310h6a678d5_0
    pandas                         1.5.3                py310h1128e8f_0
    polars                         0.18.2                        pypi_0     pypi
    probeinterface                 0.2.21                        pypi_0     pypi
    tifffile                       2023.12.9                     pypi_0     pypi
    # document
    jupyter                        1.0.0                         pypi_0     pypi
    jupyter-contrib-nbextensions   0.7.0                         pypi_0     pypi
    nbsphinx                       0.9.3                         pypi_0     pypi
    sphinx                         7.2.6                         pypi_0     pypi
    sphinx-markdown-builder        0.6.6                         pypi_0     pypi
    sphinx-rtd-theme               2.0.0                         pypi_0     pypi
"""

import re
import subprocess
import textwrap
from pathlib import Path

PACKAGES = {
    'environment': ['pip', 'python'],
    'required': ['bokeh', 'matplotlib', 'numpy', 'scipy', 'typing_extensions'],
    'optional': ['bg-atlasapi', 'pillow', 'pandas', 'polars', 'probeinterface', 'tifffile'],
    'document': ['jupyter', 'jupyter-contrib-nbextensions', 'nbsphinx', 'sphinx', 'sphinx-markdown-builder', 'sphinx-rtd-theme'],
}

PACKAGE = tuple[str, str, str, str]


def conda_list() -> list[PACKAGE]:
    ret = []
    for line in subprocess.getoutput('conda list').split('\n'):
        if line.startswith('#'):
            continue

        part = re.split(r' +', line)
        ret.append(part)

    return ret


def filter_packages(envs: list[PACKAGE]) -> dict[str, list[PACKAGE]]:
    def _find(name):
        for package in envs:
            if name == package[0]:
                return package
        return name, '', '', ''

    return {
        group: [
            _find(name)
            for name in packages
        ]
        for group, packages in PACKAGES.items()
    }


def build_output(envs: dict[str, list[PACKAGES]]) -> str:
    format = '%-30s %-20s %15s %8s'

    ret = []
    for group, packages in envs.items():
        ret.append(f'# {group}')
        for package in packages:
            ret.append(format % tuple(package))
    return '\n'.join(ret)


def update_install_rst(file: Path, content: str):
    if not file.exists():
        return

    tmp = file.with_suffix('.rst_tmp')

    with file.open('r') as source:
        with tmp.open('w') as output:
            testing_library_versions = False
            code_block = False

            for line in source:
                line = line.rstrip()
                if line.startswith('testing library versions'):
                    testing_library_versions = True
                    code_block = False
                    print(line, file=output)
                elif not testing_library_versions:
                    print(line, file=output)
                elif line.startswith('::'):
                    code_block = True
                    print(line, file=output)
                    print('', file=output)
                    print(textwrap.indent(content, '    '), file=output)
                elif not code_block:
                    print(line, file=output)
                elif len(line) > 0 and not line.startswith(' '):
                    code_block = False
                    testing_library_versions = False
                    print(line, file=output)

    print('updated', file)
    tmp.rename(file)


def update_install_md(file: Path, content: str):
    if not file.exists():
        return

    tmp = file.with_suffix('.md_tmp')

    with file.open('r') as source:
        with tmp.open('w') as output:
            testing_library_versions = False
            code_block = False

            for line in source:
                line = line.rstrip()
                if line.startswith('## testing library versions'):
                    testing_library_versions = True
                    code_block = False
                    print(line, file=output)
                elif not testing_library_versions:
                    print(line, file=output)
                elif line.startswith('```default'):
                    code_block = True
                    print(line, file=output)
                    print(content, file=output)
                elif not code_block:
                    print(line, file=output)
                elif line.startswith('```'):
                    code_block = False
                    testing_library_versions = False
                    print(line, file=output)

    print('updated', file)
    tmp.rename(file)


content = build_output(filter_packages(conda_list()))
print(content)
update_install_rst(Path('source') / 'install.rst', content)
update_install_md(Path('wiki') / 'install.md', content)
