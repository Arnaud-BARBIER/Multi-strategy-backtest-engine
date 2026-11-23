"""Generate French and English home pages from the FORGE preview pages."""
from pathlib import Path

root = Path(__file__).resolve().parents[2] / 'docs'


def french_home(source):
    source = source.replace('states/', 'next/states/')
    source = source.replace("'hero.jpg'", "'next/hero.jpg'").replace('"hero.jpg"', '"next/hero.jpg"')
    source = source.replace('href="../"', 'href="ancien.html"').replace('href="../', 'href="')
    assert 'next/next/' not in source and '../' not in source
    return source


def english_home(source):
    source = source.replace('../states/', '../next/states/')
    source = source.replace('../hero.jpg', '../next/hero.jpg')
    source = source.replace('href="../../', 'href="../')
    assert '../next/next/' not in source and '../../' not in source
    return source


for destination, source, transform in (
    ('index.html', 'next/index.html', french_home),
    ('en/index.html', 'next/en/index.html', english_home),
):
    output = root / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(transform((root / source).read_text()))
    print(f'docs/{destination} updated')
