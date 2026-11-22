"""Copy docs/next/index.html to docs/index.html (the site's home) with paths rewritten for the docs root."""
from pathlib import Path

root = Path(__file__).resolve().parents[2] / 'docs'
s = (root / 'next/index.html').read_text()
s = s.replace('states/', 'next/states/').replace("'hero.jpg'", "'next/hero.jpg'").replace('"hero.jpg"', '"next/hero.jpg"')
s = s.replace('href="../"', 'href="ancien.html"').replace('href="../', 'href="')
assert 'next/next/' not in s and '../' not in s
(root / 'index.html').write_text(s)
print('docs/index.html updated')
