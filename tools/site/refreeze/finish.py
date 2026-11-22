"""Place frozen states into docs/next/states, keep hand-set rects, rebuild p1, brand FORGE, hide the stale banner."""
import html, json, re, shutil, sys
SP = sys.argv[1]
OUT = '/Users/arnaudbarbier/Quant reaserch/Multi-strategy-backtest-engine/docs/next/states'
names = sys.argv[2:]
old = json.load(open(f'{OUT}/targets.json'))
shutil.copytree(OUT, f'{SP}/states_prev', dirs_exist_ok=True)
# rects measured by hand in earlier sessions (layout closes panels after freezing, or the element is a group)
MANUAL = {'blank': ['data'], 'tick': ['deu'], 'a1': ['equity'], 'a2': ['equity', 'tiles'],
          'bench': ['equity', 'equityLow', 'options']}
targets = dict(old)
STYLE = '<style>#server-stale-banner{display:none!important}</style>'
def fix(h):
    h = h.replace('Pod Lab', 'FORGE')
    return h.replace('</head>', STYLE + '</head>', 1) if STYLE not in h else h
for name in names:
    d = json.load(open(f'{SP}/flow/{name}.json')); d = json.loads(d) if isinstance(d, str) else d
    r = dict(d['rects'])
    if 'params' in r and 'params_end' in r:
        x, y, w, _ = r['params']; ex, ey, ew, eh = r.pop('params_end')
        r['params'] = [min(x, ex), y, max(x + w, ex + ew) - min(x, ex), ey + eh - y]
    for k in MANUAL.get(name, []):
        if k in old.get(name, {}):
            if r.get(k) != old[name][k]: print(f'  {name}.{k}: measured {r.get(k)} -> kept {old[name][k]}')
            r[k] = old[name][k]
    for k, v in r.items():
        if old.get(name, {}).get(k) not in (None, v) and k not in MANUAL.get(name, []): print(f'  {name}.{k}: {old[name][k]} -> {v}')
    targets[name] = {**old.get(name, {}), **r}
    open(f'{OUT}/{name}.html', 'w').write(fix(d['html']))
    print(name, len(d['html']) // 1024, 'KB')
if 'ai' in names:   # p1 = the AI panel as the visitor left it, with answer 1 pasted in the field
    h = open(f'{OUT}/ai.html').read()
    a1 = html.escape(open(f'{SP}/ai/answer1.md').read(), quote=False)
    h2, n = re.subn(r'(<textarea[^>]*id="aipaste-text"[^>]*>)(</textarea>)', lambda m: m.group(1) + a1 + m.group(2), h, count=1)
    assert n == 1, 'textarea not found'
    open(f'{OUT}/p1.html', 'w').write(h2); print('p1 from ai + answer 1')
    p = targets['ai']['prompt']; targets['ai']['promptmd'] = [p[0], p[1], 281, p[3]]
json.dump(targets, open(f'{OUT}/targets.json', 'w'), indent=1)
