"""Record the demo page's automatic mode (?auto) as a video, headless.

usage: record.py URL OUT.mp4 [WORKDIR]
Brave headless sends a frame each time the page repaints (CDP screencast) with its timestamp; ffmpeg
rebuilds the video at the real pace from those timestamps. Stops when the page sets window.__autoDone."""
import base64, json, os, shutil, subprocess, sys, time, urllib.request
import websocket

url, out = sys.argv[1], sys.argv[2]
work = sys.argv[3] if len(sys.argv) > 3 else out + '.frames'
W, H = int(os.environ.get('REC_W', 1600)), int(os.environ.get('REC_H', 1000))
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
brave = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'
proc = subprocess.Popen([brave, '--headless=new', '--disable-gpu', '--remote-debugging-port=9342',
                         f'--user-data-dir={work}/profile', f'--window-size={W},{H}', '--hide-scrollbars', 'about:blank'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            page = next(t for t in json.load(urllib.request.urlopen('http://127.0.0.1:9342/json')) if t['type'] == 'page'); break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=1, suppress_origin=True)
    n = [0]

    def send(method, **params):
        n[0] += 1; ws.send(json.dumps({'id': n[0], 'method': method, 'params': params})); return n[0]

    frames, done_id, started = [], None, False
    send('Page.enable'); send('Runtime.enable')
    send('Emulation.setDeviceMetricsOverride', width=W, height=H, deviceScaleFactor=1, mobile=False)
    send('Page.navigate', url=url)
    send('Page.startScreencast', format='jpeg', quality=88, maxWidth=W, maxHeight=H, everyNthFrame=1)
    last_poll, deadline = 0, time.time() + 900
    while time.time() < deadline:
        try:
            msg = json.loads(ws.recv())
        except Exception:
            msg = {}
        if msg.get('method') == 'Page.screencastFrame':
            p = msg['params']
            send('Page.screencastFrameAck', sessionId=p['sessionId'])
            path = f'{work}/f{len(frames):06d}.jpg'
            open(path, 'wb').write(base64.b64decode(p['data']))
            frames.append((p['metadata']['timestamp'], path))
        elif msg.get('id') == done_id:
            if msg.get('result', {}).get('result', {}).get('value') is True:
                break
            done_id = None
        if done_id is None and time.time() - last_poll > 1:
            last_poll = time.time(); done_id = send('Runtime.evaluate', expression='window.__autoDone === true', returnByValue=True)
    time.sleep(0.5)
finally:
    proc.kill()

# Each frame lasts until the next one; the last one holds 2 s.
lines = []
for i, (ts, path) in enumerate(frames):
    dur = (frames[i + 1][0] - ts) if i + 1 < len(frames) else 2.0
    lines += [f"file '{os.path.abspath(path)}'", f'duration {max(dur, 0.001):.4f}']
lines.append(f"file '{os.path.abspath(frames[-1][1])}'")
open(f'{work}/list.txt', 'w').write('\n'.join(lines) + '\n')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', f'{work}/list.txt',
                '-vf', 'fps=30,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
                '-movflags', '+faststart', out], check=True)
print(len(frames), 'frames', f'{frames[-1][0] - frames[0][0]:.1f} s', '->', out, os.path.getsize(out) // 1024, 'KB')
