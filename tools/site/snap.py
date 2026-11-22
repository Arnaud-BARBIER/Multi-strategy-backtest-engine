"""Drive Pod Lab headless and save frozen HTML states.
steps: ["wait",s] | ["key",name,code] | ["eval",js] | ["save",js,path] (js returns a string) | ["shot",path] | ["click",x,y] | ["move",x,y]"""
import json, os, subprocess, sys, time, urllib.request, base64
import websocket
url, profile, steps = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
W, H = int(os.environ.get("SNAP_W", 1600)), int(os.environ.get("SNAP_H", 1000))   # phone check: SNAP_W=375 SNAP_H=812
brave = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
proc = subprocess.Popen([brave, "--headless=new", "--disable-gpu", "--remote-debugging-port=9341", f"--user-data-dir={profile}",
                         "--window-size=1600,1000", "--hide-scrollbars", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            page = next(t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9341/json")) if t["type"] == "page"); break
        except Exception: time.sleep(0.2)
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=2, suppress_origin=True)
    n = [0]
    def send(m, **p):
        n[0] += 1; ws.send(json.dumps({"id": n[0], "method": m, "params": p})); return n[0]
    def pump(sec, want=None):
        end = time.time() + sec
        while time.time() < end:
            try: msg = json.loads(ws.recv())
            except Exception: continue
            m = msg.get("method")
            if m == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]; print("EXC:", (d.get("exception", {}).get("description") or d.get("text"))[:300])
            elif want and msg.get("id") == want:
                return msg.get("result", {})
    send("Runtime.enable"); send("Page.enable")
    send("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2 if W < 800 else 1, mobile=W < 800)
    send("Page.navigate", url=url); pump(6)
    for step in steps:
        k = step[0]
        if k == "wait": pump(step[1])
        elif k in ("eval", "save"):
            r = pump(60, send("Runtime.evaluate", expression=step[1], returnByValue=True, awaitPromise=True)) or {}
            v = r.get("result", {}).get("value")
            if k == "save":
                open(step[2], "w").write(v if isinstance(v, str) else json.dumps(v)); print("SAVE", step[2], len(v or ""))
            else: print("EVAL", step[1][:50].replace("\n"," "), "=>", json.dumps(v)[:800])
        elif k == "shot":
            r = pump(15, send("Page.captureScreenshot", format="png")); open(step[1], "wb").write(base64.b64decode(r["data"])); print("SHOT", step[1])
        elif k == "key":   # a real key press: ["key", "ArrowDown", keyCode]
            for t in ("rawKeyDown", "keyUp"):
                pump(0.05, send("Input.dispatchKeyEvent", type=t, key=step[1], code=step[1], windowsVirtualKeyCode=step[2], nativeVirtualKeyCode=step[2]))
        elif k == "click":
            for t in ("mouseMoved", "mousePressed", "mouseReleased"):
                pump(0.15, send("Input.dispatchMouseEvent", type=t, x=step[1], y=step[2], button="left", clickCount=1))
finally:
    proc.kill()
