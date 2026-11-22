"""Headless Brave driver: steps = ["nav",url] | ["wait",s] | ["eval",js] | ["click",css] (real mouse click on the element centre) | ["key",key,code] | ["shot",path]."""
import json, os, subprocess, sys, time, urllib.request, base64, websocket
profile, steps = sys.argv[1], json.load(open(sys.argv[2]))
W, H = int(os.environ.get("SNAP_W", 1600)), int(os.environ.get("SNAP_H", 1000))
proc = subprocess.Popen(["/Applications/Brave Browser.app/Contents/MacOS/Brave Browser", "--headless=new", *(["--disable-gpu"] if os.environ.get("NOGPU") else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]), "--remote-debugging-port=9342",
    f"--user-data-dir={profile}", f"--window-size={W},{H}", "--hide-scrollbars", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try: page = next(t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9342/json")) if t["type"] == "page"); break
        except Exception: time.sleep(0.2)
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=2, suppress_origin=True); n = [0]
    def send(m, **p): n[0] += 1; ws.send(json.dumps({"id": n[0], "method": m, "params": p})); return n[0]
    def pump(sec, want=None):
        end = time.time() + sec
        while time.time() < end:
            try: msg = json.loads(ws.recv())
            except Exception: continue
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]; send("Page.screencastFrameAck", sessionId=p["sessionId"])
                if REC is not None:
                    path = f"{REC['dir']}/f{len(REC['frames']):06d}.jpg"; open(path, "wb").write(base64.b64decode(p["data"]))
                    REC["frames"].append((p["metadata"]["timestamp"], path))
                continue
            if msg.get("method") == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]; t = (d.get("exception", {}).get("description") or d.get("text"))
                if "offsetHeight" not in t: print("EXC:", t[:200])
            elif want and msg.get("id") == want: return msg.get("result", {})
    def ev(js): return (pump(60, send("Runtime.evaluate", expression=js, returnByValue=True, awaitPromise=True)) or {}).get("result", {}).get("value")
    REC = None
    send("Runtime.enable"); send("Page.enable")
    pump(0.5, send("Browser.setPermission", permission={"name": "fullscreen", "allowWithoutSanitization": True}, setting="granted"))
    send("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    for s in steps:
        k = s[0]
        if k == "nav": send("Page.navigate", url=s[1]); pump(5)
        elif k == "wait": pump(s[1])
        elif k == "rec_start":
            import shutil; d = s[1]; shutil.rmtree(d, ignore_errors=True); os.makedirs(d); REC = {"dir": d, "frames": []}
            send("Page.startScreencast", format="jpeg", quality=90, maxWidth=W, maxHeight=H, everyNthFrame=1); pump(0.3)
        elif k == "rec_stop":
            send("Page.stopScreencast"); pump(0.5); fr = REC["frames"]; speed = s[2] if len(s) > 2 else 1.0
            lines = []
            for i, (ts, path) in enumerate(fr):
                dur = ((fr[i + 1][0] - ts) if i + 1 < len(fr) else 0.6) / speed
                lines += [f"file '{path}'", f"duration {max(dur, 0.001):.4f}"]
            lines.append(f"file '{fr[-1][1]}'"); open(REC["dir"] + "/list.txt", "w").write("\n".join(lines) + "\n")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", REC["dir"] + "/list.txt",
                "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-an", "-movflags", "+faststart", s[1]], check=True)
            print("REC", s[1], len(fr), "frames", f"{fr[-1][0] - fr[0][0]:.1f} s real", os.path.getsize(s[1]) // 1024, "KB"); REC = None
        elif k == "raw": print("RAW", pump(5, send(s[1], **s[2])))
        elif k == "save":
            v = ev(s[1]); open(s[2], "w").write(v if isinstance(v, str) else json.dumps(v)); print("SAVE", s[2], len(v or ""))
        elif k == "eval": print("EVAL", s[1][:60].replace("\n", " "), "=>", json.dumps(ev(s[1]))[:600])
        elif k == "click":
            r = ev(f"(()=>{{const e=document.querySelector({json.dumps(s[1])}); if(!e) return null; e.scrollIntoView({{block:'center'}}); const b=e.getBoundingClientRect(); return [b.x+b.width/2,b.y+b.height/2]}})()")
            if not r: print("CLICK missing", s[1]); continue
            for t in ("mouseMoved", "mousePressed", "mouseReleased"):
                pump(0.1, send("Input.dispatchMouseEvent", type=t, x=r[0], y=r[1], button="left", clickCount=1))
            print("CLICK", s[1])
        elif k == "key":
            for t in ("rawKeyDown", "keyUp"): pump(0.05, send("Input.dispatchKeyEvent", type=t, key=s[1], code=s[1], windowsVirtualKeyCode=s[2], nativeVirtualKeyCode=s[2]))
            print("KEY", s[1])
        elif k == "shot":
            r = pump(20, send("Page.captureScreenshot", format="png")); open(s[1], "wb").write(base64.b64decode(r["data"])); print("SHOT", s[1])
finally: proc.kill()
