"""Live browser tester for the SkinLens TFLite model (standard library only, runs locally).

  python -P -E scripts/serve.py [--model model_effb0/tflite/skinlens_fp16.tflite] [--port 8000]
then open http://127.0.0.1:8000 (keep it running: a new model from train.py / export_tflite.py is picked up automatically) : paste an image URL, upload/drag a photo, or take a webcam snapshot.
The page shows the 224x224 crop the model actually sees and the calibrated probabilities.
Uses exactly the same preprocessing as predict.py (= the app contract in PIPELINE.md).
"""
import argparse, base64, io, json, os, sys, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ROOT, outputs, preprocess, severity_of

MAX_BYTES = 20 * 1024 * 1024
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SkinLensLocalTester/1.0"

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SkinLens Live Test</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mut:#667085;--line:#e3e6eb;--acc:#2f6fdf;--bar:#2f6fdf;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e8ebf0;--mut:#9aa3b2;--line:#2b3240;--acc:#6b9bff;--bar:#6b9bff;--warn:#f59e0b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:900px;margin:0 auto;padding:20px 16px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 16px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.row{display:flex;gap:8px;flex-wrap:wrap}input[type=text]{flex:1;min-width:200px;padding:9px 10px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--fg);font:inherit}
button{padding:9px 14px;border:0;border-radius:7px;background:var(--acc);color:#fff;font:inherit;cursor:pointer}button.sec{background:transparent;color:var(--acc);border:1px solid var(--line)}
#drop{margin-top:10px;border:2px dashed var(--line);border-radius:8px;padding:16px;text-align:center;color:var(--mut);cursor:pointer}#drop.on{border-color:var(--acc)}
video{width:100%;max-width:360px;border-radius:8px;display:none;margin-top:10px}
.res{display:grid;grid-template-columns:224px 1fr;gap:16px;align-items:start}@media(max-width:560px){.res{grid-template-columns:1fr}}
.res img{width:224px;height:224px;border-radius:8px;border:1px solid var(--line);background:var(--bg)}
.top{font-size:22px;font-weight:600;text-transform:capitalize}.sev{font-size:15px;font-weight:500;color:var(--mut);margin-left:6px}.warn{color:var(--warn);font-size:13px;margin:2px 0 8px}
.bar{display:grid;grid-template-columns:64px 1fr 48px;gap:8px;align-items:center;margin:5px 0;font-size:14px}
.track{height:10px;background:var(--bg);border-radius:5px;overflow:hidden}.fill{height:100%;background:var(--bar);border-radius:5px}
.mut{color:var(--mut);font-size:12px}.err{color:#d92d20}#hist{display:flex;gap:6px;flex-wrap:wrap}#hist img{width:56px;height:56px;border-radius:6px;cursor:pointer;border:1px solid var(--line)}
</style></head><body><main>
<h1>SkinLens Live Test</h1><p class="sub">Model: <b id="mname"></b>. Runs locally. Not a medical diagnosis.</p>
<div class="card"><div class="row"><input id="url" type="text" placeholder="Paste an image URL (https://...jpg)"><button id="go">Analyze URL</button></div>
<div id="drop">Drop a photo here, paste one (Ctrl+V), or <u>click to choose a file</u><input id="file" type="file" accept="image/*" hidden></div>
<div class="row" style="margin-top:10px"><button class="sec" id="cam">Start webcam</button><button class="sec" id="snap" style="display:none">Take snapshot</button></div>
<video id="v" autoplay playsinline muted></video></div>
<div class="card" id="out"><span class="mut">The result appears here.</span></div>
<div class="card"><div class="mut" style="margin-bottom:6px">History (click to show again)</div><div id="hist"></div></div>
</main><script>
const $=id=>document.getElementById(id), out=$('out'), hist=[];
fetch('/info').then(r=>r.json()).then(i=>$('mname').textContent=i.model);
function show(d){
  if(d.model)$('mname').textContent=d.model;
  if(d.error){out.innerHTML='<span class="err">'+d.error+'</span>';return}
  const bars=d.ranked.map(([c,p])=>`<div class="bar"><span>${c}</span><div class="track"><div class="fill" style="width:${(p*100).toFixed(1)}%"></div></div><span>${(p*100).toFixed(1)}%</span></div>`).join('');
  out.innerHTML=`<div class="res"><img src="${d.crop}" alt="model input"><div><div class="top">${d.ranked[0][0]} ${(d.ranked[0][1]*100).toFixed(0)}%${d.severity?`<span class="sev">severity: ${d.severity[0]} ${(d.severity[1]*100).toFixed(0)}%</span>`:''}</div>
  ${d.low?'<div class="warn">Low confidence. Use a clear, well-lit, front-facing close-up of the face.</div>':''}${bars}
  <div class="mut" style="margin-top:8px">${d.src} · original ${d.size} · inference ${d.ms} ms · the image on the left is the 224×224 center crop the model sees</div></div></div>`;
}
async function send(body){
  out.innerHTML='<span class="mut">Analyzing...</span>';
  try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();show(d);
    if(!d.error){hist.unshift(d);const im=new Image();im.src=d.crop;im.title=d.ranked[0][0];im.onclick=()=>show(d);$('hist').prepend(im)}}
  catch(e){show({error:String(e)})}
}
const sendFile=f=>{const r=new FileReader();r.onload=()=>send({data:r.result,name:f.name});r.readAsDataURL(f)};
$('go').onclick=()=>{const u=$('url').value.trim();if(u)send({url:u})};$('url').onkeydown=e=>{if(e.key==='Enter')$('go').click()};
const drop=$('drop');drop.onclick=()=>$('file').click();$('file').onchange=e=>e.target.files[0]&&sendFile(e.target.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('on')};drop.ondragleave=()=>drop.classList.remove('on');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('on');const f=e.dataTransfer.files[0];if(f)sendFile(f);else{const u=e.dataTransfer.getData('text/uri-list')||e.dataTransfer.getData('text');if(u){$('url').value=u;send({url:u})}}};
document.onpaste=e=>{for(const it of e.clipboardData.items){if(it.type.startsWith('image/')){sendFile(it.getAsFile());return}}const t=e.clipboardData.getData('text');if(t&&/^https?:/.test(t)&&document.activeElement!==$('url')){$('url').value=t;send({url:t})}};
$('cam').onclick=async()=>{try{const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}});const v=$('v');v.srcObject=s;v.style.display='block';$('snap').style.display='';$('cam').style.display='none'}catch(e){show({error:'Webcam: '+e})}};
$('snap').onclick=()=>{const v=$('v'),c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);send({data:c.toDataURL('image/jpeg',0.95),name:'webcam'})};
</script></body></html>"""


class Model:
    """Holds the TFLite interpreter and reloads it when train.py/export_tflite.py write a new model.

    export_tflite.py writes model_info.json last, so a reload happens once that file is newer than the
    .tflite file. The model is loaded from bytes, so no file handle stays open (train.py can move model/).
    """
    def __init__(self, path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self.Interpreter = Interpreter
        self.path = path
        self.info_path = os.path.join(os.path.dirname(path), "model_info.json")
        self.name = os.path.basename(path)
        self.it, self.key, self.loaded_at = None, None, None
        self.lock = Lock()   # the interpreter is not thread-safe
        self.refresh()
        if self.it is None:
            print(f"no model yet at {path} - waiting; it loads automatically after train.py / export_tflite.py", flush=True)

    def _stamp(self):
        try:
            m, i = os.stat(self.path).st_mtime_ns, os.stat(self.info_path).st_mtime_ns
        except OSError:
            return None
        return (m, i) if i >= m else None        # info older than model = export still running

    def refresh(self):
        key = self._stamp()
        if key is None or key == self.key:
            return
        try:
            info = json.load(open(self.info_path))
            it = self.Interpreter(model_content=open(self.path, "rb").read(), num_threads=4)
            it.allocate_tensors()
        except Exception as e:
            print(f"model reload failed ({e}); keeping the previous one", flush=True)
            return
        with self.lock:
            self.info, self.it, self.key = info, it, key
            self.inp = it.get_input_details()[0]
            self.out, self.sout = outputs(it, info)
            self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(key[0] / 1e9))
        print(f"loaded model {self.path} (written {self.loaded_at})", flush=True)

    def __call__(self, raw):
        self.refresh()
        if self.it is None:
            raise RuntimeError(f"no model at {self.path} yet - run train.py or export_tflite.py")
        from PIL import Image
        orig = Image.open(io.BytesIO(raw))
        size = f"{orig.width}x{orig.height}"
        x = preprocess(io.BytesIO(raw))
        with self.lock:
            self.it.set_tensor(self.inp["index"], x[None].astype(self.inp["dtype"]))
            t = time.perf_counter()
            self.it.invoke()
            ms = round(1000 * (time.perf_counter() - t), 1)
            prob = self.it.get_tensor(self.out["index"])[0].copy()
            sev_prob = self.it.get_tensor(self.sout["index"])[0].copy() if self.sout else None
            info = self.info
        buf = io.BytesIO()
        Image.fromarray(x).save(buf, "JPEG", quality=90)
        order = np.argsort(-prob)
        return {"ranked": [[info["classes"][i], float(prob[i])] for i in order],
                "severity": severity_of(info["classes"][order[0]], sev_prob, info),
                "low": bool(prob[order[0]] < info["suggested_min_confidence"]),
                "crop": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "size": size, "ms": ms,
                "model": f"{self.name} ({self.loaded_at})"}


def fetch(url):
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("image larger than 20 MB")
    return data


def make_handler(model):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/info":
                model.refresh()
                if model.it is None:
                    return self._send(200, json.dumps({"model": f"{model.name} (not found yet, waiting for export)"}), "application/json")
                return self._send(200, json.dumps({"model": f"{model.name} ({model.loaded_at})", "classes": model.info["classes"]}), "application/json")
            self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, "not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 3 * MAX_BYTES:
                    raise ValueError("upload too large")
                req = json.loads(self.rfile.read(n))
                if req.get("url"):
                    raw, src = fetch(req["url"]), req["url"][:80]
                else:
                    raw, src = base64.b64decode(req["data"].split(",", 1)[1]), req.get("name", "upload")
                res = model(raw)
                res["src"] = src
                sev = f" ({res['severity'][0]} {res['severity'][1]:.2f})" if res.get("severity") else ""
                print(f"{src}: {res['ranked'][0][0]} {res['ranked'][0][1]:.2f}{sev}", flush=True)
            except Exception as e:
                res = {"error": f"Could not analyze this image: {e}"}
            self._send(200, json.dumps(res), "application/json")

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "model_effb0", "tflite", "skinlens_fp16.tflite"))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    model = Model(args.model)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model))
    print(f"SkinLens live test: http://127.0.0.1:{args.port}  (model {model.name}, Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
