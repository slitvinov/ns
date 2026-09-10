#!/usr/bin/env pvpython
"""
Time slider from ONE scene.

The leading-order field is self-similar.  A point with fixed similarity
coordinates (X, eta) sits at r = r0 sqrt(lam), z = z0 lam^D with
lam = tau/tau0, and its velocity is the tau0 velocity times lam^-A, up to a
factor lam^-h on the azimuthal component (h = 0.01: 3.3% over
tau 0.5 -> 0.02).  So the streamlines at any time are the tau0 streamlines
scaled by (sqrt(lam), sqrt(lam), lam^D), and speed is scaled by lam^-A.
One GLB, two numbers per time.

The anisotropy is a factor lam^-h on the aspect ratio: 6% over the whole
slider, invisible.  The paper's Figure 1 exaggerates it on purpose, and so
does the "exaggerate" checkbox (?x=1): the axial scale then uses
D_vis = 1/2 - h_vis (--h-vis, default 0.15) instead of D.  Speed scaling and
the label's t are untouched; only the shape is a lie, and it says so.

    pvpython ns_render.py out/core.xdmf2 --ribbon --color speed --gray \
             --color-range 0 S0 --export out/core.glb --no-html
    pvpython ns_time.py out/core.glb --tau0 0.5 --s0 S0 --smax SMAX --h-vis 0.15 -o out/site/index.html

--gray makes COLOR_0.r = speed / S0; the page applies the palette in a shader,
so speed (linear or log, on one scale for all times) and distance from the
axis are both switchable without re-rendering.  The GLB is embedded, so the
page opens from disk.
"""

import argparse
import base64
import os

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; overflow: hidden;
               font: 14px -apple-system, "Helvetica Neue", Arial, sans-serif; color: #333; }
  #c { display: block; width: 100%; height: 100%; touch-action: none; -webkit-user-select: none; user-select: none; }
  #bar { touch-action: manipulation; }
  #bar { position: absolute; left: 50%; bottom: 22px; transform: translateX(-50%);
         display: flex; align-items: center; gap: 14px; padding: 8px 14px;
         background: rgba(255,255,255,.85); border-radius: 8px; user-select: none; white-space: nowrap; }
  #bar input[type=range] { width: min(46vw, 420px); }
  #play { width: 30px; height: 26px; border: 1px solid #bbb; border-radius: 5px; background: #fff; cursor: pointer; }
  #lbl { font-variant-numeric: tabular-nums; min-width: 5.5em; white-space: pre; }
  #col { border: 1px solid #bbb; border-radius: 5px; background: #fff; padding: 3px 6px; font: inherit; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="bar">
  <button id="play" title="play / pause (space)">&#9654;</button>
  <input id="s" type="range" min="0" max="1000" step="1" value="0">
  <span id="lbl"></span>
  <select id="col" title="color"><option value="0">color: speed</option><option value="1">color: radius</option></select>
  <label id="loglbl" title="logarithmic speed scale"><input id="log" type="checkbox"> log</label>
  <label id="xlbl" title="exaggerate the radius-vs-height anisotropy: l_z ~ tau^(1/2 - h_vis) instead of tau^(1/2 - h), as in the paper's Figure 1"><input id="x" type="checkbox"> exaggerate</label>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TrackballControls.js"></script>
<script>
const GLB_B64 = "__GLB__";
const TAU0 = __TAU0__, TAU1 = __TAU1__;   // scene time, and the end of the slider
const A = __A__, D = __D__;               // 1/2 + h, 1/2 - h
const H = __H__, HVIS = __HVIS__, DVIS = 0.5 - HVIS;   // exaggerated axial exponent
const S0 = __S0__, SMAX = __SMAX__;       // COLOR_0.r = speed / S0; palette spans [0, SMAX]
const LOGMIN = SMAX / 60;
const LOOP_MS = 8000;

function b64ToBuf(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u.buffer; }

const qs = new URLSearchParams(location.search);
let colorMode = +(qs.get("c") || 0), logScale = qs.get("log") === "1", exag = qs.get("x") === "1";
let v = 0;                                 // slider position in [0, 1]
if (qs.get("t")) { const tau = 1 - +qs.get("t"); v = Math.log(tau / TAU0) / Math.log(TAU1 / TAU0); v = Math.min(1, Math.max(0, v)); }
document.getElementById("col").value = String(colorMode);
document.getElementById("log").checked = logScale;
document.getElementById("x").checked = exag;

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xffffff, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, 1, 0.001, 100);
const controls = new THREE.TrackballControls(camera, canvas);
// touch: slower, more damping; a mouse gets the snappier settings
const touch = matchMedia("(pointer: coarse)").matches;
controls.rotateSpeed = touch ? 1.2 : 3.0; controls.zoomSpeed = touch ? 0.6 : 1.2; controls.panSpeed = 0.8;
controls.dynamicDampingFactor = touch ? 0.12 : 0.18;
canvas.addEventListener("touchmove", e => e.preventDefault(), { passive: false });
scene.add(new THREE.AmbientLight(0xffffff, 0.42));
const head = new THREE.DirectionalLight(0xffffff, 0.78); head.position.set(0.35, 0.55, 1.0);
camera.add(head); scene.add(camera);

const HOT = [[0.00,.13,.22,.50],[0.22,.17,.54,.84],[0.45,.30,.78,.83],[0.62,.86,.90,.75],[0.78,.96,.78,.45],[0.90,.88,.52,.24],[1.00,.70,.18,.12]];
const RAD = [[0.00,.97,.85,.66],[0.10,.87,.62,.33],[0.22,.55,.52,.60],[0.34,.36,.47,.72],[0.52,.17,.54,.84],[0.76,.30,.78,.83],[1.00,.58,.90,.88]];
function paletteTexture(stops) {
  const n = 256, d = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const f = i / (n - 1); let k = 0; while (k < stops.length - 2 && f > stops[k + 1][0]) k++;
    const a = stops[k], b = stops[k + 1], w = (f - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) d[4 * i + c] = Math.round(255 * (a[c + 1] + w * (b[c + 1] - a[c + 1])));
    d[4 * i + 3] = 255;
  }
  const tx = new THREE.DataTexture(d, n, 1, THREE.RGBAFormat); tx.needsUpdate = true; return tx;
}
const TEX = [paletteTexture(HOT), paletteTexture(RAD)];
const U = { palette: { value: TEX[colorMode] }, mode: { value: colorMode }, logScale: { value: logScale ? 1 : 0 },
            rmax: { value: 1 }, speedFactor: { value: S0 / SMAX }, logLo: { value: Math.log(LOGMIN / SMAX) } };
function makeMaterial() {
  const m = new THREE.MeshPhongMaterial({ vertexColors: true, specular: 0x2a2a2a, shininess: 45, side: THREE.DoubleSide });
  m.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vR; uniform float rmax;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvR = length(position.xy) / rmax;");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vR; uniform sampler2D palette; uniform int mode; uniform int logScale; uniform float logLo; uniform float speedFactor;")
      .replace("#include <color_fragment>",
        "float s = clamp(vColor.r * speedFactor, 0.0, 1.0);\n" +           // speed / SMAX at this time
        "if (logScale == 1) s = clamp((log(max(s, 1e-4)) - logLo) / (-logLo), 0.0, 1.0);\n" +
        "float u = (mode == 1) ? clamp(vR, 0.0, 1.0) : s;\n" +
        "diffuseColor.rgb = texture2D(palette, vec2(u, 0.5)).rgb;");
  };
  return m;
}
function applyColor() {
  U.palette.value = TEX[colorMode]; U.mode.value = colorMode; U.logScale.value = logScale ? 1 : 0;
  document.getElementById("loglbl").style.opacity = colorMode === 0 ? 1 : 0.35;
}
document.getElementById("col").addEventListener("change", e => { colorMode = +e.target.value; applyColor(); });
document.getElementById("log").addEventListener("change", e => { logScale = e.target.checked; applyColor(); });
document.getElementById("x").addEventListener("change", e => { exag = e.target.checked; setTime(v); });

let group = null, home = null, playing = false;
const slider = document.getElementById("s"), lbl = document.getElementById("lbl"), play = document.getElementById("play");

// time -> geometry: scale (sqrt(lam), sqrt(lam), lam^D), speed x lam^-A.
// Exaggerated: lam^DVIS on z, so the column visibly turns slender; speed is still the real one.
function setTime(vv) {
  v = vv;
  const tau = TAU0 * Math.pow(TAU1 / TAU0, v), lam = tau / TAU0;
  if (group) group.scale.set(Math.sqrt(lam), Math.sqrt(lam), Math.pow(lam, exag ? DVIS : D));
  U.speedFactor.value = Math.pow(lam, -A) * S0 / SMAX;
  slider.value = Math.round(1000 * v);
  lbl.textContent = `t = ${(1 - tau).toFixed(3)}` + (exag ? `  h = ${HVIS} (real ${H})` : "");
}
slider.addEventListener("input", () => setTime(+slider.value / 1000));
function togglePlay() { playing = !playing; play.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;"; }
play.addEventListener("click", togglePlay);
window.addEventListener("keydown", e => {
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowRight") setTime(Math.min(1, v + 0.01));
  else if (e.key === "ArrowLeft") setTime(Math.max(0, v - 0.01));
  else if (e.key === "r" && home) { camera.position.copy(home.pos); camera.up.copy(home.up); controls.target.copy(home.tgt); controls.update(); }
});

function frameCamera(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3()), ctr = box.getCenter(new THREE.Vector3());
  const h = Math.max(size.z, size.x * 1.2, size.y * 1.2);
  const dist = 0.5 * h / Math.tan(0.5 * camera.fov * Math.PI / 180) * 1.15;
  const az = -62 * Math.PI / 180, el = 18 * Math.PI / 180;
  camera.position.set(ctr.x + dist * Math.cos(el) * Math.cos(az), ctr.y + dist * Math.cos(el) * Math.sin(az), ctr.z + dist * Math.sin(el));
  camera.up.set(0, 0, 1);
  camera.near = dist / 100; camera.far = dist * 100; camera.updateProjectionMatrix();
  controls.target.copy(ctr); controls.update();
  home = { pos: camera.position.clone(), up: camera.up.clone(), tgt: ctr.clone() };
}

new THREE.GLTFLoader().parse(b64ToBuf(GLB_B64), "", gltf => {
  const root = gltf.scene;
  const box = new THREE.Box3().setFromObject(root), sz = box.getSize(new THREE.Vector3());
  U.rmax.value = 0.5 * Math.max(sz.x, sz.y);                 // outer radius of the tau0 scene
  root.traverse(o => {
    if (!o.isMesh) return;
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    o.material = o.geometry.attributes.color ? makeMaterial()
      : new THREE.MeshPhongMaterial({ color: 0x9aa1a6, specular: 0x2a2a2a, shininess: 45 });
  });
  group = root; scene.add(group);
  frameCamera(group);                                        // framed at tau0, the largest
  applyColor(); setTime(v);
});

function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); controls.handleResize(); }
window.addEventListener("resize", resize); resize();

let last = performance.now();
(function loop(now) {
  if (playing) { let nv = v + (now - last) / LOOP_MS; if (nv > 1) nv -= 1; setTime(nv); }
  last = now; controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop);
})(last);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", help="scene at tau0, exported with --color speed --gray --color-range 0 S0")
    ap.add_argument("-o", "--out", default="out/site/index.html")
    ap.add_argument("--tau0", type=float, default=0.5, help="time of the scene, tau0 = 1 - t0")
    ap.add_argument("--tau1", type=float, default=0.001, help="end of the slider")
    ap.add_argument("--s0", type=float, required=True, help="speed encoded as COLOR_0.r = 1 in the GLB")
    ap.add_argument("--smax", type=float, default=None, help="top of the color scale (default s0 (tau0/tau1)^A)")
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--h-vis", type=float, default=0.15,
                    help="h used for the axial scale when 'exaggerate' is checked (shape only; default 0.15)")
    a = ap.parse_args()
    A, D = 0.5 + a.h, 0.5 - a.h
    smax = a.smax if a.smax else a.s0 * (a.tau0 / a.tau1) ** A
    with open(a.glb, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    page = (HTML.replace("__TITLE__", "Navier–Stokes blowup core").replace("__GLB__", b64)
                .replace("__TAU0__", repr(a.tau0)).replace("__TAU1__", repr(a.tau1))
                .replace("__A__", repr(A)).replace("__D__", repr(D))
                .replace("__H__", repr(a.h)).replace("__HVIS__", repr(a.h_vis))
                .replace("__S0__", repr(a.s0)).replace("__SMAX__", repr(smax)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(page)
    print(f"wrote {a.out}  ({len(page)/1e6:.1f} MB; t from {1-a.tau0:.3f} to {1-a.tau1:.3f}, "
          f"speed scale [0, {smax:.3g}]; exaggerated l_z/l_r x{(a.tau0/a.tau1)**a.h_vis:.2f} vs real x{(a.tau0/a.tau1)**a.h:.2f})")


if __name__ == "__main__":
    main()
