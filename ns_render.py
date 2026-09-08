#!/usr/bin/env pvpython
"""
Render the blowup core of "Finite Time Blowup for Navier-Stokes" (OPENAI, 2026)
as stream tubes: inward spiral plus axial stretching away from the dividing
layer near z = 0.

    pvpython ns_render.py out/core.xdmf2 -o out/core.png --taper --label
    pvpython ns_render.py out/core.xdmf2 --export out/core.glb    # + core.html

--export writes a binary glTF (triangulated, per-vertex colors, normals) and a
self-contained HTML viewer with the GLB embedded, so the scene opens in any
browser straight from disk.  No text overlays; three.js is pulled from a CDN.

Three things this has to get right, all of them physics of the field:

1. The dividing layer sits slightly BELOW z = 0, at eta = -eta0 atanh(bias),
   because of the deliberate upward bias of Sec. 2.1.  u_z vanishes there, so a
   streamline cannot cross it: seeds must be placed on both sides or the
   picture comes out one-sided.  Seeds ON the layer trace planar inward
   spirals -- the wide flaring cyan lines of the schematic.

2. Beyond X_out the flow is exactly azimuthal (the heat exterior of Sec. 2.3),
   so seeds there trace closed circles.  Correct, not a bug.

3. Arclength per turn scales with radius, so a single MaximumStreamlineLength
   makes the near-axis lines coil many more times than the outer ones and the
   result is a tangle.  Each radial band therefore gets its own length, fixed by
   a target number of turns.

ParaView footguns, three of them:
* ProgrammableSource.UpdatePipeline() execs its Script against __main__'s
  globals, and ParaView's preamble there does
      from ...numpy_interface.algorithms import *
  which replaces min/max/abs -- in this module AND inside the Script's own
  namespace -- with versions that read the second positional argument as an
  axis.  Hence _mn/_mx below, and no bare min/max anywhere in SEED_SCRIPT.
* paraview.simple auto-resets the camera on the first Render() after a Show(),
  silently discarding camera settings made before it.  Burn one Render() first.
* vtkGLTFExporter only writes COLOR_0 when the mapper does NOT interpolate
  scalars before mapping; otherwise you get TEXCOORD_0 plus a LUT texture.
"""

import argparse
import base64
import json
import math
import os
import struct
import xml.etree.ElementTree as ET

from paraview.simple import *  # noqa: F403

_mn, _mx = min, max          # captured before any pipeline update

# color stops (fraction, r, g, b).  R_STOPS: distance from the axis, tan on
# the axis to pale cyan far out.  HOT_STOPS: cool -> hot, for speed and the like.
R_STOPS = [(0.00, 0.97, 0.85, 0.66), (0.10, 0.87, 0.62, 0.33), (0.22, 0.55, 0.52, 0.60),
           (0.34, 0.36, 0.47, 0.72), (0.52, 0.17, 0.54, 0.84), (0.76, 0.30, 0.78, 0.83),
           (1.00, 0.58, 0.90, 0.88)]
HOT_STOPS = [(0.00, 0.13, 0.22, 0.50), (0.22, 0.17, 0.54, 0.84), (0.45, 0.30, 0.78, 0.83),
             (0.62, 0.86, 0.90, 0.75), (0.78, 0.96, 0.78, 0.45), (0.90, 0.88, 0.52, 0.24),
             (1.00, 0.70, 0.18, 0.12)]


SEED_SCRIPT = """
import math
import vtk

R, Zt = {R!r}, {Zt!r}
r0, r1, n = {r0!r}, {r1!r}, {n!r}
eta_up, eta_dn = {eta_up!r}, {eta_dn!r}
phase = {phase!r}
flat, spread = {flat!r}, {spread!r}
GOLD = math.pi * (3.0 - math.sqrt(5.0))

pts = vtk.vtkPoints()
for j in range(n):
    r = (r0 + (r1 - r0) * ((j + 0.5) / n)) * R
    th = (j + phase) * GOLD
    if flat is not None:
        # on the layer, fanned slightly above/below it so the planar spirals
        # drift apart instead of stacking into one ring
        eta = flat + spread * (((j % 5) - 2) / 2.0)
    else:
        lo, hi = eta_up if (j % 2 == 0) else eta_dn   # alternate the two lobes
        nhalf = ((n + 1) // 2) or 1                   # no min/max here
        eta = lo + (hi - lo) * (((j // 2) + 0.5) / nhalf)
    pts.InsertNextPoint(r * math.cos(th), r * math.sin(th), eta * Zt)

out = self.GetPolyDataOutput()
out.SetPoints(pts)
verts = vtk.vtkCellArray()
for k in range(pts.GetNumberOfPoints()):
    verts.InsertNextCell(1)
    verts.InsertCellPoint(k)
out.SetVerts(verts)
"""


# One Programmable Filter: join each seed's two halves, normalised-arclength
# taper, then a tapered tube or ribbon (ParaView's proxies hide the factors).
RIBBON_SCRIPT = """
import numpy as np
import vtk
from vtk.util import numpy_support as vn

WMAX, WFAC, POW, DEFN = {wmax!r}, {wfac!r}, {pow!r}, {defn!r}
STRAND, SIDES = {strand!r}, {sides!r}

inp = self.GetInputDataObject(0, 0)
# BOTH-direction tracing gives two polylines per seed meeting at the seed;
# merge that shared point and join them into one strand, else the taper
# pinches every strand to a point at its seed.
cl = vtk.vtkCleanPolyData(); cl.SetInputData(inp); cl.PointMergingOn(); cl.SetTolerance(0.0)
sp = vtk.vtkStripper(); sp.SetInputConnection(cl.GetOutputPort()); sp.JoinContiguousSegmentsOn()
sp.Update()
pd = vtk.vtkPolyData(); pd.ShallowCopy(sp.GetOutput())
pts = vn.vtk_to_numpy(pd.GetPoints().GetData())
taper = np.zeros(pd.GetNumberOfPoints())
lines = pd.GetLines(); lines.InitTraversal(); ids = vtk.vtkIdList()
while lines.GetNextCell(ids):
    n = ids.GetNumberOfIds()
    if n < 2:
        continue
    idx = np.array([ids.GetId(k) for k in range(n)])
    seg = np.linalg.norm(np.diff(pts[idx], axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] > 0:
        s = s / s[-1]
    taper[idx] = np.sin(np.pi * s) ** POW          # 0 at both ends, 1 mid-strand
arr = vn.numpy_to_vtk(taper.astype(np.float32), deep=True); arr.SetName("taper")
pd.GetPointData().AddArray(arr)
pd.GetPointData().SetActiveScalars("taper")
if pd.GetPointData().GetArray("Normals") is not None:
    pd.GetPointData().SetActiveNormals("Normals")   # twist with the flow

if STRAND == "tube":
    # round, tapered: radius = Radius*(1+(RadiusFactor-1)*taper) -> points at the ends
    tf = vtk.vtkTubeFilter()
    tf.SetInputData(pd)
    tf.SetRadius(WMAX / WFAC)
    tf.SetVaryRadiusToVaryRadiusByScalar()
    tf.SetRadiusFactor(WFAC)
    tf.SetNumberOfSides(SIDES)
    tf.CappingOn()
    tf.Update()
    self.GetPolyDataOutput().ShallowCopy(tf.GetOutput())
else:
    rf = vtk.vtkRibbonFilter()
    rf.SetInputData(pd)
    rf.SetWidth(WMAX / WFAC)      # VaryWidth maps taper 0..1 onto [Width, Width*WidthFactor]
    rf.SetVaryWidth(1)
    rf.SetWidthFactor(WFAC)
    rf.SetAngle(0.0)
    rf.SetUseDefaultNormal(1 if DEFN else 0)
    rf.SetDefaultNormal(0.0, 0.0, 1.0)
    rf.Update()
    self.GetPolyDataOutput().ShallowCopy(rf.GetOutput())
"""


# ----------------------------------------------------------------------------
# glTF -> GLB -> HTML
# ----------------------------------------------------------------------------

def pack_glb(gltf_path, glb_path):
    """Merge the exporter's many inline buffers into one and wrap as GLB."""
    with open(gltf_path) as f:
        g = json.load(f)
    blob = bytearray()
    offsets = []
    for b in g["buffers"]:
        data = base64.b64decode(b["uri"].split(",", 1)[1])
        while len(blob) % 4:
            blob += b"\0"
        offsets.append(len(blob))
        blob += data
    for bv in g["bufferViews"]:
        bv["byteOffset"] = bv.get("byteOffset", 0) + offsets[bv["buffer"]]
        bv["buffer"] = 0
    while len(blob) % 4:
        blob += b"\0"
    g["buffers"] = [{"byteLength": len(blob)}]
    g.pop("cameras", None)                     # the viewer frames the scene itself
    for nd in g.get("nodes", []):
        nd.pop("camera", None)
    js = json.dumps(g, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(glb_path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)        # JSON
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))  # BIN
    return g, total


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; overflow: hidden;
               font: 15px/1.3 -apple-system, "Helvetica Neue", Arial, sans-serif;
               color: #2b2f36; }
  #c { display: block; width: 100%; height: 100%; }
</style>
</head>
<body>
<canvas id="c"></canvas>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TrackballControls.js"></script>
<script>
const GLB_B64 = "__GLB__";

function b64ToBuf(s) {
  const bin = atob(s), n = bin.length, u8 = new Uint8Array(n);
  for (let i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
  return u8.buffer;
}

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xffffff, 1);
// VTK's COLOR_0 are already display-referred; r128 treats vertex colors as
// linear, so leave outputEncoding Linear or they get gamma-lifted twice.

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, 1, 0.001, 100);
// TrackballControls: no polar clamp, no fixed up-axis -- rotate freely.
const controls = new THREE.TrackballControls(camera, canvas);
controls.rotateSpeed = 3.0; controls.zoomSpeed = 1.2; controls.panSpeed = 0.8;
controls.staticMoving = false; controls.dynamicDampingFactor = 0.18;

// ParaView's lighting model: flat ambient + a headlight riding on the camera.
// Vertex colors reach the screen at (ambient + diffuse) ~ 1.0, i.e. unmuted.
scene.add(new THREE.AmbientLight(0xffffff, 0.42));
const head = new THREE.DirectionalLight(0xffffff, 0.78);
head.position.set(0.35, 0.55, 1.0);          // slightly above-right of the eye
camera.add(head); scene.add(camera);

let home = null;
function frame(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3()), ctr = box.getCenter(new THREE.Vector3());
  const h = Math.max(size.z, size.x * 1.2, size.y * 1.2);
  const dist = 0.5 * h / Math.tan(0.5 * camera.fov * Math.PI / 180) * 1.15;
  const az = -62 * Math.PI / 180, el = 13 * Math.PI / 180;
  camera.position.set(ctr.x + dist * Math.cos(el) * Math.cos(az),
                      ctr.y + dist * Math.cos(el) * Math.sin(az),
                      ctr.z + dist * Math.sin(el));
  camera.up.set(0, 0, 1);
  camera.near = dist / 100; camera.far = dist * 100; camera.updateProjectionMatrix();
  controls.target.copy(ctr); controls.update();
  home = { pos: camera.position.clone(), up: camera.up.clone(), tgt: ctr.clone() };
}
window.addEventListener("keydown", e => { if (e.key === "r" && home) {
  camera.position.copy(home.pos); camera.up.copy(home.up);
  controls.target.copy(home.tgt); controls.update(); } });

new THREE.GLTFLoader().parse(b64ToBuf(GLB_B64), "", gltf => {
  gltf.scene.traverse(o => {
    if (!o.isMesh) return;
    const g = o.geometry;
    if (!g.attributes.normal) g.computeVertexNormals();
    o.material = new THREE.MeshPhongMaterial({
      vertexColors: !!g.attributes.color, color: g.attributes.color ? 0xffffff : 0x9aa1a6,
      specular: 0x2a2a2a, shininess: 45, side: THREE.DoubleSide });
  });
  scene.add(gltf.scene);
  frame(gltf.scene);
});

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  controls.handleResize();
}
window.addEventListener("resize", resize); resize();

(function loop() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
</script>
</body>
</html>
"""


def write_html(glb_path, html_path, title, ntri):
    with open(glb_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    page = HTML.replace("__TITLE__", title).replace("__GLB__", b64)
    with open(html_path, "w") as f:
        f.write(page)
    return len(page)


def export_scene(view, displays, glb_path):
    """Per-vertex colors (see footgun 3), export, pack, and report."""
    import vtk
    for d in displays:
        d.InterpolateScalarsBeforeMapping = 0
    Render()
    tmp = glb_path + ".inline.gltf"
    e = vtk.vtkGLTFExporter()
    e.SetRenderWindow(view.GetRenderWindow())
    e.SetFileName(tmp)
    e.SetInlineData(True)
    e.SetSaveNormal(True)
    e.Write()
    g, total = pack_glb(tmp, glb_path)
    os.remove(tmp)
    ntri = 0
    attrs = set()
    for m in g["meshes"]:
        for p in m["primitives"]:
            attrs |= set(p["attributes"])
            if "indices" in p:
                ntri += g["accessors"][p["indices"]]["count"] // 3
    for d in displays:
        d.InterpolateScalarsBeforeMapping = 1
    return total, ntri, sorted(attrs), len(g["meshes"])


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vti", help="core.xdmf2 (XDMF2 + .raw companions)")
    ap.add_argument("-o", "--out", default="out/core.png")
    ap.add_argument("--size", type=int, nargs=2, default=[1500, 1500])
    ap.add_argument("--zscale", type=float, default=1.0,
                    help="display-only vertical exaggeration (the paper's Fig. 1 "
                         "does this too); the physics is untouched")
    ap.add_argument("--color-range", type=float, nargs=2, default=None,
                    help="fix the color scale (lo hi) instead of the data range; "
                         "use one range across frames so speed shows the tau^-A growth")
    ap.add_argument("--gray", action="store_true",
                    help="grey ramp instead of a palette, so an exported GLB carries the "
                         "scalar itself (COLOR_0.r = value / range) and the page applies "
                         "the palette in a shader")
    ap.add_argument("--color", default="r",
                    choices=["r", "X", "speed", "log10_speed", "u_theta", "p", "eta"],
                    help="r is the physical radius; X = r^2/2q shrinks with |z| and "
                         "makes the ends of the column look like the core")
    ap.add_argument("--tube", type=float, default=0.013,
                    help="tube radius as a fraction of the box radius")
    ap.add_argument("--sides", type=int, default=8, help="tube facets")
    ap.add_argument("--taper", action="store_true",
                    help="tubes only: vary radius with speed")
    ap.add_argument("--ribbon", action="store_true",
                    help="tapered strands (points at both ends) instead of plain tubes")
    ap.add_argument("--strand", default="tube", choices=["tube", "ribbon"],
                    help="tapered strand cross-section: round tubes (shaded like the "
                         "reference) or flat ribbons")
    ap.add_argument("--ribbon-width", type=float, default=0.012,
                    help="max ribbon width as a fraction of the box radius")
    ap.add_argument("--ribbon-pow", type=float, default=0.7,
                    help="taper exponent: width ~ sin(pi s)^pow along the strand")
    ap.add_argument("--flat-ribbons", action="store_true",
                    help="orient ribbons horizontally instead of twisting with the flow")
    ap.add_argument("--decimate", type=float, default=0.0,
                    help="DecimatePolyline target reduction, 0..1, before tubing")
    ap.add_argument("--ninner", type=int, default=24)
    ap.add_argument("--nmiddle", type=int, default=24, help="second, nested helix band")
    ap.add_argument("--nouter", type=int, default=28)
    ap.add_argument("--next", type=int, default=0,
                    help="seeds in the purely azimuthal exterior")
    ap.add_argument("--nmid", type=int, default=44,
                    help="seeds ON the dividing layer: u_z = 0 there, so these "
                         "trace planar inward spirals -- the clearest view of "
                         "the spin-up mechanism of Sec. 2.1")
    ap.add_argument("--turns", type=float, nargs=3, default=[4.0, 4.5, 1.15],
                    help="target turns for the inner, outer and exterior bands")
    ap.add_argument("--layer-turns", type=float, default=2.2,
                    help="target turns for the seeds on the dividing layer")
    ap.add_argument("--layer-spread", type=float, default=0.12,
                    help="fan the layer seeds this far above/below it (fraction "
                         "of the box half-height) so the spirals separate")
    ap.add_argument("--eta-up", type=float, nargs=2, default=[0.08, 0.46])
    ap.add_argument("--eta-dn", type=float, nargs=2, default=[-0.12, -0.52])
    ap.add_argument("--azimuth", type=float, default=-62.0)
    ap.add_argument("--elevation", type=float, default=18.0)
    ap.add_argument("--zoom", type=float, default=0.90)
    ap.add_argument("--label", action="store_true",
                    help="annotate the two mechanisms, as in the schematic")
    ap.add_argument("--no-arrow", action="store_true", help="omit the axis arrowhead")
    ap.add_argument("--export", default=None,
                    help="write a .glb and a self-contained .html viewer beside it")
    ap.add_argument("--no-html", action="store_true",
                    help="with --export: write only the .glb")
    ap.add_argument("--state", default=None, help="also save a .pvsm for the GUI")
    a = ap.parse_args()

    src = OpenDataFile(os.path.abspath(a.vti))
    src.UpdatePipeline()
    b = src.GetDataInformation().GetBounds()
    R, Zt = _mn(b[1], b[3]), b[5]

    geom = src
    if a.zscale != 1.0:
        geom = Transform(Input=src)
        geom.Transform.Scale = [1.0, 1.0, a.zscale]
        geom.UpdatePipeline()

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = a.size
    view.UseColorPaletteForBackground = 0
    view.Background = view.Background2 = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1

    # profile constants travel as <Information Name= Value=/> in the XDMF
    info = {e.get("Name"): float(e.get("Value"))
            for e in ET.parse(a.vti).iter("Information")
            if e.get("Name") and e.get("Value")}
    eta_star = info.get("eta_star", -0.0226)
    eta_max = info.get("eta_max", 0.95)
    flat_eta = eta_star / eta_max     # the layer, as a fraction of the box half-height

    # X_out is at r/R = sqrt(1/Xmax_fac) = sqrt(1/1.6) = 0.79 of the box:
    # inside, the flow is poloidal; outside, exactly azimuthal.
    # (name, r0/R, r1/R, n, turns, phase, tube scale, taper, seed-on-layer)
    bands = [("inner", 0.05, 0.22, a.ninner, a.turns[0], 0.00, 0.9, True, None),
             ("middle", 0.24, 0.46, a.nmiddle, a.turns[1], 0.21, 1.0, True, None),
             ("outer", 0.48, 0.70, a.nouter, a.turns[1], 0.37, 1.0, True, None)]
    if a.nmid > 0:
        bands.append(("layer", 0.70, 0.985, a.nmid, a.layer_turns, 0.53,
                      1.5, True, flat_eta))      # the wide cyan inflow blades
    if a.next > 0:
        bands.append(("exterior", 0.85, 0.97, a.next, a.turns[2], 0.71,
                      0.58, False, None))

    tubes, displays = [], []
    for (name, r0, r1, n, turns, phase, tscale, taper, flat) in bands:
        if n <= 0:
            continue
        seeds = ProgrammableSource(registrationName=f"seeds_{name}")
        seeds.OutputDataSetType = "vtkPolyData"
        seeds.Script = SEED_SCRIPT.format(
            R=R, Zt=Zt * a.zscale, r0=r0, r1=r1, n=n, phase=phase,
            eta_up=tuple(a.eta_up), eta_dn=tuple(a.eta_dn), flat=flat,
            spread=a.layer_spread)
        seeds.UpdatePipeline()

        st = StreamTracerWithCustomSource(registrationName=f"trace_{name}",
                                          Input=geom, SeedSource=seeds)
        st.Vectors = ["POINTS", "u"]
        st.IntegrationDirection = "BOTH"
        st.IntegratorType = "Runge-Kutta 4-5"
        st.IntegrationStepUnit = "Cell Length"
        st.InitialStepLength = 0.2
        st.MinimumStepLength = 0.01
        st.MaximumStepLength = 0.5
        st.MaximumSteps = 200000
        # arclength per turn ~ 2 pi r_mid, so a per-band length equalises turns
        st.MaximumStreamlineLength = turns * 2.0 * math.pi * 0.5 * (r0 + r1) * R
        st.TerminalSpeed = 1e-14
        st.MaximumError = 1e-6
        st.UpdatePipeline()

        # why each half-strand stopped: 1 left the box, 4 hit the length cap,
        # 5 ran out of steps, 6 reached a stagnation point
        rt = servermanager.Fetch(st).GetCellData().GetArray("ReasonForTermination")
        if rt is not None:
            from collections import Counter
            hist = Counter(int(rt.GetTuple1(i)) for i in range(rt.GetNumberOfTuples()))
            names = {1: "box", 4: "length", 5: "steps", 6: "stagnation"}
            print(f"  {name:8s} stops: " + ", ".join(
                f"{names.get(k, k)} {v}" for k, v in sorted(hist.items())))

        lines = st
        if a.decimate > 0.0:
            lines = DecimatePolyline(registrationName=f"thin_{name}", Input=st)
            lines.TargetReduction = a.decimate
            lines.UpdatePipeline()

        if a.ribbon:
            tb = ProgrammableFilter(registrationName=f"ribbon_{name}", Input=lines)
            tb.OutputDataSetType = "vtkPolyData"
            tb.Script = RIBBON_SCRIPT.format(
                wmax=a.ribbon_width * R * tscale, wfac=25.0, pow=a.ribbon_pow,
                defn=bool(a.flat_ribbons), strand=a.strand, sides=a.sides)
            tb.UpdatePipeline()
        else:
            tb = Tube(registrationName=f"tube_{name}", Input=lines)
            tb.Scalars = ["POINTS", "speed"]
            tb.Vectors = ["POINTS", "u"]
            tb.NumberofSides = a.sides
            tb.Radius = a.tube * R * tscale
            if a.taper and taper:
                tb.VaryRadius = "By Scalar"
                tb.RadiusFactor = 5.0
            tb.UpdatePipeline()

        if a.color == "r":
            rc = Calculator(registrationName=f"r_{name}", Input=tb)
            rc.ResultArrayName = "r"
            rc.Function = "sqrt(coordsX*coordsX+coordsY*coordsY)"
            rc.UpdatePipeline()
            tb = rc
        d = Show(tb, view)
        ColorBy(d, ("POINTS", a.color))
        d.SetScalarBarVisibility(view, False)
        d.Ambient, d.Diffuse = 0.30, 0.78
        d.Specular, d.SpecularPower = 0.30, 45
        tubes.append((name, lines, tb))
        displays.append(d)

    # the axis of rotation, spanning the actual tube extent, with an arrowhead
    zlo = _mn(tb.GetDataInformation().GetBounds()[4] for _, _, tb in tubes)
    zhi = _mx(tb.GetDataInformation().GetBounds()[5] for _, _, tb in tubes)
    rmax = _mx(tb.GetDataInformation().GetBounds()[1] for _, _, tb in tubes)
    grey = [0.60, 0.66, 0.69]
    axis = Line(registrationName="axis")
    axis.Point1 = [0.0, 0.0, 1.08 * zlo]
    axis.Point2 = [0.0, 0.0, 1.10 * zhi]
    axis.Resolution = 2
    stem = Tube(registrationName="axis_tube", Input=axis)
    stem.Radius = 0.0030 * R
    stem.NumberofSides = 12
    ds = Show(stem, view)
    ds.ColorArrayName = ["POINTS", ""]
    ds.AmbientColor = ds.DiffuseColor = grey
    ds.SetScalarBarVisibility(view, False)
    if not a.no_arrow:
        cone = Cone(registrationName="axis_arrow")
        cone.Resolution = 24
        cone.Radius = 0.012 * R
        cone.Height = 0.045 * R
        cone.Direction = [0.0, 0.0, 1.0]
        cone.Center = [0.0, 0.0, 1.10 * zhi + 0.5 * cone.Height]
        dc = Show(cone, view)
        dc.ColorArrayName = ["POINTS", ""]
        dc.AmbientColor = dc.DiffuseColor = grey

    lo, hi = None, None
    for _, _, tb in tubes:
        ai = tb.GetPointDataInformation().GetArray(a.color)
        if ai is None:
            continue
        r = ai.GetRange(0)
        lo = r[0] if lo is None else _mn(lo, r[0])
        hi = r[1] if hi is None else _mx(hi, r[1])
    lo, hi = (0.0, 1.0) if lo is None else (lo, hi)
    if a.color_range:
        lo, hi = a.color_range
    lut = GetColorTransferFunction(a.color)
    stops = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)] if a.gray \
        else (R_STOPS if a.color == "r" else HOT_STOPS)
    lut.RGBPoints = [v for (f, r_, g_, b_) in stops for v in (lo + f * (hi - lo), r_, g_, b_)]
    lut.ColorSpace = "RGB" if a.gray else "Lab"

    if a.label:
        for txt, pos in (("inward spiral", [0.045, 0.565]),
                         ("axial stretching", [0.60, 0.885]),
                         ("dividing layer  u_z = 0", [0.045, 0.455])):
            t = Text(registrationName=txt[:12])
            t.Text = txt
            td = Show(t, view)
            td.WindowLocation = "Any Location"
            td.Position = pos
            td.Color = [0.13, 0.14, 0.16]
            td.FontSize = 26

    # paraview.simple auto-resets the camera on the first Render() after a
    # Show(); burn that render first, or the camera settings below are lost.
    Render()

    az, el = math.radians(a.azimuth), math.radians(a.elevation)
    dist = 8.0 * _mx(R, Zt * a.zscale)
    ctr = [0.0, 0.0, 0.5 * (zlo + zhi)]
    view.CameraFocalPoint = ctr
    view.CameraPosition = [ctr[0] + dist * math.cos(el) * math.cos(az),
                           ctr[1] + dist * math.cos(el) * math.sin(az),
                           ctr[2] + dist * math.sin(el)]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelScale = _mx(
        0.5 * (zhi - zlo), rmax * a.size[1] / a.size[0]) / a.zoom
    Render()
    print(f"  camera parallel scale {view.CameraParallelScale:.5g}  "
          f"(z half-extent {0.5*(zhi-zlo):.5g}, r max {rmax:.5g})")

    SaveScreenshot(os.path.abspath(a.out), view,
                   ImageResolution=a.size, TransparentBackground=0)
    counts = ", ".join(f"{nm} {st.GetDataInformation().GetNumberOfCells()}"
                       for nm, st, _ in tubes)   # half-strands (2 per seed)
    print(f"  wrote {a.out}   box r<={R:.4g} |z|<={Zt:.4g}   [{counts}]   "
          f"{a.color} in [{lo:.4g}, {hi:.4g}]")

    if a.export:
        glb = os.path.abspath(a.export)
        nbytes, ntri, attrs, nmesh = export_scene(view, displays, glb)
        print(f"  wrote {a.export}   {nbytes/1e6:.1f} MB   {nmesh} meshes   "
              f"{ntri:,} triangles   attributes {attrs}")
        if not a.no_html:
            html = os.path.splitext(glb)[0] + ".html"
            hbytes = write_html(glb, html, "Navier–Stokes blowup core", ntri)
            print(f"  wrote {html}   {hbytes/1e6:.1f} MB  (self-contained; open in a browser)")

    if a.state:
        SaveState(os.path.abspath(a.state))
        print(f"  wrote {a.state}  (open in the ParaView GUI)")


if __name__ == "__main__":
    main()
