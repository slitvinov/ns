Streamlines of the leading-order blowup core in "Finite Time Blowup for
Navier-Stokes" (OPENAI, 2026), Sec. 3.1, in the style of the figure OpenAI
posted with the paper (img/openai-reference.png).

Live 3D:  https://slitvinov.github.io/ns/

  pvpython ns_core.py --t 0.9 --n 161 --Ce 8 --Xe 0.06 --Xout 0.6 --Xmax 0.3 \
                      --eta0 0.45 --eta1 0.8 --bias 0.35 -o out/core.xdmf2
  pvpython ns_render.py out/core.xdmf2 -o out/core.png --ribbon --decimate 0.6 \
                        --export out/core.glb

ns_core.py    builds the field, writes XDMF2 + raw float32 (numpy, scipy)
ns_render.py  ParaView streamlines -> PNG, GLB, and a self-contained
              three.js page (core.html)
ns_time.py    the live page: ONE scene at t = 0.5 plus a time slider to
              t = 0.999.  The field is self-similar, so the scene at time t
              is the t = 0.5 scene scaled by (sqrt(lam), sqrt(lam), lam^D),
              lam = (1-t)/0.5, with speed x lam^-A -- exact up to a factor
              lam^-h on the swirl (6% at the very end).  Color: speed on one
              scale for all times (saturates from t = 0.98 on), linear or
              log, or distance from the axis (?c=1).  The GLB is embedded.

  S0=$(...)   # 99.9th percentile of speed at t = 0.5
  pvpython ns_render.py out/core.xdmf2 --ribbon --decimate 0.35 --sides 12 \
                        --color speed --gray --color-range 0 $S0 --export out/core.glb --no-html
  pvpython ns_time.py out/core.glb --s0 $S0 -o site/index.html

WHAT IS PLOTTED

Instantaneous streamlines of the leading-order field u^(0) of Sec. 3.1 at
t = 0.9, nu = 1.  Cylindrical (r, theta, z); tau = 1 - t; h = 0.01.

  A = 1/2 + h,   D = 1/2 - h
  z = q^D eta,   tau = q (1 - eta^2),   X = r^2 / (2q)         (eq. 3.2)
  u_theta = q^-A E(X),   u_z = q^-A U(X, eta),   u_r = V0(X, eta) / r
  p = q^-2A Pi(X),   Pi(X) = -int_X^inf E(x)^2 / (2x) dx      (eq. 4.3)

E, U, V0 are built in the paper's Appendices A-C and have no closed form.
Here they are replaced by explicit functions with the properties Sec. 3.1
requires (smooth on the axis, E > 0, E ~ X^(-1/2-h) at large X, U = V0 = 0
beyond a fixed X, u_z of opposite sign on the two sides of a layer near
z = 0 and nonzero at z = 0):

  E(X)    = C_E sqrt(2X) (1 + 2X/X_e)^-(1+h)
  psi     = q^(1-A) C_psi X B(X/X_out) g(eta)        Stokes streamfunction
  u_r     = -(1/r) d_z psi,   u_z = (1/r) d_r psi     (div u = 0 exactly)
  B(s)    = exp(1 - 1/(1 - s^2)) for |s| < 1, 0 otherwise
  g(eta)  = (tanh(eta/eta0) + b) exp(-(eta/eta1)^4)

  C_E = 8, C_psi = 1, X_e = 0.06, X_out = 0.6, eta0 = 0.45, eta1 = 0.8, b = 0.35
  box: r <= sqrt(2 X_max tau), X_max = 0.3;  |z| <= 0.95 tau^D;  161^3 nodes

Streamlines are integrated both ways from seeds on four radial bands (0.05-
0.22, 0.24-0.46, 0.48-0.70 of the box radius on both sides of the layer;
0.70-0.985 on the layer), stopped at the box or after a fixed number of
turns, and drawn as tubes with radius ~ sin(pi s)^0.7 in normalised arclength.
Color in the still image = distance from the axis r; in the time slider,
speed on one fixed scale, or r.  Constants were chosen to match
the reference figure; only the form and the scaling laws are the paper's.

Not shown: the pulses and corrections of Secs. 6-9, i.e. the proof.
