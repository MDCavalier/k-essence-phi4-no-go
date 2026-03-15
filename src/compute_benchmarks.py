#!/usr/bin/env python3
"""
Publication-quality figures and benchmark tables for the paper.
Key insight: Mpb2 = (M_Pl * beta)^2 ~ 10^{-2} for slow roll.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from scipy.optimize import brentq
from scipy.integrate import quad

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'text.usetex': False, 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.5, 'lines.linewidth': 1.8,
})

# ── Core functions ─────────────────────────────────────────────────
def U(y):
    y = np.asarray(y, dtype=float)
    return -np.log(np.cos(np.clip(y, -1.5699, 1.5699)))

def F_func(y):
    y = np.atleast_1d(np.float64(y))
    r = np.zeros_like(y)
    s = np.abs(y) < 0.05
    r[s] = 2.0/y[s]**4 + 1.0/y[s]**2 + 7./15.
    l = ~s
    r[l] = np.tan(y[l])**2 / (U(y[l]) * y[l]**4)
    return float(r[0]) if r.size == 1 else r

def solve_master(R):
    R = float(R)
    if R < 1e-12: return R
    u = R if R < 1 else (R/4)**(1./3.)
    for _ in range(30):
        f = u*(1+2*u)**2 - R; fp = (1+2*u)**2 + 4*u*(1+2*u)
        u = max(u - f/fp, 1e-20)
        if abs(f) < 1e-14*(R+1e-30): break
    return u

def cs2(u): return (1+2*u)/(1+6*u)

def Phi(y):
    y = float(y)
    return y/2 - y**3/12 if abs(y) < 0.05 else U(y)/np.tan(y)

def L_func(y):
    y = float(y)
    return -4./y + 2*y/3 if abs(y)<0.05 else 2/(np.sin(y)*np.cos(y)) - np.tan(y)/U(y) - 4/y

def G_func(y):
    y = float(y)
    return -2./y + 2*y/3 if abs(y)<0.05 else 2/(np.sin(y)*np.cos(y)) - 2*np.tan(y)/U(y)

def eps_V(y, Mpb2): return Mpb2/2 * np.tan(y)**2/U(y)**2
def eps(y, u, Mpb2): return eps_V(y, Mpb2)/(1+2*u)
def r_obs(y, u, Mpb2): return 16*eps(y,u,Mpb2)*np.sqrt(cs2(u))

def ns_obs(y, u, C, Mpb2):
    e = eps(y, u, Mpb2); yp = -2*e*Phi(y)
    L = L_func(y); G = G_func(y)
    denom = 1 + 4*u/(1+2*u)
    up_u = L*yp/denom
    eta = G*yp - 2*u*up_u/(1+2*u)
    s = -2*u/((1+2*u)*(1+6*u))*up_u
    return 1 - 2*e - eta - s

def fNL_eq(u):
    if u<1e-12: return 0.
    return -(10./3.)*u*(5+26*u)/((1+2*u)*(1+6*u)) + (80./81.)*u**2/(1+2*u)**2

def fNL_or(u):
    if u<1e-12: return 0.
    return (10./3.)*u*(1+10*u)/((1+2*u)*(1+6*u))

def efolds(y1, y0, C, Mpb2):
    def integ(y):
        uu = solve_master(C*F_func(y))
        return (1+2*uu)*U(y)/(np.tan(y)*Mpb2)
    N, _ = quad(integ, y0, y1, limit=200)
    return N

def find_yend(C, Mpb2):
    def f(y):
        uu = solve_master(C*F_func(y))
        return eps(y, uu, Mpb2) - 1
    try: return brentq(f, 0.01, 1.56, xtol=1e-12)
    except: return 0.01


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Thermal potential and structure function
# ═══════════════════════════════════════════════════════════════════
print("Fig 1...")
fig, (a1,a2) = plt.subplots(1, 2, figsize=(12, 4.5))
y1 = np.linspace(0.01, 1.55, 500)
a1.plot(y1, U(y1), 'b-', lw=2.2)
a1.fill_between([np.pi/2-0.025, np.pi/2+0.025], 0, 6, color='red', alpha=0.12)
a1.annotate('Thermal\nhorizon', xy=(np.pi/2, 4.2), fontsize=10, ha='center', color='darkred', style='italic')
a1.set_xlabel(r'$y = \beta\phi$'); a1.set_ylabel(r'$U(y)$')
a1.set_title('(a) Thermal potential $U(y)=-\\ln(\\cos y)$', fontsize=12)
a1.set_xlim(0, 1.65); a1.set_ylim(0, 5.5)

y2 = np.linspace(0.08, 1.52, 500)
F2 = np.array([F_func(yy) for yy in y2])
a2.semilogy(y2, F2, 'b-', lw=2.2)
ym = y2[np.argmin(F2)]
a2.axvline(ym, color='gray', ls=':', alpha=0.5)
a2.annotate(f'min at $y\\approx{ym:.2f}$', xy=(ym+0.05, min(F2)*1.5), fontsize=9, color='gray')
a2.set_xlabel(r'$y = \beta\phi$'); a2.set_ylabel(r'$F(y)$')
a2.set_title('(b) Structure function', fontsize=12)
a2.set_xlim(0.08, 1.52)
fig.tight_layout(w_pad=3); fig.savefig('/home/claude/fig1_potential_structure.pdf'); plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: c_s^2 and f_NL vs u
# ═══════════════════════════════════════════════════════════════════
print("Fig 2...")
fig, (a1,a2) = plt.subplots(1, 2, figsize=(12, 4.5))
ua = np.logspace(-2, 2, 500)
a1.semilogx(ua, cs2(ua), 'b-', lw=2.2)
a1.axhline(1./3., color='red', ls='--', alpha=0.5); a1.axhline(1., color='green', ls='--', alpha=0.5)
a1.annotate(r'$c_s^2\to 1/3$', xy=(50, 0.36), fontsize=10, color='red')
a1.annotate(r'$c_s^2\to 1$', xy=(0.015, 0.96), fontsize=10, color='green')
a1.set_xlabel(r'$u$'); a1.set_ylabel(r'$c_s^2$'); a1.set_title('(a) Sound speed', fontsize=12)
a1.set_ylim(0.25, 1.08)

fe = np.array([fNL_eq(uu) for uu in ua])
fo = np.array([fNL_or(uu) for uu in ua])
a2.semilogx(ua, fe, 'b-', lw=2.2, label=r'$f_{\rm NL}^{\rm equil}$')
a2.semilogx(ua, fo, 'r-', lw=2.2, label=r'$f_{\rm NL}^{\rm orth}$')
a2.axhline(-565./81., color='b', ls=':', alpha=0.3); a2.axhline(25./9., color='r', ls=':', alpha=0.3)
a2.axhline(0, color='k', alpha=0.15)
a2.set_xlabel(r'$u$'); a2.set_ylabel(r'$f_{\rm NL}$'); a2.set_title('(b) Non-Gaussianity', fontsize=12)
a2.set_ylim(-10, 5); a2.legend(fontsize=11, loc='right')
fig.tight_layout(w_pad=3); fig.savefig('/home/claude/fig2_cs_fNL.pdf'); plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Master equation u(y) for various C
# ═══════════════════════════════════════════════════════════════════
print("Fig 3...")
fig, ax = plt.subplots(figsize=(7, 5))
y3 = np.linspace(0.06, 1.5, 400)
Cv = [1e-2, 1e-1, 1, 10, 1e2, 1e4]
labs = [r'$\mathcal{C}=10^{-2}$',r'$\mathcal{C}=10^{-1}$',r'$\mathcal{C}=1$',
        r'$\mathcal{C}=10$',r'$\mathcal{C}=10^{2}$',r'$\mathcal{C}=10^{4}$']
cols = ['#2166ac','#4393c3','#92c5de','#f4a582','#d6604d','#b2182b']
for C,l,c in zip(Cv,labs,cols):
    ax.semilogy(y3, [solve_master(C*F_func(yy)) for yy in y3], color=c, lw=2, label=l)
ax.set_xlabel(r'$y$'); ax.set_ylabel(r'$u(y)$')
ax.set_title(r'Master equation: $u(1+2u)^2=\mathcal{C}\,F(y)$', fontsize=13)
ax.legend(ncol=2, fontsize=10); ax.set_xlim(0.06,1.5); ax.set_ylim(1e-2,1e5)
fig.tight_layout(); fig.savefig('/home/claude/fig3_master_equation.pdf'); plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: (n_s, r) plane
# ═══════════════════════════════════════════════════════════════════
print("Fig 4: scanning parameter space...")
fig, ax = plt.subplots(figsize=(7.5, 5.5))

ell2 = Ellipse((0.9649,0.004), 4*0.0044, 0.044, fc='royalblue', alpha=0.08,
               ec='royalblue', lw=1.0, ls='--', label=r'Planck+BK18 $2\sigma$')
ell1 = Ellipse((0.9649,0.004), 2*0.0044, 0.022, fc='royalblue', alpha=0.15,
               ec='royalblue', lw=1.5, label=r'Planck+BK18 $1\sigma$')
ax.add_patch(ell2); ax.add_patch(ell1)

# Correct range: Mpb2 ~ 10^{-3} to 10^{-1}
Mpb2s = np.logspace(-3.5, -0.5, 30)
Cs = np.logspace(-2, 5, 80)
ys = np.linspace(0.3, 1.45, 20)

ns_a, r_a, u_a = [], [], []
for Mpb2 in Mpb2s:
    for C in Cs:
        for yy in ys:
            try:
                uu = solve_master(C*F_func(yy))
                e = eps(yy, uu, Mpb2)
                if e > 0.08 or e < 1e-6: continue
                n = ns_obs(yy, uu, C, Mpb2)
                rv = r_obs(yy, uu, Mpb2)
                if 0.93 < n < 1.0 and 0 < rv < 0.12:
                    ns_a.append(n); r_a.append(rv); u_a.append(uu)
            except: pass

ns_a = np.array(ns_a); r_a = np.array(r_a); u_a = np.array(u_a)
print(f"  {len(ns_a)} viable points found")

if len(ns_a) > 0:
    sc = ax.scatter(ns_a, r_a, c=np.log10(u_a+0.01), cmap='magma', s=5, alpha=0.5,
                    vmin=-1, vmax=2.5, zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(r'$\log_{10}(u_*)$', fontsize=12)

# Draw constant-y_* curves at Mpb2 = 0.01
Mpb2_ref = 0.01
for yy in [0.5, 0.8, 1.1, 1.4]:
    nt, rt = [], []
    for C in np.logspace(-2, 6, 300):
        uu = solve_master(C*F_func(yy))
        e = eps(yy, uu, Mpb2_ref)
        if e > 0.06 or e < 1e-6: continue
        n = ns_obs(yy, uu, C, Mpb2_ref); rv = r_obs(yy, uu, Mpb2_ref)
        if 0.93 < n < 1.0 and 0 < rv < 0.12:
            nt.append(n); rt.append(rv)
    if len(nt)>3:
        ax.plot(nt, rt, 'k-', lw=0.8, alpha=0.5)
        ax.annotate(f'$y_*\\!={yy}$', xy=(nt[len(nt)//3], rt[len(nt)//3]),
                    fontsize=8, alpha=0.7)

ax.axhline(0.036, color='red', ls='--', lw=1.4, alpha=0.7, label=r'$r<0.036$ (BK18)')
ax.set_xlabel(r'$n_s$'); ax.set_ylabel(r'$r$')
ax.set_xlim(0.935, 1.00); ax.set_ylim(0, 0.10)
ax.set_title(r'Model predictions in the $(n_s,\,r)$ plane', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
fig.tight_layout(); fig.savefig('/home/claude/fig4_ns_r.pdf'); plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Representative inflationary trajectory
# ═══════════════════════════════════════════════════════════════════
print("Fig 5: trajectory...")
Mpb2_ex = 0.008; C_ex = 10.0
ye = find_yend(C_ex, Mpb2_ex)
print(f"  y_end = {ye:.6f}")

yt = np.linspace(ye+0.001, 1.50, 800)
Nt = np.zeros(len(yt)); ut = np.zeros(len(yt)); ct = np.zeros(len(yt))
et = np.zeros(len(yt)); nt_arr = np.zeros(len(yt))

for i, yv in enumerate(yt):
    ut[i] = solve_master(C_ex*F_func(yv))
    ct[i] = np.sqrt(cs2(ut[i]))
    et[i] = eps(yv, ut[i], Mpb2_ex)
    nt_arr[i] = ns_obs(yv, ut[i], C_ex, Mpb2_ex)
    try: Nt[i] = efolds(yv, ye, C_ex, Mpb2_ex)
    except: Nt[i] = Nt[max(i-1,0)]

Nmax = max(Nt)
print(f"  N_max = {Nmax:.1f}")

# Find N=55 marker
if Nmax > 55:
    i55 = np.argmin(np.abs(Nt - 55))
else:
    i55 = np.argmax(Nt)
    print(f"  WARNING: N_max < 55, using N={Nt[i55]:.1f}")

y55 = yt[i55]; u55 = ut[i55]; e55 = et[i55]; c55 = ct[i55]; n55 = nt_arr[i55]
r55 = 16*e55*c55; f55 = fNL_eq(u55)
print(f"  At N={Nt[i55]:.1f}: y={y55:.4f} u={u55:.4f} eps={e55:.4e} cs={c55:.4f} "
      f"ns={n55:.5f} r={r55:.4e} fNL={f55:.2f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

axes[0,0].plot(Nt, yt, 'b-', lw=2)
if Nmax > 55: axes[0,0].axvline(55, color='gray', ls=':', alpha=0.5)
axes[0,0].set_xlabel(r'$N$'); axes[0,0].set_ylabel(r'$y=\beta\phi$')
axes[0,0].set_title('(a) Field evolution', fontsize=12)

axes[0,1].semilogy(Nt, ut, 'b-', lw=2)
axes[0,1].axhline(1, color='orange', ls='--', alpha=0.4)
if Nmax > 55: axes[0,1].axvline(55, color='gray', ls=':', alpha=0.5)
axes[0,1].set_xlabel(r'$N$'); axes[0,1].set_ylabel(r'$u$')
axes[0,1].set_title('(b) Control variable', fontsize=12)

axes[1,0].plot(Nt, ct, 'b-', lw=2)
axes[1,0].axhline(1/np.sqrt(3), color='red', ls='--', alpha=0.5)
if Nmax > 55: axes[1,0].axvline(55, color='gray', ls=':', alpha=0.5)
axes[1,0].set_xlabel(r'$N$'); axes[1,0].set_ylabel(r'$c_s$')
axes[1,0].set_title('(c) Sound speed', fontsize=12); axes[1,0].set_ylim(0.55, 1.0)

axes[1,1].semilogy(Nt, et, 'b-', lw=2)
axes[1,1].axhline(1, color='red', ls='--', alpha=0.5)
if Nmax > 55: axes[1,1].axvline(55, color='gray', ls=':', alpha=0.5)
axes[1,1].set_xlabel(r'$N$'); axes[1,1].set_ylabel(r'$\epsilon$')
axes[1,1].set_title('(d) Slow-roll parameter', fontsize=12)

for ax in axes.flat: ax.set_xlim(0, min(Nmax*1.05, 120))
fig.suptitle(rf'Inflationary trajectory ($\mathcal{{C}}={C_ex}$, $M_{{\rm Pl}}^2\beta^2={Mpb2_ex}$)',
             fontsize=14, y=1.01)
fig.tight_layout(); fig.savefig('/home/claude/fig5_trajectory.pdf'); plt.close()


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK TABLES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*85)
print("TABLE: Structure functions")
print("="*85)
print(f"{'y':>6} {'U(y)':>12} {'F(y)':>14} {'L(y)':>12} {'Phi(y)':>12}")
for yv in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    print(f"{yv:6.1f} {U(yv):12.6f} {F_func(yv):14.6e} {L_func(yv):12.6f} {Phi(yv):12.6f}")

print("\n" + "="*85)
print("TABLE: Observables scan (y*=1.1, Mpb2=0.01)")
print("="*85)
print(f"{'C':>10} {'u*':>10} {'eps*':>12} {'cs*':>8} {'ns':>10} {'r':>12} {'fNL':>8}")
for Cv in [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]:
    uu = solve_master(Cv*F_func(1.1))
    e = eps(1.1, uu, 0.01); c = np.sqrt(cs2(uu))
    n = ns_obs(1.1, uu, Cv, 0.01); rv = 16*e*c
    print(f"{Cv:10.2f} {uu:10.4f} {e:12.4e} {c:8.4f} {n:10.5f} {rv:12.4e} {fNL_eq(uu):8.2f}")

# Strong-coupling cutoff check
print("\n" + "="*85)
print("STRONG-COUPLING CUTOFF Lambda_sc / H")
print("="*85)
P_R = 2.1e-9
for uu_test in [0.1, 0.5, 1.0, 5.0, 10.0]:
    c_s = np.sqrt(cs2(uu_test))
    for eps_test in [0.001, 0.005, 0.01]:
        ratio = 2.0 / (4*np.pi * 1.0 * eps_test * P_R**0.5) / ((1+2*uu_test)*(1+6*uu_test))**0.25
        print(f"  u={uu_test:5.1f}, eps={eps_test:.3f}: Lambda_sc/H ~ {ratio:.2e}")

print("\nDone. Figures in /home/claude/fig*.pdf")
