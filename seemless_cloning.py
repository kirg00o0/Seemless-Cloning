import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg 

import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.transform import resize

### Teil 1: Laplace-Operator

def laplacian_sparse(N, M):
    """Laplace-Operator auf N×M-Gitter als Sparse-Matrix."""
    D2_N = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N), format="csr")
    D2_M = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(M, M), format="csr")
    I_N = sp.eye(N, format="csr")
    I_M = sp.eye(M, format="csr")
    # Kronecker-Summe (5-Punkt-Laplace)
    return sp.kron(I_M, D2_N, format="csr") + sp.kron(D2_M, I_N, format="csr")

# Hilfsfunktionen: Skalen / Konvertierung (brauchen wir später)
def _is_01(arr: np.ndarray):
    return np.nanmax(arr) <= 1.5

def _to255(arr: np.ndarray):
    arr = arr.astype(np.float64, copy=False)
    return arr * 255.0 if _is_01(arr) else arr

def _from255(arr255: np.ndarray, like: np.ndarray):
    if _is_01(like):
        return np.clip(arr255 / 255.0, 0.0, 1.0)
    return np.rint(np.clip(arr255, 0.0, 255.0)).astype(np.uint8)

### Teil 2: Poisson-Cloning (Grau & RGB)

def build_system_for_patch(g_patch: np.ndarray, fstar_patch: np.ndarray):
    """Erzeuge lineares System für Poisson-Cloning eines Graupatches."""
    H, W = g_patch.shape
    if H < 3 or W < 3:
        raise ValueError("Patch muss mind. 3×3 sein.")
    inner = [(r, c) for r in range(1, H-1) for c in range(1, W-1)]
    idx_of = {rc: k for k, rc in enumerate(inner)}
    A = lil_matrix((len(inner), len(inner)), dtype=np.float64)
    b = np.zeros(len(inner), dtype=np.float64)

    def sum_n(a, r, c):
        return a[r-1, c] + a[r+1, c] + a[r, c-1] + a[r, c+1]

    for k, (r, c) in enumerate(inner):
        A[k, k] = 4.0
        for rr, cc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if 1 <= rr <= H-2 and 1 <= cc <= W-2:
                A[k, idx_of[(rr, cc)]] = -1.0
        # rechte Seite: Delta_g + Dirichlet-Randbeitrag aus f*
        b[k] = 4.0 * g_patch[r, c] - sum_n(g_patch, r, c)
        if r-1 == 0:   b[k] += fstar_patch[r-1, c]
        if r+1 == H-1: b[k] += fstar_patch[r+1, c]
        if c-1 == 0:   b[k] += fstar_patch[r, c-1]
        if c+1 == W-1: b[k] += fstar_patch[r, c+1]

    def put_solution(x: np.ndarray) -> np.ndarray:
        out = fstar_patch.copy()
        for k2, (ri, ci) in enumerate(inner):
            out[ri, ci] = x[k2]
        return out

    return csr_matrix(A), b, put_solution

def poisson_clone_gray(f_star: np.ndarray, g: np.ndarray, top_left: tuple[int, int], rtol= 1e-6, maxiter = 50_000):
    """Poisson-Cloning eines Graubild-Patches."""
    f_star_255, g_255 = _to255(f_star), _to255(g)
    Hf, Wf = f_star_255.shape
    Hg, Wg = g_255.shape
    r0, c0 = top_left
    if not (0 <= r0 and 0 <= c0 and r0 + Hg <= Hf and c0 + Wg <= Wf):
        raise ValueError("Patch liegt außerhalb des Zielbildes.")
    fpatch = f_star_255[r0:r0+Hg, c0:c0+Wg].copy()
    A, b, put_solution = build_system_for_patch(g_255, fpatch)
    x, info = cg(A, b, maxiter=maxiter, rtol=rtol, atol=0.0)
    if info != 0:
        print(f"Warnung: CG (Poisson) info={info}")
    solved = put_solution(x)
    out = f_star_255.copy()
    out[r0:r0+Hg, c0:c0+Wg] = solved
    return _from255(out, f_star)

def poisson_clone_rgb(f_star_rgb: np.ndarray, g_rgb: np.ndarray, top_left: tuple[int, int], **kw):
    return np.stack([poisson_clone_gray(f_star_rgb[..., c], g_rgb[..., c], top_left, **kw)
                     for c in range(3)], axis=-1)

### Teil 3: Mixed Gradients (RGB)

NBR4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # rechts, links, unten, oben

def lap_mix_directional(src: np.ndarray, tar: np.ndarray, y: int, x: int,
                         ngb_flags: tuple[bool, bool, bool, bool]):
    """Richtungsweise Summe der größeren Gradientenbeträge (Quelle/Ziel)."""
    r, l, d, u = ngb_flags
    # Quelle
    grs = (src[y, x] - src[y, x+1]) if r else 0.0
    gls = (src[y, x] - src[y, x-1]) if l else 0.0
    gds = (src[y, x] - src[y+1, x]) if d else 0.0
    gus = (src[y, x] - src[y-1, x]) if u else 0.0
    # Ziel
    grt = (tar[y, x] - tar[y, x+1]) if r else 0.0
    glt = (tar[y, x] - tar[y, x-1]) if l else 0.0
    gdt = (tar[y, x] - tar[y+1, x]) if d else 0.0
    gut = (tar[y, x] - tar[y-1, x]) if u else 0.0
    # Richtungweise Auswahl
    v_r = grt if abs(grs) < abs(grt) else grs
    v_l = glt if abs(gls) < abs(glt) else gls
    v_d = gdt if abs(gds) < abs(gdt) else gds
    v_u = gut if abs(gus) < abs(gut) else gus
    return v_r + v_l + v_d + v_u

def build_L_and_index(mask: np.ndarray):
    """Laplacian-Matrix und Index-Mapping für Mixed-Gradients-Methode."""
    H, W = mask.shape
    idx = -np.ones((H, W), dtype=int)
    ys, xs = np.where(mask)
    idx[ys, xs] = np.arange(len(ys))
    K = len(ys)
    A = lil_matrix((K, K), dtype=np.float64)
    for y, x in zip(ys, xs):
        i = idx[y, x]
        deg = 0
        for dy, dx in NBR4:
            yn, xn = y + dy, x + dx
            if 0 <= yn < H and 0 <= xn < W:
                deg += 1
                if mask[yn, xn]:
                    j = idx[yn, xn]
                    A[i, j] = -1.0
        A[i, i] = deg
    return csr_matrix(A), idx

def build_b_mixed(src: np.ndarray, tar: np.ndarray, mask: np.ndarray, idx: np.ndarray):
    """Rechte Seite für Mixed Gradients."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    b = np.zeros((idx >= 0).sum(), dtype=np.float64)
    for y, x in zip(ys, xs):
        i = idx[y, x]
        flags = []
        for dy, dx in NBR4:
            yn, xn = y + dy, x + dx
            flags.append(0 <= yn < H and 0 <= xn < W and mask[yn, xn])
        r, l, d, u = flags
        b_i = lap_mix_directional(src, tar, y, x, (r, l, d, u))
        for (dy, dx), flag in zip(NBR4, (r, l, d, u)):
            yn, xn = y + dy, x + dx
            if 0 <= yn < H and 0 <= xn < W and not flag:  # Nachbar außerhalb der Maske
                b_i += tar[yn, xn]
        b[i] = b_i
    return b

def mixed_clone_rgb(f_star_rgb: np.ndarray, g_rgb: np.ndarray, mask_g: np.ndarray, top_left: tuple[int, int], rtol= 1e-6, maxiter= 100_000):
    """Mixed-Gradients-Cloning eines RGB-Patches."""
    Hf, Wf, _ = f_star_rgb.shape
    Hg, Wg, _ = g_rgb.shape
    r0, c0 = top_left
    assert 0 <= r0 and 0 <= c0 and r0 + Hg <= Hf and c0 + Wg <= Wf, "Patch außerhalb!"

    mask = np.zeros((Hf, Wf), bool)
    mask[r0:r0+Hg, c0:c0+Wg] = (mask_g > 0)

    g_embed = f_star_rgb.copy()
    g_embed[r0:r0+Hg, c0:c0+Wg, :] = np.clip(g_rgb, 0.0, 1.0)

    L, idx = build_L_and_index(mask)
    out = np.array(f_star_rgb, dtype=np.float64)
    ys, xs = np.where(mask)
    for c in range(3):
        b = build_b_mixed(g_embed[..., c], f_star_rgb[..., c], mask, idx)
        x, info = cg(L, b, maxiter=maxiter, rtol=rtol, atol=0.0)
        if info != 0:
            print(f"Warnung: CG (Mixed, Kanal {c}) info={info}")
        for y, x_ in zip(ys, xs):
            out[y, x_, c] = x[idx[y, x_]]
    return np.clip(out, 0.0, 1.0)

def mixed_clone_rgb_rect(f_star_rgb: np.ndarray, g_rgb: np.ndarray, top_left: tuple[int, int], **kw):
    Hg, Wg, _ = g_rgb.shape
    mask_g = np.ones((Hg, Wg), bool)
    return mixed_clone_rgb(f_star_rgb, g_rgb, mask_g, top_left, **kw)

# noch mehr Hilfsfunktionen: Laden/Skalieren
def load_rgb(path: str) -> np.ndarray:
    return img_as_float(io.imread(path))

def fit_source_to_target(g: np.ndarray, f_star: np.ndarray, max_scale= 1.0):
    Hf, Wf = f_star.shape[:2]
    Hg, Wg = g.shape[:2]
    scale = min(Hf / Hg, Wf / Wg, max_scale)
    if scale < 1.0:
        g = resize(g, (int(Hg * scale), int(Wg * scale), g.shape[2]),
                   preserve_range=True, anti_aliasing=True)
    return np.clip(g, 0, 1)

### Teil 4: Demos

def demo_laplacian():
    N, M = 5, 7
    Delta = laplacian_sparse(N, M)
    plt.figure(figsize=(4, 4))
    plt.spy(Delta, markersize=4)
    plt.title("Struktur des Laplace-Operators (N=5, M=7)")
    plt.tight_layout()

def demo_bear_water():
    f_star = load_rgb('images/sky.jpg')
    g_src  = load_rgb('images/bird.jpg')
    g_src  = fit_source_to_target(g_src, f_star)
    Hf, Wf = f_star.shape[:2]
    Hg, Wg = g_src.shape[:2]
    r0, c0 = (Hf - Hg)//2, (Wf - Wg)//2

    out_plain = f_star.copy()
    out_plain[r0:r0+Hg, c0:c0+Wg] = g_src

    out_poisson = poisson_clone_rgb(f_star, g_src, (r0, c0))
    out_mixed   = mixed_clone_rgb_rect(f_star, g_src, (r0, c0))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(out_plain);  plt.axis('off'); plt.title('ohne Cloning')
    plt.subplot(1, 3, 2); plt.imshow(out_poisson);plt.axis('off'); plt.title('Poisson')
    plt.subplot(1, 3, 3); plt.imshow(out_mixed);  plt.axis('off'); plt.title('Mixed')
    plt.tight_layout()

def demo_plane_bird():
    f_star = load_rgb('images/bird.jpg')
    g_src  = load_rgb('images/plane.jpg')
    g_src  = fit_source_to_target(g_src, f_star)

    Hf, Wf = f_star.shape[:2]
    Hg, Wg = g_src.shape[:2]
    r0 = max(0, (Hf - Hg)//2)
    c0 = Wf - Wg - 20  # rechtsbündig

    out_plain = f_star.copy()
    out_plain[r0:r0+Hg, c0:c0+Wg] = g_src

    out_poisson = poisson_clone_rgb(f_star, g_src, (r0, c0))
    out_mixed   = mixed_clone_rgb_rect(f_star, g_src, (r0, c0))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(out_plain);  plt.axis('off'); plt.title('ohne Cloning')
    plt.subplot(1, 3, 2); plt.imshow(out_poisson);plt.axis('off'); plt.title('Poisson')
    plt.subplot(1, 3, 3); plt.imshow(out_mixed);  plt.axis('off'); plt.title('Mixed')
    plt.tight_layout()

def main():
    demo_laplacian()
    demo_bear_water()
    #demo_plane_bird()
    plt.show()

if __name__ == "__main__":
    main()
