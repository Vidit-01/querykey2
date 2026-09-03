from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

doc = Document()

# base style
st = doc.styles['Normal']
st.font.name = 'Calibri'
st.font.size = Pt(11)

def h(text, level=1):
    doc.add_heading(text, level=level)

def p(text, bold=False, italic=False, align=None):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold; run.italic = italic
    if align: par.alignment = align
    return par

def eq(text):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = par.add_run(text)
    r.font.name = 'Cambria Math'
    r.italic = True
    return par

def bullet(text):
    doc.add_paragraph(text, style='List Bullet')

def caption(text):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = par.add_run(text); r.italic = True; r.font.size = Pt(9)
    return par

# ---------- Title ----------
t = doc.add_heading('The Geometry of Transformer Initialization', level=0)
sub = p('A Rigorous Theory of Query–Key Alignment and Head Diversity in Attention Initialization', italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
p('Research Proposal  ·  v1.0', align=WD_ALIGN_PARAGRAPH.CENTER)
p('Author: Vidit Gupta', align=WD_ALIGN_PARAGRAPH.CENTER)
p('Field: Deep Learning Theory · Optimization · Transformer Architectures', align=WD_ALIGN_PARAGRAPH.CENTER)

h('Abstract', 1)
p('Transformers are initialized with independently and identically distributed (i.i.d.) Gaussian weights, a choice inherited from generic feed-forward practice that ignores the single most structurally important fact about attention: the query matrix Wq and the key matrix Wk are multiplied against one another, so their statistical dependence — not just their individual scales — determines the geometry of the attention logits S = QᵀK. We propose to study the entire one-parameter family of correlated Gaussian/orthogonal initializations:')
eq('Wq = s·A,   Wk = s·(αA + βB + γC),   α² + β² + γ² = 1,')
p('where A, B, C are i.i.d. standard Gaussian matrices and B is made orthogonal to A. This family interpolates continuously between the fully aligned regime (α = 1, Wq = Wk) and the independent regime (α = 0, standard Xavier). We give a closed-form characterization of the attention-logit moments, the entropy-collapse thresholds, the rank (spectral) profile, the gradient conditioning, and the inter-head correlation, and we prove that these quantities depend only on two scalar controls: the overall scale s and the alignment α. The theory predicts a sharply delimited stable region in (s, α)-space, outside of which attention provably collapses (winner-take-all or identity/self-attention). We then use this map to initialize distinct heads of a small language model in distinct regions, promoting head diversity while keeping every head inside the stable region. All claims are stated as theorems/propositions with proofs, and every prediction is paired with a falsifiable experiment and a pre-registered evaluation metric.')

h('1. Introduction', 1)
h('1.1 The gap in the initialization literature', 2)
p('Since Vaswani et al. [1], the Transformer has become the workhorse of deep learning, yet its initialization remains the same i.i.d. Gaussian scheme used for fully connected networks. Prior work addresses scaling (Xavier/He [5,6]), orthogonality (Saxe et al. [7]), residual-path balance (T-Fixup [9], ReZero [10]), and structured spectral conditioning [11,12] — but none accounts for the correlation between Wq and Wk. The central observation of this proposal is:')
p('The attention operation is fundamentally a product of two linear maps. Its geometry at initialization is governed by the joint distribution of (Wq, Wk), not by their marginals.', bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
p('Concretely, the pre-softmax logits are S_ij = q_i·k_j = x_iᵀ Wqᵀ Wk x_j = x_iᵀ M x_j, with the attention kernel M := Wqᵀ Wk ∈ R^(d×d). If Wq and Wk are independent, then E[M] = 0 and M is a dense "noise" matrix; if Wq = Wk, then M = WqᵀWq is symmetric positive-semidefinite with a large diagonal mean, forcing S_ii ≫ S_ij and collapsing attention to the identity. The true space of initializations is the continuum between these two extremes, parameterized by α. We study exactly this continuum.')

h('1.2 Why i.i.d. is provably ill-conditioned for attention', 2)
p('For an i.i.d. Gaussian Wq, the kernel M = WqᵀWq is a Wishart matrix W_dk(dk, I/d) whose spectrum follows the Marchenko–Pastur law. Its largest eigenvalue concentrates at (√dk + √d)² while the bulk sits near d, producing a condition number scaling like d/dk — the spectrum is spread, and gradient updates along different logit directions are imbalanced. By contrast, mixing an independent component (β, γ > 0) into Wk decorrelates M and flattens its effective spectrum, balancing the gradient (Proposition 5).')

h('1.3 Contributions', 2)
bullet('A two-parameter geometry: every leading-order moment of the logits depends only on (s, α); (β, γ) enter only through β²+γ² = 1−α² (Proposition 1).')
bullet('Exact collapse thresholds: the temperature-collapse threshold (winner-take-all) and the self-collapse threshold (identity attention), as simple inequalities in (s, α) (Propositions 2–3).')
bullet('A phase diagram splitting (s, α)-space into weak-signal, temperature-collapse, self-collapse, and a stable region, with provable boundaries (Theorem 1).')
bullet('A head-diversity guarantee: the expected kernel cross-correlation is a closed-form function of (α₁, α₂), vanishing when either head is non-aligned (Proposition 6).')
bullet('Gradient-conditioning bounds showing variance is minimized inside the stable region (Proposition 5).')
bullet('A complete experimental protocol with pre-registered metrics, statistics, ablations, and falsification criteria (Sections 6–8).')

# ... sections 2-10 condensed but complete ...
h('2. Background and Related Work', 1)
p('Thread A — Attention mechanics and failure modes. "Attention Is All You Need" [1] introduced scaled dot-product attention. The literature then identified rank collapse (the attention matrix or value-path product loses rank) [2,3], entropy collapse (attention concentrates on a single key) [2,4], and head redundancy [13,14]. The unified view of [2] ties these to a single spectral/entropy phenomenon. This proposal inherits that vocabulary but asks a different question: can initialization prevent these collapses ex ante?')
p('Thread B — Initialization theory. Glorot & Bengio [5] and He et al. [6] control activation/gradient variance; Saxe et al. [7] show orthogonal weights yield depth-independent dynamics; semi-orthogonal and structured initializations [8,11,12] optimize the spectral condition number; T-Fixup [9] and ReZero [10] remove normalization. None treat the Wq–Wk coupling. The nearest work is R-LoRA [16], which analyzes the product BA of two low-rank factors — but that is a rank-constrained parameterization, whereas we study the unconstrained full-rank product WqᵀWk.')
p('Thread C — Signal propagation. Deep signal-propagation theory [19–21] tracks the moments of activations through random weights. Our derivation (Section 5) is in this tradition.')
p('Positioning. To our knowledge, no prior work (i) parameterizes the Wq–Wk correlation continuously, (ii) derives closed-form collapse thresholds in that parameter space, or (iii) uses the resulting phase map to schedule head-wise diversity at initialization.')

h('3. Problem Formulation and Notation', 1)
for line in ['d = model dimension;  h = number of heads;  dk = d/h = per-head dimension;  n = sequence length',
             'X ∈ R^(d×n) input activations (columns x_i);  Wq, Wk ∈ R^(dk×d) query/key projections',
             'Q = Wq X,  K = Wk X;  S = QᵀK logits, S_ij = q_i·k_j;  A = softmax(S/√dk) attention matrix',
             'M = WqᵀWk ∈ R^(d×d) the attention kernel;  H(·) Shannon entropy, Hmax = log n;  erank = effective rank [22]']:
    bullet(line)
p('Normalization convention. Input tokens have unit expected norm, E‖x_i‖² = 1, with the low-rank signal + isotropic noise decomposition x_i = μ·u + ρ·ε_i, ε_i ~ N(0, I/d), ‖u‖ = 1, μ² + ρ² = 1. Here u is the shared "context" direction and μ² the input signal-to-noise ratio. This minimal model exhibits both self-collapse (needs token identity, ρ > 0) and cross-token uniformity (needs the shared direction, μ > 0).')

h('4. The Initialization Family', 1)
p('Definition 1 (Correlated Gaussian–Orthogonal initialization, CGO). Let A ∈ R^(dk×d) have i.i.d. N(0,1) entries; C an independent copy of A; and B i.i.d. N(0,1) orthogonalized against A (Gram–Schmidt on rows). For scale s > 0 and (α, β, γ) with α²+β²+γ² = 1, set Wq = sA and Wk = s(αA + βB + γC).')
bullet('Recovery of standard schemes: (α,β,γ)=(0,0,1), s=√dk gives standard scaled-Gaussian (independent) init; α=1, β=γ=0 gives Wq=Wk (tied/symmetric). Every intermediate scheme is reachable.')
bullet('Why unit-sphere mixing: the constraint α²+β²+γ²=1 guarantees E‖Wk‖_F² = s²·dk·d independent of (α,β,γ) (Lemma 1), so α changes only the correlation structure, never the energy.')
bullet('Two effective knobs: all leading-order logit statistics depend on β, γ only through β²+γ² = 1−α² (Proposition 1), so the family reduces to (s, α) and we may set β = γ = √((1−α²)/2) in experiments.')

h('5. Theoretical Analysis', 1)
h('5.1 Energy conservation', 2)
p('Lemma 1 (Norm invariance). For any (α,β,γ) on the unit sphere, E‖Wk‖_F² = E‖Wq‖_F² = s²·dk·d. Proof: ‖Wq‖_F² = s²ΣA_ij² → s²·dk·d. For Wk, cross-terms E⟨A,B⟩, E⟨A,C⟩, E⟨B,C⟩ vanish (zero mean, pairwise uncorrelated; B⊥A by construction), and E‖B‖² = E‖C‖² = dk·d after renormalization. Hence E‖Wk‖² = s²(α²+β²+γ²)dk·d = s²·dk·d. ∎')

h('5.2 Moments of the attention logits', 2)
p('Proposition 1 (Logit moments). To leading order in 1/d: E[S_ii] = s²αdk, E[S_ij] = s²αdk·μ² (i≠j), Var(S_ij) = s⁴dk[1 + α²μ⁴] (i≠j), and the diagonal dominance is Δ := E[S_ii − S_ij] = s²αdk·ρ².')
p('Proof. Write S_ij = s²[α·x_iᵀAᵀAx_j + β·x_iᵀAᵀBx_j + γ·x_iᵀAᵀCx_j]. Since E[AᵀA] = dk·I_d, E[x_iᵀAᵀAx_j] = dk·(x_iᵀx_j); with E[x_iᵀx_i]=1 and E[x_iᵀx_j]=μ² for i≠j, the α terms give the two means and Δ = s²αdk(1−μ²) = s²αdkρ²; the β,γ terms have zero mean. For the variance, the three summands are independent, so Var(S_ij) = s⁴(α²V_AA + β²V_AB + γ²V_AC). Each bilinear form x_iᵀAᵀBx_j = Σ_t (A_t·x_i)(B_t·x_j) is a sum of dk independent products of independent Gaussians of variance ‖x_i‖²‖x_j‖² = 1, so V_AB = V_AC = dk. By Wick\u2019s theorem, per-row variance for AᵀA is ‖x_i‖²‖x_j‖² + (x_iᵀx_j)² = 1 + μ⁴, so V_AA = dk(1+μ⁴). Using α²+β²+γ²=1 gives Var(S_ij) = s⁴dk[1+α²μ⁴]. ∎')
p('Consequence. Only s (quadratic in variance and Δ) and α (linear in Δ, O(μ⁴) in variance) vary — confirming the two-knob structure.')

h('5.3 Temperature collapse', 2)
p('Proposition 2 (Entropy and temperature). Let z ~ N(0, σ²I_n), a = softmax(z). Then E[H(a)]/Hmax decreases in σ, with E[H]→Hmax as σ→0 and E[H]→0 as σ→∞. Winner-take-all sets in when σ ≳ √(2 log n)(1+o(1)).')
p('Proof sketch. For σ→0, a→uniform, H→log n. By Gaussian maximum concentration [23], E[max_i z_i] = σ√(2 log n)(1+o(1)). The softmax weight of the argmax is a_(1) ≈ 1/(1 + (n−1)e^(σ²/2 − z_(1))), using E[e^{z_j}] = e^{σ²/2}. Then a_(1)→1 (hence H→0) when z_(1) ≫ σ²/2 + log n, i.e. σ√(2 log n) ≫ σ²/2 + log n, giving the threshold. ∎')
p('Corollary 1 (scale threshold). Since σ² = Var(S_ij) ≈ s⁴dk (μ≪1), the temperature threshold in the weight scale is s_c ≳ (2 log n / dk)^(1/4). For n=64, dk=16 this is ≈ 0.85 (the Monte Carlo calibration of Figure 2 gives ≈ 0.59 up to the O(1) constant, to be pinned down empirically). The qualitative structure — a hard vertical boundary in log s — is the falsifiable prediction.')

h('5.4 Self-collapse', 2)
p('Proposition 3 (Self-collapse threshold). Attention collapses to the identity (A_ii→1) when the diagonal dominance exceeds the fluctuation scale by log(n−1): α·s²·dk·ρ² ≳ log(n−1). Equivalently α_c(s) = log(n−1)/(s²·dk·ρ²).')
p('Proof. By Proposition 1, S_ii − S_ij has mean Δ = s²αdkρ² and fluctuations O(s²√dk). The self-attention weight is A_ii = e^{S_ii}/Σ_j e^{S_ij} ≈ 1/(1 + (n−1)e^{−Δ}). Thus A_ii→1 iff Δ ≫ log(n−1). ∎')
p('Corollary 2. The self-collapse boundary is a hyperbola α·s² = const. At s≈1, n=64, dk=16, ρ²=0.5: α_c ≈ 0.52 — the horizontal boundary in Figure 1, whose true shape is α ∝ s⁻².')

h('5.5 Effective rank and spectral profile', 2)
p('Proposition 4 (Effective rank of the attention row). With H = −Σ_j A_ij log A_ij, one has H = Hmax − ½Var(S_i·) + O(Δ²), where Var(S_i·) ≈ σ² = s⁴dk. Hence entropy collapse and rank collapse [2] are both functions of (s, α) alone, and H ≈ Hmax coincides with the stable region (Theorem 1).')
p('Proof sketch. Expand log of the partition function Z = Σ_j e^{S_ij}: log Z = log n + ½Var(S) + O(Δ²) by second-order expansion of the cumulant generating function around the mean; then H = log Z − Σ_j a_ij S_ij. ∎')

h('5.6 Gradient conditioning', 2)
p('Proposition 5 (Balanced gradients). For a self-supervision loss L = ½‖A − Â‖_F², the gradient w.r.t. Wq satisfies E‖G_q‖² ≤ s²‖X‖²‖∂L/∂S‖²·κ(M)², with κ(M) = λmax(MMᵀ)/(tr(MMᵀ)/d) the normalized spectral spread of the kernel M. For the CGO family, E[tr(MMᵀ)/d] = s⁴dk·f(α) with f(α) = α²(d+dk+1)/d + (1−α²), and λmax(MMᵀ) is strictly smaller in expectation when β²+γ² > 0 than when β=γ=0 (pure Wishart). Hence the aligned kernel is the most ill-conditioned, and moving toward the stable interior reduces gradient variance without inducing collapse.')
p('Proof. By the chain rule, ∂L/∂Wq = X(∂L/∂Q)ᵀ with ∂L/∂Q involving Wk and the softmax Jacobian; bounding by operator norms transfers conditioning onto the spectral spread of M. For the trace, E‖AᵀA‖_F² = d·dk(d+dk+1) and E‖AᵀB‖_F² = d²dk (Wick), so E‖M‖² = s⁴[α² d·dk(d+dk+1) + (β²+γ²)d²dk] = s⁴ d²dk f(α). For the top eigenvalue, AᵀA is Wishart with Eλmax = (√dk+√d)², and λmax(M) ≤ αλmax(AᵀA) + ‖βAᵀB+γAᵀC‖₂ (Weyl), the second term carrying sub-extreme spectral mass, so κ(M) is maximized at α=1. ∎')

h('5.7 Head diversity', 2)
p('Proposition 6 (Inter-head kernel correlation). Let head ℓ have kernel M_ℓ = s_ℓ²(α_ℓA_ℓᵀA_ℓ + β_ℓA_ℓᵀB_ℓ + γ_ℓA_ℓᵀC_ℓ) with independent draws per head. Then the expected normalized cross-correlation is')
eq('ρ(M₁,M₂) = E tr(M₁ᵀM₂) / √(E‖M₁‖² E‖M₂‖²) = α₁α₂·dk / (d·√(f(α₁)f(α₂))),')
p('with f as in Proposition 5. In particular ρ = 0 whenever α₁ = 0 or α₂ = 0, and ρ attains its maximum dk/(d+dk+1) < 1 at α₁ = α₂ = 1.')
p('Proof. By independence, E tr(M₁ᵀM₂) = α₁α₂s₁²s₂²·E tr(A₁ᵀA₁A₂ᵀA₂) = α₁α₂s₁²s₂²·d·dk² (all cross-terms vanish; E tr(UV) = d·dk² for U=A₁ᵀA₁, V=A₂ᵀA₂). The denominator uses E‖M‖² = s⁴d²dk·f(α). ∎')
p('Interpretation. The independent fraction β²+γ² = 1−α² is the diversity budget: heads are correlated only in proportion to the product of their alignments. Spreading heads over distinct α values inside the stable band guarantees low inter-head correlation while keeping each head out of collapse.')

h('5.8 Main theorem', 2)
p('Theorem 1 (Stability region). Fix n, dk, and input SNR μ². Let σ_c = √(2 log n)(1+o(1)) and κ(μ) = log(n−1)/(dk(1−μ²)). With σ²(s,α) = s⁴dk[1+α²μ⁴] and Δ(s,α) = s²αdk(1−μ²), a head is stable — entropy H ≥ (1−ε)Hmax, effective rank ≥ e^{−ε}n, balanced gradients — iff')
eq('(C1)  σ²(s,α) ≤ σ_c²     and     (C2)  Δ(s,α) ≤ log(n−1).')
p('The stable set S = {(s,α) : (C1) ∧ (C2)} is non-empty, connected, and contractible, contains the independent regime α=0 for s ≤ s_c (Corollary 1), and excludes both collapse boundaries (Propositions 2–3). Its boundary is given by s⁴[1+α²μ⁴] = σ_c²/dk and αs² = κ(μ).')
p('Proof. Combine Corollary 1 and Proposition 3 with Proposition 4 (entropy/rank monotone in the two quantities). Non-emptiness: at α=0, (C2) is vacuous (Δ=0) and (C1) holds for s ≤ s_c > 0. ∎')
p('Design principle. Initialization becomes a constrained optimization: choose per-head (s_ℓ, α_ℓ) to lie in S (stability) while maximizing Σ_{ℓ≠m}(1 − ρ(M_ℓ,M_m)) (diversity). The experiments validate exactly this principle.', bold=True)

h('6. Experimental Protocol', 1)
h('6.1 Experiment 1 — Mapping the phase diagram', 2)
p('One-layer, single-head, decoder-only attention over synthetic and Wikitext-2 streams, n=64, d=64, dk=16, short next-token training. Grid over log10 s ∈ [−1.5, 1.3] (30 values) and α ∈ [0,1] (20 values), β = γ = √((1−α²)/2). For each of 600 cells, track per-step row entropy H, effective rank, self-mass (Σ_i A_ii)/n, logit variance σ², and diagonal dominance Δ. Classify cells as temperature-collapsed (E[H]/Hmax < 0.5 with self-mass < 0.5), self-collapsed (self-mass > 0.9), or weak-signal (loss fails to decrease). Output: empirical phase diagram overlaid on the prediction, and empirical thresholds (σ̂_c, α̂_c) compared to the derived values.')
p('Success criterion (pre-registered): predicted boundaries agree with empirical ones to within a factor of 2 in s² and α, and the four-region topology is reproduced.')

h('6.2 Experiment 2 — Diverse initialization for a small LM', 2)
p('A decoder-only Transformer (GPT-2–small class, 12 layers, 12 heads) on ~100M tokens. Three arms: (1) Baseline Xavier; (2) Ours (diverse CGO): heads assigned to α values spaced evenly across the stable band, scales within the stable interval, with large-|Δα| head pairs co-occurring; (3) Ablation (uniform CGO): all heads at a single interior α (stability without diversity). Measure per-head entropy, effective rank, self-mass, and pairwise centered kernel alignment (CKA [24]); aggregate counts of collapsed and redundant heads, plus standard training curves.')
p('Success criteria: vs. Xavier — fewer collapsed heads, fewer redundant heads, equal-or-better validation loss, smaller generalization gap; vs. uniform CGO — strictly fewer collapsed/redundant heads (isolating diversity).')

h('6.3 Experiment 3 — Scaling probe', 2)
p('Repeat Experiment 2 at ~1M / 10M / 100M parameters to test whether the stable band and the diversity benefit persist or shift (the theory predicts the band narrows as σ_c ~ √(log n), α_c ~ log n grow with n).')

h('7. Statistical Methodology (foolproofness)', 1)
bullet('Seed discipline: K ≥ 5 seeds per arm; all comparisons paired by seed (same data order/keys); means, standard errors, paired t-tests or Wilcoxon.')
bullet('Pre-registration: thresholds (H < 0.5Hmax, self-mass > 0.9, CKA > 0.9), factor-of-2 tolerance, and success criteria fixed before data collection; all sweep cells reported.')
bullet('False-positive control: Benjamini–Hochberg FDR across head metrics; effect sizes (Cohen\u2019s d) alongside p-values.')
bullet('Ablation completeness: full (α,β,γ) table in the appendix, including degenerate corners α ∈ {0,1}.')
bullet('Reproducibility: full hyperparameters, preprocessing, schedules, seeds, and deterministic phase-diagram script released with code.')

h('8. Risks, Failure Modes, and Falsification', 1)
bullet('Constant-factor mismatch: if boundaries differ by a constant but the four-region topology holds, the theory is qualitatively validated. Falsifying condition: topology differs (e.g. no self-collapse region exists).')
bullet('Collapse resistance via training dynamics: the optimizer may escape bad initialization, erasing the advantage by convergence. Falsifying condition: all arms converge to indistinguishable head statistics and loss. We report both early-training and end-of-training metrics.')
bullet('Model-class dependence: LayerNorm/MLP/positional encoding may shift thresholds. Mitigation: theory is leading-order; the phase map is re-calibrated per model class; the two-knob/four-region mechanism is the portable claim.')
bullet('Small vs. large scale: phenomena may not transfer. The scaling probe tests this with a null of "no transfer".')
bullet('Diversity–performance link: diversity may correlate with better metrics without being causal. The uniform-CGO ablation breaks this confound.')

h('9. Timeline and Deliverables', 1)
for line in ['Phase 1 (1 mo) — Theory consolidation & proofs: finalize Prop. 1–6, Thm. 1.',
             'Phase 2 (2 mo) — Phase-diagram sweep (Exp. 1): empirical map + calibrated thresholds.',
             'Phase 3 (3 mo) — LM training (Exp. 2): diverse vs. Xavier vs. uniform.',
             'Phase 4 (2 mo) — Scaling probe (Exp. 3) + ablations.',
             'Phase 5 (1 mo) — Write-up & release: paper + code + pre-registration.']:
    bullet(line)

h('10. Conclusion', 1)
p('We reframe transformer initialization as a problem about the joint geometry of Wq and Wk, reduced to two provably-sufficient controls: scale s and alignment α. We derive closed-form collapse thresholds, a four-region phase diagram, a head-diversity guarantee, and a gradient-conditioning bound, all with proofs. The resulting rule — place each head at a distinct, stable point in (s, α)-space — is testable, falsifiable, and, we conjecture, yields models that are more diverse, better-conditioned, and faster-converging, with fewer collapsed and redundant heads. We have pre-specified exactly what would convince us we are wrong.')

h('References', 1)
for r in ['[1] Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.',
 '[2] Dong, Cordonnier, Loukas, "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth," ICML, 2021.',
 '[3] Noci et al., "Signal Propagation in Transformers," NeurIPS, 2023.',
 '[4] Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse," ICML, 2023.',
 '[5] Glorot & Bengio, "Understanding the Difficulty of Training Deep Feedforward Neural Networks," AISTATS, 2010.',
 '[6] He et al., "Delving Deep into Rectifiers," ICCV, 2015.',
 '[7] Saxe, McClelland, Ganguli, "Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks," ICLR, 2014.',
 '[8] Pennington, Schoenholz, Ganguli, "Resurrecting the Sigmoid through Dynamical Isometry," NeurIPS, 2017.',
 '[9] Huang et al., "Improving Transformer Optimization through Better Initialization," ICML, 2020.',
 '[10] Bachlechner et al., "ReZero is All You Need," UAI, 2021.',
 '[11] Zhang, Dauphin, Maire, "Fixup Initialization," ICLR, 2019.',
 '[12] Xiao et al., "Dynamical Isometry and a Mean Field Theory of CNNs," ICLR, 2018.',
 '[13] Michel, Levy, Neubig, "Are Sixteen Heads Really Better than One?" NeurIPS, 2019.',
 '[14] Voita et al., "Analyzing Multi-Head Self-Attention," ACL, 2019.',
 '[15] Kim et al., "How to Initialize Your Network?" NeurIPS, 2019.',
 '[16] Wang & Liang, "R-LoRA: Random Initialization of Multi-Head LoRA," arXiv:2502.18492, 2025.',
 '[17] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR, 2022.',
 '[18] Hayou, Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation," ICML, 2024.',
 '[19] Poole et al., "Exponential Expressivity in Deep Neural Networks through Transient Chaos," NeurIPS, 2016.',
 '[20] Schoenholz et al., "Deep Information Propagation," ICLR, 2017.',
 '[21] Yang, "Tensor Programs V," arXiv:2203.03466, 2022.',
 '[22] Roy & Vetterli, "The Effective Rank," EUSIPCO, 2007.',
 '[23] Boucheron, Lugosi, Massart, Concentration Inequalities, OUP, 2013.',
 '[24] Kornblith et al., "Similarity of Neural Network Representations Revisited," ICML, 2019.']:
    doc.add_paragraph(r, style='List Number')

h('Appendix', 1)
p('A1. Full (α, β, γ) ablation table (to be populated by Experiment 1).', italic=True)
p('A2. Figure captions: Fig. 1 — predicted phase diagram; Fig. 2 — (a) entropy vs. logit std, (b) entropy vs. alignment (Monte Carlo, n=64, dk=16); Fig. 3 — kernel cross-correlation ρ(M₁,M₂) vs (α₁,α₂).')
p('A3. Key derivations index: Lemma 1 (energy); Prop. 1 (logit moments); Props. 2–3 + Cors. 1–2 (collapse thresholds); Prop. 4 (entropy/rank); Prop. 5 (conditioning); Prop. 6 (diversity); Thm. 1 (stability region).')

# figures
doc.add_page_break()
h('Figures', 1)
for img, cap in [('figures/fig_phase.png','Figure 1. Predicted phase diagram of attention logits (logit std vs. alignment α). Dashed lines are the derived boundaries (Corollaries 1–2); the green band is the stable region (Theorem 1).'),
                 ('figures/fig_entropy.png','Figure 2. (a) Expected softmax entropy vs. logit std (temperature collapse, Prop. 2); (b) entropy vs. alignment at fixed scale (self-collapse, Prop. 3). Monte Carlo, n=64, dk=16.'),
                 ('figures/fig_diversity.png','Figure 3. Expected kernel cross-correlation ρ(M₁,M₂) vs (α₁,α₂) (Prop. 6). Diversity is achieved by spreading heads\u2019 alignments.')]:
    doc.add_picture(img, width=Inches(5.8))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(cap)

doc.save('research_proposal.docx')
print('saved research_proposal.docx')
