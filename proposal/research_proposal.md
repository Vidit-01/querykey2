# The Geometry of Transformer Initialization
## Exact Query-Key Moment Identities and a Falsifiable Initialization Atlas

**Research Proposal** · v2.0 (mathematical audit revision)  
**Author:** Vidit Gupta  
**Field:** Deep Learning Theory · Optimization · Transformer Architectures
**Status:** Proposal - exact finite-width identities + preregistered experiments

---

## Abstract

The query and key projections of a Transformer enter attention through their product, so their **joint initialization** matters in addition to their marginal variances. We propose a controlled family

$$
W_q=sA,\qquad W_k=s(\alpha A+\beta B+\gamma C),\qquad
\alpha^2+\beta^2+\gamma^2=1,
$$

where $A$ and $C$ are fan-in-scaled Gaussian matrices and $B$ is a random matrix whose row space is orthogonal to the row space of $A$. The scale convention is $\operatorname{Var}(A_{ab})=\operatorname{Var}(C_{ab})=1/d$, so $s=1$ is the fan-in Gaussian reference, not $s=\sqrt{d_k}$. The standard scaled dot-product logits are

$$
L=\frac{Q^\top K}{\sqrt{d_k}}
  =\frac{X^\top W_q^\top W_kX}{\sqrt{d_k}}.
$$

For fixed inputs, we derive exact finite-width expressions for the mean and variance of every logit, including the finite-$d$ difference between the orthogonal component $B$ and the independent Gaussian component $C$. We also give exact energy identities, exact back-propagation identities, and an exact cross-head common-bias statistic. We do **not** claim that these moment identities imply a sharp trainability theorem. Softmax scores are correlated, row entropy is not matrix rank, and initialization quality depends on residual paths, normalization, depth, data, and optimization. Accordingly, the main empirical contribution is a preregistered atlas over the constrained four-tuple $(s,\alpha,\beta,\gamma)$, followed by a controlled real-language-model test. The atlas first verifies the mathematics, then maps diffuse attention, low-entropy concentration, self-locking, gradient starvation, and empirically trainable regions. A real-model study compares standard independent initialization, a scale-only control, a homogeneous structured initialization, and a within-layer diverse portfolio while matching marginal weight variance and all non-query/key parameters.

---

## 1. Motivation and Positioning

### 1.1 Why joint query-key initialization is a real degree of freedom

For a head with model width $d$, head width $m=d_k$, and token matrix $X\in\mathbb R^{d\times n}$,

$$
Q=W_qX,\qquad K=W_kX,\qquad
L=\frac{Q^\top K}{\sqrt m}
  =\frac{X^\top M X}{\sqrt m},\qquad M:=W_q^\top W_k.
$$

Two initializations can give $W_q$ and $W_k$ identical marginal variances while producing different distributions for $M$. This changes the mean diagonal preference, logit fluctuations, spectral concentration, and the softmax operating point. The correct object of study is therefore the joint law of $(W_q,W_k)$.

This direction is related to, but distinct from, several established lines of work. T-Fixup and residual-scaling methods control signal and gradient propagation across depth [9-11]. Mimetic initialization explicitly initializes the query-key product to resemble structures observed in pretrained models [12]. Bao et al. connect attention localization to the query-key eigenspectrum [13]. Giorlandino and Goldt derive trainability diagrams for deep Transformers by jointly considering attention scale and residual strength [14]. Zhai et al. connect pathologically low attention entropy to training instability [4]. These works mean that this proposal must not claim to be the first to study $W_q^\top W_k$ or initialization phase diagrams.

### 1.2 The narrower contribution

The proposed contribution is:

1. a reproducible, energy-matched interpolation among a shared query-key component, a row-space-orthogonal component, and an independent Gaussian component;
2. exact finite-width logit moment formulas for that interpolation;
3. an empirical atlas over all admissible $(s,\alpha,\beta,\gamma)$, including negative $\alpha$ and finite-width effects that distinguish $\beta$ from $\gamma$;
4. a causal real-model comparison that separates scale, structure, and within-layer head heterogeneity.

### 1.3 Claim discipline

To prevent theory from outrunning the evidence, every statement is assigned one of three statuses.

- **Exact:** proved under the stated finite-dimensional random-matrix model.
- **Asymptotic or surrogate:** valid only under an explicit approximation such as nearly isotropic token Gram matrices or i.i.d. Gaussian score surrogates.
- **Empirical hypothesis:** tested by the atlas or language-model experiment; not presented as a theorem.

The exact results concern energy, first and second moments, matrix rank facts, gradients, and a cross-head mean-bias statistic. “Good initialization zones,” trainability, and benefits from head diversity remain empirical hypotheses.

---

## 2. Notation and Failure Modes

| Symbol | Meaning |
|---|---|
| $d$ | model width |
| $m=d_k$ | width of one attention head |
| $n$ | sequence length |
| $X=[x_1,\ldots,x_n]\in\mathbb R^{d\times n}$ | input tokens |
| $W_q,W_k\in\mathbb R^{m\times d}$ | query and key projections |
| $Q=W_qX,\ K=W_kX$ | query and key matrices |
| $L=Q^\top K/\sqrt m$ | scaled logits |
| $P=\operatorname{softmax}_{\rm row}(L)$ | attention matrix |
| $M=W_q^\top W_k$ | query-key kernel |
| $v_i=\|x_i\|_2^2/d$ | normalized token energy |
| $c_{ij}=x_i^\top x_j/d$ | normalized token similarity |
| $H(p)=-\sum_jp_j\log p_j$ | row entropy |
| $N_{\rm eff}(p)=e^{H(p)}$ | effective number of attended keys |

Three different phenomena must not be conflated:

1. **Diffuse or near-uniform attention:** on each row, probability is nearly uniform over the unmasked keys. In unmasked attention, $P\approx\mathbf1\mathbf1^\top/n$ has matrix rank one. Causal prefix-uniform attention is lower triangular and full rank, although repeated prefix averaging can still homogenize representations. Row entropy alone therefore does not determine matrix rank or representation rank collapse [2,3,14].
2. **Low-entropy concentration:** one or a few keys dominate a row. This is the entropy-collapse failure mode studied by Zhai et al. [4].
3. **Self-locking:** $P_{ii}$ is large because the diagonal logit has a systematic advantage. $P\approx I$ is full rank, so self-locking is not matrix rank collapse.

The row effective support $N_{\rm eff}$ and the singular-value effective rank of $P$ are different statistics and will be reported separately.

For a matrix $R$ with singular values $\sigma_r$ and $p_r=\sigma_r/\sum_j\sigma_j$, define

$$
\operatorname{erank}(R)=\exp\!\left(-\sum_r p_r\log p_r\right),\qquad
\operatorname{srank}(R)=\frac{\|R\|_F^2}{\|R\|_2^2}.
$$

---

## 3. A Well-Defined Correlated Gaussian-Orthogonal Family

### 3.1 Base matrices

Assume $d>m$. Draw

$$
A_{ab},C_{ab}\sim\mathcal N(0,1/d),
\qquad G_{ab}\sim\mathcal N(0,1/(d-m)),
$$

with entries i.i.d. within each matrix and with $A,C,G$ mutually independent. Since $A$ has full row rank almost surely, define the projector onto its row space and its orthogonal complement:

$$
\Pi_A=A^\top(AA^\top)^{-1}A,\qquad
\Pi_A^\perp=I_d-\Pi_A,\qquad B=G\Pi_A^\perp.
$$

Then

$$
AB^\top=0
$$

exactly. The earlier expression $A(A^\top A)^{-1}A^\top B$ is invalid when $m<d$ because $A^\top A$ is singular. In implementation, no explicit inverse or $d\times d$ projector is needed: obtain an orthonormal basis $U$ for the columns of $A^\top$ by QR and compute $B=G-(GU)U^\top$.

If $d\ge2m$, $B$ can also have full row rank almost surely. The theory below only requires $d>m$.

### 3.2 Query-key family and coefficient domain

Define

$$
W_q=sA,\qquad W_k=s(\alpha A+\beta B+\gamma C),
\qquad s>0,\quad \alpha^2+\beta^2+\gamma^2=1.
$$

The four reported values are constrained; they do not form an unconstrained four-dimensional Cartesian box. Because $B$ and $C$ are symmetric about zero, the signs of $\beta$ and $\gamma$ are distributionally redundant for seed-averaged initialization statistics. The main atlas therefore uses

$$
\alpha\in[-1,1],\qquad
\phi\in[0,\pi/2],\qquad
\beta=\sqrt{1-\alpha^2}\cos\phi,\qquad
\gamma=\sqrt{1-\alpha^2}\sin\phi.
$$

Thus the admissible atlas has three independent coordinates $(s,\alpha,\phi)$ while retaining all four coefficients in every record. The sign of $\alpha$ is not redundant: positive $\alpha$ induces self-bias, while negative $\alpha$ induces anti-self bias.

Important corners are:

- **independent Gaussian:** $(\alpha,\beta,\gamma)=(0,0,1)$;
- **row-space orthogonal:** $(0,1,0)$;
- **tied:** $(1,0,0)$;
- **anti-tied:** $(-1,0,0)$.

Under this convention, $s=1$ is the fan-in Gaussian reference. For an implementation whose baseline entry standard deviation is $\sigma_0$, its equivalent scale is $s_0=\sigma_0\sqrt d$.

For identification, the $\beta=0$ slice is reported separately as the clean correlated-Gaussian family. Varying $\phi$ introduces the row-space-orthogonal mechanism and is analyzed as a structural ablation; results from the two mechanisms are never pooled into a claimed two-knob law.

---

## 4. Exact Finite-Width Results

### 4.1 Energy and marginal scale

**Lemma 1 (energy matching; exact).** For every admissible $(\alpha,\beta,\gamma)$,

$$
\mathbb E\|W_q\|_F^2=\mathbb E\|W_k\|_F^2=s^2m,
\qquad
\mathbb E[(W_k)_{ab}^2]=\frac{s^2}{d}.
$$

Moreover,

$$
\mathbb E[W_q^\top W_k]=s^2\alpha\,\frac md I_d,
\qquad
\mathbb E[W_qW_k^\top]=s^2\alpha I_m.
$$

*Proof.* Each row of $A$ and $C$ has expected squared norm one. Conditional on $A$,

$$
\mathbb E_G[BB^\top\mid A]
=\frac{\operatorname{tr}(\Pi_A^\perp)}{d-m}I_m=I_m.
$$

Rotational invariance gives $\mathbb E[(\Pi_A^\perp)_{jj}]=(d-m)/d$, hence $\mathbb E[B_{ab}^2]=1/d$. The $A$-$B$ cross term is zero exactly because $AB^\top=0$; cross terms involving $C$ vanish in expectation. The remaining identities follow from $\mathbb E[A^\top A]=(m/d)I_d$ and $\mathbb E[AA^\top]=I_m$. $\square$

This lemma is the fairness constraint for all comparisons: changing $(\alpha,\beta,\gamma)$ does not change the expected marginal weight energy.

### 4.2 Mean and variance of scaled logits

For fixed inputs, define

$$
L_{ij}=\frac{x_i^\top W_q^\top W_kx_j}{\sqrt m},
\qquad v_i=\frac{\|x_i\|^2}{d},\qquad c_{ij}=\frac{x_i^\top x_j}{d}.
$$

**Proposition 1 (conditional logit moments; exact).**

$$
\mathbb E[L_{ij}\mid X]=\alpha s^2\sqrt m\,c_{ij},
$$

and

$$
\operatorname{Var}(L_{ij}\mid X)
=s^4\!\left[
\alpha^2(v_iv_j+c_{ij}^2)
+\beta^2 g_{ij}
+\gamma^2v_iv_j
\right],
$$

where

$$
g_{ij}
=\frac{d\big((d+1)v_iv_j-2c_{ij}^2\big)}
{(d-1)(d+2)}.
$$

*Proof sketch.* For one Gaussian row $a\sim\mathcal N(0,I_d/d)$, Isserlis' theorem gives

$$
\operatorname{Var}\big((a^\top x_i)(a^\top x_j)\big)
=v_iv_j+c_{ij}^2.
$$

The independent $A$-$C$ product has variance $v_iv_j$. For the orthogonal term, conditional on $A$,

$$
\operatorname{Var}\!\left[
\frac{1}{\sqrt m}\sum_{t=1}^m(a_t^\top x_i)(b_t^\top x_j)
\middle|A,X\right]
=\frac{\|Ax_i\|^2}{m}\,
\frac{x_j^\top\Pi_A^\perp x_j}{d-m}.
$$

Taking expectation over the uniformly random rank-$m$ projector $\Pi_A$ and using its second moment yields $g_{ij}$. All cross-covariances among the three channels vanish. A full projector-moment derivation is given in Appendix A. $\square$

Several corrections follow immediately.

1. The standard $1/\sqrt m$ attention scaling removes the spurious factor $m$ from the logit variance.
2. The orthogonal and independent channels agree only to leading order:

$$
g_{ij}=v_iv_j+O(1/d).
$$

Thus $\beta$ and $\gamma$ are not exactly interchangeable at finite width.
3. The proposition gives marginal moments. It does not make the logits independent; logits sharing a query, key, input, or base matrix are correlated.

### 4.3 Nearly isotropic token-Gram approximation

To interpret the exact formulas, suppose LayerNorm-like inputs satisfy $v_i\approx1$ and $c_{ij}\approx r$ for $i\ne j$, with $0\le r<1$. Then

$$
\mathbb E[L_{ii}]\approx\alpha s^2\sqrt m,\qquad
\mathbb E[L_{ij}]\approx\alpha s^2\sqrt m\,r,
$$

so the expected self-versus-cross gap is

$$
\Delta_{\rm th}
\approx \alpha s^2\sqrt m(1-r).
$$

For $i\ne j$,

$$
\operatorname{Var}(L_{ij})
\approx s^4\big(1+\alpha^2r^2\big)+O(s^4/d).
$$

These are conditional-weight approximations based on the observed Gram matrix. They are not universal data-distribution laws. In the atlas, $v_i$ and $c_{ij}$ are measured from every activation batch and the exact conditional formulas are evaluated directly.

### 4.4 Entropy: a local identity and an explicit surrogate

Let $z\in\mathbb R^n$, $\bar z=n^{-1}\sum_jz_j$, and $y=z-\bar z\mathbf1$. Softmax is invariant to the row mean.

**Proposition 2 (entropy near the uniform point; exact local expansion).**

$$
H(\operatorname{softmax}z)
=\log n-\frac{1}{2n}\|y\|_2^2+O(\|y\|_2^3)
\qquad\text{as }\|y\|_2\to0.
$$

Therefore small row-centered logit variance produces high entropy. It does **not** by itself imply a useful attention map; in the unmasked case, exactly uniform attention has matrix rank one.

For a masked row, the same expansion is applied only to its active support $S_i$, replacing $n$ by $k_i=|S_i|$. Rows with $k_i=1$ carry no entropy information.

For an i.i.d. Gaussian score surrogate $z_j\sim\mathcal N(0,\tau_n^2)$, Random-Energy-Model asymptotics place the entropy-condensation transition at the scale

$$
\tau_n\asymp\sqrt{2\log n}.
$$

This is an asymptotic surrogate, not a theorem for attention logits: real scores are correlated and $n=64$ is finite. The proposal therefore uses the measured row-centered standard deviation

$$
\widehat\tau^2
=\frac1n\sum_j(L_{ij}-\bar L_i)^2
$$

as a coordinate, but calibrates all finite-$n$ entropy boundaries by Monte Carlo and replication rather than asserting a universal constant.

### 4.5 Self-locking: exact deterministic reference

If one row has a diagonal advantage $\Delta$ and equal off-diagonal logits,

$$
z_i=\mu+\Delta,\qquad z_j=\mu\quad(j\ne i),
$$

then exactly

$$
p_{\rm self}
=\frac{1}{1+(n-1)e^{-\Delta}}.
$$

Consequently, the deterministic gap needed for a target self-mass $\eta$ is

$$
\Delta_\eta=\log\!\left(\frac{(n-1)\eta}{1-\eta}\right).
$$

For $n=64$, $\Delta_{0.5}=\log63\approx4.14$, while $\Delta_{0.9}=\log567\approx6.34$. Thus the old condition $\Delta\gtrsim\log(n-1)$ corresponds to only 50% self-mass in the noise-free reference, not 90% identity collapse. Random fluctuations and score correlations can shift the boundary in either direction, so the real boundary is estimated empirically.

For causal attention at query position $i$, $n-1$ must be replaced by the number of other unmasked keys available to that query. Self-locking thresholds are therefore position dependent and are summarized both by position and by sequence-level aggregates.

### 4.6 Matrix spectrum and gradients

**Proposition 3 (rank and nonzero spectrum; exact/asymptotic).** Since $M=W_q^\top W_k$,

$$
\operatorname{rank}(M)\le m.
$$

When $m<d$, the ordinary $d\times d$ condition number of $M$ is infinite. It is therefore invalid to describe $M$ as full rank or to compare its ordinary condition number across treatments.

At the tied corner, $M=s^2A^\top A$. Its nonzero eigenvalues equal those of $s^2AA^\top$. If $m/d\to c\in(0,1)$, the Marchenko-Pastur support is

$$
s^2(1-\sqrt c)^2
\le\lambda_{\rm nonzero}(M)
\le s^2(1+\sqrt c)^2,
$$

and the asymptotic nonzero-eigenvalue condition number is

$$
\kappa_+(M)\longrightarrow
\left(\frac{1+\sqrt c}{1-\sqrt c}\right)^2.
$$

No theorem in this proposal claims that adding $B$ or $C$ must reduce the top singular value. The atlas measures $\|M\|_2$, stable rank, singular-value effective rank, and spectral concentration directly.

**Proposition 4 (back-propagation identities; exact).** Let $\mathcal L$ be any scalar loss and $D=\partial\mathcal L/\partial L\in\mathbb R^{n\times n}$. Then

$$
\frac{\partial\mathcal L}{\partial W_q}
=\frac1{\sqrt m}KD^\top X^\top,\qquad
\frac{\partial\mathcal L}{\partial W_k}
=\frac1{\sqrt m}QD X^\top.
$$

Hence

$$
\left\|\frac{\partial\mathcal L}{\partial W_q}\right\|_F
\le\frac{\|K\|_2\|D\|_F\|X\|_2}{\sqrt m},
\qquad
\left\|\frac{\partial\mathcal L}{\partial W_k}\right\|_F
\le\frac{\|Q\|_2\|D\|_F\|X\|_2}{\sqrt m}.
$$

For a row $p=\operatorname{softmax}(z)$, the softmax Jacobian is

$$
J(p)=\operatorname{diag}(p)-pp^\top,\qquad \|J(p)\|_2\le\frac12.
$$

Both a saturated one-hot row and, for large $n$, a nearly uniform row can have small Jacobian scales, for different reasons. These identities do not imply that a particular $(s,\alpha,\beta,\gamma)$ minimizes gradient variance; gradient health is an experimental outcome.

### 4.7 Cross-head common bias is not a diversity guarantee

Consider two independently drawn heads, possibly with different coefficients. Define

$$
f(\alpha)=1+\alpha^2\frac{m+1}{d}.
$$

**Proposition 5 (ratio-of-expectations common-bias statistic; exact).**

$$
\mathbb E[M_\ell]=s_\ell^2\alpha_\ell\frac md I_d,
\qquad
\mathbb E\|M_\ell\|_F^2=s_\ell^4m f(\alpha_\ell).
$$

Therefore

$$
R_{12}:=
\frac{\mathbb E\langle M_1,M_2\rangle_F}
{\sqrt{\mathbb E\|M_1\|_F^2\,
\mathbb E\|M_2\|_F^2}}
=\frac{\alpha_1\alpha_2m}
{d\sqrt{f(\alpha_1)f(\alpha_2)}}.
$$

This is a ratio of expectations, not $\mathbb E[\cos(M_1,M_2)]$. It measures the shared isotropic mean induced by positive alignment. If kernels are centered by their ensemble means, independent heads satisfy

$$
\mathbb E\langle M_1-\mathbb EM_1,\,
M_2-\mathbb EM_2\rangle_F=0
$$

for every pair of alignments. Consequently, varying $\alpha$ does not mathematically guarantee functional head diversity. It supplies different self-bias priors. Whether those priors produce diverse learned attention maps is the central empirical hypothesis, measured with attention-map similarity, centered kernel alignment, output diversity, and head ablation.

### 4.8 What the exact theory supports

The exact results support two screening coordinates:

$$
\Delta_{\rm th}=\alpha s^2\sqrt m(1-r)
\quad\text{and}\quad
\tau_{\rm th}^2\approx s^4(1+\alpha^2r^2),
$$

with exact finite-width corrections evaluated from Proposition 1. They suggest where self-bias and concentration may occur. They do **not** establish a sharp stable region, an “if and only if” trainability theorem, or a two-parameter reduction valid for all finite widths and data.

> **Design principle.** Use exact moments to organize the search, but let replicated forward and training experiments determine which constrained $(s,\alpha,\beta,\gamma)$ regions are usable.

---

## 5. Operational Atlas: What “Good” and “Bad” Mean

### 5.1 Metrics

For each head and batch, record:

- measured logit mean, variance, row-centered scale $\widehat\tau$, and diagonal gap $\widehat\Delta$;
- the active key set $S_i$, its size $k_i$, and normalized row entropy $H(P_{i\cdot})/\log k_i$ for $k_i>1$;
- effective support $N_{\rm eff}(P_{i\cdot})/k_i$;
- maximum mass $\max_jP_{ij}$ and self-mass $P_{ii}$;
- distance from active-support uniformity

$$
d_{\rm unif}
=\left(\frac1n\sum_i\|P_{i\cdot}-u_i\|_2^2\right)^{1/2},
\qquad
u_{ij}=\frac{\mathbf1\{j\in S_i\}}{k_i};
$$

- $\operatorname{erank}(P)/n$ and $\operatorname{srank}(P)/n$;
- output-token cosine similarity after applying attention to independently initialized values;
- $\|J(P_{i\cdot})\|_F$ and $\|J(P_{i\cdot})\|_2$;
- $\|M\|_2$, $\operatorname{erank}(M)$ over nonzero singular values, and spectral concentration;
- query/key gradient norms, update-to-weight ratios, loss, and loss slope during training.

### 5.2 Preregistered descriptive labels

These labels describe initialization behavior; they are not synonyms for downstream quality.

- **Diffuse:** the lower 95% bootstrap confidence bound for median normalized row entropy is at least $0.95$, and the upper confidence bound for distance from uniform is at most $0.10$.
- **Concentrated:** the upper 95% confidence bound for median normalized entropy is at most $0.50$, or the lower confidence bound for median maximum mass is at least $0.80$.
- **Self-locked:** the lower confidence bound for median self-mass is at least $0.80$.
- **Gradient-starved:** the upper confidence bound for the normalized query/key update-to-weight ratio or softmax-Jacobian scale is below a threshold fixed by the baseline pilot.
- **Screened candidate:** none of the preceding pathologies is established, normalized entropy lies between $0.60$ and $0.95$, and attention-matrix effective rank is above a pilot-fixed floor.
- **Boundary/uncertain:** confidence intervals cross one or more label thresholds.

Rows with $k_i=1$ are excluded from normalized-entropy, self-locking, and Jacobian-starvation labels. Confidence intervals resample paired seeds as the independent clusters; heads, batches, and token rows remain nested within seed. Metrics are either aggregated once per seed or analyzed by a hierarchical cluster bootstrap. Threshold sensitivity at $\pm0.05$ for probability/entropy cutoffs is reported. A point is called **empirically trainable** only after it passes a training test.

### 5.3 The phase map is an empirical object

The atlas will overlay the exact moment coordinates and the i.i.d.-Gaussian entropy surrogate, but colored zones will be generated from replicated measurements. No rectangular “stable region” is drawn by assumption. Boundaries are reported with uncertainty and can depend on $d,m,n$, token correlation, architecture, depth, normalization, and residual scaling.

---

## 6. Experiment 1 - Sweep and Map the Initialization Atlas

### 6.1 Stage 1A: unit-test the mathematics

Before mapping behavior, verify Lemma 1 and Propositions 1, 4, and 5 in code.

1. Use $d\in\{32,64,128\}$, $m/d\in\{1/8,1/4,1/2\}$, and fixed token pairs spanning $c_{ij}\in\{-0.5,0,0.5,1\}$.
2. Test the four corners and at least 20 interior coefficient points.
3. Use at least 20,000 matrix draws per small configuration, increasing draws until $\operatorname{SE}(\widehat\mu)\le0.01\sigma_{\rm pred}$ and $\operatorname{SE}(\widehat{\operatorname{Var}})\le0.01\operatorname{Var}_{\rm pred}$.
4. Require every predicted mean and variance to lie inside a simultaneous 99% Monte Carlo confidence interval after Holm correction.
5. Verify gradient identities by central finite differences in float64 at 20 random configurations, requiring relative error below $10^{-5}$ away from zero-gradient coordinates.
6. Require $\|AB^\top\|_F/(\|A\|_F\|B\|_F)<10^{-6}$, component energies within 1% of target, and marginal entry variances within 2% of $s^2/d$.
7. Verify that empirical entrywise query-key correlation is within 0.02 of $\alpha$, and that $(\alpha,\beta,\gamma)=(0,0,1)$ at $s=s_0$ reproduces the repository initializer's logit distribution.
8. Confirm the custom initializer runs after global initialization, survives checkpoint construction, and is not overwritten before the first forward pass.

Failure of this stage blocks all downstream claims and triggers a code/theory correction.

### 6.2 Stage 1B: discovery sweep on the constrained domain

Reference configuration:

$$
d=64,\qquad m=16,\qquad n=64.
$$

Use the grid

$$
\log_2s\in[-2,2]\ \text{(17 values)},\quad
\alpha\in[-1,1]\ \text{(13 values)},\quad
\phi\in[0,\pi/2]\ \text{(9 values)}.
$$

At $\alpha=\pm1$, all $\phi$ values represent the same coefficient point, so retain one endpoint value there. This gives

$$
17\,[11\times9+2]=1{,}717
$$

unique admissible cells. Every row stores $(s,\alpha,\beta,\gamma)$ explicitly. At each cell:

- use 64 paired seeds in discovery;
- evaluate synthetic Gaussian inputs with pairwise-correlation settings $r\in\{0,0.25,0.5\}$;
- evaluate real activation batches captured immediately after LayerNorm from an independently initialized baseline model;
- use causal masks and unmasked attention as separate strata;
- report the forward metrics from Section 5.1 and a preregistered single-batch backward probe. Update-to-weight trajectories and loss slopes are reserved for Stage 1D.

The signs of $\beta$ and $\gamma$ are checked at 50 selected cells as a symmetry unit test. They are not included in the main grid because they do not change the seed-averaged initialization law.

### 6.3 Stage 1C: adaptive boundary refinement and replication

Fit a probabilistic surrogate classifier to the discovery labels using

$$
(\log s,\alpha,\phi,1/d,m/d,\log n,\text{mask},
\widehat{v_iv_j},\widehat{c_{ij}^2},
\widehat\Delta,\widehat\tau)
$$

as features. The model is a visualization/interpolation tool, not evidence by itself.

1. Select points with highest boundary uncertainty and points where the exact-moment coordinates disagree with the learned boundary.
2. Add at least 500 adaptive cells with 256 seeds each.
3. Replicate the final map at $(d,m,n)\in\{(128,16,128),(128,32,128),(256,32,256)\}$.
4. Hold out 20% of cells from all surrogate fitting and report calibration, balanced accuracy, and boundary error.

### 6.4 Stage 1D: trainability validation

Select points without looking at their training outcomes:

- 15 diffuse;
- 15 concentrated;
- 15 self-locked;
- 15 screened candidates;
- 20 boundary/uncertain points;
- all four canonical corners at matched scales.

If labels overlap, assign a mutually exclusive stratum in this precedence order: self-locked; concentrated but not self-locked; diffuse; gradient-starved only; screened candidate; boundary/uncertain.

Train a one-block decoder on two tasks: a controlled associative-recall task and next-token prediction on a fixed TinyStories subset. Use identical data order, all non-query/key weights, optimizer state, and compute budget across paired treatments.

Primary Stage-1D outcome: loss decrease per token over the first fixed training window. Secondary outcomes: divergence rate, gradient/update statistics, final validation loss, and whether the atlas label predicts failure. The atlas is considered useful only if screened candidates outperform the pooled pathology groups on the preregistered primary outcome with a confidence interval excluding zero.

### 6.5 Experiment 1 outputs

- a three-coordinate atlas over $(s,\alpha,\phi)$ with 1,717 unique cells and the corresponding $(\beta,\gamma)$ recorded;
- 2D slices in $(s,\alpha)$ for every $\phi$;
- uncertainty bands, not hand-drawn hard boundaries;
- exact-versus-empirical moment residual plots;
- transfer maps across width, head dimension, sequence length, masking, and real activations;
- a frozen list of candidate portfolios for Experiment 2.

---

## 7. Experiment 2 - Controlled Test on a Real Language Model

### 7.1 Model and data

Use a decoder-only Transformer in the 30M-60M parameter range, with at least 8 layers and 8 heads, trained on a fixed TinyStories or comparable open corpus split. Fix tokenizer, context length, batch size, optimizer, learning-rate schedule, residual scaling, normalization, dropout, token budget, and evaluation cadence before unblinding final results.

The main run uses at least 100M training tokens. A smaller pilot is used only for variance and power estimation; pilot seeds are excluded from confirmatory analysis.

### 7.2 Treatments

Treatment 1 retains the repository scale $s_0$. Treatments 2-5 use a common preregistered scale $s_\star$, or Treatments 4-5 use exactly the same per-head scale multiset. Within every scale-matched comparison, $\mathbb E[(W_q)_{ab}^2]$, $\mathbb E[(W_k)_{ab}^2]$, parameter count, and every non-query/key initialization match. Thus Treatment 1 versus 2 isolates scale, Treatment 2 versus 3 isolates structure, and Treatment 4 versus 5 isolates coefficient allocation.

1. **Architecture baseline:** the repository's standard independent query/key initialization.
2. **Scale-only control:** independent Gaussian initialization $(0,0,1)$ at the best preregistered atlas scale.
3. **Homogeneous structured:** every head uses the same best screened structured point.
4. **Within-layer diverse portfolio:** each layer receives distinct screened points spanning $(\alpha,\phi)$ while remaining inside replicated candidate regions.
5. **Globally matched, within-layer homogeneous control:** each layer uses one portfolio point for all its heads, while portfolio points cycle across layers so the model-wide histogram of $(s,\alpha,\beta,\gamma)$ matches Treatment 4. This isolates within-layer heterogeneity from the global coefficient distribution.
6. **Orthogonal-endpoint ablation:** $(0,1,0)$ at the matched candidate scale, if it passes Stage 1D.

Head order is randomly permuted per layer and seed. The permutation is fixed across checkpoints and recorded.

### 7.2.1 Per-head implementation and integration checks

For a conventional linear projection stored as `[output, input]`, construct $A_h,B_h,C_h\in\mathbb R^{m\times d}$ independently for each head and assign

$$
W_q[hm:(h+1)m,:]=W_{q,h},\qquad
W_k[hm:(h+1)m,:]=W_{k,h}.
$$

For fused QKV projections, first locate the global query and key blocks, then apply the same head-row slices inside each block. GPT-2-style `Conv1D` stores the transpose, so the corresponding head slices occupy columns. The first real-model implementation uses ordinary multi-head attention; grouped-query or multi-query attention is deferred because query heads do not have one-to-one key heads.

Before training:

- compare framework query/key tensors and logits with an independent manual projection;
- verify reshape and head ordering with one-hot test tensors;
- preserve native bias handling and all value/output projections;
- measure logits after positional transforms such as RoPE and after masking;
- apply row-space orthogonalization independently per head, never globally across heads;
- save and reload an initialization checkpoint, then require bitwise-equal query/key tensors.

### 7.3 Confirmatory outcomes

The single primary outcome is held-out validation cross-entropy at the fixed token budget. The two primary contrasts are:

- diverse portfolio versus architecture baseline;
- diverse portfolio versus globally matched within-layer homogeneous control.

Holm correction controls the family-wise error rate across these two contrasts. Report paired mean differences, 95% confidence intervals, and standardized effect sizes. If the confidence interval lies inside a preregistered equivalence margin of $\pm0.01$ nats, conclude practical equivalence rather than “no difference.”

Secondary outcomes:

- tokens and wall-clock time to reach a fixed validation loss;
- divergence/NaN rate and gradient clipping frequency;
- early loss slope;
- normalized row entropy, maximum mass, self-mass, and attention-matrix effective rank by layer/head;
- pairwise attention-map Jensen-Shannon divergence and centered kernel alignment;
- output diversity and per-head ablation loss;
- persistence of the assigned initialization geometry over training.

A Treatment 4-5 difference supports a causal effect of within-layer coefficient heterogeneity, but does not establish attention-map similarity as the causal mediator.

### 7.4 Seeds, power, and paired design

Use paired seeds: treatments share data order and every initial random tensor not mathematically forced to differ by the query/key treatment. Run an external pilot with at least 3 seeds per arm to estimate the standard deviation of paired validation-loss differences. Freeze a power calculation for 80% power at the smallest practically relevant effect, with a minimum of 8 confirmatory seeds per treatment. If the power calculation requires more than the available budget, increase the budget or declare the confirmatory comparison infeasible; do not silently cap an underpowered study.

No treatment-specific optimizer tuning is allowed in the confirmatory study. A separate robustness appendix may evaluate a shared learning-rate grid, clearly labeled exploratory.

### 7.5 Generalization checks

After the confirmatory model is complete, replicate the winning and baseline treatments at:

- a second model width/head count;
- a second context length;
- one deeper model with residual scaling held fixed;
- one different open text corpus if compute permits.

These are transfer tests, not opportunities to redefine the candidate region.

---

## 8. Statistical and Reproducibility Protocol

1. **Freeze the protocol.** Commit coefficient grids, seeds, labels, primary outcomes, exclusions, and analysis code before final runs.
2. **Separate discovery from confirmation.** Atlas fitting, threshold tuning, and portfolio selection use discovery data only. Confirmatory language-model test data are evaluated after treatments are frozen.
3. **Report uncertainty for maps.** Every cell includes sample count, median, bootstrap interval, and label confidence. Uncertain cells remain uncertain.
4. **Use paired analyses.** Analyze per-seed paired differences. Inspect their distribution; use a permutation or Wilcoxon test if a Gaussian paired model is visibly inappropriate.
5. **Control multiplicity.** Holm correction for confirmatory contrasts; Benjamini-Hochberg false-discovery control for large exploratory head-wise families.
6. **Report effect sizes and equivalence.** A non-significant $p$-value is not evidence of equality.
7. **Track compute and failures.** Include failed runs, OOMs, NaNs, clipping, and restarts in the released run manifest.
8. **Release artifacts.** Publish environment lockfile, exact model configuration, data hashes, initialization code, unit tests, raw per-seed metrics, and scripts that regenerate every figure.

---

## 9. Falsification Criteria

The proposal is designed to be refutable.

1. **Mathematical implementation failure:** Monte Carlo moments or finite-difference gradients fail the Stage-1A tolerances. The corresponding derivation or implementation is wrong.
2. **No finite-width structural effect:** after controlling $s$ and $\alpha$, $\phi$ has no reproducible effect beyond Monte Carlo error. Then the orthogonal-versus-independent split is unnecessary for the tested widths.
3. **No atlas transfer:** boundaries do not transfer across replicated $(d,m,n)$ settings after conditioning on measured $\widehat\tau$, $\widehat\Delta$, and input Gram statistics. Then there is no portable initialization atlas of the proposed form.
4. **No trainability validity:** screened candidates do not improve the preregistered Stage-1D early-loss outcome relative to pathology groups. Then the forward atlas is not a useful trainability screen.
5. **No real-model benefit:** adjusted confidence intervals establish practical equivalence to, or harm relative to, both the architecture baseline and the matched homogeneous control. Then the practical head-portfolio hypothesis is unsupported.
6. **Scale explains everything:** adjusted equivalence intervals establish that the scale-only control and structured treatments are practically equivalent on confirmatory outcomes. Then joint structure adds no demonstrated value beyond scale selection.
7. **No within-layer allocation effect:** the diverse portfolio is equivalent to, or worse than, the globally matched within-layer homogeneous control under the preregistered adjusted intervals. Then within-layer coefficient heterogeneity is unsupported.

Negative results are still informative: they identify which degrees of freedom can be removed from future initialization design.

---

## 10. Timeline and Deliverables

| Phase | Duration | Deliverable / gate |
|---|---|---|
| 1. Mathematical and code validation | 3 weeks | Stage-1A tests pass; otherwise stop and repair |
| 2. Discovery atlas | 5 weeks | 1,717-cell map with uncertainty |
| 3. Boundary replication and trainability | 6 weeks | replicated atlas + frozen portfolios |
| 4. Real-model pilot and power analysis | 3 weeks | confirmatory seed count frozen |
| 5. Confirmatory language-model study | 8 weeks | paired primary analysis |
| 6. Transfer tests and release | 5 weeks | paper, code, raw metrics, preregistration |

---

## 11. Conclusion

This proposal studies Transformer initialization through the joint geometry of query and key projections without claiming more than the mathematics supports. The correlated Gaussian-orthogonal family is dimensionally valid, fan-in scaled, energy matched, and parameterized on its true constrained domain. Its logit mean and variance are available exactly at finite width, including the distinction between row-space-orthogonal and independent components. Entropy concentration, self-locking, rank propagation, gradient health, trainability, and learned head diversity are treated as separate phenomena.

The experimental program follows the intended idea in two stages: first sweep and map the admissible $(s,\alpha,\beta,\gamma)$ initialization domain; then test frozen candidate regions in a real language model. Exact unit tests gate the sweep, replicated confidence intervals define the map, and matched controls isolate scale, structure, and within-layer heterogeneity. A positive result would provide a reproducible initialization atlas and a causal test of head-wise portfolios. A negative result would still identify which apparent geometric degrees of freedom do not matter in practice.

---

## References

[1] Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.

[2] Dong, Cordonnier, and Loukas, "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth," *ICML*, 2021.

[3] Noci, Anagnostidis, Biggio, Orvieto, Singh, and Lucchi, "Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse," *NeurIPS*, 2022.

[4] Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse," *ICML*, 2023.

[5] Glorot and Bengio, "Understanding the Difficulty of Training Deep Feedforward Neural Networks," *AISTATS*, 2010.

[6] He et al., "Delving Deep into Rectifiers," *ICCV*, 2015.

[7] Saxe, McClelland, and Ganguli, "Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks," *ICLR*, 2014.

[8] Pennington, Schoenholz, and Ganguli, "Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry," *NeurIPS*, 2017.

[9] Huang, Perez, Ba, and Volkovs, "Improving Transformer Optimization Through Better Initialization," *ICML*, 2020.

[10] Bachlechner et al., "ReZero is All You Need: Fast Convergence at Large Depth," *UAI*, 2021.

[11] Zhang, Dauphin, and Maire, "Fixup Initialization: Residual Learning without Normalization," *ICLR*, 2019.

[12] Trockman and Kolter, "Mimetic Initialization of Self-Attention Layers," *ICML*, 2023.

[13] Bao, Hataya, and Karakida, "Self-attention Networks Localize When QK-eigenspectrum Concentrates," *ICML*, 2024.

[14] Giorlandino and Goldt, "Two Failure Modes of Deep Transformers and How to Avoid Them: A Unified Theory of Signal Propagation at Initialisation," *arXiv:2505.24333*, 2025.

[15] Qi et al., "Taming Transformer without Using Learning Rate Warmup," *arXiv:2505.21910*, 2025.

[16] Michel, Levy, and Neubig, "Are Sixteen Heads Really Better than One?" *NeurIPS*, 2019.

[17] Voita et al., "Analyzing Multi-Head Self-Attention," *ACL*, 2019.

[18] Roy and Vetterli, "The Effective Rank: A Measure of Effective Dimensionality," *EUSIPCO*, 2007.

[19] Boucheron, Lugosi, and Massart, *Concentration Inequalities*, Oxford University Press, 2013.

[20] Yang, "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer," *arXiv:2203.03466*, 2022.

---

## Appendix A. Orthogonal-Channel Variance Derivation

For a uniformly random rank-$m$ orthogonal projector $\Pi$ in $\mathbb R^d$,

$$
\mathbb E[\Pi_{ij}\Pi_{kl}]
=a\,\delta_{ij}\delta_{kl}
+b(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}),
$$

with

$$
a=\frac{m(md+m-2)}{d(d-1)(d+2)},\qquad
b=\frac{m(d-m)}{d(d-1)(d+2)}.
$$

The row space of a Gaussian $A$ is uniform on the Grassmannian, and its singular values are independent of its row-space orientation. Therefore

$$
\mathbb E[A^\top A\mid\Pi_A]=\Pi_A.
$$

Using $B=G\Pi_A^\perp$,

$$
g_{ij}
=\mathbb E_A\left[
\frac{\|Ax_i\|^2}{m}\,
\frac{x_j^\top\Pi_A^\perp x_j}{d-m}
\right].
$$

Evaluating the projector moment gives

$$
g_{ij}
=\frac{d\big((d+1)v_iv_j-2c_{ij}^2\big)}
{(d-1)(d+2)}.
$$

For $i=j$, $c_{ii}=v_i$, so $g_{ii}=d\,v_i^2/(d+2)$. For fixed normalized tokens and $d\to\infty$, $g_{ij}\to v_iv_j$.

