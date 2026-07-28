Running lab log for the stag-hunt language-emergence project. Newest entries on top.

---

## 2026-07-27 — First genuine symbolic communication (lever sweep results)

All 8 lever runs completed (array 35404344; the two sticky "TIMEOUT"s are a Slurm exit artifact — all runs finished 3000 updates). Full probes on the final checkpoints, 500 eval episodes per condition. Figure: `stag-hunt-files/2026-07-26_blind-levers/blind_levers_summary.png`. Data dir reorganized into dated experiment folders (see its README).

| arm / seed | eval joint stag: none / random / muted | accuracy: none / random / muted | MI color / region (bits) | verdict |
| --- | --- | --- | --- | --- |
| long s0 | 25% / 26% / 25% | 50% / 48% / 64% | .005 / .011 | convention |
| long s1 | 26% / 26% / 27% | 43% / 43% / 59% | .002 / .003 | convention |
| sticky s0 | 31% / 30% / 8% | 52% / 48% / 41% | .003 / .006 | convention |
| sticky s1 | 48% / 46% / 24% | 93% / 92% / 84% | .003 / .002 | convention (high-performing) |
| talk5 s0 | 32% / 31% / 28% | 85% / 83% / 83% | .002 / .002 | convention |
| talk5 s1 | 24% / 25% / 24% | 50% / 49% / 53% | .002 / .004 | convention |
| **aux s0** | **62% / 11% / 1%** | 83% / 58% / 43% | **1.00 / 0.98** | **causal bidirectional communication** |
| **aux s1** | **29% / 1% / 30%** | 97% / 29% / 91% | **0.97 / 0.95** | **causal communication + convention fallback** |

Readings:

1. **The signaling+listening biases produced the project's first real language.** Both aux seeds saturate MI at ~1 bit *in both directions* (the full clue each way), and the random-message control — the decisive probe — collapses captures 62%→11% (s0) and 29%→1% (s1). Aux s0 hits 62% eval capture with mean return 2.49, far above the ~50% ceiling of any non-communicative rendezvous strategy; in-training it was still climbing at 60%. Honest caveat: the signaling bias *buys* MI directly, so MI alone proves nothing — the emergent part is that listeners came to causally *rely* on the content (intervention collapse), which the bias does not directly enforce (the listening bias pushes sensitivity to messages, not correct use of their meaning).
2. **The rendezvous-convention loophole is real and is what every other arm learned.** Six of six non-aux runs show null random controls despite accuracies up to 93% — above-chance targeting with causally dead channels, exactly the camping/sweeping convention predicted last night. First-presence accuracy is officially demoted to a screening metric; verdicts require interventions.
3. **Sticky and talk5 didn't ground symbols; they accelerated convention formation** (sticky s1 had the earliest capture-rate takeoff). Persistence and credit concentration alone don't break the speaker-listener deadlock — the deadlock had to be attacked from both ends simultaneously (aux).
4. Curious detail: aux s1 keeps a convention as *fallback* — muting the channel leaves it at 29% (convention performance) while random symbols actively mislead it to 1%. Aux s0 is fully channel-dependent (muted → 1%). Two different equilibria under the same config.
5. The mute-vs-random dissociation in long/sticky (muted often *raises* accuracy) is the input-distribution-shift artifact again, in the opposite direction — third instance; the random control remains the only trustworthy single probe.

Next questions (not yet launched): does aux communication survive *removing* the biases (anneal coefs to 0 — is the protocol self-sustaining once formed)? Does it survive reintroducing hares (phase 2, risk-dominance)? Can aux-trained symbols transfer to a bias-free partner? And the deferred ideas: message decay sweep, uncertainty-gated speaking, population training.

### Interpretation check + codebooks (same day, discussion)

Extracted the learned codebooks from the aux checkpoints (P(symbol | clue), 300 episodes): seed 0 uses a **binary partition with synonyms** ({3,4} = red/west vs {silence,1,2} = blue/east); seed 1 grounded **silence as a word** (silence = red/west, any symbol = blue/east) — which retroactively explains its mute-intervention profile (forced silence is a valid, sometimes-true utterance; random symbols are lies). Vocab size is confirmed non-binding: 1 bit needed per direction, so the 5-symbol space collapses into two equivalence classes.

Agreed interpretation of the aux result: the signaling bias *buys* the code, the listening bias *buys* attention; what **emerged** through task reward is the semantics — the listener's correct, reward-relevant decoding (evidence: performance far above any bias's direct optimum; seed 1's convention-fallback-under-mute vs misled-under-random asymmetry).

**Critical confound found in the codebooks: both roles use the identical partition — because the agents share weights.** There is effectively one network talking to itself; the listener owns the speaker's code by construction, so comprehension was partially free. Decision (user): **decouple the agents** (separate networks per agent) and **randomize clue assignment** per episode (either agent can hold color or region; the `private_clue` slot layout already marks the held attribute), forcing each network to learn all four speak/decode mappings and making the code attribute-indexed rather than speaker-indexed. Complementary assignment keeps referents inferable without attribute marking (stops being true at 3+ attributes — a future compositionality lever). Frozen-speaker transfer is **postponed** (user: still exploring setups).

## 2026-07-27 (later) — Decoupled agents + randomized clues launched

Code: `--separate-actors` (two `RecurrentActor`s, per-agent BPTT, both checkpoint formats supported by the analysis), `EnvConfig.randomize_clue_assignment` (+ assignment bit in the critic state), signaling-MI loss and all diagnostics now grouped by **attribute holder** instead of agent index (`mi_color`/`mi_region` keys). 33 tests passing.

Array `stag-decoupled` (job 35489336), blind phase-1 game, 4000 updates × 64, `--array=0-5%4`: `phase1_dec_{aux,none}_seed{0,1,2}` — aux = both biases at 0.1, none = no biases (the move-away-from-bias control). Both arms decoupled + randomized. Predictions: `none` → rendezvous conventions again (possibly harder to form — conventions must also be attribute-indexed now); `aux` → the decisive test of whether scaffolded emergence survives true two-mind separation; mirrored codebooks should NOT reappear.

### Preliminary results (evening; 3/6 runs done — figure `stag-hunt-files/dec_preliminary_summary.png`)

- **Emergence survives decoupling: 3/3 aux seeds** reach 88–94% targeting accuracy and ~19–44% capture (still climbing), with training-MI saturated at the 2-bit ceiling (both attribute directions, both independent networks, randomized roles). Takeoff needs ~1500–2500 updates vs ~500–800 shared — consistent with the listener now having to learn the speaker's code instead of owning it.
- **Sequencing signature:** in every aux seed MI saturates ~500–1000 updates *before* accuracy/captures rise — the bias installs the code first; comprehension (the emergent part) arrives later and brings performance with it.
- **Bias-free control much weaker under randomization:** `none` s0 only ~35–46% accuracy by update 4000 (attribute-indexed conventions are harder for two independent nets to coordinate); s1/s2 near chance early. Randomized assignment weakened the non-communicative shortcut, as hoped.
- Verdicts still pending eval interventions (training MI is the optimized quantity); analysis running on the finished runs. Remaining: aux s1 (~2800/4000), none s1/s2 (early). Also agreed: adopt **vectorized episode collection** after this array finishes — profiling showed the trainer is latency-bound (policy calls at batch 1–2; GPU ~idle; batch-64 forward costs the same as batch-1), expected ~2–2.5× end-to-end.

## 2026-07-28 — Decoupled sweep final: two independent minds, one shared language

All 6 runs terminal (aux s1 timed out at 3127/4000 on a slow node — analyzed at checkpoint 3000; results unaffected). Runs + figures filed in `stag-hunt-files/2026-07-27_decoupled/` (`dec_summary.png`).

| run | eval joint: none / random / muted | accuracy: none / random / muted | eval MI c/r | verdict |
| --- | --- | --- | --- | --- |
| dec aux s0 | 24% / **0%** / 24% | 94% / — / 77% | 0.97 / 0.96 | **causal communication** + silence fallback |
| dec aux s1 | 24% / **0.2%** / 24% | 98% / 33% / 72% | 0.97 / 0.96 | **causal communication** + silence fallback |
| dec aux s2 | **51%** / 12% / 0.2% | 94% / 53% / 25% | 1.00 / 0.88 | **causal communication**, fully channel-dependent |
| dec none s0 | 25% / 25% / 22% | 43% / 41% / 48% | ~0.002 | convention |
| dec none s1 | 27% / 27% / 25% | 33% / 32% / 34% | ~0.0005 | weak convention |
| dec none s2 | 25% / 24% / 22% | 44% / 39% / 61% | ~0.002 | convention |

Findings:

1. **Scaffolded emergence fully survives decoupling: 3/3 vs 0/3.** With separate networks and randomized roles, every biased seed developed causal bidirectional communication (random-message collapse to 0–12% captures); every bias-free seed produced a convention with a dead channel.
2. **The shared-lexicon result (per-network codebooks, 400 episodes):** within each aux seed, the two *independently initialized* networks converged on the *same* code with near-identical probabilities — s0/s1: silence = red/west, any symbol = blue/east; s2: {sil,3,4} = red/west vs {1,2} = blue/east. The code is **attribute-agnostic** (same partition for color and region; the referent is inferred from complementarity), and **silence is grounded as a word in all three seeds**. The mirrored-partition structure seen under weight sharing was therefore *not* an artifact — it's the convergent economical solution; but now it's genuinely *negotiated* between two minds rather than inherited from shared parameters.
3. **Open observation — an execution gap:** aux s0/s1 target at 94–98% accuracy yet capture in only ~24% of eval episodes (≈ the convention arms' rate; only s2 converts knowledge into 51% captures). With positions hidden, *finding each other at the agreed stag* within the horizon is now the binding constraint, not knowing where to go. Candidate follow-ups: longer horizon at eval, movement-efficiency analysis, or whether s2's higher capture rate reflects a faster meeting protocol worth characterizing.
4. Sequencing signature held in all three aux seeds: MI saturates 500–1000 updates before accuracy/captures move — code first, comprehension later.

**Infrastructure:** vectorized episode collection adopted (post-array, as agreed): NumPy observation encoder (exact-parity tested), lockstep stepping of all envs with one batched forward per agent per timestep, single H2D/D2H per step; replay-determinism test; 35 tests green. Benchmark: 1.7 s/update at production settings on the MIG slice vs 4.1–4.5 s/update the decoupled runs averaged on full H100s — expect 4000-update runs in ~1–1.5 h now.

**Open next steps:** bias anneal → self-sustaining protocol test; frozen-speaker transfer (infrastructure now ready, user-postponed); phase-2 hare reintroduction; execution-gap characterization; deferred ideas (message decay, uncertainty-gated speaking, 3+ attributes for compositionality pressure).

## 2026-07-28 — Why doesn't language emerge without the bias? (8-stag sweep launched)

Decision (user): slow down and interrogate the no-bias failure rather than push
new capabilities. Central manipulation: an **8-stag game** (4 colors × 2
regions) that raises the value of communication and starves the alternatives —
chance targeting drops 25% → 12.5%, the rendezvous-convention capture ceiling
drops to ~25%, and the EV table becomes asymmetric (color clue = 2 bits, region
= 1 bit): no info EV 0.5, region-only 1.0, color-only 2.0, bidirectional 4.0
vs stag = 4. Vocabulary (4+silence) is now *exactly sufficient* for one-shot
color naming — mild efficient-coding pressure for free.

Array `stag-8stag` (job 35658979), all arms decoupled + randomized clues +
blind + vectorized trainer (new `--n-colors/--n-regions/--vocab-size` flags):

- `big8_none` ×3 (6000 updates): the question — does 8× communication value
  ignite emergence without any bias? (Prediction: no — the deadlock is a
  gradient-structure problem, not a payoff-size problem. If it DOES emerge,
  the whole bias story reduces to an economics story.)
- `big8_aux` ×3 (6000): positive control at higher entropy; does the lexicon
  grow beyond a binary partition to name 4 colors?
- `sig_only` ×2 / `lis_only` ×2 (2×2 game, 4000): deadlock decomposition —
  which half (informative speaking vs attentive listening) is the binding
  constraint? Prediction: both fail alone (code-without-audience /
  attention-without-content), confirming two-sidedness.

### Results (same day; all runs done in 34–80 min each thanks to vectorization; figure `2026-07-28_8stag-deadlock/deadlock_8stag_summary.png`)

| arm | eval joint: none / random / muted | accuracy: none / random | verdict |
| --- | --- | --- | --- |
| **sig_only s0** | **75% / 24% / 25%** | 96% / 75% | **causal + graceful fallback — project best** |
| **sig_only s1** | **73% / 22% / 27%** | 97% / 71% | same |
| lis_only s0 | 7% / 1% / **0%** | 35% / 19% | no code; obligate attention to babble |
| lis_only s1 | 27% / 22% / **0%** | 43% / 45% | convention; mute-paralysis |
| big8_none ×3 | 3–5% / ≈same / ≈same | 11–14% ≈ chance | dead flat for 6000 updates |
| big8_aux s0,s2 | 2–4% / ~1% / **0%** | 9–12% | MI bias never engaged; listening pathology only |
| big8_aux s1 | 5% / 2% / 4% | 23% | late MI takeoff (2.5/3 bits), unconverged |

Findings — the answer to "why no emergence without the bias":

1. **The deadlock is asymmetric: production is the missing gradient, comprehension is free.** The signaling bias alone yields the strongest communication yet (75%/73% captures, return ≈ 3.0, MI ≈ 0.9 bits/direction): given informative symbols, listeners learn to decode them from task reward alone. The listening bias alone produces nothing (attention to noise ≠ content). So the two-sided-deadlock model was half right — only the speaker side truly lacks a gradient path (a speaker's informative symbol is credited only through listener behavior that doesn't exist yet; a listener attending to informative symbols is rewarded immediately).
2. **Unforced listeners are robust listeners.** sig_only agents under random messages keep 71–75% *accuracy* (they discount incoherent streams and fall back to the convention, captures → ~23%), whereas every listening-bias run is an *obligate* listener — muting collapses big8_aux and lis_only runs to literally 0/500. The listening bias caused the earlier "execution gap": forced sensitivity ≈ fragility. **Drop the listening bias going forward.**
3. **Payoff size does not break the deadlock.** 8× communication value (big8_none): perfectly flat at the 12.5% chance floor for 6000 updates, null interventions. The failure is gradient-structural, full stop.
4. **The MI bias itself stalls at 4 colors (2/3 seeds)** — hypothesis: the message-entropy bonus (0.05) now directly opposes the code. At 2×2 the synonym-partition lexicon delivered max MI *at high entropy* (the objectives were compatible — that's why synonyms emerged); naming 4 colors among 5 signals requires near-deterministic messages, so entropy and MI now trade off and the uniform-babble solution is a local optimum. Fix to test: message entropy ~0.01 or annealed.

**Proposed next (pending user):** `sig_only` at 8 stags with message entropy 0.01 vs 0.05 — the single most promising configuration for strong, minimally-scaffolded language; then bias-anneal on a sig_only checkpoint (self-sustainability), then frozen-speaker transfer.

### sig8 results (same day, array 35730280; user approved 8-stag sig-only)

Entropy-conflict hypothesis **confirmed on the speaker side**: message entropy 0.05 → MI stuck at 0.00 in 2/3 seeds (same stall as big8_aux); entropy 0.01 → MI engages in **3/3 seeds** (2.47 / 1.90 / 1.24 of the 3-bit ceiling by q4). But performance lags the code: best runs reach only 22–25% accuracy (chance 12.5%) and 9–13% captures at 6000 updates, still climbing ~linearly. At 2×2, comprehension came fast and free; at 4×2 the listener's reward-driven decoding is the slow step (4 speak/decode mappings, rare captures, partial codes: seed 2 plateaued at 1.24 bits — likely region + partial color only). Extended runs launched: `sig8L` (job 35736459) = entropy 0.01, **16000 updates**, 3 seeds. Codebook/intervention analysis of the 6000-update runs in progress.

Also under discussion (user proposal, design phase): **free-goal game** — remove the designated target, let agents choose and negotiate the goal (communication = intention, not injected fact). Key design constraint identified: with symmetric full observability and equal-value stags, a Schelling convention kills the need for language — decision-relevant *private* information must remain (candidate designs: private positions / private per-stag value draws / private perception). Measurement shifts to MI(message; captured stag), intent-decoding probes, and the unchanged intervention contract. Code: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/stag-hunt-language-emergence`, artifacts: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/stag-hunt-files`. Detailed run-by-run record: `docs/experiments/2026-07-23-risk-curriculum.md` in the repo.

Related: [[Language Emergence with Stag Hunt Game]], [[Stag Hunt Language Emergence - Experiment Design]], [[Stag Hunt Language Emergence - Episode and Agent Architecture]]

---

## 2026-07-26 — Design review, phase plan, presence-based phase-1 game

### New finding from the old data: cooperation without any information transfer

Re-analysis of the cooperative run `cw3s_hare10_seed0` (eval, 500 episodes): 108 `joint_stag` vs 306 `failed_stag`. Even if every failure were a joint attempt at the wrong stag, targeting accuracy is ≤ 26% — and **25% is exactly chance** for two agents guessing independently from their private clues (each picks between its 2 clue-consistent stags; they coincide on the correct one 1/4 of the time). Mean eval return 0.93 is below the trivial hare payoff of 1.0.

Interpretation: the "movement-carried coordination" we celebrated is likely just *meet-and-pounce-together* — synchronization was solved, but **zero bits about the target are being transferred, symbolically or positionally**. The old logs can't separate joint-wrong-stag from solo window expiry, so this was invisible until now; a targeting-accuracy diagnostic is added as of today (see below).

### Expected return for collaboration (analytic)

With 4 stags (2 colors × 2 regions), stag = 4, clues jointly identify the target:

| Information exchanged | P(both at correct stag) | EV of stag attempt | vs hare = 2 | vs hare = 1 |
| --- | --- | --- | --- | --- |
| none (independent guessing) | 1/4 | 1.0 | loses | ties |
| one clue (unidirectional) | 1/2 | 2.0 | ties | wins |
| both clues (bidirectional) | 1 | 4.0 | wins | wins |

(Ignores failure risk and travel time — both only make stag attempts worse.)

Consequences:
- Quantitatively explains the observed phase boundary (2/3 seeds cooperate at hare = 1.0, 0/3 at hare = 2.0): at hare = 1.0 even zero-communication pouncing ties the hare, so cooperation bootstraps *without* language — matching the chance-level accuracy above. At hare = 2.0, cooperation only pays once communication is **bidirectional**, but no gradient reaches the channel before cooperation pays. Chicken-and-egg.
- At default payoffs, unidirectional communication only *ties* the hare. For language to be strictly load-bearing: raise stag relative to hare, or enlarge the target space (8 stags drop the guessing EV to 0.5).

### Evaluation of the five design questions (discussed 2026-07-26)

1. **Channel too small (vocab 4, length 1)?** Not the current bottleneck: task needs 1 bit per direction, channel offers ~2 bits/step × 30 steps, and MI ≈ 0.001 shows the first bit was never used. Decision: scale the channel with task information content *later* (more stags / timing), not preemptively.
2. **Expected return for collaboration** — computed above; EV reference lines to be added to plots.
3. **Remove hares early** — accepted, best idea of the set. Phase 1 becomes a pure cooperative referential game (no defect option, no absorbing hare state); reintroduce hares later as an annealed phase 2. Note: hare is not "competition" (not zero-sum), it is an absorbing defect action that also destroys the partner's pending attempt.
4. **Timing component (private hunt window)** — good pressure for state-dependent messages, deferred until a symbolic channel is causally active at all (phase 3+).
5. **Better than commit window?** Yes for phase 1: **presence-based capture** (both agents standing on the stag cell captures it; no INTERACT/tick synchronization at all) plus **non-terminal wrong attempts**. Terminal failure was the deepest sampling bottleneck all along. Trial-and-error bypass is bounded: ~5–9 steps per attempt on the 7×7 grid, horizon 30 → ~3 attempts, and with hidden positions agents can't even coordinate the search order. The commit window remains the right mechanic for phase 2 where "waiting at the stag" is a meaningful risky signal.

### Phase plan

- **Phase 1 (now):** no hares, presence capture, 2×2 stags, with vs without `observe_other_position`. With positions visible, leader–follower solves it without symbols (correct stag is always in the leader's candidate pair → guaranteed within horizon) — that arm is the control. With positions hidden, the channel is the only way to beat chance targeting. Readout: first-presence targeting accuracy (chance 25% / unidirectional 50% / bidirectional 100%) and MI(clue; message), plus mute + random-message interventions ([[Stag Hunt Language Emergence - Experiment Design]] contract: never trust the mute control alone — input-distribution-shift false positive seen in `cw3s_hare10_seed1`).
- **Phase 2:** reintroduce hares with annealed hare payoff (0 → 1 → 2) or warm-start from a cooperative phase-1 checkpoint; does the protocol survive risk-dominance?
- **Phase 3:** more attributes/stags (channel scaled to match), timing window with asymmetric private information.

### Code changes (2026-07-26)

- `EnvConfig.capture_mode = "interact" | "presence"`; `n_hares = 0` now allowed (hare payoff-ordering check waived when hare-free); presence mode requires `commit_window = 0`.
- Presence resolution: solo on hare captures it; both agents on correct stag = terminal joint capture; both on wrong stag = **non-terminal** (`failed_stag_reward` per step, 0 by default). INTERACT is a no-op (action space unchanged for model compatibility).
- New diagnostics threaded through env → trainer metrics → `analyze_language.py`: `first_joint_presence` ("correct"/"wrong"), `wrong_presence_steps`, per-update and eval-time `first_presence_accuracy`.
- New trainer flags: `--n-hares`, `--capture-mode`, `--hide-other-position`, `--horizon`.
- Tests: 27 passing, incl. presence semantics, hare-free shapes, PettingZoo API in presence mode.

### Runs launched

Phase-1 pilots (queued sequentially, single-CPU node): `phase1_{obs,blind}_seed{0,1}` — no hares, presence capture, batch 64, 800 updates, message entropy 0.05, PPO, no shaping and no curriculum (the point is that the simplified game shouldn't need them). Artifacts in `stag-hunt-files/`. Early sanity check: a random policy samples joint captures at ~2–3% of episodes (vs 0.03% in the old terminal-INTERACT game, ~100×), first-presence accuracy at chance before learning — the redesign unblocks sampling as predicted.

Same four runs also submitted as a Slurm job array (`sbatch_phase1.sh`, job 35380392, partition `kempner_h100`, account `kempner_bsabatini_lab`, `--array=0-3%4`), outputs `phase1_{obs,blind}_seed{0,1}_h100` — sbatch works fine from the interactive compute node, and Slurm jobs survive the session, unlike the detached queues that have died twice before. The interactive MIG copies were killed once the array was confirmed running; all four array tasks completed in ~38 min each.

### Phase-1 pilot results (same day, 800 updates × 64 episodes, eval 500 episodes)

| run | joint stag (eval) | first-presence accuracy | MI(clue; msg) | random-msg control |
| --- | --- | --- | --- | --- |
| obs seed 0 | 28.6% | **64%** | ~0.001 | no effect (63%) |
| obs seed 1 | 21.2% | **46%** | ~0.002 | no effect (46%) |
| blind seed 0 | 3.8% | 26.6% (chance) | ~0.0002 | no effect |
| blind seed 1 | 7.4% | 26.4% (chance) | ~0.0004 | no effect |

Readings:
- **The phase-1 game works as designed.** Both obs seeds learn cooperation with zero shaping/curriculum (the old setup needed a commit window + two anneals + batch 128 and still failed 1/3 of seeds at hare=1.0). Sampling at random init is ~2–3% joint captures vs 0.03% before.
- **Obs arm = movement protocol, quantified.** Accuracy 46–64% sits at/above the unidirectional (one-clue) signature of 50%, far above the 25% chance floor — the first demonstrated *information transfer* in this project, carried by leader-following. The symbolic channel stays causally inert (MI ≈ 0.001, random-message control null; seed 0's mute effect is the known input-distribution-shift artifact — random control is the decisive one).
- **Blind arm = clean negative at 800 updates.** Both seeds at chance accuracy; they learn to *meet* more (joint rate 4–7% vs 2% at init — synchronization without targeting) but no bits flow, symbolic or otherwise. Still rising slowly at update 800.
- Accuracy in the obs arm was still climbing at the end → longer runs may approach the bidirectional 100% regime; the blind arm likely needs substantially longer training and/or stronger channel pressure (higher message entropy for longer, larger batch, or warm-start from an obs checkpoint).

### Why the channel doesn't bootstrap: credit dilution (evening discussion)

Silence rate is flat at 18–22% in all four runs — exactly the 1/5 of a uniform 5-symbol head. The message policy never leaves its initialization: alive (entropy bonus) but shapeless. Three dilution mechanisms explain it:

1. **Temporal**: a useful symbol at t≈2 pays off at capture t≈20+; its credit competes with the variance of all ~60 intervening decisions.
2. **Repetition**: a symbol is emitted every step, so an episode's "utterance" is a ~30-symbol sequence (5³⁰ protocol space) and gradient smears over all of them. A Lewis game has 1 symbol, 1 response, immediate reward — we embedded that game in a sequential problem and divided per-symbol credit by ~30.
3. **Two-sided deadlock**: speaker gradient is zero until the listener attends to messages; the listener has no reason to attend to uniform noise. Escape requires amplifying a triple coincidence (informative symbol + correct reaction + capture) — and only ~5% of blind episodes end in reward at all.

The obs arm avoids all three because movement is a grounded, continuously-reinforced signal — the act *is* the message.

Agreed levers (2026-07-26, user-selected options 1–3 of 4): sticky messages, talk-then-hunt phase, positive signaling/listening biases (Eccles et al. 2019 style); brute-force longer training as the control arm.

Deferred ideas from user, revisit after the lever runs:
- **Uncertainty-gated speaking** — agents talk only when internally uncertain; makes silence informative, but needs a calibrated uncertainty signal (phase 3).
- **Message latency/decay** — a received symbol persists for a bounded window (e.g. 3–4 steps) instead of one step or forever: the natural sweep axis between the current channel and sticky, and re-enables meaningful message *sequences* once content gets richer. The cross-episode variant (message survives into later episodes) is misaligned with per-episode target resampling but becomes interesting with population training / per-partner memory.

### Lever implementation (same day)

- `EnvConfig.sticky_messages`: partner's most recent non-silent symbol persists in `received_message` (silence no longer overwrites) — collapses the protocol space toward one persistent utterance and removes the listener's memory burden.
- `EnvConfig.talk_phase_steps=K`: first K steps freeze movement and capture resolution; only messages flow. Concentrates message credit at episode start.
- Trainer `--signaling-coef`: differentiable batch estimate of MI(private clue; message policy) per speaker role, added to the objective; `--listening-coef`: L1 divergence between the move policy given real vs muted messages (extra BPTT forward), pushing listeners to attend. Both logged per update (`signaling_mi_bits`, `listening_l1`).
- 31 tests passing.

### Methodological catch #2: the rendezvous-convention loophole (2026-07-26, late)

First-presence accuracy above 25% — even near 100% — does **not** by itself prove communication in the blind arm. Because the two clue-consistent candidate pairs intersect exactly at the correct stag, agents can learn complementary *rendezvous conventions* (e.g. one camps at a fixed color-indexed candidate in its region, the other sweeps its color pair): every joint presence is then automatically correct, no bits exchanged, up to ~50% capture rate with a dead channel. The 25/50/100 interpretation grid only applies to *camping* strategies. Verdicts must come from the intervention probes (random-message control especially — conventions are untouched by it, real protocols collapse) plus MI. Wave-1 live numbers (long: acc 40–60%; sticky seed 1: acc 92%, joint 45%) are consistent with either explanation until those probes run.

### Runs launched: `stag-levers` array (job 35404344)

Blind phase-1 game, 3000 updates × 64 episodes, 2 seeds per arm, `--array=0-7%4`: `phase1_blind_{long,sticky,talk5,aux}_seed{0,1}`. `long` is the pure sample-count control; `aux` uses signaling 0.1 + listening 0.1. Key comparison: if sticky/talk5/aux beat `long` on first-presence accuracy, the bottleneck is credit structure, not sample count.
