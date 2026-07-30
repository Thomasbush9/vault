Synthesis of the stag-hunt language-emergence programme, written 2026-07-29 as a
thinking document. The chronological record is in [[Stag Hunt Language Emergence - Lab Log]];
this page is the compressed "what we tried, what we know, what to do next".

---

# 1. The game as it currently stands

7×7 gridworld, 2 agents, PettingZoo `ParallelEnv`, MAPPO with per-agent GRU
actors and centralised critics.

- **8 stags**, each with a public (colour, region) signature — 4 colours × 2 regions.
- The target is defined by a (colour, region) pair. **Each agent privately holds
  one attribute** (which one is randomised per episode), so neither can identify
  the target alone. Agents cannot see each other (`--hide-other-position`).
- **Capture = both agents standing on the correct stag** (`capture_mode=presence`),
  worth 4.0. Wrong-stag joint presence is non-terminal and free. No hares.
- **Channel**: one symbol per agent per timestep from a vocabulary of 4 plus
  silence, delivered to the partner on the next step. Horizon 30.
- Chance targeting is 12.5%; one clue gives 50% (colour) or 25% (region); both
  clues give 100%.

# 2. Everything we tried, and what happened

| # | Intervention | Outcome |
| --- | --- | --- |
| 1 | Risk curriculum on `failed_stag_reward` | 0/5 seeds cooperate |
| 2 | Joint proximity shaping | No gain; PPO ≡ REINFORCE |
| 3 | `commit_window` (arm-and-join) | Sampling ×6, no amplification |
| 4 | Slow anneal + batch 128 | Cooperation at hare=1.0 only — later shown to be **chance-level targeting**, i.e. meet-and-pounce, zero information transfer |
| 5 | **Phase-1 redesign**: no hares, presence capture | Obs arm cooperates (46–64% targeting) but coordination is carried by *seeing each other*, not symbols |
| 6 | Anti-dilution levers: sticky messages, talk-then-hunt phase | Faster **rendezvous conventions**, channel still causally dead |
| 7 | **Eccles-style aux biases** (signaling MI + listening L1) | **First causal symbolic communication** (MI ≈ 1 bit both directions, interventions collapse it) — but scaffolded |
| 8 | Decoupled agents + randomised clue roles | Emergence survives true two-mind separation, 3/3 seeds; two independent nets converge on one lexicon |
| 9 | Deadlock decomposition (signaling-only vs listening-only) | Deadlock is **asymmetric**: signaling-only is the minimal sufficient scaffold (75% captures); **listening bias is harmful** (obligate attention, mute → 0) |
| 10 | 8× payoff for communication | Flat at chance for 6000 updates — the failure is gradient-structural, not economic |
| 11 | Message-entropy sweep (0.05 vs 0.01) | 0.05 kills the code in 2/3 seeds; **0.01 is the standard** |
| 12 | Brute time (40k updates) vs **reward density** (batch 128) | Density wins decisively: batch 128 splits the frozen colour pair in 3/3 seeds, 86% captures; batch 64 never escapes even with more episodes |
| 13 | **Bias anneal** from a converged checkpoint | Zero decay over 4000 bias-free updates → the *scaffolded* language is self-sustaining |
| 14 | **Bias-free 2×2**: private co-observation × tied production/comprehension embeddings | **Complete null**, MI = 0 in 12/12 runs. Diagnosis: targeting accuracy is flat whether neither, one, or BOTH agents see the target |
| 15 | **Shared fog** (co-observation drawn per episode = common knowledge) | Near-null. Common-knowledge hypothesis **falsified**; and combined with #16 it *weakens* the code, because free information removes the need to talk |
| 16 | **Solo scouting reward** (0.5 for standing alone on the correct stag) | **First bias-free causal communication**: 43–87% captures, 74–84% of them destroyed by the distribution-matched swap test, 3/3 seeds |
| 17 | Solo-reward anneal (running, array 36240683) | Pending — is the *bias-free* language also self-sustaining? |

# 3. Honest assessment: has language emerged?

**Signalling: yes. Language: no.** Both halves matter.

What we can defend: there is now **causal, referential information transfer that
emerged from task reward alone**, with no auxiliary loss and no curriculum. The
evidence is the swap test (replay another episode's message stream at matched
statistics): 74–84% of captures disappear when only the *content* is wrong. The
channel is about the target, not about position or timing (MI 0.18 bits with the
held clue vs 0.04 with speaker position, 0.02 with episode phase).

Why it nevertheless isn't language, and why the scepticism is right:

1. **Bandwidth is tiny.** Best-step MI is 0.68 of 1 bit for region and 0.47 of 2
   for colour. Agents transmit ~1–1.5 bits where 3 would fully specify the target.
   Codes are partial, with categories merged (red/blue collapse onto one symbol).
2. **No structure, and the setup cannot show any.** Utterances are a *single
   symbol* per timestep, and the vocabulary (4 + silence) is as large as the
   attribute cardinality (4 colours, 2 regions). So an optimal code is
   holophrastic — one symbol per meaning — and **composition can never pay**.
   We have never measured compositionality because the design makes it
   impossible, not because agents failed at it.
3. **What the biased runs achieved was over-specification, not richness.** The
   2.9-of-3-bits codes came from optimising MI directly; the task only ever needs
   *the complement of what your partner already knows*.
4. Everything achieved so far is a fixed lexicon of ≤5 holophrastic labels. That
   is animal signalling, not language.

So the setup is, at present, **an excellent instrument for studying the
emergence of signalling** — and it has produced a real, defensible result there —
but it is structurally incapable of exhibiting the properties that make language
interesting (productivity, compositionality, syntax).

# 4. What we know that should survive any redesign

**Scientific.**
- The blocker is not payoff size, not channel capacity, not training time.
- The deadlock is asymmetric: comprehension is nearly free given informative
  symbols; **production** is what lacks a gradient.
- Forcing listeners to attend (listening bias) produces *fragile* listeners;
  unforced ones degrade gracefully.
- **Reward density beats training time** for completing a code.
- The real precondition is that **individual competence pays**: in a
  joint-only-payoff game no agent has any reason to act on private information,
  so there is nothing to decode and nothing for a symbol to influence. A small
  solo payoff unlocks everything. (Matches Inoue & Wakabayashi 2025.)
- **Free information is the enemy of language**: making the target observable
  half the time measurably weakened the code.
- Scaffolded language, once formed, is self-sustaining when the scaffold is removed.

**Methodological — the measurement contract.**
- `first_presence_accuracy` is a *screening* metric only. Rendezvous conventions
  reach 93% accuracy with a causally dead channel.
- **Mute and random interventions are unreliable** — they shift the input
  distribution, and a GRU trained on its own babble degrades for reasons
  unrelated to meaning. Five sightings of this artifact. The **swap test**
  (`scripts/probe_shuffle.py`) is the verdict-grade probe.
- **Pooled MI hides brief protocols.** `analyze_language.py` pools over all 30
  timesteps and diluted a real 0.68-bit signal to 0.21. Use
  `scripts/probe_timing.py`.
- Non-zero MI *and* a swap-test collapse together are what license the word
  "communication".
- Shared weights let one network talk to itself; decoupled actors plus
  randomised roles are mandatory.
- Practical: 2 training runs per GPU (contexts time-slice badly), batch 128.

# 5. Options for what to try next

**A. Make the setup capable of language (the structural fix).** Give the
referent space more entries than the vocabulary can name in one symbol, and
allow multi-symbol utterances. Concretely: add a third attribute (say shape) for
4 × 4 × 4 = 64 stags, keep the vocabulary at 4, and let an utterance be 2–3
symbols. Now a holophrastic code is impossible and composition is the only way
to succeed, so we can finally measure topographic similarity and positional
disentanglement. This is the single change that would let the project make a
claim about *language* rather than signalling. Combine with the solo reward,
which we now know is the precondition.

**B. Ritualization (the naturalistic fix, and my favourite for the framing you
like).** Remove the dedicated channel entirely; let signals grow out of
instrumentally useful behaviour — an orientation or gesture visible only within a
radius, initially with its own payoff, plus a cheap stylised variant so the
pressure is toward compression. Metrics come straight from ethology: stereotypy
(entropy collapse of the signal's form), emancipation (does it fire outside its
original context?), iconicity (does the ritualised form still point stag-ward?).
Precedents: Quinn (2001) for the no-dedicated-channel design; Halina, Rossano &
Tomasello (2013) for the truncation-under-anticipation mechanism;
Scott-Phillips, Kirby & Ritchie (2009) for the signalhood problem itself.

**C. Finish characterising what we have.** The solo-reward anneal (running), a
longer solo run to see whether the partial code completes, and a frozen-speaker
transfer to a naive partner (does the code function as a *convention* a new agent
can learn?). Cheap, and it firms up a publishable signalling result.

**D. Multi-round / cumulative episodes.** Respawn a new target after each capture
so reward is dense *within* a trajectory and codes must be reusable across
targets in one recurrent state. Mechanism-backed by the density result; also
kills "follow-me" conventions.

**E. Population / turnover.** Multiple agents with partner mixing, so a code has
to be shared rather than idiosyncratic — the standard route to stable
conventions, and it makes transmission (rather than invention) measurable.

My recommendation, if the goal is language rather than signalling: **A now, B
next**, with C running in the background because it is nearly free. A tells us
whether this environment can produce structure at all; B is the more original
scientific story and the one that fits the naturalistic framing.

# 6. Literature: mechanisms we have not tried (reviewed 2026-07-30)

Three targeted reviews, all citations verified.

## 6.1 Our deadlock is a *known algorithmic* failure, not only a task-design failure

- **Havrylov & Titov (2017), "Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols," NeurIPS 2017.** Head-to-head comparison of REINFORCE versus straight-through Gumbel-Softmax for training a discrete channel: the relaxation converges *much* faster and yields more effective protocols. This is our failure mode in a controlled experiment.
- **Bogin, Geva & Berant (2018), "Emergence of Communication in an Interactive World with Consistent Speakers," NeurIPS EmeCom workshop, arXiv:1809.00549.** States plainly that "training using policy gradient fails" in a harder interactive environment, and diagnoses *speaker inconsistency* — an unstable message→meaning map gives the listener no stable target, so the credit reaching the speaker is noise.
- **Eccles et al. (2019), NeurIPS** — the source of our biases — motivates them as a fix for a **joint exploration problem** in temporally-extended RL. Note for framing: the paper makes **no biological claim** whatsoever.

Reading: much of what we attributed to the world may be attributable to training a discrete channel with RL-only credit assignment.

## 6.2 The fix that fits our architecture exactly

- **Vanneste et al., "Learning to Communicate Using Counterfactual Reasoning" (MACC), arXiv:2006.07200; journal version in *Neural Computing and Applications* (2024), doi:10.1007/s00521-024-10598-0.** Applies a **COMA-style counterfactual baseline specifically to MESSAGE actions**: the speaker's advantage for the message it sent is computed by marginalising over the messages it could have sent, holding the partner fixed. Builds on **Foerster et al. (2018), COMA, AAAI**.
- Why it fits: we already have centralised critics (CTDE) and a tiny vocabulary, so the marginalisation is cheap. It is a pure credit-assignment change — **no information-theoretic objective, no gradients crossing between agents**.
- Alternatives: **DIAL** (Foerster et al. 2016, NeurIPS) and **straight-through Gumbel-Softmax** (Jang et al. 2017; Maddison et al. 2017) give an exact low-variance gradient but require the listener's loss to backpropagate into the speaker — training-time gradient sharing between agents, which is biologically implausible and a bigger conceptual concession than a counterfactual baseline. **Jaques et al. (2019), "Social Influence as Intrinsic Motivation," ICML** rewards *causally influencing* the partner rather than being statistically informative — closer to what we want than MI, but needs a model of the partner's policy.

## 6.3 What reliably makes a channel necessary

The property common to every robust success: **strict, movement-irresolvable information asymmetry** — no sequence of actions lets an agent perceive what the partner holds. Archetype: **Hanabi** (Bard et al. 2019, *Artificial Intelligence*) — you can see everyone's hand but your own. Our design already has this, which is good.

What we may be missing is that a **50/50 split under-constrains necessity**: each agent's half already biases it toward a subset, so partial, non-communicative solutions exist (our rendezvous conventions). Stronger forcing structures from the literature:
- **Guide/follower**: one agent has the whole target but cannot act, the other can act but knows nothing. See the MADDPG cooperative-communication task (**Lowe et al. 2017, NeurIPS**), **Mul, Bouchacourt & Bruni (2019), arXiv:1908.05135**, and **CoMON (Patel et al., ICCV 2021, arXiv:2110.05769)**.
- **Temporal-exclusivity bottleneck**: the switch riddle in Foerster et al. (2016) — information is available to only one agent at a time and must be *persisted* through the channel.
- **Resnick et al. (2020), "Capacity, Bandwidth, and Compositionality in Emergent Language Learning," AAMAS** — there is a band of capacity/bandwidth that induces structure; too much lets agents cheat. Our vocabulary (4) versus attribute cardinality (4) is exactly the over-provisioned regime.

Also: **Cao et al. (2018), "Emergent Communication through Negotiation," ICLR** — prosocial (shared-reward) agents use an ungrounded channel, self-interested ones do not. And **Kottur et al. (2017), EMNLP** (best short paper) — "natural language does not emerge naturally"; unconstrained agents find effective but degenerate codes.

## 6.4 Is the MI bias an evolutionary mechanism? (direct answer: partly)

The **architecture** — fixed production, learned comprehension — is well grounded:
- **Byrne et al. (2017), *Animal Cognition* 20:755–769** — great-ape gesture *forms* look largely innate, while their deployment is flexible.
- **Guilford & Dawkins (1991), "Receiver psychology and the evolution of animal signals," *Animal Behaviour* 42:1–14** — receiver psychology is a separate selective landscape from signal production.
- Fixed action patterns (Lorenz/Tinbergen); **Skyrms (2010)** — signal meaning is conventional and *arbitrary* by default.

But the **mechanism** is not: nothing in biology optimises mutual information online. Evolution fixes an **arbitrary** display over generations; plasticity sits on the receiver side. So the honest position is that our MI loss models an innate *propensity to signal informatively*, which is close to assuming what we want to explain — and Eccles et al. never claimed otherwise.

**Genuine gap found:** **Evtimova et al. (2018), ICLR, arXiv:1705.10369** froze a *randomly initialised* sender and found performance "significantly lags" joint training — but a frozen random network is not the same as a **fixed, injective, arbitrary code**. Nobody appears to have run the clean condition: hardwire a systematic but arbitrary clue→symbol bijection, never train it, and learn only comprehension. That is cheap for us and would let us claim, without circularity, "given an innate arbitrary display, meaning is learnable from task reward alone."

## 6.5 Validation of our own methodology

**Lowe et al. (2019), "On the Pitfalls of Measuring Emergent Communication," AAMAS** independently defines *positive signalling* versus *positive listening* and shows they dissociate — which is exactly the artifact we rediscovered the hard way, and their recommended causal-intervention test is what our swap probe implements. Our measurement contract is citable, not idiosyncratic.

# 6. Open questions worth holding onto

- Does the bias-free code complete (to ~3 bits) with more time, or is ~1 bit an
  equilibrium because it is *task-sufficient*? (The anneal/cont runs speak to this.)
- Why did the colour direction stay weak (0.47 of 2 bits) while region saturated?
  Asymmetric attribute cardinality, or asymmetric value of the information?
- Is the solo payoff a discovery scaffold only (like the signaling bias turned
  out to be), or load-bearing at equilibrium?
- Phase-2 risk (reintroducing hares) has never been tested against a working
  language — does communication survive a tempting safe option?
- Silence is repeatedly grounded as a meaningful word. Is that an artifact of
  vocabulary size, or a genuine finding about zero-cost signals?

---

Related: [[Stag Hunt Language Emergence - Lab Log]], [[Language Emergence with Stag Hunt Game]], [[Stag Hunt Language Emergence - Experiment Design]], [[Stag Hunt Language Emergence - Episode and Agent Architecture]]
