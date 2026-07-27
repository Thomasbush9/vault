# Stag Hunt Language Emergence — Experiment Design

Related: [[Language Emergence with Stag Hunt Game]]

Implementation: [[Stag Hunt Language Emergence - Repository Setup]]

Episode and agents: [[Stag Hunt Language Emergence - Episode and Agent Architecture]]

## Assessment

The Stag Hunt is a promising basis for studying language emergence because communication can play an instrumental role in choosing between a safe individual strategy and a risky but more rewarding cooperative strategy.

The current setup does not yet force agents to communicate, however. Agents might coordinate through shared observations, their visible movements, parameter sharing, or a fixed policy such as “always go to the stag.” Successful cooperation would therefore not, by itself, demonstrate language emergence.

The experiment should first establish that messages are causally necessary for coordination. Questions about dialogue, compositionality, populations, and language evolution can then be added as separate experimental stages.

## Primary research question

> Does emergent discrete communication help agents select the payoff-dominant cooperative equilibrium as the risk of failed coordination increases?

A useful prediction is:

> Communication will matter most at intermediate risk. At low risk, agents may cooperate without much reassurance; at very high risk, exploration may collapse into the safe equilibrium.

## Minimal environment

Use a small partially observable grid world:

- Two agents.
- A short, fixed episode horizon.
- One or more stags and hares.
- A stag can be captured only when both agents execute `capture` at the same target within a narrow time window.
- A hare can be captured individually.
- Capturing a reward ends the episode, or the horizon is tight enough that an agent cannot capture a hare and subsequently help capture the stag.
- Start with an open grid. Walls should be introduced only after the basic communication result works.

### Payoffs

Give each agent an individual payoff rather than only a single team reward. A simple matrix is:

| Agent A / Agent B | Stag | Hare |
|---|---:|---:|
| **Stag** | 4, 4 | 0, 2 |
| **Hare** | 2, 0 | 2, 2 |

The exact values can change, but they should preserve:

$$
R_{\text{both stag}} > R_{\text{hare}} > R_{\text{failed stag}}
$$

Risk can be manipulated through:

- The payoff for failed stag hunting.
- The reward for successful stag hunting.
- Uncertainty about whether the partner will cooperate.
- The timing tolerance for joint capture.

A single shared team reward would still define a useful cooperative-control problem, but individual payoffs better preserve the equilibrium-selection structure of the Stag Hunt.

## Make communication necessary

The agents should receive complementary private information. For example, targets could be defined by `colour × region`:

- Agent A observes which colour is valuable.
- Agent B observes which region is safe.
- Neither agent can identify the correct stag alone.
- Both must communicate their private information and converge on the same target.

Without private information, agents can learn to walk directly toward the stag. If only one agent has relevant information, the task requires a sender and receiver but not genuinely two-way communication.

Physical movements also form a communication channel. Agents may point, guide, hesitate, or reveal their intentions through their trajectories. This can either be treated as an interesting form of embodied communication or controlled by hiding movement until commitment. It should not be silently conflated with use of the symbolic channel.

## Communication versus dialogue

Two agents emitting messages is not necessarily dialogue. If both agents simultaneously state their private information once, the result is two-way communication without sequential conversational dependence.

To require functional dialogue, introduce a process such as:

1. Agent A proposes a target.
2. Agent B privately inspects the relevant route or resource.
3. Agent B confirms or rejects the proposal.
4. Agent A maintains or revises the plan.
5. Both agents commit to an action.

Evidence for dialogue would require showing that:

- Replies depend on earlier messages, controlling for the replying agent's own observation.
- Later proposals change after rejection or correction.
- Removing, replacing, or reordering a conversational turn reduces performance.

For the first experiment, “emergent two-way communication” is probably the more defensible description.

## Communication channel

- Use a small vocabulary of discrete symbols.
- Allow one sampled token per communication step, or a short bounded sequence.
- Use the same discrete channel at training and evaluation.
- Avoid giving the receiver continuous logits that can carry much more information than the visible discrete message.
- Optionally assign a small message cost to test communicative efficiency.
- Treat vocabulary size and maximum message length as experimental controls.

A large vocabulary allows agents to memorize a separate symbol for every complete state. If compositionality is of interest, communication bandwidth should be restricted relative to the size of the meaning space.

## Training

Centralized training with decentralized execution is a reasonable starting point. A centralized critic may see global training state, but each actor must receive only its legitimate local observation, history, and incoming messages during execution.

Parameter sharing can make coordination artificially easy. Initially it may reduce implementation difficulty, but the final analysis should include either separate agent networks or evaluation with unseen partners.

Temporally extended multi-agent reinforcement learning has a difficult joint-exploration problem. A curriculum could begin with:

1. Easy joint capture and low failure cost.
2. Increased information asymmetry.
3. Narrower timing tolerance.
4. Increased risk.
5. Reduced communication bandwidth.

Final evaluation should use the unshaped target environment.

Positive-signalling or positive-listening auxiliary objectives can be useful if agents ignore the channel, but they should be reported as a separate training condition because they explicitly encourage communication.

## Essential controls and ablations

A performance improvement when a communication channel is available is not sufficient evidence that agents use it. Evaluate each trained policy under:

| Condition | Purpose |
|---|---|
| Normal learned messages | Main treatment |
| No communication channel | Learning baseline |
| Muted or constant messages | Tests whether trained agents require variable messages |
| Messages shuffled between episodes | Preserves message frequency while destroying contextual meaning |
| Counterfactually replaced tokens | Tests whether a particular message changes receiver behavior |
| One-way communication | Tests whether reciprocal information transfer is necessary |
| Two-way communication | Main reciprocal condition |
| Oracle information | Upper bound and environment sanity check |
| No symbolic channel, visible movement | Measures non-verbal coordination |
| Hidden movement until commitment | Isolates the symbolic channel |

The strongest evidence is an intervention:

> Holding the environment state fixed, replacing a message systematically changes the receiver's destination or commitment and changes expected reward.

This distinguishes two different phenomena:

- **Positive signalling:** messages contain information about the sender's observation or behavior.
- **Positive listening:** the receiver causally changes behavior in response to messages.

Mutual information alone mostly establishes signalling, not listening.

## Measurements

### Task behavior

- Mean individual return.
- Mean joint return.
- Stag-selection rate.
- Successful joint-capture rate.
- Failed-coordination rate.
- Hare-selection rate.
- Time to agreement or capture.
- Payoff asymmetry or fairness, if agents have different incentives.

### Causal communication

- Reward decrease when messages are muted.
- Reward decrease when messages are shuffled.
- Change in receiver action distribution under token intervention.
- Mutual information between messages and private observations.
- Mutual information between messages and the partner's later actions, conditioned on public state.
- Vocabulary usage and message entropy.
- Performance as a function of available communication rounds.

### Linguistic structure

Compositionality is meaningful only if the underlying meaning space is factorized, such as `colour × region × prey type`.

Possible measurements include:

- Accuracy on held-out attribute combinations.
- Topographic similarity.
- Positional disentanglement.
- Bag-of-symbols disentanglement.
- Token-to-feature decoding.
- Synonymy and polysemy.
- Success on unseen maps.
- Success with unseen partners.

Task generalization and compositionality should be measured separately. A protocol can generalize without being compositionally organized.

## Population and evolutionary questions

Training multiple agents does not by itself model language evolution. An evolutionary or cultural-transmission claim requires mechanisms such as:

- Population turnover.
- Selection and reproduction.
- Iterated learning between generations.
- A transmission bottleneck.
- New learners acquiring conventions from trained agents.

A relevant follow-up experiment would be:

1. Train a communication protocol in the Stag Hunt environment.
2. Measure its compositionality and causal usefulness.
3. Freeze one trained agent.
4. Replace its partner with a fresh learner.
5. Measure the fresh learner's acquisition speed.
6. Repeat with several independently emerged protocols.
7. Test whether more compositional protocols are learned faster and survive partner replacement better.

This directly connects the experiment with findings that compositionality is not necessary for generalization but may improve transmission to new learners.

Population evaluation should also include cross-play: pair agents that did not train together and test whether their protocols form population-level conventions rather than private codes between fixed partners.

## Important confounds

- **Movement as communication:** trajectories can reveal intent without symbols.
- **Parameter sharing:** identical policies may coordinate more easily than independent agents.
- **Shared rewards:** these can remove or alter the individual strategic incentives of the Stag Hunt.
- **Oversized bandwidth:** agents may transmit arbitrary state encodings rather than develop reusable structure.
- **Centralized information leakage:** global information must not enter decentralized execution policies.
- **Cheap memorization:** a small meaning space permits holistic one-symbol-per-state protocols.
- **Ceiling effects:** if stag hunting is too easy, communication adds nothing.
- **Floor effects:** if it is too risky, agents may never explore cooperation.
- **Seed variance:** different runs may converge on different equilibria and protocols.
- **Best-run reporting:** results should include all seeds and uncertainty intervals.
- **Learning versus evolution:** within-lifetime optimization and intergenerational transmission are different processes.

## Minimal experiment plan

### Stage 1 — Environment validation

- Implement the open-grid environment.
- Verify that an oracle-information policy solves the task.
- Establish a genuine performance gap between oracle and no-communication agents.
- Tune rewards so neither stag nor hare is an overwhelming default.

### Stage 2 — Core result

- Train communication-on and communication-off agents.
- Use three levels of coordination risk.
- Run at least 10 random seeds per central condition if compute permits.
- Report distributions and confidence intervals rather than only the best run.

### Stage 3 — Causal evaluation

- Mute messages.
- Shuffle messages.
- Replace individual tokens counterfactually.
- Measure resulting changes in actions, destinations, and rewards.
- Decode which private variables the messages represent.

### Stage 4 — Reciprocal communication

- Compare no communication, one-way communication, and two-way communication under complementary private information.
- Do not claim dialogue unless later messages conditionally depend on earlier messages and turn-level interventions matter.

### Stage 5 — Generalization and transmission

- Test held-out maps and attribute combinations.
- Test unseen agent pairings through cross-play.
- Replace one agent with a fresh learner and measure protocol acquisition speed.

## Suggested initial experimental matrix

For a compact first study:

| Risk | No communication | Two-way communication | Oracle information |
|---|---:|---:|---:|
| Low | ✓ | ✓ | optional |
| Medium | ✓ | ✓ | ✓ |
| High | ✓ | ✓ | optional |

Run causal message ablations on the trained communicating agents. If two-way communication succeeds, add a one-way condition specifically at the medium-risk setting rather than immediately multiplying every condition.

## Relevant work

- [Compositionality and Generalization In Emergent Languages](https://aclanthology.org/2020.acl-main.407/) — separates generalization, compositionality, and ease of transmission.
- [Emergent Language Generalization and Acquisition Speed are not tied to Compositionality](https://aclanthology.org/2020.blackboxnlp-1.2/) — cautions against assuming compositionality necessarily provides particular functional benefits.
- [On the Pitfalls of Measuring Emergent Communication](https://arxiv.org/abs/1903.05168) — explains why signalling correlations and increased reward do not by themselves establish causal communication.
- [Biases for Emergent Communication in Multi-agent Reinforcement Learning](https://arxiv.org/abs/1912.05676) — introduces positive-signalling and positive-listening biases for difficult temporally extended tasks.
- [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://ojs.aaai.org/index.php/AAAI/article/view/11492) — demonstrates grounded symbolic and non-verbal communication in an embodied multi-agent environment.
- [Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning](https://proceedings.mlr.press/v97/jaques19a.html) — studies counterfactual influence over other agents as a coordination and communication signal.
- [Emergent Communication through Negotiation](https://arxiv.org/abs/1804.03980) — examines cheap talk, cooperation, and prosocial versus self-interested incentives.

## Bottom line

Keep the Stag Hunt, but add complementary private information, formalize the individual payoff structure, prevent alternative coordination shortcuts, and treat causal message interventions as part of the main experiment rather than an optional analysis.

The clean progression is:

$$
\text{cooperation} \rightarrow \text{causal communication} \rightarrow \text{two-way exchange} \rightarrow \text{dialogue} \rightarrow \text{transmission and evolution}
$$
