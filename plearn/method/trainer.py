"""RLVB (EM): the trainer (Algorithm 1 in plearn/method/README.md).

A turn at history h: reasoning z (in the thinking block) -> belief line beta (one line) -> action u. The posterior
environment (the --domain module) gives the teacher at every history: the update trace z*, the correct belief line beta*
and the teacher policy pi*, a Boltzmann policy over exact lookahead values that use the agent's measured answer accuracy.
The reasoning is the agent's own: z is a latent variable (EM). At each history the current model writes K thoughts z_k ~ p(z|h);
each is weighted by how well it leads to the teacher's outputs, w_k = p(beta*, u*|h, z_k) / sum_j p(beta*, u*|h, z_j) (the
posterior of z given beta* and u*, by importance sampling from p(z|h)), and with u* ~ pi*
    Loss = - sum_k [ (w_k - 1/K) log p(z_k|h) / |z_k| + w_k log p(beta*|h,z_k) + w_k log p(u*|h,z_k,beta*) ].
This is the EM gradient of log p(beta*, u*|h) with z summed out. The thought term keeps its expectation with the baseline
1/K (the score of the model's own samples has mean zero) and is a per-token mean, so its long sum does not drown the belief
line and the action. The teacher's update trace z* is never trained (it stays in the plan for logs). With K = 1 only beta*
and u* are trained.
The teacher as a critic (--objective critic): no teacher string is a target. After each own thought z_k the model also writes
its own belief line b_k and action a_k; the teacher scores their meaning, R_k = S(b_k; P*) + log pi*(a_k | h): the teacher's
log-likelihood of the turn, where S is the log score of the belief line under the posterior (belief_score, the line read as a
uniform distribution over the set it names, log 0 floored at -200) and pi* the teacher's policy at this history
(action_logp: the Boltzmann over the looks' lookahead values at tau, the probability of answering now, and the stop rule when
the answer is certain; a line the helpers cannot do or the teacher never takes scores as a look that changes nothing, an
answer outside the posterior as a worthless one, both down to LOGP_FLOOR). With alpha = 1, exp(R) = exp(S) pi*(a) is the
teacher's likelihood of the agent's own outputs: the E-step is Bayes' rule with it, q(x|h) ~ p(x|h) exp(R(h, x) / alpha) for
x = (z, b, a), by self-normalised importance sampling from the prior p itself: w_k = softmax_k(R_k / alpha). The M-step is
the projection argmin KL(q || pi):
    Loss = - sum_k w_k [ log p(z_k|h) + log p(b_k|h,z_k) + log p(a_k|h,z_k,b_k) ],   w_k >= 0.
A thought cut at --max_think ends with the </think> that the protocol forces (as on the benchmark): that text is not the
model's own and has no term in log p(z_k|h); a thought that ends by itself keeps its </think>.
--think_cost K1,K2,... (concise thinking as the ability rises): a price of Kn nats per 1,000 thought tokens in
phase n. The samples of one history that the teacher scores alike (the same R: the same belief and action by the teacher's
measure) keep their total weight and share it in proportion to exp(-Kn |z| / 1000), so among the thoughts that reach the same
belief and action the shorter gets more weight; the weight of each belief and action is the E-step's as before (length_tiebreak).
--length_reward C1,C2,... (correct reasoning within the test's thinking budget, not the best of the samples under
the training cap): in phase n the E-step adds Cn times a length term to each own turn's R, the length reward of Kimi k1.5 over the
history's own turns (length_reward): lam = 0.5 - (|z| - min |z|) / (max |z| - min |z|); a correct turn (accepted action, line within
--em_more_joint nats of the exact line) gets lam, so the shorter correct thought gets more; a wrong turn gets min(0, lam), so a short
wrong turn gets no reward and a long one a penalty. The rows log em_q_budget, the budget measure of the target (an offline proxy, not
the success): the E-step weight on correct turns whose thought ends by itself within --eval_max_think tokens (budget_share), and
em_q_correct, the weight on correct turns at any length.
With --em_m M > 1 each thought z_k gets M own decisions (b_km, a_km) and w = softmax over all K M samples: the same q, with
M decisions explored after each thought (a thought's total weight sum_m w_km estimates its likelihood E[exp(R / alpha) | z_k]
from M decisions, not one). --em_j J > 0: the M-step fits J draws from q per history (systematic resampling of w, each draw
with weight 1/J) instead of every sample with its weight; the expected gradient is the same, at a cost of J sequences.
--em_more R: the samples estimate q only where they reach the turns q prefers. At a history where no own turn takes an action
the teacher accepts (pi* gives it less than 1/--accept of the probability of its best action; at a certain answer, every turn
that does not submit that answer), the self-normalised weights lift the best of the excluded turns, and the M-step moves the
model away from the turns q prefers. There the E-step draws K M more own turns from the prior, up to R times. A history that
still has no accepted turn has no estimate of q and is left out of the M-step (the loss keeps the divisor of the full batch,
so the other histories keep their weight).
--think mean: the thought term of the M-step is the mean log-likelihood per thought token; the belief line and the action keep
their sum. The terms of KL(q || pi) over (z, b, a) have one minimizer, q, so their weights change only the path of the M-step.
In the own-turn batches the summed thought term (about 1,000 tokens against 50) was 82 to 94 percent of each clipped step,
but two halves of a batch's histories gave thought gradients with no common direction (cosine -0.01 at update 1, 0.11 at
update 25), while the decision gradients had one (0.53, 0.18): the step was mostly sampling noise, and the decision moved
at 26 to 42 percent of the speed it has alone.
--neg_answer W: with --accept_only a history where no own turn takes an action the teacher accepts has
no target and is left out of the M-step, so nothing lowers an answer exactly where every own turn answers too early (8B games:
11.5 of 32 histories per update, median; MiniGrid test answers with all 3 rows possible in 298-299 of 300 plays). With W > 0 each
own turn there that answers (all rejected) is pushed down with weight W / (its number) per history: the bounded unlikelihood
-log(1 - p(a | h, z, b)) of the answer line's tokens (no terminator) after the own thought and belief line, factor p / (1 - p) capped at 1 (loss_neg); the
thought and the belief line get no term, and the step is the decision block's. --set_keep keeps these histories (keep_set).
Logged: neg_items, neg_p (their mean p(a) before the step), loss_neg. METHOD is unchanged (W = 0).
--neg_thought L (push down the thought): the same histories, but the push-down acts on the THOUGHT: at update 1-2
of a --neg_answer run every rejected answer had p >= 1 - 1e-6 after its own thought and belief line (the
thought fixes the answer), so a push on the answer tokens has no gradient there. Each distinct own thought that led to a rejected
answer (neg_rows, at most NEG_MAX) enters the thought block with weight -L / (their number) per history: the mirror of an accepted
history's +1, through the same clipped thought step (its own </think> too: the stop of the thought is part of the early answer). For a
near-certain token the gradient of log p is ~0, so the push acts on the thought's uncertain tokens. The trust region is the phase's
--kl_stop. Logged: neg_thoughts (items per update). METHOD is unchanged (L = 0).
--line_score none: the turn score is log pi*(u | h) alone; the belief line is not scored.
--no_belief_line 1 (24 Nov 2026, the user: "that script only remove the belief reward, but i want the belief line to be completely
removed"; "we suspect that the belief line is noisy"): with --line_score none, the turn has no belief line at all. The prompt has the
action-only rules of the one-line turn (plearn/tasks/domain.py action_only on the domain's system text, the tests' one_line prompt)
and no carried line; each own turn is z </think> a (M actions sampled after each thought), scored by log pi*(a | h) alone, and the M-step
item is that text (turn_item). The teacher, the sample sets and the update are unchanged. Test with the one-line turn (TURN=one_line).
--line_score exact, --carry (v8c): the belief line is written in the environment's belief language (the form of the
teacher's line; read exactly or it gets the floor, so no reader of prose), scored by the log score with floor --line_floor, and
carried into the next turn as the agent's state (the belief is recursive, b_t = update(b_{t-1}, o_t), and the context is the
agent's only state across turns). The carried line is the agent's own earlier estimate, not evidence: the prompt says so and
that the evidence overrules it (plearn/api/episode.py carried; beliefs must be subject to correction), and
the teacher scores each new line against b from the evidence alone, so a line that copies an earlier misread scores low. The floor m ranks the imperfect lines, and the ranking must be right in the large scenes too: with |Theta| ~ 1000
hypotheses the vacuous line scores -log|Theta| ~ -7, and a wrong claim that keeps 40 percent of the mass scores -0.6 m, so under
m = 10 the base model's habit (claim 'normal' for every parcel) scored level with saying nothing and the agent never learned
caution (v8c, 25 updates: wrong claims 0.5 of its lines as in the base, success 0.16 / 0.24 as the old method). With m >> log
|Theta| the E-step ranks exact > vacuous > wrong claim; the vacuous line is the first lesson and the right one, the second
lesson is the factor |B| / |supp| between the vacuous line and a narrower correct line, which m does not touch. A line outside
the language gets the floor (Domain.belief_score), so the safe statement is the vacuous line of the language, not prose.
--line_score decision: the same language, the line scored by the act it implies. Under the log score a line that
leaves out half of the posterior costs about m / 2 = 25 nats and a line wider than the support log(|b| / |supp|), about 1 nat, so the
E-step learns caution first and the action only as a tie-break: in the ablation study the rows with the belief term wrote lines 3.2
to 5.6 times the support (the base model 2.75, the naive row 1.4 to 3.4), with little gain in exact lines, and the success did not
follow the belief. Every decision problem has its own proper scoring rule, the value of the act that a belief implies (Gruenwald and
Dawid 2004; Gneiting and Raftery 2007). Here S_dec(b) = log pi*(a_b | h), a_b the act the teacher takes when the line is its posterior
(the domain's line_belief: its hypotheses that agree with the line, those the evidence excluded included; decision_score): a line
wider than the support implies another look where the answer is known (the stop rule gives such a look LOGP_FLOOR), a narrower one
an early or a wrong answer, each at the cost the teacher's policy gives that act; a line that changes no decision costs nothing.
Both terms of R are then log-likelihoods of acts under pi*. S_dec is proper but not strictly: lines with the same act tie, so R
adds --line_mix times the log score, and the sum of a strictly proper and a proper score is strictly proper (the exact line stays
the one best line). MineSweeper's line_belief keeps the mine sets that agree with the line among those the evidence allows (the
sets the evidence excluded are too many to list), so there a wider line costs only its share of the log score. The validation
reports this check of the line on its probes: decision_regret, the teacher's best act against the act the line
implies (max log pi* - S_dec, 0 for the exact line), and decision_ok, the share with regret 0, beside belief_regret and belief_ok.
--line_wide: the log score is not symmetric. A line that leaves out posterior mass x pays m x, a line k times
wider than the exact line only log k, so where the lines tie under S_dec (MiniGrid, the tabletop blocks, every MineSweeper state)
the E-step preferred the open line to a wrong one: the 4B decision rows wrote 'Possible: any' (or all) on 30 to 38 percent of
their validation turns after the first, the base model on 17 percent (open_line counts it per task). The flag adds the mirror term -m U_b(outside e), U_b the line read as uniform on its
options and e the teacher's exact line (line_wide, the domain's line_excess): both errors pay the same floor m for their share, a
line two times too wide about m / 2 as a line that leaves out half of the posterior, and the best line under the agent's own
uncertainty keeps an option when it is more likely possible than not (not, as before, unless it is about 98 percent sure the
option is impossible). The exact line keeps the score 0, so it is still the one best line; the term needs no new constant, and
every teacher reads it with its own reader. S_dec now prices the wrong confident claims that m was set high for.
--m_step blocks: the M-step objective KL(q || pi) over the own sample (z, b, a) has two blocks of tokens, the thought (~1,000
tokens, its gradient mostly sampling noise: the cosine between two halves of a batch is about 0) and the decision (~40 tokens,
a shared direction: cosine 0.2–0.5). One clipped step on the sum is 82–94 percent thought, and the decision block moves at a
quarter to a third of its own speed (grad_split.py, 2026-10-24); v8d's lines did not move in 14 updates. The blocks option
takes one clipped step on each block (one forward pass, two gradients): the same objective and the same minimizer, each block
at its own pace, the thought still fitted (unlike --think mean, which left the thought unfitted and drifting). The row logs
grad_norm (thought), grad_norm_dec and loss_dec. v8d-blocks s0 at update 25 (2026-10-25): success 0.34, premature 0.50, mean
turns 5.8 against 0.10-0.20 / 0.88-0.98 / 2.5-3.2 under the joint step: the seeking lesson transfers once the decision block
has its own step; the lines improve at the same slot-by-slot pace as before. The decision block's step alone (the thought not
fitted) gave success 0.24 at update 50 and worse reading (2026-10-26): the thought's step stays.
--proposal base: the prior is the base model pi_0 and the K samples come from it, so q = pi_0 exp(R / alpha) / Z, the
optimum of E[R] - alpha KL(pi || pi_0), is a fixed target; where the teacher prefers nothing, q = pi_0 and the model learns
the base model. The policy is the model under its sampler, in the E-step, in the agent's own episodes and at test: the thought
under SAMPLER (temperature 0.6, top-k 20, top-p 0.95, Qwen's thinking-mode recommendation), the belief line and the action under
DECISION (the same temperature, no truncation), so that every legal action keeps a probability above zero and the target can
lift it: under top-p a look has probability exactly zero at a state where the policy submits with 97%, and then no sample and
no weight can reach it (the own-episode batches showed the target with all its mass on a premature submit at such states). The
M-step likelihood is the token probability under the same two samplers (token_logp), so the fixed point is the deployed policy. (Samples
from the truncated sampler with a raw-model likelihood moved the model toward the truncation, KL 0.025 nats per token, a change
R never asked for.) One such step reaches
q at best, which only re-weights what pi_0 proposes. --phase N: policy mirror descent, pi_{n+1} = argmax E_pi[R] - alpha KL(pi || pi_n)
~ pi_n exp(R / alpha): the sampling adapter is refreshed every N updates only, so within a phase the prior pi_n is frozen and the
target is fixed (each dump and log row names its prior_step); across phases the support of good samples grows with the policy.
--phase_set H (--role actor, with --phase): one sample set per phase. Inside phase n the target q_n ~ pi_n exp(R / alpha) is
fixed, so samples from pi_n stay valid for the whole phase. At the phase start the actor plays episodes for H histories (all
their turns are candidates), runs the E-step at all of them, and writes the phase's batches as draws of --B histories from the
set, in passes without replacement. The phase is one mirror-descent iteration with one sample set, as the usual batch form of the
method; with H = 200, B = 32 and phases of 25 updates, each history serves 4 updates. --set_draw is: the batches come by
importance sampling instead, as prioritized replay: history h with p_h = (1 - P) / H + P sig_h / sum(sig), sig_h half the L1
distance between its E-step weights and equal weights, its loss times 1 / (H p_h); the phase's objective is the same in
expectation and its gradient noise lower when sig follows the size of the histories' gradients (in two dumped batches,
the top quarter of the histories held 70-74 % of the signal, a third had nearly equal weights, among them every first turn).
--kl_stop E (a phase must not go far in a wrong direction): the limit of a phase's step. The KL to pi_n keeps the
target q_n near pi_n, but nothing kept the fit near it: the M-step fits a few samples of q_n per history for --phase updates, and
it also fits their noise (in a run of an older trainer on Odd Parcel: after the phase from update 51 all five tasks tested at
every phase end were below their own means over the phase ends, by 3-11 points, z -3.6 together; the next phase raised them
again). After each update the learner measures KL(pi_n || pi_phi) of the next-token distributions, exactly, at every position of
own turns of pi_n (the probe: --kl_probe turns of the phase's first batch, one per thought, spread over its histories; kl_states,
with pi_n put back for a second forward pass, prior_policy) and, above E nats per position, skips the rest of the phase: the model
after that update is the next prior (PPO's target-KL stop of its epochs on one batch, as OpenAI's Spinning Up: the KL over the
batch's own states, the limit 1.5 x the target 0.01; MPO's KL limit on the same M-step). The rows log kl_prior and, at the update
that stops a phase, kl_stop_left (the updates skipped). --kl_log 1 measures and logs kl_prior the same way and never stops a phase
(the training dynamics of a run without the limit, e.g. the ablation study's).
--set_keep L (dynamic sampling, as DAPO's, arXiv 2503.14476): the batches of a phase come only from the histories
of its sample set that give the M-step a gradient, the first L of them in the set's random order (keep_set): the E-step fits the
history and an own turn takes an action the teacher accepts (after the --em_more rounds). DAPO removes the prompts whose group gives
no gradient (every answer right or every answer wrong) and fills its batch with others; here a history with no accepted own turn
already gets no weight (--accept_only), so without the filter it only takes a place in a batch. The filter keeps the histories whose
samples are acceptable but not the teacher's best, where the model can still improve: a first rule that also skipped those (no own
turn with the teacher's best action) left out every MiniGrid history and a third of the Sokoban and FrozenLake ones of a
large model's dumped batches (the model would get better at what it can already do and not at exploring the harder
tasks). A skipped history is not sampled again; in the same dumps 80 percent of the histories had an accepted own turn, so a set of
2L histories holds more than L of them. The set's row logs set_drawn, set_accepted, set_kept and the kept and drawn histories of each
task (set_by_task); the phase's first update row logs set_accept_share.
--proposal policy: the prior is the current model (EM for log E_pi[exp(R / alpha)]), with no pull to pi_0.
Warm-up (--episodes 0): histories of the teacher policy mixed with random evidence actions (--mix). Own episodes
(--episodes 1): histories of the agent's own episodes; the teacher's action replaces the agent's with probability
alpha_k = beta0 * beta_decay**k, and the agent's answers update the answer-accuracy counts. In its own episodes and in
evaluation the agent writes its action freely, as on a benchmark; the environment parses it (parse_action), an answer is
scored by its own text, and an action the environment cannot carry out uses up the decision. The legal strings are the
teacher's: they define pi*, never what the agent may write. Reasoning for acting and
evaluation comes from a vLLM server that reloads the trainer's LoRA after every update. Under torchrun the update is data
parallel: rank 0 runs the environment, the sampler and the evaluation and sends each batch; every rank scores and
backpropagates its share of the histories, and the LoRA gradients are summed over the ranks.
--accept_only (the action must at least be acceptable): the E-step's weights go only to the own turns whose
action the teacher accepts (estep_weights); a turn with a rejected action gets weight 0, whatever its belief line. In the smoke run,
where every line of a history left out part of the posterior, the line's floor gave the weight to a rejected action (a line with
one of three cells beat lines with none). With --em_more_joint D a history is covered only when one own turn has an accepted
action and a line within D nats of the exact line; else more rounds, then the history is left out. --leave_out 0 (one more round,
no leave-out, the best of the 16 thoughts): after the more rounds (--em_more 1: 16 reasonings, 64
turns) the history is fitted all the same, with its accepted turns (estep_weights). A history with no positive target gets no
weight (no_target; an unverified target is never a positive one): no accepted turn at all (em_no_accept), or
accepted turns whose belief lines all have none of the posterior (em_wrong_line: in a measured sample set, a content
error, readable lines that kept the carried line or missed what a picture showed; the more round did not mend them). The E-step
leaves it out of the M-step, whatever --leave_out, and keeps it in the dump.
--readout W (W > 0: on): the belief readout, one linear head on the policy's final hidden state with two outputs, both
functions of the exact posterior and free of any vocabulary: the readiness of the turn, U(b*) = max_y b*(y) (the teacher's stop
quantity), read at the state that writes the belief line's first token (Bernoulli cross-entropy, a proper score), and a
relative score of a written action, read at its last token, whose softmax over the candidates of one history is fitted to pi*
renormalized on them (listwise cross-entropy: a decision is a preference among alternatives, not a grade). The head reads
detached states (no gradient into the LoRA: an attached head gave no gain over three seeds, 2026-10-26) and is fitted on a
buffer of recent updates (--readout_inner Adam steps per update). --act_pick M: at test time M action lines per turn, the one
the head prefers is taken (on two checkpoints the teacher's mass on the pick rose by 0.055 and 0.07 over the first sample).
--em_more_joint D: with --em_more, a history counts as covered only when an own turn has an accepted action and a belief line
within D nats of the exact line (the mode of q is among the samples); otherwise more own turns, then the history is left out.
Several GPUs (torchrun, --vllm_gpu with the same ids): one trainer rank and one vLLM server per GPU; rank 0 plays the episodes,
samples through all servers (each history stays on one server) and sends the batch; each rank fits its share of the histories,
and in the per-block M-step each block's gradient is summed over the ranks before its clipped step, so the update is the
single-GPU update on the whole batch (--B histories; 16 per GPU keeps each rank's load at the single-GPU runs'). The readout
lives on rank 0 and reads rank 0's share. dp_param_spread in the rows checks that the ranks stay identical.
--temper EPS (way 2 of the doc tab "SPMD test and better learning"; replaces --em_more and its leave-out): at each history the
E-step takes the largest step the samples can carry, w^lam ~ exp(lam R / alpha) with lam_h = max {lam in [0, 1] : sum_i w_i^lam
log(N w_i^lam) <= EPS}, the sample estimate of KL(q^lam || pi_k) (at most log N): the weights stay spread over about N e^-EPS
turns. Where the weights at lam = 1 meet the bound, lam_h = 1 and the update is the settled one; a history with no accepted turn
takes a small step (near lam = 0 the policy gradient with the mean score as baseline) instead of none. The rows log em_lambda
(mean lam_h), em_lambda_lt1 (share of histories with lam_h < 1).
--credit share (way 1): the score of an own decision d_i goes to every own thought in proportion to how probable that thought
makes it, w_ij = w_i r_ij, r_ij = pi_k(d_i | h, z_j) / sum_l pi_k(d_i | h, z_l): given the K thoughts, the K M decisions are a sample
of the mixture (1/K) sum_l pi_k(d | h, z_l), and w_ij is the deterministic-mixture (balance heuristic) importance weight of the
pair (z_j, d_i); a decision's total weight w_i is unchanged. pi_k(d_i | h, z_j) is a teacher-forced score of the decision tokens
under the phase prior (the base model, or the LoRA as it was at the first update of the phase: the vLLM adapter), the DECISION
sampler. Only the distinct own decisions with an accepted action are shared (--share_max of them, the highest R); the others keep
their own thought. No teacher text is a target. The rows log credit_moved (the weight moved off the decision's own thought).
--update grpo ("RLVB (GRPO)", the ablation of the update rule): the same sample sets and histories (the
E-step's leave-out rules and --set_keep's filter unchanged, so a history with no acceptable own turn is left out exactly as with
em: it has no turn to push up, and the two runs train on the same histories), the same K M own turns per history, the same scores R (with the length term) and the same
two-block M-step; only the weights change. In place of the E-step's q, each own turn gets its group-relative advantage
(grpo_advantages): R minus the mean of the history's other own turns (leave-one-out, as RLOO), the scores first raised to at least
the group's best minus C and the advantages limited to [-C, C] nats (--grpo_clip, so a turn at the score floor cannot lift the
others), and a turn whose action the teacher rejects never positive (RLVB (EM)'s accept rule). Each weight is the advantage over the
group's size, times one factor per batch (grpo_weights, --grpo_norm 1) that gives every batch the total absolute weight of
--update em (with AdamW and the gradient-norm clip a constant factor would not change the step; the factor evens out how much each
batch counts). The loss -sum w log p(turn) is the policy-gradient surrogate; no value model is learned. The samples of a phase are
fitted about four times while the model moves from the phase prior pi_n that drew them, so each block of an item stops by PPO's
clip rule (ppo_keep, --grpo_eps 0.2): no gradient once its log p per token (over the tokens score() counts, block_counts) is
log(1 - eps) below pi_n's for a negative weight, or log(1 + eps) above for a positive one (pi_n's log p of each item once per
phase, prior_policy). --line_target
masks an unverified line on a positive weight only (a turn pushed down keeps all its tokens). No draws (--em_j) are made; the
E-step weights are still computed, for the logs. The rows log grpo_scale, grpo_adv_abs, grpo_adv_neg and grpo_adv_pos (the shares
of own turns pushed down and up), grpo_adv_zero_hist (groups with no signal), grpo_stop_t and grpo_stop_d (blocks the clip rule
stopped, summed over the ranks) and grpo_items. --temper and --think_cost act on the E-step weights only, so grpo refuses them.
The sample sets do not depend on --update, so PL_SET_FROM may reuse an RLVB (EM) run's first set (tests/method/test_grpo_posterior.py).
--fl_view_value (default 0, the fix): FrozenLake's teacher valued a move by the best walk through known-safe cells to a
cell whose view it then scored as an answer, log c(n) + E[log U'] (plearn/tasks/frozenlake/teacher.py, Belief._view_lookahead), but
the game has no answer: only reaching the goal pays. Once the goal is known (U = 1) a walk to any cell that shows fog was worth
log c(n) + log 1, exactly what reaching the goal in as many moves is worth, and the walk on to the goal was never counted, so the
teacher tied or preferred sightseeing over finishing, and its greedy policy could wander and never reach the goal; with the goal
unknown a view was worth its best guess's accuracy, again without the walk still needed (two looks of one cell at the same distance
tie). Against the exact full-horizon value of the teacher's own greedy continuation (enumerated over the posterior) on 148
training histories, its top action was the best in 57% of them (38% with the goal known, 69% unknown; mean regret 0.047 with it
known). Now a move is worth the return it leads to (Belief.lookahead): with the goal known, reaching it along a safe shortest route;
otherwise an exact depth-d expectimax over the posterior whose leaf is each hypothesis's own shortest walk to its goal. Measured
after the fix on the same 148 histories (the reference recomputed with the fixed teacher): the top action is the best in 90% (100% with the goal known, 84% unknown), mean regret 0.0003, Spearman 0.96 (0.74 before); from the start of 300 training scenes the greedy teacher wins all (as before) in 3.9 actions (6.7), return 0.926 (0.876), and pi* wins 98.5% (85.3%, 13.2% out of time), return 0.801 (0.684); from those 148 histories the greedy teacher now wins 98% with the goal known (87%). A plan costs 2.8 ms (7.3 ms before; one process, warm), plus ~0.07 s once per start cell (walks()). Only FrozenLake changes (the other games' plans are byte-identical); the test-time path does not use the lookahead (plearn evaluate).
--fl_view_value 1 restores the old values, for runs made before the fix (plearn.tasks.FL_VIEW_VALUE). PL_SET_FROM refuses a saved set
of a run with another --fl_view_value (a set.json without it: 1) when the run has FrozenLake (tests/method/test_fl_walk_value.py).
--sk_guess_value (default 0, the fix): Sokoban's teacher (plearn/tasks/sokoban/teacher.py, Belief._guess_lookahead)
valued heading for a target as a guess among the k candidates, u(c(n + L), 1 / k) = log c(n + L) - log k, and an exploring walk
(no push) to a view by its chance of showing the target, the rest again such a guess. The game has no guess: it ends only when the
box is on the target (a win) or after T actions; a box on another cell shows that cell is not the target and the search goes on, so
log k (1.1 nats for 3 candidates, against 0.02 an action) undervalued every plan that leaves candidates to be found later, above
all pushes, whose views on the way and whose chance of landing the box on the target counted for nothing; the teacher preferred
the walk whose view resolves most now, even one that leaves the player on the wrong side of the box (seed 823435504 at t = 2: the
walk UP had log pi* -0.07 and the push LEFT -2.9, the reference 0.59 and 0.90). Against the exact full-horizon value of the
teacher's own greedy play (enumerated over the posterior) on 442 training histories, its top move was the best in 91% (68% while
0.1 <= U < 0.5, mean regret 0.026 there), Spearman 0.85, and 72 pairs of moves were reversed by more than 0.05. Now a move is
worth the return it leads to (Belief.lookahead): an exact expectimax over the posterior to the end of the game (after each move:
the box on a cell that may hold the target wins with its chance, else the new view shows the target or excludes its cells; with
the target known, its exact shortest solution; a dead box or the limit: the floor u(c(T), 0)). Not to the run's depth: with depth
2 and each candidate's solution length at the leaf the greedy teacher stood still where a view or a push had to come first (it won
88% from start states, pi* 86%). Measured after the fix on the same 442 histories (the reference recomputed with the fixed
teacher): the top move is the best in 99.5% (98% while 0.1 <= U < 0.5), mean regret 0.0000 (0.026 there before), Spearman 1.00,
no reversed pair, the looks in the accept window 84% (83% before), the gap from the best look to the next level 0.20 nats (0.20);
from the start of 300 training rooms the greedy teacher wins 99.7% (as before) in 6.1 actions (6.5), return 0.883 (0.877), and
pi* wins 99.7% (96.7%), return 0.789 (0.769). The values depend only on (state, possible targets, time) and are shared by an
episode's plans: a room's first plan costs 27 ms on average (median 2.4, p99 361, max 417; before 2.5 / 0.8 / 11 / 11), each later
plan 0.14 ms (0.23); 4.7 ms a plan on average over greedy games (0.6). Only Sokoban changes (the other games' plans are
byte-identical); with the target known the plans are as before (exact solution lengths).
--sk_guess_value 1 restores the old values, for runs made before the fix (plearn.tasks.SK_GUESS_VALUE). PL_SET_FROM refuses a saved
set of a run with another --sk_guess_value (a set.json without it: 1) when the run has Sokoban (tests/method/test_sk_return_value.py).
--plan_past_limit (default 0, the fix): near the end of a game two lookaheads valued plans that end after the game's
action limit. The limits, measured on the engines: MiniGrid offers T = 64 looks (look_strings), then one turn to answer (turns
= T + 1; the engine's own max_steps, 480, never binds); popgym's MineSweeper ends the game on a MOVE that is action 2 n^2 (32 on
easy; no answer after it), not on a FIRE that is action 2 n^2 (popgym tests a FIRE's limit on the time before it), after which the
answer comes on the next turn. MiniGrid (Belief.lookahead) valued every pose a plan reaches, also after look T; MineSweeper
(Belief.lookahead, values_two) valued a move by the best FIRE at any distance and a second reveal at any distance, so at action 29
the teacher moved where its plan could not be finished (spec 970564884: RIGHT, FIRE, then a move as action 32 ended the game with no
answer; the reference value of RIGHT was 0 against 0.107 for FIRE). Now a plan's poses come by look T (MiniGrid) and its FIREs by
action 2 n^2, a move as action 2 n^2 is worth the floor u(c, 0) and values_two's second reveal must fit in the actions left after
the first (its cache is keyed on the time too). Measured on the study's histories (the reference recomputed under each setting):
MineSweeper easy, 239 histories: the 17 at t >= 28 now have the top action best in 82% (59%), mean regret 0.0009 (0.035), Spearman
0.63 (0.01); the others 81% (80%), regret 0.0056 (0.0061); from 300 training boards pi* wins 77.3% (71.5%), return 0.271 (0.258), the
greedy teacher as before (79.7%). MiniGrid S4R3, 302 histories: the 28 at t >= 56 now have the top action best in 100% (46%),
mean regret 0.0000 (0.0053), Spearman 1.00 (-0.20) (spec 835189679 at t = 58: the old teacher turned and walked on, the new one
answers, reference 0.155 against 0.137); the other 274 as before (99%, regret 0.0001). The plan cost is unchanged (MineSweeper
7.3 ms, 7.1 before).
The same omission, also fixed under this switch: TableTop's looks had no time check (an invalid turn counts in t, so after
invalid turns the lookahead valued looks after the budget of n looks), and RoboQuest's worlds (plearn/method/teacher.py
Belief.looks) were planned to the domain's T (5) while a world allows W.T looks (4 for OddParcel and a MarkedMugs of 4): 10 of 238
RoboQuest histories change. The other games' plans are byte-identical; their lookaheads already end at the limit (mastermind,
sokoban, frozenlake, the proworlds teachers, parcels).
--plan_past_limit 1 restores the old values, for runs made before the fix (plearn.tasks.PLAN_PAST_LIMIT). PL_SET_FROM refuses a saved
set of a run with another --plan_past_limit (a set.json without it: 1) when the run has MiniGrid, MineSweeper, TableTop or RoboQuest
(tests/method/test_action_limit.py).
These three lookaheads read the game through the belief's episode, which is the live episode at plan time and the copy that
Domain.freeze makes for the plan (FrozenLake, Sokoban, MiniGrid), so a line's S_dec computed after the episode has
gone on is the one at its history (tests/method/test_fl_walk_value.py, test_sk_return_value.py). Sokoban's value table (Ep._vals)
is shared by the frozen copies (copy.copy): its values depend only on (state, possible targets, time); slim() drops plan["B"],
so it is never pickled into an actor batch.
--tau_look (default 0 = --tau): the temperature of pi*'s Boltzmann among the looks only (estimator.targets, Planner.act_logp,
the waste_lp of Planner.__call__, Planner.implied). Among looks the values differ by about one step of time cost (0.02 in log value,
0.2 nats at tau 0.1), while an answer before the evidence is enough costs about 25 nats and the accept window is 3 nats, so the
score taught "do not answer early" about 100 times more strongly than "choose a good look". With --tau_look 0.02 one wasted step costs
1 nat. The answer-or-look sigmoid (pC), the stop rule when the answer is identified, the Boltzmann among the answers and wrong_lp keep
--tau, so the penalty on a premature answer is unchanged. A domain with look_scale = False keeps --tau (mastermind: its lookahead
separates guesses whose real difference is about a third of a guess); TableTop's looks are all tied, so nothing changes there.
--tau_look_play 0 (default): only the scores of own turns and the accept rule use --tau_look; the teacher's played moves (plan tgt:
--beta_expert) keep --tau, so the histories are distributed as in RLVB (GRPO). 1: tgt uses --tau_look too (a
sharper teacher plays better moves; other histories). The batches hold the own turns' scores, so PL_SET_FROM refuses a saved set with
another --tau_look or --tau_look_play (a set.json without them: 0). Logs: look_gap (nats from the best looks to the next level),
look_accept (the share of the looks accepted) and tau_look_on (the share of scored histories where --tau_look applied); look_stats.
Not in METHOD (tests/method/test_tau_look.py).
--stall_cap C (part 3 of the reward repair; default 0 = off): a stalling step is rejected. A look that is neither one of the
teacher's best looks (the highest log pi* among the looks) nor one with real expected evidence (look_info: the expected drop of the
answer's entropy, above STALL_INFO; each of the five game teachers computes it from its own outcome model, Tabletop from
Belief.outcomes) gets log pi* = min(its own, best - C), best being the highest log pi* of any action; so does an action that is not
legal (waste_lp). A look loses only its small time cost against the best one, so before it was accepted (accepted: best - log 20,
3.0 nats); with C = 3.5 it is rejected, and a group of such own turns gets the teacher's row (--teacher_row). The rule only lowers a
value: the answers, the best looks and the looks with evidence keep theirs, and the teacher's own moves (tgt) are unchanged.
MineSweeper: a cursor move counts only when it heads for the best cell (it is then one of the best looks). Planner(stall_cap),
cap_stalls; plan["stall"] holds the lowered values, and the log stall_rejected is the share of own turns rejected by the cap alone.
PL_SET_FROM refuses a set made with another C (tests/method/test_stall_cap.py).
--stall_play 1 (with --stall_cap): the teacher's own moves (tgt) come from the capped scores too (capped_tgt: a lowered look keeps
exp(new - old) of its probability). At tau 0.1 the teacher's choice among moves that differ by one or two steps of time cost is
soft, so without it the teacher itself takes a move the cap rejects at about a third of its steps in the four navigation games
(offline check: MineSweeper 30%, MiniGrid 37%, FrozenLake 45%, Sokoban 34%; with it 3-5%, and its games are shorter).
"""
from __future__ import annotations
import argparse, contextlib, copy, importlib, inspect, io, json, math, os, random, re, statistics, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from plearn.method import estimator as E
from plearn.api.episode import CARRY_MARK, carried   # the carried line and its rule (turn_prompt; the analysis scripts read CARRY_MARK)

CLOSE, TERM = "\n</think>\n\n", "\n"
SAMPLER = {"T": 0.6, "top_p": 0.95, "top_k": 20, "presence": 0.0}    # the policy's sampler for the thought (Qwen's thinking-mode recommendation), set from the arguments
DECISION = {"T": 0.6, "top_p": 1.0, "top_k": -1, "presence": 0.0}    # the belief line and the action: the same temperature, the full support (every legal action has probability > 0)
# "presence" (--score_presence 1): the sampler's presence penalty PL_PRESENCE_PENALTY, which the learner then has in its score too
# (presence_requests, presence_shift). 0: the score has no penalty, whatever the sampler's penalty is
LOGP_FLOOR = -200.0    # log pi* of an action the teacher's policy excludes (the belief score's miss penalty has the same size)
TIE_SLACK = 44         # sampler_logp: how many values past the k-th its top-k form reads (ties of the k-th value must be among them)
ANCHOR_HINT = (" Before the action, write one line starting with 'Possible:' that lists the (option, transformation) pairs "
               "still consistent with every core seen so far.")


def f2(v): return f"{v:.2f}"


PROC = None            # the model's processor, set when the domain's observations are images
ATTN = {"name": "sdpa", "checked": False, "busy": False}   # --attn: the learner's attention and its one log-prob check (attn_check)
ATTN_CHECK_N = 4       # --attn: the pairs of the first score() call that the check scores twice (the flash attention, then sdpa)
ATTN_TOL = float(os.environ.get("PL_ATTN_TOL", "0.02"))   # --attn: the largest mean |log p difference| per counted token (nats) that passes
MSTEP = {}             # --m_step blocks: the two blocks' losses and gradient norms of the last update, for the log row
CARRY = False          # --carry: the agent's own belief line of its last turn enters its next prompt
EXACT_LINE = False     # --line_score exact: the prompt states the environment's belief language (the domain's BELIEF_HINT)
NO_BELIEF_LINE = False  # --no_belief_line (24 Nov 2026, the user: "i want the belief line to be completely removed"): no belief line at all


def after_line(pre, line):
    """The prefix of the action after the belief line: pre + line + TERM. With --no_belief_line the turn has no belief line: the
    action follows </think> at once (pre)."""
    return pre if NO_BELIEF_LINE else pre + line + TERM


def no_line_prompt(text):
    """--no_belief_line: the prompt neither asks for a belief line (the two-line rules, the 'Possible:' hint) nor carries one."""
    return not any(k in text for k in (CARRY_MARK, "reply with exactly two lines", "Line 1:", "Line 2:", "write one line starting with 'Possible:'"))


def turn_prompt(D, tok, e):
    mod = sys.modules[type(D).__module__]
    if hasattr(D, "system_text"): system = D.system_text(e)            # a domain whose system text depends on the episode
    elif EXACT_LINE and hasattr(mod, "SYSTEM_EXACT"): system = mod.SYSTEM_EXACT    # the two-line protocol with the belief language
    else: system = mod.SYSTEM + (getattr(mod, "BELIEF_HINT", None) if EXACT_LINE else None or getattr(mod, "ANCHOR_HINT", ANCHOR_HINT))
    if NO_BELIEF_LINE:                                                # --no_belief_line: the action-only rules (the
        from plearn.tasks.domain import action_only, system_text      # one-line turn of the tests, plearn/tasks/domain.py one_line)
        system = action_only(system_text(D, e))                       # and no carried line (below)
    if PROC is not None and getattr(D, "images", False):              # observations as images, interleaved with the text
        from plearn.models.sampler import tag_prompt
        note = getattr(e, "note", None) if CARRY and not NO_BELIEF_LINE else None
        content, images = D.user_content(e, note=note) if "note" in inspect.signature(D.user_content).parameters else D.user_content(e)
        text = PROC.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": content}],
                                        tokenize=False, add_generation_prompt=True)
        assert not NO_BELIEF_LINE or no_line_prompt(text), "--no_belief_line: the prompt asks for or carries a belief line"
        return tag_prompt(text, images)
    text = D._history(e)
    if CARRY and not NO_BELIEF_LINE and getattr(e, "note", None):     # the agent's own last belief line, before the closing request
        head, sep, tail = text.rpartition(" Decide now.")
        note = carried(e.note)
        text = head + note + "\nDecide now." + tail if sep else text + note
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)      # ends with "<think>\n"
    assert not NO_BELIEF_LINE or no_line_prompt(out), "--no_belief_line: the prompt asks for or carries a belief line"
    return out


def anchor_answer(anchor, answers, D=None, e=None):
    """The answer a written anchor implies: the option with the most surviving hypotheses, ties to the smallest."""
    if D is not None and hasattr(D, "anchor_answer"): return D.anchor_answer(anchor, answers, e)
    s = E.note_set(anchor)
    if s in (("any",), ("unknown",)) or not s: return answers[0]
    counts = {}
    for item in s:
        a, poses = (item[0], item[1]) if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple) else (item, ())
        a = a if a in answers else (str(a).split() or [""])[-1]
        if a in answers: counts[a] = counts.get(a, 0) + max(1, len(poses))
    if not counts: return answers[0]
    return min(counts, key=lambda a: (-counts[a], answers.index(a)))


def anchor_support(anchor, answers, D=None, e=None):
    """The answers a written anchor still allows."""
    if D is not None and hasattr(D, "anchor_support"): return D.anchor_support(anchor, answers, e)
    s = E.note_set(anchor)
    if s in (("any",), ("unknown",)) or not s: return set(answers)
    out = set()
    for item in s:
        a = item[0] if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple) else item
        a = a if a in answers else (str(a).split() or [""])[-1]
        if a in answers: out.add(a)
    return out



# --------------------------------------------------------------------- monitor metrics, logged on W&B.
# Diagnostics only: they change no update.
def _exact_n(B):
    """|B*|: the number of answers the exact posterior still allows (0 while no answer can be right yet)."""
    if hasattr(B, "answer_counts"): return sum(1 for v in B.answer_counts().values() if v > 1e-12)
    return None


def answered(eps):
    """The episodes that ended with an answer (SUBMIT, or the turn limit, where the belief counts as the answer). A game that ends
    without one (a goal reached, a hole, a mine) has no commit_ident: the answer metrics leave it out."""
    return [e for e in eps if getattr(e, "commit_ident", None) is not None]


def premature_rate(eps):
    """The share of answers given before the evidence identified the target (None when no episode answered)."""
    a = answered(eps); return statistics.fmean(not e.commit_ident for e in a) if a else None


def monitor_eval_metrics(decs, eps, D_eval, line_floor):
    """1-5, 9, 12 on the evaluation episodes (the agent alone)."""
    out = {}; by_id = {id(e): e for e in eps}; t_of = {}; rows = []
    for r in decs:                                                   # decisions come in turn order per episode
        e = by_id.get(r["ep"]); t = t_of.get(r["ep"], 0); t_of[r["ep"]] = t + 1
        B = r["plan"]["B"]; ans = r["plan"]["answers"]
        line_n = len(anchor_support(r["anchor"], ans, D_eval, e))
        prem = bool(r["commit"]) and not r["identified_before"]
        reg = (belief_best(D_eval, B, r["plan"]) - belief_score(D_eval, B, r["anchor"], r["plan"])) if hasattr(D_eval, "belief_score") else None
        rows.append({"t": t, "U": float(B.U()), "ident": bool(B.identified()), "commit": bool(r["commit"]), "prem": prem, "line_one": line_n == 1,
                     "exact": bool(r["anchor_ok"]), "n_star": _exact_n(B), "correct": bool(e.correct) if (e is not None and r["commit"]) else None, "reg": reg})
    if not rows: return out
    fm = lambda xs: float(np.mean(xs)) if len(xs) else None
    for name, lo, hi in (("0_50", 0.0, 0.5), ("50_90", 0.5, 0.9), ("90_100", 0.9, 1 - 1e-9), ("100", 1 - 1e-9, 2.0)):      # 1: the stop rule as a curve
        xs = [x["commit"] for x in rows if lo <= x["U"] < hi]; out[f"submit_hazard_U_{name}"] = fm(xs); out[f"submit_n_U_{name}"] = len(xs)
    pr = [x for x in rows if x["prem"]]; out["premature_submits"] = len(pr)
    out["premature_exact_line"] = fm([x["exact"] for x in pr])                                                # 2: decision failure
    out["premature_overconfident_line"] = fm([x["line_one"] and not x["ident"] for x in pr])                  # 3: reading failure
    out["self_consistency"] = fm([x["commit"] == x["line_one"] for x in rows])                                # 4: the action follows the line
    g = [x for x in pr if x["correct"] is not None and x["n_star"] is not None]                               # 5: early guesses against chance in B*
    out["guess_acc_vs_chance"] = (fm([x["correct"] for x in g]) - fm([1.0 / x["n_star"] if x["n_star"] else 0.0 for x in g])) if g else None
    for t in range(5):                                                                                         # 9: per turn index (4 = 4 and later)
        xs = [x for x in rows if (x["t"] == t if t < 4 else x["t"] >= 4)]
        out[f"premature_t{t}"] = fm([x["prem"] for x in xs]); out[f"decisions_t{t}"] = len(xs)
        rg = [x["reg"] for x in xs if x["reg"] is not None]; out[f"belief_regret_t{t}"] = fm(rg)
    rs = np.random.default_rng(0); n = len(eps)                                                               # 12: bootstrap 95% intervals over the episodes
    cols = {"success": [float(bool(e.correct)) for e in eps], "premature": [float(not e.commit_ident) for e in answered(eps)], "mean_t": [float(e.t) for e in eps]}
    idx = rs.integers(0, n, size=(1000, n))
    for k, v in cols.items():
        if not v: continue
        i = idx if len(v) == n else rs.integers(0, len(v), size=(1000, len(v)))   # premature: over the answered episodes only
        b = np.asarray(v)[i].mean(1); out[f"{k}_ci95_lo"], out[f"{k}_ci95_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return out


def repeat_loop(z):
    """A thought's tail repeats: the same text of 20-400 characters 3 times at its end, or half of its last 40 lines (10 or more) are copies.
    A log metric only (think_repeat_n / think_repeat_of); no loss or sampling change."""
    t = z.rstrip()
    if any(t[-p:] == t[-2 * p:-p] == t[-3 * p:-2 * p] for p in range(20, 401) if len(t) >= 3 * p): return True
    ln = [x.strip() for x in t.splitlines() if x.strip()][-40:]
    return len(ln) >= 10 and len(set(ln)) <= len(ln) / 2


def monitor_train_metrics(crit_h, rows_h, w, fit, decs, eps):
    """6, 7, 8, 10, 11 on one update (rank 0: every history's E-step weights)."""
    out = {}; fm = lambda xs: float(np.mean(xs)) if len(xs) else None
    sp, share, ess_f, coll, kl = [], [], [], [], []
    for h in fit:
        wh = np.asarray(w[h], dtype=np.float64); ch = crit_h[h][:len(wh)] if crit_h else []
        if len(ch):
            R = np.array([c["R"] for c in ch]); S = np.array([c["S"] for c in ch]); A = np.array([c["A"] for c in ch])
            sp.append(float(R.max() - np.median(R)))                                                          # 6: R spread
            vs, va = float(S.var()), float(A.var())
            if vs + va > 0: share.append(vs / (vs + va))                                                       # 6: the share of the line term
        n = len(wh)
        if n:
            ess_f.append(float(1.0 / (wh ** 2).sum()) / n); coll.append(float(wh.max() > 0.9))                # 7: ESS fraction, collapse
            nz = wh > 0; kl.append(float((wh[nz] * np.log(wh[nz] * n)).sum()))                                 # 8: KL(q || pi) on the samples
    out.update({"R_spread": fm(sp), "R_term_share": fm(share), "em_ess_frac": fm(ess_f), "w_collapse": fm(coll), "kl_q_pi": fm(kl)})
    th = [r[6] for rh in rows_h for r in rh if len(r) > 6]
    out["think_truncated"] = fm([not st for st in th])                                                         # 10: own thoughts cut by the length limit
    zs_ = [r[1] for rh in rows_h for r in rh if len(r) > 1 and isinstance(r[1], str)]                           # own thoughts that end in a repeat loop
    out["think_repeat_n"] = sum(repeat_loop(z) for z in zs_); out["think_repeat_of"] = len(zs_)
    if decs is not None and eps:                                                                               # 11: training episodes the teacher never acted in
        helped = {r["ep"] for r in decs if r.get("teacher_acted")}; alone = [e for e in eps if id(e) not in helped]
        out["agent_only_success"] = fm([float(bool(e.correct)) for e in alone]); out["agent_only_n"] = len(alone)
    return out


class Planner:
    """The ideal decision at the current history: exact anchor, action target pi*, and the exact thought."""
    def __init__(self, D, rel, tau, depth, tau_look=0.0, tau_look_play=0, stall_cap=0.0, stall_play=0):
        self.D, self.rel, self.tau, self.depth = D, rel, tau, depth
        self.tau_look, self.tau_look_play = float(tau_look or 0.0), int(tau_look_play or 0)   # --tau_look, --tau_look_play (0: tau)
        self.stall_cap = float(stall_cap or 0.0)                        # --stall_cap (0: off; the training planners only)
        self.stall_play = int(stall_play or 0)                          # --stall_play: the teacher's own moves (tgt) from the capped scores

    def look_tau(self, Dk):
        """The temperature of the Boltzmann among the looks for the teacher Dk: --tau_look when it is set and the domain does not
        opt out (Domain.look_scale = False: mastermind), else --tau."""
        return self.tau_look if self.tau_look > 0 and getattr(Dk, "look_scale", True) else self.tau

    def __call__(self, e):
        D = self.D; B = D.belief(e); Dk = getattr(e, "_teacher", D); T, lam = Dk.T, Dk.lam   # a mixed run: the episode's own teacher
        looks = D.look_strings(e) if B.t < T else {}
        answers = D.answer_options(e); post = D.answer_posterior(e) or {}
        best = None
        if answers:
            if post: best = max(answers, key=lambda y: (post.get(y, 0.0), -answers.index(y)))
            elif B.U() > 0: best = answers[0]
        commits = {y: Dk.commit_string(y) for y in answers}
        depth = D.depth(B) if hasattr(D, "depth") else self.depth
        tl = self.look_tau(Dk)                                          # --tau_look: the scores of own turns; the played moves (tgt) too with --tau_look_play 1
        tgt, Q, pC, qy = E.targets(B, looks, answers, commits, post, self.rel, lam, T, self.tau, depth, tl if self.tau_look_play else None)
        cm = sum(tgt.pop(s, 0.0) for s in commits.values())
        if answers and cm > 0: tgt[commits[best if best is not None else answers[0]]] = cm
        anchor = D.belief_note(e) if hasattr(D, "belief_note") else E.note_text(post, answers)
        r = self.rel.mean(B); c0 = math.exp(-lam * B.t / T)
        qC = self.rel.util(c0, r * (1.0 if post else 1.0 / len(answers))) if answers else None
        lines = [D.state_text(e), f"Best answer now: {best if best is not None else 'none'}; reliability {f2(r)}"]
        parts = ["submit " + (f2(qC) if qC is not None else "not possible")]
        if looks:
            groups = {}
            for k in looks: groups.setdefault(round(Q[k], 2), []).append(k)
            order = sorted(groups, reverse=True)
            for v in order[:2]:
                names = [D.look_name(k, e) if getattr(D, "look_name_takes_episode", False) else D.look_name(k) for k in sorted(groups[v], key=str)]
                more = f" and {len(names) - 4} more" if len(names) > 4 else ""
                parts.append(f"{' / '.join(names[:4])}{more}: {f2(v)}")
            rest = sum(len(groups[v]) for v in order[2:])
            if rest: parts.append(f"{rest} other looks: {f2(order[2])} or less")
        else:
            parts.append("no look possible")
        lines.append("Values: " + "; ".join(parts))
        values = {s: Q[k] for k, s in looks.items()}; values.update({commits[y]: v for y, v in qy.items()})   # the critic's V
        waste = self.rel.util(math.exp(-lam * (B.t + 1) / T), r)   # a decision that changes nothing: answering one step later
        # the critic's action score: log pi*(a | h), the teacher's policy itself (E.targets: the stop rule when the answer is certain,
        # the Boltzmann over the looks' values at tau, the answer probability pC), at the level of a look's value group
        logp, waste_lp, wrong_lp = self.scores(looks, commits, Q, pC, qy, waste, self.rel.util(c0, 0.0), tl)
        stall = None                                                    # --stall_cap: stalling looks and actions that are not legal
        if self.stall_cap > 0: logp, waste_lp, stall = cap_stalls(logp, waste_lp, looks, self.stall_cap, lambda a: look_info(B, a))
        if self.stall_play and stall: tgt = capped_tgt(tgt, logp, stall)   # --stall_play: the teacher plays by the capped scores too
        if hasattr(Dk, "freeze"): B = Dk.freeze(B)                     # the game at this history, for scores made later
        return {"anchor": anchor, "tgt": tgt, "exact": "\n".join(lines), "B": B, "best": best, "post": post, "values": values, "waste": waste,
                "logp": logp, "waste_lp": waste_lp, "wrong_lp": wrong_lp, "tl_on": bool(looks) and tl != self.tau,
                "legal": list(looks.values()) + list(commits.values()), "looks": looks, "commits": commits, "answers": answers, "truth": D.label(e),
                "dom": getattr(e, "_teacher_name", None),   # a mixed run: the episode's teacher by name (plans are pickled)
                **({"stall": stall} if stall is not None else {})}   # --stall_cap: the values it lowered (score_turn's stall log)

    def scores(self, looks, commits, Q, pC, qy, waste, v0, tl=None):
        """The critic's action scores from the values Q of the looks, qy of the answers and pC (the lookahead's): log pi* of every legal string (act_logp), of a decision that
        changes nothing worth `waste` (waste_lp) and of an answer outside the posterior worth v0 (wrong_lp). tl: the temperature of
        the Boltzmann among the looks (look_tau; None: tau); pC, the answers and wrong_lp keep tau."""
        tl = tl or self.tau
        logp, F = self.act_logp(looks, commits, Q, pC, qy, tl), LOGP_FLOOR
        lg = lambda x: math.log(x) if x > 0 else F
        if looks:
            groups = sorted({round(Q[k], 3) for k in looks}); qL = groups[-1]; z = sum(math.exp((g - qL) / tl) for g in groups)
            waste_lp = max(F, min(lg(1 - pC), lg(1 - pC) + (round(waste, 3) - qL) / tl - math.log(z)))   # a log-probability: at
            # most the share of the looks (where every look is worth far less than a decision that changes nothing, as the guesses at a
            # solved Mastermind state, the Boltzmann term alone would give an unreadable line about +1e7, more than any action)
        else:
            waste_lp = F
        if qy:
            qC = max(qy.values()); zy = sum(math.exp((v - qC) / self.tau) for v in qy.values())
            # v0: an answer outside the posterior, accuracy 0 (0 in the accuracy value; its floor in the log value)
            wrong_lp = max(F, min(lg(pC), lg(pC) + (v0 - qC) / self.tau - math.log(zy)))
        else:
            wrong_lp = F
        return logp, waste_lp, wrong_lp

    def act_logp(self, looks, commits, Q, pC, qy, tl=None):
        """log pi*(a | h) of every legal string from the parts of E.targets: the stop rule and the answer probability pC, the
        Boltzmann over the looks' values at tl (look_tau; None: tau) at the level of a look's value group (its group's score), the
        answers' values at tau."""
        logp, F = {}, LOGP_FLOOR; tl = tl or self.tau
        lg = lambda x: math.log(x) if x > 0 else F
        if looks:
            groups = sorted({round(Q[k], 3) for k in looks}); qL = groups[-1]; z = sum(math.exp((g - qL) / tl) for g in groups)
            for k, s in looks.items(): logp[s] = max(F, lg(1 - pC) + (round(Q[k], 3) - qL) / tl - math.log(z))
        if qy:
            qC = max(qy.values()); zy = sum(math.exp((v - qC) / self.tau) for v in qy.values())
            for y, v in qy.items(): logp[commits[y]] = max(F, lg(pC) + (v - qC) / self.tau - math.log(zy))
        return logp

    def implied(self, Dk, plan, Bb):
        """The act the teacher takes when its posterior is Bb, a belief the agent stated (Dk.line_belief): the legal strings of the
        plan with the highest log pi* at Bb (act_logp, at the plan's history and time), ties all kept. Dk: the plan's own teacher."""
        post = Bb.post() if hasattr(Bb, "post") else {y: Dk.answer_prob(Bb, y) for y in plan["answers"]}
        depth = self.D.depth(Bb) if hasattr(self.D, "depth") else self.depth
        _, Q, pC, qy = E.targets(Bb, plan["looks"], plan["answers"], plan["commits"], post, self.rel, Dk.lam, Dk.T, self.tau, depth)
        lp = self.act_logp(plan["looks"], plan["commits"], Q, pC, qy, self.look_tau(Dk))
        if not lp: return []
        top = max(lp.values()); return [s for s, v in lp.items() if v >= top - 1e-9]


def collect_histories(D, tok, nrng, specs, n, planner, mix, keep_eps=None):
    """Histories of the environment under an even mixture of the target policy and random looks, with their plans."""
    out = []
    while len(out) < n:
        e = D.start([next(specs)])[0]
        while len(out) < n:
            plan = planner(e)
            if plan["tgt"]:
                out.append((turn_prompt(D, tok, e), plan))
                if keep_eps is not None: keep_eps.append(copy.copy(e))
            looks = D.look_strings(e) if e.t < getattr(e, "_teacher", D).T else {}
            if not looks: break
            if nrng.random() < mix:
                ks = list(plan["tgt"]); ps = np.array([plan["tgt"][k] for k in ks])
                s = ks[int(nrng.choice(len(ks), p=ps / ps.sum()))]; inv = {v: k for k, v in looks.items()}
                if s not in inv: break                  # the target commits here: this history ends
                D.do_look(e, inv[s])
            else:
                D.do_look(e, list(looks)[int(nrng.integers(len(looks)))])
            e.t += 1
        D.close([e])
    return out


def keep_dir(dump_em):
    """The folder that keeps every trained batch (the rollouts of every RL run):
    <dump_em>/../batches (plearn/train.py always sets --dump_em to <out>/case-<tag>/em), so a run's own argv does not change.
    PL_KEEP_ROLLOUTS=0 or no --dump_em: no folder (the batch is removed, as before)."""
    if os.environ.get("PL_KEEP_ROLLOUTS", "1") == "0" or not dump_em: return ""
    return os.path.join(os.path.dirname(os.path.normpath(dump_em)), "batches")


def keep_batch(src, dump_em, step, log=None):
    """After update `step` (--role learner): move the queue's batch_<step>.pkl (the actor's samples, histories, scores and images of
    that update) to keep_dir() in place of the old delete; one log row with the folder's size, for the disk check. Returns the new
    path, or "" when the batch is removed (no folder) or the move fails (a line, the run goes on)."""
    import shutil
    d = keep_dir(dump_em)
    if not d:
        try: os.remove(src)
        except OSError: pass
        return ""
    dst = os.path.join(d, os.path.basename(src))
    try:
        os.makedirs(d, exist_ok=True); shutil.move(src, dst)
    except OSError as e:
        print(f"KEEP BATCH FAIL update {step}: {e}", flush=True); return ""
    mb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 2**20
    print(f"KEEP BATCH update {step}: {d} {mb:.0f} MiB", flush=True)
    if log is not None: log.write(json.dumps({"kept_batch": step, "kept_dir": d, "kept_mib": round(mb)}) + "\n"); log.flush()
    return dst


def keep_last_saves(out, tag, step, keep, protect):
    """PL_KEEP_LAST (keep the 25 / 50 saves and always the last 2 updates; older ones are deleted as the run
    goes on): with PL_KEEP_LAST = K > 0 the learner saves the full state (adapter + Adam) after every update; after
    update `step` this removes the saves of the older updates, so only the last K updates stay (a kept update among them counts as
    one of the K), besides every update n with protect(n) (the
    --eval_every / PL_SAVE_STEPS / last-update saves, kept as before). Only files of updates <= step are touched (a file of a later
    update is from another attempt of the same run and stays). Returns the removed update numbers, oldest first."""
    pat = re.compile(re.escape(tag) + r"-(lora|opt|readout)-(\d+)\.pt")
    found = {}
    try: names = os.listdir(out)
    except OSError: return []
    for f in names:
        m = pat.fullmatch(f)
        if m and int(m.group(2)) <= step: found.setdefault(int(m.group(2)), []).append(f)
    last = set(sorted(found)[-keep:]) if keep > 0 else set()          # the last `keep` updates, a kept step among them counts too
    gone = sorted(n for n in found if not protect(n) and n not in last)
    for n in gone:
        for f in found[n]:
            try: os.remove(os.path.join(out, f))
            except OSError: pass                                        # another rank of this node removed it first
    return gone


def dump_rollouts(out, step, D, eps, decs):
    """Every evaluation episode in full, for inspection: at each decision the prompt, the agent's thought, belief line and
    action, the legal actions and the teacher at that history (z*, beta*, pi*); the episode's images as files."""
    from plearn.models.sampler import IMAGES, prompt_key, untag
    d = os.path.join(out, f"step{step}"); os.makedirs(d, exist_ok=True)
    by = {}
    for r in decs: by.setdefault(r["ep"], []).append(r)
    for k, e in enumerate(eps):
        rs = by.get(id(e), [])
        for j, im in enumerate(IMAGES.get(prompt_key(rs[-1]["prompt"]), []) if rs else []): im.save(os.path.join(d, f"ep{k}_img{j}.png"))
        steps = [{"prompt": untag(r["prompt"]), "n_images": len(IMAGES.get(prompt_key(r["prompt"]), [])), "thought": r["thought"],
                  "belief_line": r["anchor"], "action": r["action"], "parsed": r.get("look") if r.get("look") is not None else
                  ("answer" if r.get("commit") else "invalid"), "answer": r.get("answer"), "legal": r["plan"]["legal"],
                  "teacher": {"trace": r["plan"]["exact"], "belief_line": r["plan"]["anchor"], "pi": r["plan"]["tgt"]}, "pick": r.get("pick")} for r in rs]
        env = {a: repr(getattr(e, a)) for a in ("log", "angles", "acts", "vm_miss") if hasattr(e, a)}
        json.dump({"spec": e.spec, "truth": D.label(e), "correct": bool(e.correct), "steps": steps, "env": env},
                  open(os.path.join(d, f"ep{k}.json"), "w"), indent=1, default=str)


# --------------------------------------------------------------------- generation and scoring
class ServerSampler:
    """Thoughts and anchors from vLLM servers that reload the trainer's LoRA after every update: one server per GPU of gpu
    (comma-separated ids). A request goes to the server of its history (the chat text before the assistant turn), so the
    thoughts, lines and actions of one history share that server's prefix cache; a new history goes to the server with the fewest
    sequences in flight."""
    def __init__(self, model_path, gpu, temperature, out, lora_r, mem=0.45, images=False, start=True):
        self.temperature, self.dir, self.model_path, self.lora_r = temperature, os.path.join(out, "adapter"), model_path, lora_r
        self._cfg = (model_path, gpu, out, mem, images)
        if start: self._start()
        else: self.servers = []               # PL_SET_FROM: no server until the first phase the actor makes itself (start())

    def start(self):
        """--colocate: start the servers again after the learner's phase (the next adapter is loaded by the caller)."""
        self._start()

    def _start(self):
        from plearn.models.sampler import VLLMSampler, MMSampler
        model_path, gpu, out, mem, images = self._cfg
        S = MMSampler if images else VLLMSampler
        gpus = [g.strip() for g in str(gpu).split(",") if g.strip()]
        k_ = int(os.environ.get("PL_GPUS_PER_SERVER") or 0)                # speed: several smaller servers on this machine, k GPUs each
        if k_: gpus = [",".join(gpus[i:i + k_]) for i in range(0, len(gpus), k_)]
        elif os.environ.get("PL_VLLM_TP") == "1": gpus = [",".join(gpus)]     # speed: one tensor-parallel server over all GPUs
        assert not images or k_ or len(gpus) == 1, "image domains: one local server (PL_VLLM_TP=1: over all the GPUs), or PL_GPUS_PER_SERVER; others in PL_VLLM_REMOTE"
        from plearn.models.sampler import RemoteVLLMSampler, RemoteMMSampler
        far = []                                                          # 2-node run: the other machine's server (a picture server for
        for url in [u for u in os.environ.get("PL_VLLM_REMOTE", "").split(",") if u.strip()]:    # image domains: mm_server); with
            r = (RemoteMMSampler if images else RemoteVLLMSampler)(url.strip())                     # PL_REMOTE_SERVE this process starts it,
            if getattr(r, "ctl", None): r.launch(mem)                                              # before the local one: both load at once
            far.append(r)
        self.servers, todo = [None] * len(gpus), list(range(len(gpus)))
        for attempt in range(3):              # all servers start at once; one that does not come up (its port was taken before it bound
            start = [(i, S(model_path, gpus[i], gpu_memory_utilization=mem, log=os.path.join(out, "vllm.log" if i == 0 else f"vllm-{i}.log")))
                     for i in todo]           # it) starts again on a new port
            todo = []
            for i, s in start:
                try: s.wait(); self.servers[i] = s
                except RuntimeError:
                    s.stop(); todo.append(i); print(f"vLLM server on GPU {gpus[i]} did not start (attempt {attempt + 1})", flush=True)
            if not todo: break
        if todo:
            for s in self.servers:
                if s is not None: s.stop()
            raise RuntimeError(f"vLLM servers on GPUs {[gpus[i] for i in todo]} did not start in 3 attempts (see their vllm logs)")
        for r in far: r.wait(); self.servers.append(r)
        import threading
        self.S, self.route, self.busy, self.lock = self.servers[0], {}, [0] * len(self.servers), threading.Lock()
        w = [float(x) for x in os.environ.get("PL_SERVER_WEIGHTS", "").split(",") if x.strip()]   # speed: the servers'
        self.weight = w if len(w) == len(self.servers) and min(w) > 0 else [1.0] * len(self.servers)   # sizes (local first, then PL_VLLM_REMOTE)
        self.spill = float(os.environ.get("PL_ROUTE_SPILL") or 0)    # speed: see _server_of; 0 = off
        self.moved = 0

    def _server_of(self, p, n=1):
        """The server of a request's history, with n more sequences in flight there. A new history goes to the server with the
        fewest sequences in flight (verl's least-in-flight rule; the servers can differ in size, and histories in work); with
        PL_SERVER_WEIGHTS (one weight per server, e.g. "1,4": a 2-GPU local server and a 4-GPU remote one with ~4x its KV cache) the fewest
        in flight per unit of weight. Unset = the rule above, unchanged. With PL_ROUTE_SPILL x > 0, a history's next request moves to
        the least-loaded server when its own server has more than x times that load (in flight per unit of weight, this request
        included): the history's prefix cache there is lost once, but a server no longer waits idle while the other one has a queue
        (for example 128 running + 354 waiting on one node and 4 running on the other)."""
        i = p.find("<|im_start|>assistant"); key = p[:i] if i >= 0 else p
        load = lambda j: (self.busy[j] + n) / self.weight[j]
        with self.lock:
            s = self.route.get(key)
            if s is None:
                if len(self.route) > 20000: self.route.clear()
                s = self.route[key] = min(range(len(self.servers)), key=lambda j: (self.busy[j] / self.weight[j], j))   # PL_SERVER_WEIGHTS: in flight per unit of size
            elif self.spill > 0:
                b = min(range(len(self.servers)), key=lambda j: (load(j), j))
                if b != s and load(s) > self.spill * load(b): s = self.route[key] = b; self.moved += 1
            self.busy[s] += n
        return s

    def _done(self, s, n):
        with self.lock: self.busy[s] -= n

    def _complete(self, prefixes, **kw):
        if len(self.servers) == 1: return self.S.complete(prefixes, **kw)
        from concurrent.futures import ThreadPoolExecutor
        groups, n = {}, kw.get("n", 1)
        for i, pr in enumerate(prefixes): groups.setdefault(self._server_of(pr, n), []).append(i)
        out = [None] * len(prefixes)
        def run(item):
            s, idx = item
            try: return idx, self.servers[s].complete([prefixes[i] for i in idx], **kw)
            finally: self._done(s, n * len(idx))
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            for idx, res in ex.map(run, groups.items()):
                for i, r in zip(idx, res): out[i] = r
        return out

    def complete(self, prefixes, **kw):
        """The servers' complete() with each request routed to its history's server (the evaluator plays through this: with two
        machines, the evaluations use both)."""
        return self._complete(prefixes, **kw)

    def sample(self, prefixes, n, max_new, stop_think=True, adapter=True, priority=1):
        """Samples from the policy, the same in training and at test: a thought (stop_think) under SAMPLER (temperature, top-k,
        top-p), a belief line or an action under DECISION (the temperature only, so every legal line keeps a probability above
        zero). adapter=False: from the base model (the LoRA stays loaded for the next call). A thought with max_new 0 is empty:
        the prompt's <think> is closed at once (Qwen's format without thinking)."""
        if stop_think and max_new == 0: return [[("", False)] * n for _ in prefixes]   # max_new < 0: uncapped thinking
        sp = SAMPLER if stop_think else DECISION
        res = self._complete(prefixes, n=n, max_tokens=max_new, temperature=sp["T"], top_p=sp["top_p"], top_k=sp["top_k"],
                             stop=["</think>"] if stop_think else None, use_adapter=adapter, priority=priority)   # 0: an episode turn (waits for
                                                               # nothing else, comes first in the server's queue), 1: bulk (the E-step)
        return [[(c["text"], c["finish"] == "stop") for c in cs] for cs in res]

    def score(self, prefix, conts, n_prefix, adapter=True):
        s = self._server_of(prefix, len(conts))
        try: return self.servers[s].score(prefix, conts, n_prefix=n_prefix, use_adapter=adapter)
        finally: self._done(s, len(conts))

    def sync(self, model):
        from plearn.models.sampler import export_lora
        d = export_lora(model, self.dir, self.model_path, self.lora_r)
        self.load(d)

    def load(self, d):                                                # an exported adapter (--role actor: the learner's)
        from plearn.models.sampler import RemoteVLLMSampler, RemoteMMSampler
        far = None
        for s in self.servers:
            if isinstance(s, (RemoteVLLMSampler, RemoteMMSampler)):   # the other machine of a 2-node run reads its own copy
                if far is None: far = remote_copy(d)
                s.load(far)
            else: s.load(d)

    def sleep(self):                                                  # PL_VLLM_SLEEP=1: all servers free their GPU memory for the M-step
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.servers)) as ex: list(ex.map(lambda s: s.sleep(), self.servers))

    def wake(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.servers)) as ex: list(ex.map(lambda s: s.wake(), self.servers))

    def stop(self):
        """Stop every server: the local ones, and a remote one that this process started (PL_REMOTE_SERVE); --colocate: all their GPU
        memory goes back while the learner trains a phase."""
        for s in self.servers:
            if s is not None: s.stop()
        self.servers = []


def remote_copy(d):
    """A 2-node run whose machines share no disk (PL_REMOTE_SYNC=user@host:/dir; two machines): the adapter
    folder d is copied to the other machine (rsync over PL_REMOTE_SSH, e.g. 'ssh -i KEY') as /dir/<run's folder>/<adapter>, and that
    path is what its server loads. Without PL_REMOTE_SYNC the other machine reads d itself (a shared disk)."""
    spec = os.environ.get("PL_REMOTE_SYNC")
    if not spec: return d
    import shlex, subprocess
    host, root = spec.split(":", 1); d = os.path.abspath(d); ssh = shlex.split(os.environ.get("PL_REMOTE_SSH", "ssh"))
    dst = os.path.join(root, os.path.basename(os.path.dirname(d)), os.path.basename(d))
    subprocess.run(ssh + [host, "mkdir", "-p", dst], check=True, timeout=120)
    subprocess.run(["rsync", "-a", "--delete", "-e", shlex.join(ssh), d + "/", f"{host}:{dst}/"], check=True, timeout=1800)
    return dst


def encode_prefix(tok, p):
    """Token ids of a prefix and, when it carries images, the processor's image inputs (pixel values, grid sizes, token types)."""
    if PROC is not None:
        from plearn.models.sampler import IMAGES, prompt_key, untag
        imgs = IMAGES.get(prompt_key(p)); p = untag(p)
        if imgs:
            enc = PROC(text=[p], images=imgs, return_tensors="pt")
            return enc["input_ids"][0].tolist(), enc
    return tok.encode(p, add_special_tokens=False), None


THINK_END = "</think>"


def presence_requests(text, d0, offsets, ids):
    """The sampled requests of one continuation, for the presence penalty: vLLM lowers the logit of every token that the SAME
    request has generated before (logits -= penalty * output_mask, model_executor/layers/utils.py apply_penalties: the output
    tokens only, not the prompt), so the learner needs each request's own tokens. A turn is three requests (ServerSampler.sample,
    own_turns): the thought up to its </think> (the stop string's token is the request's last), the belief line, the action; the
    text the protocol puts between them is not sampled and has no penalty. text: the continuation; d0: the character where the
    decision starts (None: the whole continuation is a thought, 0: a decision alone); offsets: the tokens' character spans; ids:
    the tokens. Returns [(j0, j1, tokens, first)]: the request's tokens are ids[j0:j1], tokens its distinct token ids and first
    the index (in the continuation) of each one's first occurrence."""
    spans = []
    if d0 is None: spans.append((0, len(text)))
    else:
        if d0 > 0:
            e = text.rfind(THINK_END, 0, d0); spans.append((0, e + len(THINK_END) if e >= 0 else d0))
        nl = text.find(TERM, d0)
        if nl < 0: spans.append((d0, len(text)))
        else: spans += [(d0, nl), (nl + len(TERM), len(text))]
    out = []
    for cs, ce in spans:
        js = [j for j, (a, _) in enumerate(offsets) if cs <= a < ce]
        if not js: continue
        j0, j1 = js[0], js[-1] + 1; first = {}
        for j in range(j0, j1): first.setdefault(ids[j], j)
        out.append((j0, j1, torch.tensor(list(first)), torch.tensor(list(first.values()))))
    return out


def presence_slice(reqs, shift, a, b, device):
    """presence_requests of a batch's rows for the positions a..b-1 of score()'s right-aligned window: per row, per request that
    has positions there, (the positions' indices in the slice, the request's distinct tokens, [position, token]: the token was
    generated before that position). reqs[r]: the row's requests, or None; shift[r]: the window position of the row's first token."""
    out = []
    for rq, sh in zip(reqs, shift):
        part = []
        for j0, j1, cols, first in rq or []:
            lo, hi = max(a, j0 + sh), min(b, j1 + sh)
            if lo >= hi: continue
            pos = torch.arange(lo, hi, device=device)
            part.append((pos - a, cols.to(device), (pos - sh)[:, None] > first.to(device)[None, :]))
        out.append(part)
    return out


def presence_shift(logits, pres, penalty):
    """The logits as vLLM's sampler sees them after its presence penalty (before the temperature): float32 logits with penalty
    taken off every token its request has generated before (pres: presence_slice). The gradient flows through."""
    l = logits.float()
    if l is logits: l = l.clone()
    for r, part in enumerate(pres):
        for rows, cols, seen in part:
            l[r, rows[:, None], cols[None, :]] -= penalty * seen.to(l.dtype)
    return l


def sampler_logp(logits, tgt, sp):
    """log-probability of the target tokens under a sampler sp, as vLLM applies it: logits / T, the top-k set, the top-p set of
    the renormalised top-k probabilities (a token stays while the mass before it is below top_p), renormalised over the kept
    set. The target token always counts as kept (a sampled token can fall outside the set by numerics). With top_k -1 and
    top_p 1 this is the tempered log-softmax. The gradient flows through the kept logits; the set itself is a constant."""
    l = logits.float() / sp["T"]; k, p_ = sp["top_k"], sp["top_p"]; tg = tgt[..., None]
    if 0 < k and p_ < 1 and l.shape[-1] > 4 * (k + TIE_SLACK):
        # the same kept set from the k + TIE_SLACK largest values alone (no sort of the whole vocabulary): exact when every tie of
        # the k-th value is among them; a tie at the top-p cut may fall the other way, as in the sort below (its order of equal
        # values is not fixed either)
        v, idx = l.topk(k + TIE_SLACK, -1); kth = v[..., k - 1:k]; ink = v >= kth
        if bool((v[..., -1:] < kth).all()):
            pr = torch.softmax(v.masked_fill(~ink, float("-inf")), -1); kp = ink & ~((pr.cumsum(-1) - pr) >= p_)
            lt = l.gather(-1, tg).squeeze(-1); lse = torch.logsumexp(v.masked_fill(~kp, float("-inf")), -1)
            return lt - torch.where(((idx == tg) & kp).any(-1), lse, torch.logaddexp(lse, lt))   # the target always counts as kept
    keep = torch.ones_like(l, dtype=torch.bool) if k <= 0 else l >= l.topk(k, -1).values[..., -1:]
    if p_ < 1:
        pr = torch.softmax(l.masked_fill(~keep, float("-inf")), -1)
        ps, idx = pr.sort(-1, descending=True); drop = (ps.cumsum(-1) - ps) >= p_
        keep &= ~torch.zeros_like(keep).scatter(-1, idx, drop)
    keep.scatter_(-1, tg, True)
    return torch.log_softmax(l.masked_fill(~keep, float("-inf")), -1).gather(-1, tg).squeeze(-1)


def token_logp(logits, tgt, dec=None, pres=None):
    """log-probability of the target tokens under the policy: thought tokens under SAMPLER, decision tokens (dec True: the
    belief line and the action) under DECISION. pres (presence_slice; --score_presence 1): the sampler's presence penalty is
    taken off the logits first, as vLLM does (penalty, then temperature, top-k, top-p); every request has the same penalty."""
    if pres is not None and SAMPLER["presence"]: logits = presence_shift(logits, pres, SAMPLER["presence"])
    if dec is None or not dec.any(): return sampler_logp(logits, tgt, SAMPLER)
    if dec.all(): return sampler_logp(logits, tgt, DECISION)
    return torch.where(dec, sampler_logp(logits, tgt, DECISION), sampler_logp(logits, tgt, SAMPLER))


def score_settings(args):
    """The score reads the run's sampler: the thought's temperature, top-p and top-k (SAMPLER), the temperature of the belief line
    and the action (DECISION), and with --score_presence 1 the sampler's presence penalty (PL_PRESENCE_PENALTY). Returns the
    penalty. EVERY rank calls it at the start of train(). If the two tables were set after the point where a
    worker rank enters its job loop, only rank 0 would have the run's values: a worker rank would score its share of every batch with
    the defaults of this module (temperature 0.6, no presence penalty), and a run with temperature 1.0 would fit thoughts
    that a temperature-1.0 sampler made under a temperature-0.6 score, on (world - 1) / world of the batch
    (tests/method/test_rank_settings.py)."""
    pp = float(os.environ.get("PL_PRESENCE_PENALTY") or 0) if args.score_presence else 0.0   # the sampler's own variable (plearn/models/sampler.py)
    SAMPLER.update(T=args.temperature, top_p=args.top_p, top_k=args.top_k, presence=pp); DECISION.update(T=args.temperature, presence=pp)
    return pp


def settings_row():
    """The score's sampler settings as numbers, for the check that every rank has the same (same_settings in train())."""
    return [float(SAMPLER["T"]), float(SAMPLER["top_p"]), float(SAMPLER["top_k"]), float(SAMPLER["presence"]),
            float(DECISION["T"]), float(DECISION["top_p"]), float(DECISION["top_k"]), float(DECISION["presence"])]


def set_attn(model, name):
    """The attention implementation of a loaded model, the language model and the vision tower alike (transformers reads it at each forward)."""
    if hasattr(model, "set_attn_implementation"): model.set_attn_implementation(name); return
    seen = set(); todo = [model.config]
    while todo:
        c = todo.pop()
        if id(c) in seen: continue
        seen.add(id(c)); c._attn_implementation = name
        todo += [v for v in vars(c).values() if hasattr(v, "_attn_implementation") or type(v).__name__.endswith("Config")]


def attn_verdict(a, b, tol=None):
    """--attn: the check of the flash attention against sdpa on the same pairs. a, b: per pair the counted tokens' log-probabilities
    (score(per_token=True)) with the flash attention and with sdpa. Returns (ok, mean |difference|, max |difference|, tokens): ok when
    every value is finite and the mean difference per token is at most tol nats (ATTN_TOL, PL_ATTN_TOL)."""
    tol = ATTN_TOL if tol is None else tol
    d = torch.cat([(x.detach().float() - y.detach().float()).abs().reshape(-1) for x, y in zip(a, b)]) if a else torch.zeros(0)
    if d.numel() == 0: return False, float("nan"), float("nan"), 0
    mean, mx = float(d.mean()), float(d.max())
    return bool(math.isfinite(mean) and math.isfinite(mx) and mean <= tol), mean, mx, int(d.numel())


def attn_check(model, tok, pairs, chunk):
    """--attn flash_attention_3 / _2 (flash attention for speed; it must not change the log-probabilities): once per learner
    rank, at its first score() call, the first ATTN_CHECK_N pairs are scored without gradient with the flash attention and again with
    sdpa (the attention of every RLVB (EM) run before). The rank keeps the flash attention only when attn_verdict passes; else it uses sdpa for
    the rest of the run (no stop). The result goes to the log row (attn_flash 1/0, attn_dlogp_mean, attn_dlogp_max) and the run log.
    No collective call: the ranks reach their first score() at different points."""
    ATTN["busy"] = True
    try:
        sub_ = [(it[0], it[1], it[2] if len(it) > 2 else None, it[3] if len(it) > 3 else None) for it in pairs[:ATTN_CHECK_N]]
        with torch.no_grad():
            a = score(model, tok, sub_, chunk, per_token=True)
            set_attn(model, "sdpa")
            try: b = score(model, tok, sub_, chunk, per_token=True)
            finally: set_attn(model, ATTN["name"])
        ok, mean, mx, n = attn_verdict(a, b)
    finally:
        ATTN["busy"] = False
    if not ok: set_attn(model, "sdpa")
    print(f"ATTN CHECK {'PASS' if ok else 'FAIL'}: {ATTN['name']} vs sdpa on {len(sub_)} pairs, {n} tokens: mean |d log p| {mean:.5f}, "
          f"max {mx:.5f} (limit {ATTN_TOL}) -> {ATTN['name'] if ok else 'sdpa'} for the run", flush=True)
    ATTN.update({"checked": True, "name": ATTN["name"] if ok else "sdpa"})
    MSTEP.update({"attn_flash": float(ok), "attn_dlogp_mean": mean, "attn_dlogp_max": mx})


def score(model, tok, pairs, chunk, per_token=False, readout=False, states=False):
    """Summed log-probability of each continuation given its prefix (the continuation carries its own terminator). A pair
    may carry a third element, a character span (start, end) of the continuation whose tokens are not counted (text that
    the protocol forces, not the model's own), and a fourth, the character offset where the decision tokens start (the
    belief line and the action, scored under DECISION; before it the thought, under SAMPLER; 0: the whole continuation is a
    decision; absent or None: the whole continuation is a thought), and a fifth, "mean": the counted thought tokens enter as
    their mean (each with weight 1 / their number), the decision tokens as their sum. per_token: a list of the counted tokens'
    log-probabilities. readout: a sixth element per pair, the readout's targets {"U": float, "identified": bool, "act":
    [(start, end, log pi*, group)] in continuation characters}; the final hidden states are read (detached) at the positions
    of readout_positions and READOUT["head"] applied without gradient; returns (scores, {"ready": [logits], "rtgt": [U],
    "ident": [bool], "h_ready": [states], "act": [scores], "atgt": [log pi*], "agrp": [groups], "h_act": [states]}). states: per pair
    (the final hidden states at the positions that predict its counted tokens, which of those tokens are decision tokens), for the
    exact KL of the next-token distributions (kl_states)."""
    if ATTN["name"] != "sdpa" and not ATTN["checked"] and not ATTN["busy"] and pairs: attn_check(model, tok, pairs, chunk)
    tok.padding_side = "left"; out = []; ro_out = {"ready": [], "rtgt": [], "ident": [], "h_ready": [], "act": [], "atgt": [], "agrp": [], "h_act": []}
    for s in range(0, len(pairs), chunk):
        part = pairs[s:s + chunk]; cids, masks, decs, offs, reqs = [], [], [], [], []
        pen = bool(SAMPLER["presence"]) and not states                 # --score_presence: each row's sampled requests (presence_requests)
        for it in part:
            span = it[2] if len(it) > 2 else None; d0 = it[3] if len(it) > 3 else None
            if span is None and d0 is None and not pen: cids.append(tok.encode(it[1], add_special_tokens=False)); masks.append(None); decs.append(None); offs.append(None); continue
            enc = tok(it[1], add_special_tokens=False, return_offsets_mapping=True); cids.append(enc["input_ids"]); offs.append(enc["offset_mapping"])
            if pen: reqs.append(presence_requests(it[1], d0, enc["offset_mapping"], enc["input_ids"]))
            if span is None and d0 is None: masks.append(None); decs.append(None); continue
            sp_ = None if not span else (list(span) if isinstance(span[0], (tuple, list)) else [span])   # one span or a list of spans (an empty list = no span)
            masks.append(None if sp_ is None else [float(not any(a < e_ and b > s_ for s_, e_ in sp_)) for a, b in enc["offset_mapping"]])
            decs.append(None if d0 is None else [a >= d0 for a, b in enc["offset_mapping"]])
        pre = [encode_prefix(tok, it[0]) for it in part]
        ids = [pi + c for (pi, _), c in zip(pre, cids)]
        L = max(len(x) for x in ids); pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device=model.device)
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device=model.device)
        keep = max(len(c) for c in cids) + 1
        extra = {}
        encs = [enc for _, enc in pre if enc is not None]
        if encs:                                                     # images of the batch, in row order; token types padded like the ids
            extra["pixel_values"] = torch.cat([enc["pixel_values"] for enc in encs]).to(model.device, model.dtype)
            extra["image_grid_thw"] = torch.cat([enc["image_grid_thw"] for enc in encs]).to(model.device)
            if "mm_token_type_ids" in encs[0]:
                rows = []
                for (pi, enc), x in zip(pre, ids):
                    tt = enc["mm_token_type_ids"][0].tolist() if enc is not None else [0] * len(pi)
                    rows.append([0] * (L - len(x)) + tt + [0] * (len(x) - len(pi)))
                extra["mm_token_type_ids"] = torch.tensor(rows, device=model.device)
        chunked = os.environ.get("PL_CHUNKED_HEAD") == "1" or states   # uncapped thinking: no full-vocab logits of the whole continuation
        cap = {}; hook = (READOUT.get("norm") or find_norm(model)).register_forward_hook(lambda m, i, o: cap.__setitem__("h", o)) if (readout or chunked) else None
        try:
            logits = model(input_ids=inp, attention_mask=att, use_cache=False, logits_to_keep=1 if chunked else keep, **extra).logits[:, :-1, :]
        finally:
            if hook is not None: hook.remove()
        if readout:                                                        # the readout at its reading points (detached states, no gradient)
            hfull = cap["h"].detach()
            for r, it in enumerate(part):
                ro = it[5] if len(it) > 5 else None
                if not ro or offs[r] is None: continue
                n_pre = len(pre[r][0]); off_r = L - len(ids[r]); d0 = it[3] if len(it) > 3 else None
                kd, acts = readout_positions(offs[r], n_pre, ro, d0)
                has_u = kd is not None and ro.get("U") is not None
                idx = ([off_r + kd] if has_u else []) + [off_r + k for k, _, _ in acts]
                if not idx: continue
                hs = hfull[r, idx]; n1 = int(has_u); hd = hs.to(torch.float16).cpu()
                with torch.no_grad(): ready, actl = READOUT["head"](hs)
                if has_u:
                    ro_out["ready"].append(ready[:n1]); ro_out["rtgt"].append(torch.tensor([ro["U"]], device=ready.device, dtype=ready.dtype))
                    ro_out["ident"].append(bool(ro.get("identified"))); ro_out["h_ready"].append(hd[:n1])
                if acts:
                    ro_out["act"].append(actl[n1:]); ro_out["atgt"].append(torch.tensor([lp_ for _, lp_, _ in acts], device=ready.device, dtype=ready.dtype))
                    ro_out["agrp"].append([g for _, _, g in acts]); ro_out["h_act"].append(hd[n1:])
        tgt = inp[:, -keep + 1:]; lp = []
        dec = torch.zeros(len(part), keep - 1, dtype=torch.bool, device=model.device)        # decision tokens, right-aligned like the ids
        for r, (c, dm) in enumerate(zip(cids, decs)):
            if dm is not None: dec[r, keep - 1 - len(c):] = torch.tensor(dm, device=model.device)
        if states:                                                         # the states that predict each pair's counted tokens (kl_states)
            hk = cap["h"][:, L - keep:L - 1]
            for r, (c, mk) in enumerate(zip(cids, masks)):
                h, d = hk[r, keep - 1 - len(c):], dec[r, keep - 1 - len(c):]
                if mk is not None: m = torch.tensor(mk, device=h.device) > 0; h, d = h[m], d[m]
                out.append((h, d))
            continue
        shift = [keep - 1 - len(c) for c in cids]                          # the window position of each row's first continuation token
        pres_at = (lambda a: presence_slice(reqs, shift, a, min(a + 256, keep - 1), model.device)) if pen else (lambda a: None)
        if chunked:                                                        # the same log-probabilities: the LM head per 256-token slice, recomputed in backward
            from torch.utils.checkpoint import checkpoint
            hk, head = cap["h"][:, L - keep:L - 1], model.get_output_embeddings()
            for a in range(0, keep - 1, 256):
                lp.append(checkpoint(lambda h_, t_, d_, p_=pres_at(a): token_logp(head(h_), t_, d_, p_), hk[:, a:a + 256], tgt[:, a:a + 256], dec[:, a:a + 256], use_reentrant=False))
        else:
            for a in range(0, keep - 1, 256):                              # slices keep the float32 log-softmax small
                lp.append(token_logp(logits[:, a:a + 256], tgt[:, a:a + 256], dec[:, a:a + 256], pres_at(a)))
        lp = torch.cat(lp, 1)
        for r, (c, mk) in enumerate(zip(cids, masks)):
            x = lp[r, keep - 1 - len(c):]
            if len(part[r]) > 4 and part[r][4] == "blocks":                  # the two blocks apart: [thought sum, decision sum]
                d = dec[r, keep - 1 - len(c):]; m = torch.ones_like(x) if mk is None else torch.tensor(mk, device=x.device, dtype=x.dtype)
                out.append(torch.stack([(x * m * (~d).to(x.dtype)).sum(), (x * m * d.to(x.dtype)).sum()])); continue
            if len(part[r]) > 4 and part[r][4] == "mean":                    # the thought tokens as a mean, the decision tokens summed
                d = dec[r, keep - 1 - len(c):]; m = torch.ones_like(x) if mk is None else torch.tensor(mk, device=x.device, dtype=x.dtype)
                n = float(((~d).to(x.dtype) * m).sum().clamp(min=1.0))
                x = x * torch.where(d, torch.ones_like(x), torch.full_like(x, 1.0 / n))
            if per_token: out.append(x if mk is None else x[torch.tensor(mk, device=x.device) > 0])
            else: out.append(x.sum() if mk is None else (x * torch.tensor(mk, device=x.device, dtype=x.dtype)).sum())
    res = out if (per_token or states) else torch.stack(out)
    return (res, ro_out) if readout else res


class base_model:
    """The LoRA switched off (scale 0): the base model pi_0, for reference log-probabilities."""
    def __init__(self, model): self.mods = [m for m in model.modules() if isinstance(m, E.LoRALinear)]
    def __enter__(self):
        self.s = [m.scale for m in self.mods]
        for m in self.mods: m.scale = 0.0
    def __exit__(self, *_):
        for m, v in zip(self.mods, self.s): m.scale = v


READOUT = {}           # --readout: {"head": Readout, "norm": the final norm module}
# The metrics W&B records (only useful metrics): the M-step's losses and
# gradients, the E-step target (budget share, lengths, KL to the samples, the histories without a target), the prior's sample
# quality, the time, and per evaluation the success, premature answers, evidence actions, cut reasoning, the tabletop reading,
# the open belief lines (plearn/tasks/domain.py open_line) and the belief check. The run's own log files keep every metric;
# PL_WANDB_ALL=1 sends every metric to W&B as well.
WANDB_KEEP = re.compile(
    r"^(train/(loss_dec|loss_think|grad_norm_think|grad_norm_dec|kl_q_pi|em_ess|m_items|em_q_budget|em_q_correct|think_tokens_p"
    r"|think_tokens_q|think_truncated|think_repeat_n|think_repeat_of|em_more_hist|em_left_out|teacher_rows|stall_rejected|em_no_accept|em_wrong_line|crit_belief_ok|crit_act_best|crit_line_wide|w_premature"
    r"|w_exact_line|w_line_target|estep_read_exact|estep_read_misread|kl_prior|kl_stop_left|set_accept_share|neg_items|neg_p|loss_neg|neg_thoughts|sec|gen_sec|wall_min"
    r"|grpo_scale|grpo_adv_abs|grpo_adv_neg|grpo_adv_pos|grpo_adv_zero_hist|grpo_stop_t|grpo_stop_d|grpo_items|attn_flash|attn_dlogp_mean|attn_dlogp_max|look_gap|look_accept|tau_look_on)"
    r"|eval/(update|eval_minutes|belief_ok|belief_ok_sampled|decision_ok|decision_ok_sampled|decision_regret|decision_regret_sampled|think_tokens|groups\.train_suite\.(success|looks|thought_cut|invalid_turns|open_line)"
    r"|tasks\.[a-z0-9_]+\.(success|premature|looks|thought_cut|fact_acc|open_line)|tasks\.tabletop_[a-z]+\.(read_exact|read_misread|read_reasoning|read_wider)))$")


class Board:
    """W&B logging on rank 0 when --wandb_project (or WANDB_PROJECT) is set; WANDB_MODE=offline writes the run to disk. Every
    training row goes under train/ and every evaluation under eval/, numbers only (nested dicts flattened with '.'), and only
    the metrics of WANDB_KEEP (PL_WANDB_ALL=1: every metric). A failed call is reported once and never stops the run."""
    def __init__(self, args, tag, world, enabled=True):
        self.run, self.warned = None, False
        project = args.wandb_project or os.environ.get("WANDB_PROJECT", "")
        if not project or not enabled: return
        try:
            import subprocess
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        except Exception:
            commit = ""
        try:
            import wandb
            kw = dict(project=project, entity=args.wandb_entity or os.environ.get("WANDB_ENTITY") or None,
                      name=args.wandb_name or tag, group=args.wandb_group or os.environ.get("WANDB_GROUP") or None,
                      config={**vars(args), "world": world, "commit": commit})
            wid = os.environ.get("PL_WANDB_ID")              # a --resume_step run goes on in one W&B run, not a new one:
            rewind = bool(wid and args.resume_step) and self.past(wandb, kw, wid, args.resume_step)
            tries = ([dict(resume_from=f"{wid}?_step={args.resume_step}")] if rewind else []) + \
                    ([dict(id=wid, resume="must")] if wid else []) + [{}]   # rewind to the resumed update, else go on, else a new run
            for t in tries:
                try:
                    self.run = wandb.init(**kw, **t); break
                except Exception as ex:
                    print(f"WANDB: init with {t or 'a new run'} failed:", repr(ex)[:300], flush=True)
                    if not t: raise
            if wid and getattr(self.run, "id", wid) != wid: print(f"WANDB: PL_WANDB_ID {wid} NOT continued: a new run {self.run.id}", flush=True)
            print("wandb:", self.run.url if getattr(self.run, "url", None) else "offline", flush=True)
            self.run.define_metric("eval/update"); self.run.define_metric("eval/*", step_metric="eval/update")   # --role: an evaluation
            # arrives some updates after its own update; its charts use the evaluated update as their x axis
        except Exception as ex:
            print("WANDB: init failed, the run goes on without it:", repr(ex)[:300], flush=True)

    @staticmethod
    def past(wandb, kw, wid, step):
        """Whether W&B run wid has history after update step (unknown: yes). Only then does a resumed run rewind it; a run whose
        history ends at the resumed update or before goes on (resume "must"). Some W&B servers have no rewind ("Rewind
        is in private preview"), and a failed rewind holds the run's id, so that the resume after it fails too ("run ID ... is in
        use") and the run went on in a new W&B run."""
        try:
            api = wandb.Api(); last = api.run(f"{kw.get('entity') or api.default_entity}/{kw['project']}/{wid}").summary.get("_step")
            return last is not None and int(last) > step
        except Exception as ex:
            print(f"WANDB: the history of {wid} is unknown ({repr(ex)[:200]}); it is rewound", flush=True)
            return True

    @staticmethod
    def flat(d, pre):
        out = {}
        for k, v in d.items():
            if isinstance(v, bool): out[pre + str(k)] = float(v)
            elif isinstance(v, (int, float)) and math.isfinite(v): out[pre + str(k)] = v
            elif isinstance(v, dict): out.update(Board.flat(v, pre + str(k) + "."))
        return out

    @staticmethod
    def kept(d, prefix):
        """The flat metrics of one row that W&B records (WANDB_KEEP; PL_WANDB_ALL=1: all of them)."""
        flat = Board.flat(d, prefix + "/")
        return flat if os.environ.get("PL_WANDB_ALL") == "1" else {k: v for k, v in flat.items() if WANDB_KEEP.match(k)}

    def log(self, d, step, prefix):
        if self.run is None: return
        try: self.run.log(self.kept(d, prefix), step=step)
        except Exception as ex:
            if not self.warned: print("WANDB: log failed (reported once):", repr(ex)[:300], flush=True); self.warned = True

    def commit(self, step):
        """Write step `step` to W&B now. A log with step=N is kept open until a log of a higher step arrives, so without this
        each update would show on W&B only one update later."""
        if self.run is None: return
        try: self.run.log({}, step=step, commit=True)
        except Exception as ex:
            if not self.warned: print("WANDB: commit failed (reported once):", repr(ex)[:300], flush=True); self.warned = True

    def finish(self):
        if self.run is None: return
        try: self.run.finish()
        except Exception: pass


class Readout(torch.nn.Module):
    """--readout: two readouts of the exact posterior from the policy's final hidden state, free of any vocabulary: the
    readiness of the turn, read at the state that writes the belief line's first token (target U(b*) = max_y b*(y), the
    teacher's stop quantity), and a relative score of a written action, read at its last token. Linear on the layer-normed
    state (the probe of 2026-10-25 found a linear readout as good as an MLP). Zero at the start."""
    def __init__(self, d):
        super().__init__(); self.ln = torch.nn.LayerNorm(d); self.ready = torch.nn.Linear(d, 1); self.act = torch.nn.Linear(d, 1)
        for lin in (self.ready, self.act): torch.nn.init.zeros_(lin.weight); torch.nn.init.zeros_(lin.bias)
    def forward(self, h):
        """(readiness, action) logits at the given states. The action score is relative: over the candidates sampled for one
        history its softmax is the head's distribution over them, fitted to the teacher's pi* renormalized on the same
        candidates by the cross-entropy (listwise; invariant to a constant per history). A decision is a preference among
        alternatives, not a grade."""
        z = self.ln(h.float()); return self.ready(z)[..., 0], self.act(z)[..., 0]


def find_norm(model):
    """The language model's final norm: its output is the whole sequence of final hidden states (the LM head reads a slice)."""
    names = [n for n, m in model.named_modules() if n.endswith(".norm") and ".layers." not in n and ".blocks." not in n]
    pick = [n for n in names if "language_model" in n] or names
    assert len(pick) == 1, names
    return dict(model.named_modules())[pick[0]]


def readout_positions(offs_cont, n_pre, ro, d0):
    """Token indices (in the unpadded row ids, prefix + continuation) of the readout's reading points: for the readiness the
    token before the first decision token (the state that writes the belief line's first token); for the written action
    (continuation span) its last token, with the teacher's log pi*(u) and the history's group id.
    Returns (index or None, [(index, log pi*, group)])."""
    def last_tok(offs, en, base):
        k = max((k for k, (a, b) in enumerate(offs) if a < en), default=None); return None if k is None else base + k
    k0 = next((k for k, (a, b) in enumerate(offs_cont) if b > d0), None) if d0 is not None else None
    dec = None if k0 is None or n_pre + k0 - 1 < 0 else n_pre + k0 - 1
    act = [(last_tok(offs_cont, en, n_pre), lp, grp) for st, en, lp, grp in ro.get("act", [])]
    return dec, [x for x in act if x[0] is not None]


def auroc(score, y):
    """The area under the ROC curve of score for the binary labels y (the rank statistic, ties at their mean rank, so a constant
    score gives 1/2); None without both classes."""
    score, y = np.asarray(score, float), np.asarray(y, bool); n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0: return None
    order = np.argsort(score, kind="stable"); s = score[order]; r = np.empty(len(score)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def readout_stats(st):
    """The readout against its targets over collected outputs (score(..., readout=True)): for the readiness the mean absolute
    error to U(b*) and the AUROC for identified; for the action score, per history with two or more candidates the teacher
    tells apart, the KL(pi* || head), the top-1 agreement and the teacher's mass on the head's pick (times the candidate count)."""
    out = {}
    if st["ready"]:
        q = torch.sigmoid(torch.cat([x.detach().float().cpu() for x in st["ready"]])).numpy(); u = torch.cat([x.detach().float().cpu() for x in st["rtgt"]]).numpy()
        out["ro_ready_n"] = len(q); out["ro_ready_mae"] = float(np.abs(q - u).mean())
        if st.get("ident") and len(st["ident"]) == len(q): out["ro_ready_auc"] = auroc(q, np.array(st["ident"], bool))
    if st.get("act"):
        s = torch.cat([x.detach().float().cpu() for x in st["act"]]).numpy(); lp = torch.cat([x.detach().float().cpu() for x in st["atgt"]]).numpy()
        grp = np.array([g for gs in st["agrp"] for g in gs]); kls, top, mass, n = [], [], [], 0
        for g in np.unique(grp):
            m = grp == g
            if m.sum() < 2 or np.ptp(lp[m]) < 1e-6: continue                    # a preference needs two candidates that the teacher tells apart
            ps = np.exp(lp[m] - lp[m].max()); ps /= ps.sum(); qs = np.exp(s[m] - s[m].max()); qs /= qs.sum(); nz = ps > 1e-12
            kls.append(float((ps[nz] * (np.log(ps[nz]) - np.log(np.maximum(qs[nz], 1e-12)))).sum())); top.append(float(ps[int(qs.argmax())] >= ps.max() - 1e-9))
            mass.append(float(ps[int(qs.argmax())] * m.sum())); n += 1                 # the teacher's mass on the head's pick, against 1/n
        if n: out.update({"ro_act_groups": n, "ro_act_kl": float(np.mean(kls)), "ro_act_top1": float(np.mean(top)), "ro_act_pick_gain": float(np.mean(mass))})
    return out


def loss_items(model, tok, items, N, chunk, weights=None):
    """items: (prefix, continuation). Backpropagates -sum w log p(continuation | prefix) / N (w = 1 unless given)."""
    total = 0.0
    for s in range(0, len(items), chunk):
        lp = score(model, tok, items[s:s + chunk], chunk)
        w = torch.ones_like(lp) if weights is None else torch.tensor(weights[s:s + chunk], device=lp.device, dtype=lp.dtype)
        loss = -(w * lp).sum() / N
        loss.backward(); total += loss.item()
    return total


def ppo_keep(w, logratio, eps):
    """--update grpo, --grpo_eps: PPO's clip rule as a stop, per item and block: a block with a negative weight keeps its gradient
    while its log-probability per token is above log(1 - eps) of the phase prior's (logratio, per token), one with a positive weight
    while it is below log(1 + eps); outside, the min of PPO's clipped surrogate has no gradient. The samples of a phase are fitted
    about four times (25 updates of 32 histories over 200), so without the stop a pushed-down turn falls without a floor."""
    lo, hi = math.log(1.0 - eps), math.log(1.0 + eps)
    return ~(((w < 0) & (logratio <= lo)) | ((w > 0) & (logratio >= hi)))


def loss_items_blocks(model, tok, items, N, chunk, weights, params, readout=False, weights_t=None, ppo=None):
    """--m_step blocks: the M-step loss in two blocks from one forward pass per chunk, the thought tokens and the decision tokens
    (items carry the "blocks" mode, score() returns [thought sum, decision sum]). The gradient of each block is accumulated in
    its own buffers (the trainable parameters are the LoRA's, so a second copy is small). Returns the two losses and the two
    gradients, as lists aligned with params, and with --readout the readout's outputs and detached states of the update (for
    readout_stats and readout_fit; else None). ppo (--update grpo): (old, ntok, eps), the items' [thought, decision] log-probabilities
    under the phase prior and their token counts ([n, 2] tensors) and the clip epsilon: each block stops by ppo_keep, and the
    numbers of stopped blocks go to MSTEP (grpo_stop_t, grpo_stop_d)."""
    gt = [torch.zeros_like(p) for p in params]; gd = [torch.zeros_like(p) for p in params]; lt = ld = 0.0; stop = [0, 0]
    st = {"ready": [], "rtgt": [], "ident": [], "h_ready": [], "act": [], "atgt": [], "agrp": [], "h_act": []} if readout else None
    for s in range(0, len(items), chunk):
        res = score(model, tok, items[s:s + chunk], chunk, readout=readout); lp, ro = res if readout else (res, None)
        w = torch.tensor(weights[s:s + chunk], device=lp.device, dtype=lp.dtype)
        wt = w if weights_t is None else torch.tensor(weights_t[s:s + chunk], device=lp.device, dtype=lp.dtype)   # --line_target: the thought's own weight
        if ppo is not None:                                            # --update grpo: the stop against the phase prior, per block
            old, ntok, eps = ppo
            r = (lp.detach() - old[s:s + chunk].to(lp.device, lp.dtype)) / ntok[s:s + chunk].to(lp.device, lp.dtype).clamp(min=1.0)
            kt, kd = ppo_keep(wt, r[:, 0], eps), ppo_keep(w, r[:, 1], eps)
            stop[0] += int((~kt).sum()); stop[1] += int((~kd).sum())
            wt, w = wt * kt.to(wt.dtype), w * kd.to(w.dtype)
        loss_t = -(wt * lp[:, 0]).sum() / N; loss_d = -(w * lp[:, 1]).sum() / N
        if ro is not None:
            for k in st: st[k] += list(ro[k])
        loss_t.backward(retain_graph=True)
        for g, p in zip(gt, params):
            if p.grad is not None: g += p.grad; p.grad = None
        loss_d.backward()
        for g, p in zip(gd, params):
            if p.grad is not None: g += p.grad; p.grad = None
        lt += loss_t.item(); ld += loss_d.item()
    if ppo is not None: MSTEP.update({"grpo_stop_t": stop[0], "grpo_stop_d": stop[1], "grpo_items": len(items)})
    return lt, ld, gt, gd, st


NEG_CAP = 1.0      # --neg_answer: the cap of the factor p / (1 - p) in the gradient of -log(1 - p)
NEG_MAX = 8        # --neg_answer: at most this many distinct own answer turns per history (evenly spaced in the E-step's order), so the
                   # push-down costs about what the E-step's fitted items cost (m_items: 9-15 per history in the measured game runs)


def neg_rows(sizes, rows_h):
    """--neg_answer: per history of a batch, the own turns whose rejected answer the M-step pushes down, or None. Only a history that
    the E-step leaves out (sizes 0) and where no own turn takes an action the teacher accepts (the rows' accepted flag) has them:
    its own turns that answer (answer_text of the action line as the test reads it). Every such answer is rejected there. The same
    turn (thought, lines) counts once; more than NEG_MAX distinct turns are cut to NEG_MAX, evenly spaced."""
    out = []
    for z, rs in zip(sizes, rows_h):
        if z or not rs or any(bool(r[8]) if len(r) > 8 else True for r in rs): out.append(None); continue
        a, seen = [], set()
        for r in rs:
            k = (r[1], r[3], r[4])
            if k not in seen and answer_text(ordered(r[3], r[4])[1]) is not None: seen.add(k); a.append(r)
        if len(a) > NEG_MAX: a = [a[(i * len(a)) // NEG_MAX] for i in range(NEG_MAX)]
        out.append(a or None)
    return out


def neg_item(r):
    """--neg_answer: the scoring item of one own turn's answer (prefix, continuation, span of the characters not counted): the
    turn as the model wrote it, its own thought and lines in their own order, with only the answer line's own tokens counted.
    No terminator after the answer line: the model ends its turn there with <|im_end|>, not with TERM, so a TERM token has a
    probability near 0 and made p(a) ~1e-12 at update 1 of the first --neg_answer run (8ghav2qi: neg_p 1.19e-12, 40 items), and
    the push-down was 0. Action line second: the whole turn, the span covers the thought and the belief line; action line first:
    the turn up to the action line."""
    prompt, z, _, l1, l2 = r[:5]; d0 = len(z) + len(CLOSE)
    if ordered(l1, l2)[1] == l1 and l1 != l2:                          # the action line came first
        return (prompt, z + CLOSE + l1, (0, d0))
    return (prompt, z + CLOSE + l1 + TERM + l2, (0, d0 + len(l1) + len(TERM)))


def neg_thought_items(rs, lam, mode, hwh=1.0):
    """--neg_thought: the M-step items of one history's rejected-answer turns (neg_rows): one per distinct own thought, its thought
    and </think> as the prompt and the turn wrote them (the span of a protocol-forced </think> is not counted, as for the accepted
    items), weight -lam / (the number of distinct thoughts) times the history's draw weight. Returns (items, weights)."""
    seen, out = set(), []
    for r in rs or []:
        if r[1] in seen or not r[1]: continue
        seen.add(r[1]); z = r[1]; d0 = len(z) + len(CLOSE)
        out.append((r[0], z + CLOSE, None if (len(r) > 6 and r[6]) else (len(z), d0), d0, mode))
    return (out, [-lam / len(out) * hwh] * len(out)) if out else ([], [])


def turn_item(z, anc, a, stopped, own_term=False):
    """The M-step continuation of one own turn (thought z, belief line anc, action a): (continuation, span(s) not counted, d0).
    own_term False: z </think> anc TERM a TERM; the </think> of a thought cut at --max_think is not counted. own_term True
    (--dec_term own, METHOD; fix A): the turn as the model wrote it: no TERM after the action (the model ends
    its turn there with <|im_end|>: base log p(TERM) = -33.4, log p(<|im_end|>) = -0.0002 over 96 own turns), and the TERM between the
    two lines is in the text but not counted (the protocol puts it in the action's prefix; the model's own separator is not kept).
    kl_split v2: 7.21 of the 7.24 nats per turn that 25 updates moved the decision were these two tokens."""
    d0 = len(z) + len(CLOSE); cut = None if stopped else (len(z), d0)
    if NO_BELIEF_LINE:                                                # --no_belief_line: z </think> a, the turn as the model wrote it
        assert not anc, "--no_belief_line: an own turn with a belief line"
        return (z + CLOSE + a + TERM, cut, d0) if not own_term else (z + CLOSE + a, [cut] if cut else None, d0)   # no span = None (24 Nov fix: [] crashed score)
    if not own_term: return z + CLOSE + anc + TERM + a + TERM, cut, d0
    t0 = d0 + len(anc)
    return z + CLOSE + anc + TERM + a, ([cut] if cut else []) + [(t0, t0 + len(TERM))], d0


def line_span(z, l1, l2, own_term=False):
    """The characters of the BELIEF line in turn_item's continuation (z CLOSE l1 TERM l2 ...): the model's first line, or its second
    when it wrote the action line first (ordered)."""
    d0 = len(z) + len(CLOSE)
    if ordered(l1, l2)[0] == l1: return (d0, d0 + len(l1))
    s0 = d0 + len(l1) + len(TERM); return (s0, s0 + len(l2))


def target_item(z, l1, l2, stopped, own_term, line_target=True):
    """turn_item for the M-step, with --line_target: (continuation, spans not counted, d0, the thought's factor). A turn whose belief
    line the teacher verifies (line_target True) is the whole turn, factor 1. A turn whose line is not verified keeps only its
    accepted ACTION as a target: the belief line's characters are not counted and the thought's weight is 0 (factor 0). The line
    and the thought stay in the text, as the context of the action."""
    cont, cut, d0 = turn_item(z, l1, l2, stopped, own_term)
    if line_target: return cont, cut, d0, 1.0
    spans = list(cut) if isinstance(cut, list) else ([cut] if cut else [])
    return cont, (spans + ([] if NO_BELIEF_LINE else [line_span(z, l1, l2, own_term)])) or None, d0, 0.0   # no span = None


def loss_neg(model, tok, items, N, chunk, weights, gd, params):
    """--neg_answer: the push-down of rejected answers (unlikelihood training, Welleck et al. 2019), bounded. Each item's answer
    probability p = exp(log p(a | h, z, b)); the loss is sum w (-log(1 - p)) / N, whose gradient is w p / (1 - p) grad log p. The
    factor p / (1 - p) is detached and capped at NEG_CAP: it goes to 0 as p goes to 0 (the push stops when the answer is rare) and
    a near-certain answer pushes no harder than NEG_CAP times its log-likelihood gradient. The gradient is added to the decision
    block's buffers gd (the item's grads leave p.grad). Returns the summed loss value (-log(1 - p) terms, with their weights) and
    the items' p before the step."""
    total, ps = 0.0, []
    for s in range(0, len(items), chunk):
        lp = score(model, tok, items[s:s + chunk], chunk)
        w = torch.tensor(weights[s:s + chunk], device=lp.device, dtype=lp.dtype)
        p = lp.detach().exp().clamp(max=1 - 1e-6)
        coef = (p / (1 - p)).clamp(max=NEG_CAP)
        loss = (w * coef * lp).sum() / N                             # minimizing lowers log p(a) with the unlikelihood factor
        loss.backward()
        for g, q in zip(gd, params):
            if q.grad is not None: g += q.grad; q.grad = None
        total += float((w * -torch.log1p(-p)).sum() / N); ps += p.tolist()
    return total, ps


def kl_probe(rows_, sizes, n):
    """--kl_stop: the probe of a phase, up to n own turns of one batch (rows_ and sizes as em_update takes them), one per distinct
    thought with its first decision, taken in turn from the histories (the first thought of each history, then the second, ...), so
    that it spreads over the batch. Returns the indices into rows_."""
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(int); per = []
    for h in range(len(sizes)):
        seen, idx = set(), []
        for i in range(off[h], off[h + 1]):
            if rows_[i][1] not in seen: seen.add(rows_[i][1]); idx.append(i)
        per.append(idx)
    out = []
    for r in range(max((len(x) for x in per), default=0)):
        out += [x[r] for x in per if r < len(x)]
    return out[:n]


@torch.no_grad()
def kl_states(head, h_old, h_new, dec, t_think, t_dec, chunk=256):
    """KL(old || new) of the next-token distributions at the positions of a turn, from the two models' final hidden states there
    (score(states=True)) and their shared output layer head: the sum over the positions of sum_v p_old(v) (log p_old(v) - log
    p_new(v)), p = softmax(head(h) / T) over the whole vocabulary with T of the thought or of the decision (dec), and the number of
    positions. Exact and never negative: the expected KL over the visited states that PPO's approx_kl estimates from one sampled
    token each. That estimate fails here (measured on Qwen3.5-4B): the turns come from vLLM through the sampler's cut (top-k, top-p), so
    a sampled token can have a tiny probability in the trainer's own computation ((r - 1) - log r gave 15 nats per token), and the
    tokens that the update makes rarer were cut away and are never sampled (the mean log ratio went below 0, to -0.005 at lr 1e-5).
    No cut here: a cut set that differs between the two models would make the KL infinite."""
    s = 0.0
    for a in range(0, len(h_old), chunk):
        T = torch.where(dec[a:a + chunk], t_dec, t_think).float()[:, None]
        lo = torch.log_softmax(head(h_old[a:a + chunk]).float() / T, -1); ln = torch.log_softmax(head(h_new[a:a + chunk]).float() / T, -1)
        s += float((lo.exp() * (lo - ln)).sum())
    return s, int(len(h_old))


def phase_end(step, phase, steps):
    """The last update of the phase of update step."""
    return min(steps, ((step - 1) // phase + 1) * phase)


def kl_measured(args):
    """Whether the learner measures kl_prior after each update: --kl_stop (the measure also limits the phase) or --kl_log (it is only
    logged)."""
    return bool(args.kl_stop or getattr(args, "kl_log", 0))


def keep_set(sizes, rows_h, L, neg=False):
    """--set_keep L (dynamic sampling, as DAPO's): the histories of a phase's sample set that its batches come from, in the set's
    own (random) order: those that give the M-step a gradient, the E-step fits them (sizes > 0) and an own turn takes an action the
    teacher accepts (the rows' accepted flag), at most L of them. Any other history is skipped and not sampled again. neg
    (--neg_answer): a history with rejected answers to push down (neg_rows) gives a gradient too and is kept. Returns the kept
    histories and the number of histories with an accepted own turn."""
    acc = [bool(z and any(bool(r[8]) if len(r) > 8 else True for r in rs)) for z, rs in zip(sizes, rows_h)]
    nr = neg_rows(sizes, rows_h) if neg else [None] * len(acc)
    ok = [h for h in range(len(acc)) if acc[h] or nr[h]]
    return ok[:L], sum(acc)


def by_task(plans, idx):
    """The task of each history of a set (its plan's belief class, e.g. ('minigrid', 2)) -> the number of histories idx holds."""
    out = {}
    for h in idx:
        m = re.match(r"\(?'([^']+)'", str(plans[h].get("cls", ""))); k = m.group(1) if m else "?"; out[k] = out.get(k, 0) + 1
    return out


def deal(work, k):
    """Items 0 .. n-1 with these amounts of work dealt to k parts of about equal work (as verl's balance_batch): the largest item
    first, each to the part with the least work so far. The same input gives the same parts on every rank."""
    load, parts = [0] * k, [[] for _ in range(k)]
    for h in sorted(range(len(work)), key=lambda h: (-work[h], h)):
        r = min(range(k), key=lambda r: (load[r], r)); parts[r].append(h); load[r] += work[h]
    return [sorted(p) for p in parts]


def set_signal(R, alpha):
    """--set_draw is: the signal of a history of a phase set, half the L1 distance between its E-step weights softmax(R / alpha)
    and equal weights. 0: the target is the prior there. It bounds the part of the history's M-step gradient that comes from the
    weights (the rest is the noise of its own samples). None: the E-step left the history out (no samples)."""
    if not len(R): return None
    w = np.exp((np.asarray(R, dtype=float) - max(R)) / alpha); w /= w.sum()
    return 0.5 * float(np.abs(w - 1.0 / len(w)).sum())


def set_inclusion(sig, B, prio):
    """--set_draw is: the probability of each history of a phase set to be in one batch of B, from the signals sig (None: never
    drawn). The draw probability p_h = (1 - prio) / N + prio * sig_h / sum(sig) over the N histories with samples (the uniform part
    keeps every history possible and every weight below 1 / (1 - prio)); then pi_h = B p_h, capped at 1, the excess spread over the
    other histories in proportion to p."""
    ok = np.array([x is not None for x in sig]); n = int(ok.sum()); B = min(B, n)
    s = np.array([0.0 if x is None else x for x in sig]); tot = s[ok].sum()
    p = np.where(ok, (1 - prio) / max(1, n) + (prio * s / tot if tot > 0 else prio / max(1, n)), 0.0)
    pi, fixed = B * p, np.zeros(len(p), bool)
    while (pi > 1 + 1e-12).any():
        fixed |= pi > 1 + 1e-12; pi[fixed] = 1.0
        rest = ok & ~fixed; pi[rest] = (B - fixed.sum()) * p[rest] / p[rest].sum()
    return pi


def pps_draw(pi, groups, rnd):
    """--set_draw is: one batch by systematic sampling: history h is in it with probability pi[h], the batch holds sum(pi)
    histories and none twice (each pi <= 1). The histories are laid out group by group (a group: the episode of a history) in a
    random order, so that a batch spreads over the episodes."""
    by = {}
    for h, g in enumerate(groups):
        if pi[h] > 0: by.setdefault(g, []).append(h)
    keys = list(by); rnd.shuffle(keys); x, acc, out = rnd.random(), 0.0, []
    for k in keys:
        hs = by[k]; rnd.shuffle(hs)
        for h in hs:
            acc += pi[h]
            if x < acc: out.append(h); x += 1.0
    return out


def png_bytes(im):
    buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()


def from_png(b):
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")


def first_line(text):
    return text.strip().split("\n")[0].strip()


def ordered(a, u):
    """The belief line and the action line as the test reads them (bench_eval): when the first line is the ACTION line and the
    second is not, the ACTION line is the action and the other line is the belief line."""
    return (u, a) if a.upper().startswith("ACTION:") and not u.upper().startswith("ACTION:") else (a, u)


def norm(s): return " ".join(str(s).lower().split())


def answer_text(text):
    """The answer of an action line that answers ('ACTION: SUBMIT <answer>', read as the domain's parse_action reads it); None
    for any other line."""
    s = text.strip(); body = s[len("ACTION:"):].strip() if s.upper().startswith("ACTION:") else s
    return body[len("SUBMIT"):].strip() if body.upper().startswith("SUBMIT") else None


def parse_action(D, e, text, plan):
    """The agent's free action line as ('look', look id) | ('answer', text) | ('invalid', None). A domain may parse its own
    action language (D.parse_action); the default matches the look strings and reads 'SUBMIT <answer>'."""
    if hasattr(D, "parse_action"): return D.parse_action(e, text)
    t = norm(text)
    for k, s in plan["looks"].items():
        if norm(s) == t: return ("look", k)
    m = __import__("re").fullmatch(r"(action: weigh) (.+) vs (.+)", t)
    if m:                                                              # the same weighing with the pans swapped
        for k, s in plan["looks"].items():
            if norm(s) == f"{m.group(1)} {m.group(3)} vs {m.group(2)}": return ("look", k)
    if t.startswith("action: submit "): return ("answer", text.strip().split(None, 2)[2].strip())
    return ("invalid", None)


# --------------------------------------------------------------------- the teacher as a critic (--objective critic)
def belief_score(D, B, text, plan, miss=200.0):
    """S(b; P*) = E_{x ~ P*}[log u_b(x)]: the log score of a written belief line read as the agent's stated distribution u_b,
    uniform over the set b the line names, with the floor log 0 := -miss outside b (LOGP_FLOOR, the floor of the action score
    too): S = -P*(b) log |b| - miss P*(outside b). The log score is proper, so its best line is the posterior's support (every
    value of probability above about 1 / (miss |b|)), in the smallest set. A domain scores its own belief lines
    (D.belief_score); by default b is the set of answers anchor_support reads from the line."""
    if hasattr(D, "belief_score"): return D.belief_score(B, text, miss)
    answers = plan["answers"]; allowed = anchor_support(text, answers, D) & set(answers)
    post = plan.get("post") or {y: 1.0 / len(answers) for y in answers}
    pin = sum(p for y, p in post.items() if y in allowed)
    return -pin * math.log(max(1, len(allowed))) - miss * (1.0 - pin)


def belief_best(D, B, plan):
    """The highest belief score at this history (the support of the posterior; D.belief_best when the domain has it)."""
    return D.belief_best(B) if hasattr(D, "belief_best") else belief_score(D, B, plan["anchor"], plan)


def line_wide(D, text, plan, miss):
    """--line_wide: -miss times the share of the line's options that the teacher's exact line (plan["anchor"])
    excludes, both read by the domain's reader (D.line_excess; by default the answers anchor_support reads): the mirror of the log
    score's -miss P*(outside b), so a line wider than the exact line pays the floor for its share of excluded options as a narrower
    line pays it for the posterior mass it leaves out. 0 for the exact line and for every line inside it; added to the log score."""
    if hasattr(D, "line_excess"): return -miss * D.line_excess(plan["B"], text, plan["anchor"])
    answers = set(plan["answers"]); s = anchor_support(text, plan["answers"], D) & answers
    x = anchor_support(plan["anchor"], plan["answers"], D) & answers
    return -miss * len(s - x) / len(s) if s else 0.0


def decision_score(D, planner, plan, text):
    """--line_score decision: the score of a belief line by the act it implies, log pi*(a_b | h), with a_b the act the teacher takes
    when the line is its posterior (D.line_belief, Planner.implied; ties averaged): the proper scoring rule of the teacher's own
    decision problem. The exact line implies the teacher's own act and gets the highest score; a line that implies the same act
    scores the same. A line the domain cannot read as a belief gets LOGP_FLOOR, as an action the teacher excludes."""
    Bb = D.line_belief(plan["B"], text)
    acts = planner.implied(D, plan, Bb) if Bb is not None else []
    return float(np.mean([plan["logp"].get(s, plan["waste_lp"]) for s in acts])) if acts else LOGP_FLOOR


def teacher_of(D, plan):
    """The plan's own teacher in a mixed run (a plan names it by "dom"), else D."""
    return D.sub(plan["dom"]) if plan.get("dom") is not None and hasattr(D, "sub") else D


def decision_regret(D, planner, plan, text):
    """The validation's check of a belief line by the act it implies: the log pi* of the teacher's best act less
    decision_score, in nats. 0 for the exact line and for every line that implies the teacher's own act; the best less LOGP_FLOOR
    (about 200) for a line the domain cannot read. planner must be the teacher as it made the plan (the same reliability table)."""
    Dk = teacher_of(D, plan)
    return max(plan["logp"].values(), default=plan["waste_lp"]) - decision_score(Dk, planner, plan, text)


def action_value(D, text, plan, table="values"):
    """V(a): the teacher's lookahead value of an action line the agent wrote (a domain brings a line to the teacher's form,
    D.canonical_action). An answer outside the posterior is worth 0. Any other line (one the helpers cannot do, one not
    understood, or an evidence action the teacher never takes) uses up a decision: it is worth answering one step later.
    table "logp": log pi*(a | h) instead, the teacher's policy (Planner): a wrong answer and a wasted decision score as the
    teacher's policy scores them, down to LOGP_FLOOR."""
    D = D.sub(plan["dom"]) if plan.get("dom") is not None and hasattr(D, "sub") else D   # a mixed run: the plan's teacher reads it
    key = D.canonical_action(text) if hasattr(D, "canonical_action") else text.strip()
    vals = plan[table]; lp = table == "logp"
    if key in vals: return vals[key]
    low = {norm(k): v for k, v in vals.items()}; t = norm(key)
    if t in low: return low[t]
    if (t[len("action:"):].strip() if t.startswith("action:") else t).startswith("submit"): return plan["wrong_lp"] if lp else 0.0
    return plan["waste_lp"] if lp else plan["waste"]


def action_logp(D, text, plan): return action_value(D, text, plan, "logp")


def look_stats(plans, ratio):
    """--tau_look logs, over the scored histories: look_gap, the mean gap in nats of log pi* between the best looks and the next
    level of looks (histories with two levels or more); look_accept, the share of the looks the accept rule (ratio) accepts (each
    history with looks counts once); tau_look_on, the share of histories whose looks were scored at --tau_look (not --tau)."""
    gap, acc, on = [], [], []
    for pl in plans:
        if not pl.get("looks"): continue
        lv = sorted({round(pl["logp"][s], 9) for s in pl["looks"].values() if s in pl["logp"]}, reverse=True)
        if len(lv) > 1: gap.append(lv[0] - lv[1])
        top = max(pl["logp"].values()) - math.log(ratio)
        acc.append(statistics.fmean(pl["logp"].get(s, LOGP_FLOOR) >= top for s in pl["looks"].values()))
        on.append(float(bool(pl.get("tl_on"))))
    fm_ = lambda x: statistics.fmean(x) if x else None
    return {"look_gap": fm_(gap), "look_accept": fm_(acc), "tau_look_on": fm_(on)}


def teacher_row(r):
    """--teacher_row: an M-step row that holds the teacher's own belief line and action (its tenth field), not an own turn."""
    return len(r) > 9 and bool(r[9])


def teacher_action(plan):
    """--teacher_row: the teacher's action at a history: the most likely one in "logp", the table the accept rule reads (accepted),
    so that it is accepted by construction; a plan without that table: the most likely one of its target "tgt". The first of
    equal ones."""
    t = plan.get("logp") or plan["tgt"]
    return max(t, key=t.get)


def own_rows(rows):
    """--teacher_row: the number of own turns among the rows of a history (the teacher's turn is not one)."""
    return sum(not teacher_row(r) for r in rows)


def add_teacher_rows(rows_h, crit_h, plans, ratio, score):
    """--teacher_row 1: for every history where no own turn takes an action the teacher accepts, one more turn: the history's first
    own thought with the teacher's exact belief line (plan "anchor") and the teacher's action (teacher_action), scored by
    score(row0, plan, line, action) -> (row, scores) as an own turn is. The row gets a tenth field True and its scores the mark
    "teacher"; a turn the teacher does not accept (not expected) is not added. Returns the number of histories that got one."""
    n = 0
    for rows, crit, pl in zip(rows_h, crit_h, plans):
        if not rows or any(accepted(c["A"], pl, ratio) for c in crit): continue
        r, c = score(rows[0], pl, pl["anchor"], teacher_action(pl))
        if r[8]: rows.append(tuple(r) + (True,)); crit.append({**c, "teacher": True}); n += 1
    return n


def own_w(wh, rows, M):
    """--teacher_row: the weights of the own turns of a history, for em_ess_think (a log value). A teacher row
    is not one of the M thought samples; without this the 64 + 1 weights of such a history did not reshape to (-1, M)."""
    wh = np.asarray(wh, dtype=float)
    if len(rows) == len(wh): wh = wh[[not teacher_row(r) for r in rows]]
    return wh[:len(wh) - len(wh) % M]


def cap_teacher_adv(A, flags, cap):
    """--teacher_row_cap: the advantage of a teacher row is at most cap nats (cap <= 0, the default: no limit); every other turn is
    unchanged. Without a limit the group is balanced: the own turns of a group of n lose the teacher row's advantage in total, and
    grpo_weights divides every advantage by n, so such a history weighs less than one with mixed own turns. A limit makes the
    history mostly a push-down of the own turns."""
    A = np.asarray(A, dtype=float); f = np.asarray(flags, dtype=bool)
    if not len(A) or not f.any() or cap <= 0: return A
    return np.where(f, np.minimum(A, cap), A)


STALL_INFO = 1e-9   # --stall_cap: a look whose expected evidence is at most this shows nothing (a numerical zero)


def look_info(B, a):
    """--stall_cap: the evidence that look a is expected to bring at belief B: the expected drop of the entropy of the
    answer in nats (the mutual information of the look's next observation and the answer). The teacher's own B.look_info
    (mastermind, minesweeper, minigrid, frozenlake, sokoban); else from B.outcomes and B.post (tabletop); None when the belief
    cannot say (the look then counts as one with evidence)."""
    if hasattr(B, "look_info"): return float(B.look_info(a))
    if hasattr(B, "outcomes") and hasattr(B, "post"):
        h = lambda p: -sum(v * math.log(v) for v in p.values() if v > 0)
        return max(0.0, h(B.post()) - sum(p * h(b.post()) for p, b in B.outcomes(a)))
    return None


def cap_stalls(logp, waste_lp, looks, cap, info):
    """--stall_cap: a look that is neither one of the teacher's best looks (the highest log pi* among the looks, the
    look value group its Boltzmann puts first) nor one with real expected evidence (info(a) above STALL_INFO) gets at most
    best - cap, best being the highest log pi* of any action (the accept rule's reference); an action that is not legal (waste_lp)
    too. A value is only lowered: the answers, the best looks, the looks with evidence and every value already below the line keep
    theirs, and so does best. With cap above log(--accept) (3.0 nats at 20) such a step is rejected. info is called only for the
    looks the cap would lower. Returns (logp, waste_lp, stall); stall = {"logp": the old values of the looks it lowered,
    "waste_lp": the old waste_lp, or None when it is unchanged}."""
    if cap <= 0 or not logp: return logp, waste_lp, None
    line = max(logp.values()) - cap
    top = max((logp[s] for s in looks.values() if s in logp), default=None)
    out, old = dict(logp), {}
    for a, s in looks.items():
        v = logp.get(s)
        if v is None or v <= line or v >= top: continue                # already below the line, or one of the best looks
        x = info(a)
        if x is None or x > STALL_INFO: continue                         # a look with real expected evidence (None: cannot say)
        out[s] = line; old[s] = v
    w = min(waste_lp, line)
    return out, w, {"logp": old, "waste_lp": waste_lp if w < waste_lp else None}


def capped_tgt(tgt, logp, stall):
    """--stall_play: the teacher's own moves when it plays by the capped scores too: the probability of each look
    that cap_stalls lowered is multiplied by exp(new - old), then the target is renormalized; every other move keeps its ratio to
    the others. A look the cap left alone, an answer and the target's commit mass are not changed otherwise."""
    st = (stall or {}).get("logp") or {}
    if not st: return tgt
    out = {s: p * math.exp(logp[s] - st[s]) if s in st else p for s, p in tgt.items()}
    z = sum(out.values())
    return {s: p / z for s, p in out.items()} if z > 0 else tgt


def stall_rejected(A, plan, ratio, A_old):
    """--stall_cap log: an own turn that the teacher rejects only because of the cap (its action's log pi* before the cap, A_old,
    is accepted)."""
    return not accepted(A, plan, ratio) and accepted(A_old, plan, ratio)


def uncapped(plan):
    """--stall_cap log: the plan with the values cap_stalls lowered put back (the plan itself when nothing was lowered)."""
    st = plan.get("stall")
    if not st or (not st["logp"] and st["waste_lp"] is None): return plan
    return {**plan, "logp": {**plan["logp"], **st["logp"]}, "waste_lp": plan["waste_lp"] if st["waste_lp"] is None else st["waste_lp"]}


def accepted(A, plan, ratio):
    """An own action with log pi* = A is one the teacher accepts: pi* gives it at least 1/ratio of the probability of its best action."""
    return A >= max(plan["logp"].values(), default=LOGP_FLOOR) - math.log(ratio)


def turn_score(line_score, line_mix, S, W, Sd, A, act=1):
    """The teacher's log-likelihood score of one turn (the critic's R, score_turn): the belief line's part by --line_score, then the
    action's log pi* A. reader / exact: S + W; decision: S_dec + line_mix (S + W); none (Sd None): no line part. act 0 (--act_score 0,
    the ablation without the action reward): R is the belief line's part only, no A."""
    line = S + W if line_score in ("reader", "exact") else Sd + line_mix * (S + W) if Sd is not None else 0.0
    return line + (A if act else 0.0)


def grpo_advantages(R, acc=None, clip=20.0):
    """--update grpo (RLVB (GRPO)): the group-relative advantage of each own turn of ONE history (its group): its score R (the
    critic's R with the length term) minus the mean score of the history's OTHER own turns (leave-one-out, as RLOO; a group of one
    turn, or of equal scores, gets 0). clip > 0: the scores are first raised to at least the group's best minus clip, so one turn
    at the score floor (log pi* -200) cannot lift the advantages of the others, and every advantage is limited to [-clip, clip]
    nats; 0: neither. acc (the teacher accepts the turn's action, score_turn): a turn the teacher rejects never gets a positive
    advantage, min(A, 0), RLVB (EM)'s accept rule (an unverified action is never a positive target): in a group of rejected turns
    the least bad one would otherwise be pushed up. The scores are shifted by the group's best first, so equal scores give exactly
    0 (not a rounding residue that grpo_weights' batch factor could scale up to a full step), and |A| below GRPO_ZERO is 0. A score
    that is not finite (not expected: every part of R has a floor) counts as the group's lowest finite score minus clip (200 nats
    when clip is 0); a group with no finite score gets 0."""
    R = np.asarray(R, dtype=float); n = len(R)
    if n < 2: return np.zeros(n)
    ok = np.isfinite(R)
    if not ok.any(): return np.zeros(n)
    if not ok.all(): R = np.where(ok, R, R[ok].min() - (clip if clip and clip > 0 else -LOGP_FLOOR))
    R = R - R.max()
    Rc = np.maximum(R, -clip) if clip and clip > 0 else R
    A = Rc - (Rc.sum() - Rc) / (n - 1)
    if clip and clip > 0: A = np.clip(A, -clip, clip)
    A[np.abs(A) < GRPO_ZERO] = 0.0
    if acc is not None: A = np.where(np.asarray(acc, dtype=bool), A, np.minimum(A, 0.0))
    return A


GRPO_ZERO = 1e-9     # --update grpo: an advantage or a batch's total weight below this many nats is no signal (rounding)


def grpo_weights(A_h, n_fit, norm=True):
    """--update grpo: the M-step weights of a batch from its histories' advantages A_h: each turn's advantage over its history's
    number of turns (the group mean, as GRPO's per-group loss), times one factor for the batch. norm (--grpo_norm 1): the factor
    n_fit / sum |A / n|, so every batch has the total absolute weight of --update em's (the E-step's weights sum to 1 per fitted
    history, n_fit of them); the relative weighting of the histories inside a batch is kept. With AdamW and the gradient-norm clip
    of each block a constant factor would not change the step; the factor varies from batch to batch, so it evens out how much each
    batch counts in Adam's running averages, as GRPO's advantage normalisation does. A batch whose total is below GRPO_ZERO has no
    signal: all its weights are 0 (factor 0). Returns (the weights per history, the factor)."""
    A_n = [np.asarray(a, dtype=float) / len(a) if len(a) else np.asarray(a, dtype=float) for a in A_h]
    tot = float(sum(np.abs(a).sum() for a in A_n))
    if not tot >= GRPO_ZERO * max(1, n_fit): return [np.zeros_like(a) for a in A_n], 0.0
    sc = n_fit / tot if norm else 1.0
    return [a * sc for a in A_n], sc


def block_counts(tok, it):
    """[thought, decision]: the counted tokens of an M-step item's two blocks by score()'s own rule, so that the PPO stop of --update
    grpo divides each block's log p (score() sums the counted tokens only) by the same number of tokens: the continuation it[1]
    encoded with offsets; a token is counted unless it overlaps one of the spans it[2] (a protocol-forced </think>, the TERM
    between the lines under --dec_term own, a belief line --line_target masks), and is a decision token when it starts at or
    after it[3] (d0)."""
    span = it[2] if len(it) > 2 else None; d0 = it[3] if len(it) > 3 else None
    sp_ = [] if not span else (list(span) if isinstance(span[0], (tuple, list)) else [span])   # an empty list = no span
    n = [0, 0]
    for a, b in tok(it[1], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]:
        if any(a < e_ and b > s_ for s_, e_ in sp_): continue
        n[int(d0 is not None and a >= d0)] += 1
    return n


def estep_weights(R, acc, alpha, lam=1.0, accept_only=False):
    """The E-step weights of one history: softmax(lam R / alpha) over its own turns. accept_only (--accept_only; the action must
    at least be acceptable): only the turns whose action the teacher accepts share the weight, and a turn with a
    rejected action gets 0 whatever its belief line (where every line leaves out part of the posterior, the line's floor could
    otherwise give the weight to a rejected action). A history with no accepted turn gets no weight at all (an
    unverified action is never a positive target; the E-step leaves it out)."""
    x = lam * np.asarray(R, dtype=float) / alpha
    if not len(x): return np.zeros(0)
    if accept_only:
        a = np.asarray(acc, dtype=bool)
        if not a.any(): return np.zeros(len(x))
        x = np.where(a, x, -np.inf)
    w = np.exp(x - x.max()); return w / w.sum()


def no_target(crit, plan, ratio, floor=None):
    """Why one history has no positive target for the E-step (--accept_only; an unverified target is never a
    positive one): "no_accept" when no own turn takes an action the teacher accepts, "wrong_line" when every accepted turn's
    belief line has none of the posterior (the line score's floor -floor: a content error, the line is read), else None.
    floor None: the line is not scored."""
    acc = [c for c in crit if accepted(c["A"], plan, ratio)]
    if not acc: return "no_accept"
    if floor is not None and not any(c["S"] > -floor + 1e-6 for c in acc): return "wrong_line"
    return None


def think_cost_at(spec, step, phase):
    """--think_cost: the price of a thought in nats per 1,000 tokens at update step (1, 2, ...): the phase's value of the comma
    list (the first value in phase 1, the last value in every later phase); 0 without a list. --length_reward's C by phase alike."""
    vals = [float(x) for x in str(spec or "").split(",") if x.strip()]
    if not vals: return 0.0
    return vals[min((step - 1) // phase if phase else 0, len(vals) - 1)]


def length_tiebreak(w, R, lengths, kappa):
    """--think_cost: the E-step weights w of one history, where each group of samples that the teacher scores alike (the same R)
    keeps its total weight and shares it in proportion to exp(-kappa * thought tokens / 1000). Among the thoughts that reach the
    same belief and action the shorter gets more weight; the weight of each belief and action stays the E-step's."""
    w = np.asarray(w, dtype=float)
    if not kappa: return w
    w = w.copy(); key = np.round(np.asarray(R, dtype=float), 9); L = np.asarray(lengths, dtype=float)
    for r in np.unique(key):
        g = key == r
        if g.sum() < 2 or w[g].sum() <= 0: continue
        f = np.exp(-kappa * (L[g] - L[g].min()) / 1000.0); w[g] = w[g].sum() * f / f.sum()
    return w


def length_reward(lengths, ok, budget=0):
    """--length_reward: the length term of each own turn of one history (Kimi k1.5's length reward, arXiv 2501.12599, over the
    history's turns: for a correct answer the shorter reasoning gets more, a wrong answer never gets a positive
    length reward and a long one gets a penalty). lam = 0.5 - (L - min L) / (max L - min L); a correct turn gets lam, a wrong one
    min(0, lam); all 0 when every thought has the same length.
    budget > 0 (--length_budget: one rule for every model, the term only where a thought does not fit the test's budget):
    a correct turn whose thought fits (L <= budget) gets 0, and one over it gets -0.5 (L - budget) / (max L - budget), from 0 at
    the budget to -0.5 at the history's longest thought; a wrong turn keeps min(0, lam) as above. So the shorter of two correct
    thoughts that both fit gets no extra weight; only a thought that the test would cut pays."""
    L = np.asarray(lengths, dtype=float)
    if not len(L) or L.max() == L.min(): return np.zeros(len(L))
    lam = 0.5 - (L - L.min()) / (L.max() - L.min())
    wrong = np.minimum(0.0, lam)
    if budget and budget > 0:
        over = L > budget
        right = np.where(over, -0.5 * (L - budget) / (L.max() - budget), 0.0) if over.any() else np.zeros(len(L))
        return np.where(np.asarray(ok, dtype=bool), right, wrong)
    return np.where(np.asarray(ok, dtype=bool), lam, wrong)


def budget_share(w, ok, stopped, lengths, budget):
    """The budget measure of one history's E-step target: (the weight on correct turns whose thought ended by itself
    within budget tokens, the weight on correct turns), both over the history's total weight; None for a history with no weight.
    A budget below 0 is no budget (unlimited reasoning): a thought that ended by itself fits."""
    w = np.asarray(w, dtype=float); s = w.sum()
    if not len(w) or s <= 0: return None
    ok = np.asarray(ok, dtype=bool); fits = ok & np.asarray(stopped, dtype=bool) & ((np.asarray(lengths, dtype=float) <= budget) | (budget < 0))
    return float(w[fits].sum() / s), float(w[ok].sum() / s)


def restricted_policy(model, tok, pre, legal, chunk):
    """Probabilities of the legal action strings after each prefix, normalised over the legal set."""
    pairs, owner = [], []
    for i, (p, strings) in enumerate(zip(pre, legal)):
        for s in strings: pairs.append((p, s + TERM, None, 0)); owner.append(i)
    with torch.no_grad(): lp = score(model, tok, pairs, chunk).float().cpu().numpy() if pairs else np.zeros(0)
    out = []
    for i in range(len(pre)):
        seg = lp[[j for j in range(len(pairs)) if owner[j] == i]]
        out.append(E.softmax_np(seg) if len(seg) else np.zeros(0))
    return out


# --------------------------------------------------------------------- training
def train(args):
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))          # a plain exit runs atexit, which stops the vLLM server
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0")); lead = rank == 0; dist = None
    actor, learner = args.role == "actor", args.role == "learner"   # --role: the actor makes the batches, the learner fits them (--queue)
    global NO_BELIEF_LINE; NO_BELIEF_LINE = bool(args.no_belief_line)   # on every rank, before a worker rank's job loop (turn_item)
    assert not args.no_belief_line or (args.objective == "critic" and args.line_score == "none" and args.credit == "own" and not args.neg_answer
                                       and not args.neg_thought and not args.teacher_row and args.readout <= 0 and args.act_pick <= 1
                                       and args.line_target < 0 and args.em_more_joint <= 0 and not args.line_wide), \
        ("--no_belief_line 1: the critic with --line_score none (no belief reward), --credit own, no --neg_answer / --neg_thought / --teacher_row / "
         "--readout / --act_pick / --line_target / --em_more_joint / --line_wide (each reads or writes a belief line)")
    pp = score_settings(args)                                         # on every rank, before a worker rank's job loop
    assert not (actor or learner) or args.queue, "--role actor/learner: set --queue, the folder both processes share"
    assert not args.colocate or (args.phase_set and args.phase), "--colocate: with --phase_set and --phase (the GPUs switch at the phase ends)"
    assert not args.resume_step or (args.phase_set and args.phase and args.resume_step % args.phase == 0), "--resume_step: a --phase_set run, at a phase end"
    assert args.set_draw == "passes" or (args.phase_set and args.objective == "critic" and 0 <= args.set_prio < 1), \
        "--set_draw is: with --phase_set and the critic objective, 0 <= --set_prio < 1"
    assert not args.set_keep or (args.phase_set and args.objective == "critic" and (actor or learner)), \
        "--set_keep: the critic with --phase_set and --role actor/learner (python -m plearn train)"
    assert args.neg_answer == 0 or (args.objective == "critic" and args.m_step == "blocks" and args.accept_only), \
        "--neg_answer: the critic with --m_step blocks and --accept_only 1 (its histories are the ones accept_only leaves out)"
    assert args.neg_thought == 0 or (args.objective == "critic" and args.m_step == "blocks" and args.accept_only), \
        "--neg_thought: the critic with --m_step blocks and --accept_only 1"
    assert args.act_score or (args.objective == "critic" and args.line_score != "none"), \
        "--act_score 0: the critic with a scored belief line (--line_score not none), else R has no part left"
    assert args.accept_rule or not (args.accept_only or args.set_keep or args.em_more or args.neg_answer or args.neg_thought or args.credit == "share"), \
        "--accept_rule 0: no rule that reads the teacher's acceptance of the action (--accept_only 0, --set_keep 0, --em_more 0, --neg_answer 0, --neg_thought 0, --credit own)"
    assert args.line_target < 0 or (args.objective == "critic" and args.m_step == "blocks"), \
        "--line_target: the critic with --m_step blocks (the thought block has its own weights there)"
    assert args.update == "em" or (args.objective == "critic" and args.m_step == "blocks" and args.credit == "own" and args.neg_answer == 0
                                    and args.neg_thought == 0 and args.grpo_clip >= 0 and 0 <= args.grpo_eps < 1), \
        "--update grpo: the critic objective with --m_step blocks, --credit own, no --neg_answer / --neg_thought, --grpo_clip >= 0, 0 <= --grpo_eps < 1"
    assert args.update == "em" or (args.temper == 0 and all(float(x) == 0 for x in str(args.think_cost or "0").split(",") if x.strip())), \
        "--update grpo: no --temper and no --think_cost (they change only the E-step's weights, which grpo does not train with)"
    assert not actor or world == 1, "--role actor: one process (its vLLM servers are in --vllm_gpu and PL_VLLM_REMOTE)"
    assert not (actor or learner) or not args.phase or args.eval_every % args.phase == 0, \
        "--role actor/learner: --eval_every must be a multiple of --phase (the actor has a model only at the phase ends)"
    assert not kl_measured(args) or (args.objective == "critic" and args.phase and args.phase_set and (actor or learner) and args.kl_probe > 0), \
        "--kl_stop, --kl_log: the critic with --phase, --phase_set and --role actor/learner (python -m plearn train), --kl_probe > 0"
    if world > 1:
        import datetime, torch.distributed as dist
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0"))); dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6))
        CPU_GROUP = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=48))   # job messages on the CPU: a rank waiting for work uses no GPU (an NCCL wait spins on the GPU beside vLLM)
    dargs = json.loads(args.domain_args)
    Dmod = importlib.import_module(args.domain)
    images = bool(int(dargs["images"])) if "images" in dargs else getattr(Dmod.Domain, "images", False)   # every rank knows it (the learner
    D = Dmod.Domain(**dargs) if lead and not learner else None      # has no domain); the environment lives on rank 0 only (the actor's, with --role)
    D_run = D
    def same(a, b, pl=None):                                          # a mixed run: the plan's own teacher compares the lines
        Dk = D_run.sub(pl["dom"]) if (pl or {}).get("dom") is not None and hasattr(D_run, "sub") else D_run
        return Dk.anchor_equal(a, b) if hasattr(Dk, "anchor_equal") else E.note_set(a) == E.note_set(b)
    rseed = args.seed * 1000003 + args.resume_step if args.resume_step else args.seed   # --resume_step: new draws (no scene of the first part again)
    rng = random.Random(rseed); random.seed(rseed); torch.manual_seed(rseed); nrng = np.random.default_rng(rseed)
    dom_name = getattr(Dmod.Domain, "name", None) or D.name         # the class's name first: the learner builds no domain, and the
    tag = f"{dom_name}{('-' + args.tag) if args.tag else ''}-seed{args.seed}"   # two roles must name the run's files alike
    tok = AutoTokenizer.from_pretrained(args.model)
    if images:
        global PROC
        from transformers import AutoProcessor
        PROC = AutoProcessor.from_pretrained(args.model)
        mm_size = os.environ.get("PL_MM_SIZE") or getattr(Dmod.Domain, "mm_size", None)   # the domain's picture size (for example the lab's
        if mm_size:                                                   # pixel limits), the same in this processor and in the vLLM server
            os.environ["PL_MM_SIZE"] = mm_size; size = json.loads(mm_size); PROC.image_processor.size = size
            for k, v in (("min_pixels", size["shortest_edge"]), ("max_pixels", size["longest_edge"])):
                if hasattr(PROC.image_processor, k): setattr(PROC.image_processor, k, v)
    model = params = opt = None                                       # --role actor: no trainer model (it only samples through vLLM)
    if not actor:
        from plearn.models import attn_implementation              # --attn: sdpa (every RLVB (EM) run so far) unless the run file names a flash attention;
        ATTN["name"] = "sdpa" if args.attn == "sdpa" else attn_implementation(os.environ.get("PL_ATTN") or args.attn)   # PL_ATTN: an A100 host's FA2
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=ATTN["name"])   # Qwen3-VL and Qwen3.5 alike
        print("attention", ATTN["name"], "(checked against sdpa at the first score)" if ATTN["name"] != "sdpa" else "", flush=True)
        if not args.colocate: model = model.cuda()                    # --colocate: in CPU memory until the learner's first phase
        print("lora layers", E.add_lora(model, r=args.lora_r, all_linear=bool(args.lora_all_linear)), "all_linear", bool(args.lora_all_linear), flush=True)
        if args.init_lora:
            res = model.load_state_dict(torch.load(args.init_lora, map_location=model.device), strict=False); assert not res.unexpected_keys, res.unexpected_keys
            print("initialised LoRA from", args.init_lora, flush=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        if args.resume_step:                                          # --resume_step R: the model after update R and Adam's state then
            res = model.load_state_dict(torch.load(os.path.join(args.out, f"{tag}-lora-{args.resume_step}.pt"), map_location=model.device), strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
            f_opt = os.path.join(args.out, f"{tag}-opt-{args.resume_step}.pt")
            if os.path.exists(f_opt): opt.load_state_dict(torch.load(f_opt, map_location="cpu"))
            print(f"resumed after update {args.resume_step}: its adapter", "and Adam's state" if os.path.exists(f_opt) else
                  "(Adam's state was not saved: its moments start again)", flush=True)
    READOUT.clear()
    if args.readout > 0 and lead and not actor:                       # the belief readout (--readout): readiness per turn, action preference per candidate;
        assert not images and args.objective == "critic" and args.m_step == "blocks" and args.line_score == "exact", \
            "--readout: text domains, the critic with --m_step blocks and --line_score exact"                # rank 0 only: it reads rank 0's share
        READOUT.update({"head": Readout(model.get_output_embeddings().in_features).cuda(), "norm": find_norm(model)})
        print("readout: states from", [n for n, m in model.named_modules() if m is READOUT["norm"]], flush=True)
        if args.init_readout:                                         # older readouts also hold the retired reliability output
            sd = {k: v for k, v in torch.load(args.init_readout, map_location="cuda").items() if not k.startswith("rel.")}
            res = READOUT["head"].load_state_dict(sd, strict=False); assert not res.unexpected_keys, res.unexpected_keys
            print("initialised readout from", args.init_readout, "missing (kept at zero):", res.missing_keys, flush=True)
    if args.act_pick > 1 and lead: assert READOUT.get("head") is not None, "--act_pick: needs --readout (and the trainer model: not with --role actor)"
    assert not args.rho_rb or (args.rho == "fidelity" and args.objective == "critic" and args.episodes), "--rho_rb: --rho fidelity, the critic, episodes"
    opt_ro = torch.optim.AdamW(READOUT["head"].parameters(), lr=args.readout_lr, weight_decay=0.0) if READOUT.get("head") is not None else None
    ro_buf = {"ready": __import__("collections").deque(maxlen=args.readout_buffer),
              "act": __import__("collections").deque(maxlen=max(16, args.readout_buffer // 8))}   # act: groups (a history's candidate states, their log pi*)

    def readout_fit(st):
        """The readout's own fitting (its parameters only): the update's detached states at the reading points join a buffer of
        recent updates, and --readout_inner Adam steps fit the readiness cross-entropy on mini-batches of the buffer and the
        listwise cross-entropy of the action score on up to 64 of its candidate groups (the states move with the LoRA, so the
        buffer is short). Returns the last loss."""
        for h, t in zip(st["h_ready"], st["rtgt"]): ro_buf["ready"].append((h, t.detach().float().cpu()))
        groups = {}
        for h, t, gs in zip(st["h_act"], st["atgt"], st["agrp"]):      # this update's candidates by history
            for hi, ti, g in zip(h, t.detach().float().cpu(), gs): groups.setdefault(g, ([], []))[0].append(hi); groups[g][1].append(float(ti))
        for hs, lps in groups.values():
            if len(hs) >= 2 and max(lps) - min(lps) > 1e-6: ro_buf["act"].append((torch.stack(hs), torch.tensor(lps)))
        if not ro_buf["ready"] and not ro_buf["act"]: return None
        Xr = torch.cat([h for h, _ in ro_buf["ready"]]).cuda().float() if ro_buf["ready"] else None
        Yr = torch.cat([t for _, t in ro_buf["ready"]]).cuda() if ro_buf["ready"] else None
        head = READOUT["head"]; head.train(); last = None; ng = min(64, len(ro_buf["act"]))
        for _ in range(args.readout_inner):
            loss = torch.zeros((), device="cuda")
            if Xr is not None:
                b = torch.randint(0, len(Xr), (min(args.readout_batch, len(Xr)),), device="cuda")
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(head(Xr[b])[0], Yr[b])
            for gi in torch.randperm(len(ro_buf["act"]))[:ng].tolist():   # the action score: its softmax over a history's candidates against pi*
                H, LP = ro_buf["act"][gi]; s = head(H.cuda().float())[1]
                loss = loss - (torch.softmax(LP.cuda(), 0) * torch.log_softmax(s, 0)).sum() / ng
            opt_ro.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt_ro.step(); last = loss.item()
        opt_ro.zero_grad(); head.eval()
        return last

    jrng = np.random.default_rng(args.seed + 11)                      # the same on every rank: each draws for all histories
    own_prior = args.objective != "critic" or (args.phase or (1 if args.proposal == "policy" else 0)) > 0   # the E-step samples from the LoRA
    grpo_old = {}                                                     # --update grpo: (prompt, continuation) -> its [thought, decision] log p under the phase prior
    prior = {"sd": None, "probe": None}                               # --credit share: the LoRA at the start of the phase (the vLLM adapter);
                                                                      # --kl_stop: this rank's probe of the phase (probe_start)

    @contextlib.contextmanager
    def prior_policy():
        """The phase prior pi_k in the trainer's model: the base model when the E-step samples from it, else the LoRA as it was
        at the first update of the phase (the adapter the vLLM servers sample from); the current LoRA is put back after."""
        if not own_prior:
            with base_model(model): yield
            return
        cur = [p.data.clone() for p in params]
        for p, q in zip(params, prior["sd"]): p.data.copy_(q)
        try: yield
        finally:
            for p, c in zip(params, cur): p.data.copy_(c)

    def temper(R):
        """--temper: lam_h, the largest lam in [0, 1] with sum_i w_i log(N w_i) <= EPS for w = softmax(lam R / alpha) (bisection;
        the sample KL grows with lam)."""
        N = len(R)
        kl = lambda lam: float((lambda w: (w * torch.log((w * N).clamp_min(1e-300))).sum())(torch.softmax(lam * R / args.alpha, 0)))
        if N < 2 or kl(1.0) <= args.temper: return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if kl(mid) <= args.temper else (lo, mid)
        return lo

    def share_credit(rr, wj):
        """--credit share at one history: rr its own rows (M per thought), wj their weights. Returns the M-step candidates
        (thought row, decision row, weight), the weight moved off the decisions' own thoughts and the number of shared decisions."""
        M = args.em_m; K = len(rr) // M
        dec = sorted({(r[3], r[4]) for r in rr if len(r) > 8 and r[8]}, key=lambda d: (-max(r[5] for r in rr if (r[3], r[4]) == d), d))[:args.share_max]
        if not dec or K < 2: return [(i, i, float(x)) for i, x in enumerate(wj)], 0.0, 0, None
        pairs = [(rr[t * M][2], b + TERM + u + TERM, None, 0) for t in range(K) for b, u in dec]
        with torch.no_grad(), prior_policy(): lpd = score(model, tok, pairs, args.score_chunk).double().cpu().view(K, len(dec))
        rmat = torch.softmax(lpd, 0).numpy(); col = {d: c for c, d in enumerate(dec)}; cand, moved, r_own = [], 0.0, []
        for i, (r, x) in enumerate(zip(rr, wj)):
            c = col.get((r[3], r[4])) if len(r) > 8 and r[8] else None
            if c is None: cand.append((i, i, float(x))); continue
            cand += [(t * M, i, float(x * rmat[t, c])) for t in range(K)]; moved += float(x * (1.0 - rmat[i // M, c])); r_own.append(float(rmat[i // M, c]))
        return cand, moved, len(dec), float(np.mean(r_own))

    def probe_pairs(rows_, idx):
        """The probe's own turns as score() pairs: the thought under SAMPLER, the decision under DECISION, the </think> of a thought
        cut at --max_think not counted (as the M-step items)."""
        out = []
        for i in idx:
            r = rows_[i]; cont, sp_, d0 = turn_item(r[1], r[3], r[4], r[6], args.dec_term == "own")
            out.append((r[0], cont, sp_, d0))
        return out

    def probe_start(rows_, sizes):
        """--kl_stop at the first update of a phase: this rank's share of the probe (kl_probe) and the images of its prompts (the
        batch's images go after the update)."""
        from plearn.models.sampler import IMAGES, prompt_key
        pairs = probe_pairs(rows_, kl_probe(rows_, sizes, args.kl_probe)[rank::world])
        prior["probe"] = (pairs, {k: IMAGES[k] for k in {prompt_key(p[0]) for p in pairs} if k in IMAGES})

    def probe_kl():
        """--kl_stop after an update: the exact KL(pi_n || pi_phi) per position of the next-token distributions along the probe's own
        turns (kl_states; pi_n is the LoRA as it was at the phase's first update, prior_policy), over all ranks (every rank calls it)."""
        from plearn.models.sampler import IMAGES
        pairs, imgs = prior["probe"]; s = n = 0.0; head = model.get_output_embeddings()
        new = {k: v for k, v in imgs.items() if k not in IMAGES}; IMAGES.update(new)
        try:
            model.eval()
            with torch.no_grad():
                for p in pairs:
                    (h1, d1), = score(model, tok, [p], 1, states=True)
                    with prior_policy(): (h0, _), = score(model, tok, [p], 1, states=True)
                    s_, n_ = kl_states(head, h0, h1, d1, SAMPLER["T"], DECISION["T"]); s += s_; n += n_
        finally:
            for k in new: IMAGES.pop(k, None)
        if world > 1:
            t = torch.tensor([s, n], device=model.device, dtype=torch.float64); dist.all_reduce(t); s, n = t.tolist()
        return s / n if n else None

    def same_settings():
        """A data-parallel update: every rank scores with the same sampler settings (settings_row), else the run stops here."""
        cs = torch.tensor(settings_row(), device=model.device, dtype=torch.float64)
        al = [torch.zeros_like(cs) for _ in range(world)]; dist.all_gather(al, cs)
        assert all(torch.equal(a, al[0]) for a in al), f"the ranks score with different sampler settings: {[a.tolist() for a in al]}"

    def em_update(rows_, nh, sizes, snap=False, hw=None, kappa=0.0, lrw=0.0, okr=None, neg=None, ltg=None):
        """One update on this rank's share of the histories (the critic: dealt by their M-step work, deal(); else h % world ==
        rank; --credit share keeps h % world == rank, its credit needs the model): the own thoughts are scored, weighted by
        p(beta*, u* | h, z_k) and backpropagated (loss over the global batch size); gradients are summed over the ranks.
        rows_ holds the samples of history 0, then those of history 1, ...: sizes[h] of them (0: the E-step leaves h out; the
        loss is still divided by nh, so the other histories keep their weight). hw (--set_draw is): history h's loss times hw[h].
        kappa: the --think_cost of this update; lrw: its --length_reward; okr[i]: row i's belief line is correct (line_ok).
        ltg[i] (--line_target): row i's belief line is verified, a positive target; else the row trains its action only (target_item).
        Returns the summed losses, the gradient norm and per history (h, [[log p(beta*), log p(u*)] per thought], weights);
        with the critic, per sample [R] instead (rows carry the own belief line, the own action and R)."""
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        critic = args.objective == "critic"
        grpo = critic and args.update == "grpo"                       # --update grpo: signed advantages in place of the E-step's weights
        if world > 1: same_settings()
        model.eval()
        if snap: prior["sd"] = [p.detach().clone() for p in params]  # the first update of a phase: the LoRA is the phase prior
        if snap or not (args.phase or args.proposal == "policy"):    # --update grpo: its prior scores live one phase (without phases, one
            grpo_old.clear()                                          # update); train()'s `phase` is not bound yet in a worker rank's loop
        if snap and kl_measured(args): probe_start(rows_, sizes)      # --kl_stop, --kl_log: the phase's probe (own turns of its prior)
        if critic:                                                    # E-step: q ~ p exp(R / alpha), from the prior's own samples
            seg = [range(off[h], off[h + 1]) for h in range(nh)]      # first for every history: no model work, the same on every rank
            R = [torch.tensor([rows_[i][5] for i in s], dtype=torch.float64) for s in seg]
            acc_h = [[bool(rows_[i][8]) if len(rows_[i]) > 8 else True for i in s] for s in seg]   # the teacher accepts the turn's action (score_turn)
            ok_h = [[(a or not args.accept_rule) and (okr is None or bool(okr[i])) for a, i in zip(a_, s)] for a_, s in zip(acc_h, seg)]   # correct: accepted, line_ok (--accept_rule 0: line_ok only)
            n_tok = {}
            def nt(z):
                if z not in n_tok: n_tok[z] = len(tok.encode(z, add_special_tokens=False))
                return n_tok[z]
            L = [np.array([nt(rows_[i][1]) for i in s], dtype=float) for s in seg]   # thought tokens: the length terms, the budget measure
            if lrw:                                                   # --length_reward: R + C x the length term of each own turn
                R = [r + lrw * torch.tensor(length_reward(l_, o_, args.length_budget), dtype=torch.float64) for r, l_, o_ in zip(R, L, ok_h)]
            lam = [temper(r) if args.temper > 0 else 1.0 for r in R]    # --temper: the largest step the samples carry
            w = [estep_weights(r.numpy(), a_, args.alpha, l_, bool(args.accept_only)) for l_, r, a_ in zip(lam, R, acc_h)]
            if kappa:                                                 # --think_cost: among the samples alike, the shorter thoughts
                w = [length_tiebreak(x, r.numpy(), l_, kappa) for x, r, l_ in zip(w, R, L)]
            bud = [b for b in (budget_share(x, o_, [rows_[i][6] for i in s], l_, args.eval_max_think or args.max_think)
                               for x, o_, s, l_ in zip(w, ok_h, seg, L)) if b is not None]
            fit_L = [(x, l_) for x, l_ in zip(w, L) if len(l_) and x.sum() > 0]
            MSTEP.update({"think_cost": kappa, "length_reward": lrw,
                          "think_tokens_p": float(np.mean([l_.mean() for _, l_ in fit_L])) if fit_L else 0.0,
                          "think_tokens_q": float(np.mean([float(x @ l_) for x, l_ in fit_L])) if fit_L else 0.0,
                          "em_q_budget": float(np.mean([b[0] for b in bud])) if bud else 0.0, "em_q_correct": float(np.mean([b[1] for b in bud])) if bud else 0.0})
            gw = [None] * nh                                          # --update grpo: each own turn's weight (w stays the E-step's, for the logs)
            if grpo:                                                  # the same on every rank: every history's advantages, one batch factor
                A_h = [grpo_advantages(r.numpy(), a_ if args.accept_rule else None, args.grpo_clip)
                       for r, a_ in zip(R, acc_h)]   # --accept_rule 0: no min(A, 0)
                if any(teacher_row(r_) for r_ in rows_):               # --teacher_row: its advantage is limited (--teacher_row_cap)
                    A_h = [cap_teacher_adv(a, [teacher_row(rows_[i]) for i in sg], getattr(args, "teacher_row_cap", 0.0)) for a, sg in zip(A_h, seg)]
                gw, sc = grpo_weights(A_h, sum(1 for s in seg if len(s)), bool(args.grpo_norm))
                adv = np.concatenate([a for a in A_h if len(a)]) if any(len(a) for a in A_h) else np.zeros(0)
                MSTEP.update({"grpo_scale": sc,                       # the batch factor (--grpo_norm 1: RLVB (EM)'s total weight over the raw one)
                              "grpo_adv_abs": float(np.abs(adv).mean()) if len(adv) else 0.0,
                              "grpo_adv_neg": float((adv < -1e-9).mean()) if len(adv) else 0.0,     # share of own turns pushed down
                              "grpo_adv_pos": float((adv > 1e-9).mean()) if len(adv) else 0.0,      # share pushed up
                              "grpo_adv_zero_hist": float(np.mean([not np.abs(a).sum() for a in A_h if len(a)])) if len(adv) else 0.0})   # groups with no signal
            u = (jrng.random((nh, 1)) + np.arange(args.em_j)) / args.em_j if args.em_j else [None] * nh   # one stream for all ranks
            def draw(r, x):                                           # J draws from q per history, by systematic resampling
                if x is None or not len(r) or not r.sum(): return r
                return np.bincount(np.minimum(np.searchsorted(np.cumsum(r), x, side="right"), len(r) - 1), minlength=len(r)) / args.em_j
            if args.credit == "share" or world == 1:
                mine = [h for h in range(nh) if h % world == rank]
            else:                                                     # the histories dealt to the ranks by their M-step work, the characters
                work = [sum(len(rows_[i][0]) + len(rows_[i][1]) + len(rows_[i][3]) + len(rows_[i][4])      # of the items the J draws keep
                            for i, f in zip(seg[h], np.abs(gw[h]) if grpo else draw(w[h], u[h])) if f > (1e-9 if grpo else 1e-3))
                        + sum(len(r[0]) + len(r[1]) + len(r[3]) + len(r[4]) for r in ((neg or [None] * nh)[h] or [])) for h in range(nh)]   # --neg_answer
                parts = deal(work, world); mine = parts[rank]
                load = [sum(work[h] for h in p) for p in parts]; old = [sum(work[r::world]) for r in range(world)]
                MSTEP["m_balance"] = max(load) / max(1e-9, float(np.mean(load)))     # the slowest rank's work over the mean (1 = even);
                MSTEP["m_balance_rr"] = max(old) / max(1e-9, float(np.mean(old)))      # the same for the old split h % world
            seg, R, lam, w, u, gw = ([x[h] for h in mine] for x in (seg, R, lam, w, u, gw)); lp2 = [r.view(-1, 1).float() for r in R]
            sub_ = [rows_[i] for s in seg for i in s]; pos = [(j, k) for j, s in enumerate(seg) for k in range(len(s))]
            extra = [None] * len(mine); cand = []                     # M-step candidates per history: (thought row, decision row, weight)
            for j, s in enumerate(seg):
                if args.credit == "share" and len(s):                 # --credit share: each accepted decision's weight over the thoughts
                    c, moved, n_sh, r_own = share_credit([rows_[i] for i in s], w[j])
                else:
                    c, moved, n_sh, r_own = [(k, k, float(x)) for k, x in enumerate(gw[j] if grpo else w[j])], 0.0, 0, None
                cand.append(c); extra[j] = {"lam": lam[j], "moved": moved, "n_share": n_sh, "r_own": r_own}
            wc = [np.array([x for _, _, x in c]) for c in cand]
            wfit = wc if grpo else [draw(r, x) for r, x in zip(wc, u)]    # --update grpo: every own turn with its signed weight, no draws
        else:
            mine = [h for h in range(nh) if h % world == rank]
            seg = [range(off[h], off[h + 1]) for h in mine]
            sub_ = [rows_[i] for s in seg for i in s]; pos = [(j, k) for j, s in enumerate(seg) for k in range(len(s))]
            extra = [None] * len(mine)
            K = sizes[0] if sizes else 1; assert all(s == K for s in sizes), "the strings objective has K thoughts per history"
            pairs = [q for _, _, pre, anc, a in sub_ for q in ((pre, anc + TERM), (pre + anc + TERM, a + TERM))]
            with torch.no_grad(): lp2 = score(model, tok, pairs, args.score_chunk).float().view(-1, 2) if pairs else torch.zeros(0, 2)
            w = list(torch.softmax(lp2.sum(1).view(len(mine), K), 1).cpu().numpy()); lp2 = list(lp2.view(len(mine), K, 2))
        model.train(); opt.zero_grad()
        items_think, items_short, w_think, w_short = [], [], [], []
        w_tt = [] if (ltg is not None or any(teacher_row(r_) for r_ in rows_)) else None   # --line_target, --teacher_row: the thought block's own weights
        sw_lt = [0.0, 0.0]                                           # --line_target: the fitted weight, and its part on turns with a verified line
        for r, (j, k) in zip(sub_, pos):
            prompt, z, pre, anc, a = r[:5]
            wk = float(w[j][k]); adv = wk - 1.0 / len(w[j])
            if critic: continue                                        # the critic's items: below, from the candidates
            if len(w[j]) > 1 and z and abs(adv) > 1e-3:                        # own thought: advantage over the mean, per token
                items_think.append((prompt, z + CLOSE)); w_think.append(adv / max(1, len(tok.encode(z + CLOSE, add_special_tokens=False))))
            if wk > 1e-3: items_short += [(pre, anc + TERM, None, 0), (pre + anc + TERM, a + TERM, None, 0)]; w_short += [wk, wk]
        if critic:                       # each candidate (own thought z_t, own decision (b_i, a_i)) with its weight >= 0; a thought cut at
            mode = "blocks" if args.m_step == "blocks" else args.think     # --max_think gets its </think> from the protocol, not trained
            for j, s in enumerate(seg):
                rr = [rows_[i] for i in s]
                for (t, i, _), wf in zip(cand[j], wfit[j]):
                    if (abs(wf) <= 1e-9) if grpo else (wf <= 1e-3): continue    # --update grpo: negative weights too
                    rt, ri = rr[t], rr[i]; z, anc, a = rt[1], ri[3], ri[4]; d0 = len(z) + len(CLOSE)
                    ro = ri[7] if t == i and len(ri) > 7 and ri[7] and READOUT.get("head") is not None else None     # --readout: own pairs, the action's span
                    if ro: ro = {**ro, "act": [(d0 + len(anc) + len(TERM), d0 + len(anc) + len(TERM) + len(a), ro["act_lp"], j)] if a.strip() else []}
                    # --line_target masks an unverified line as a POSITIVE target; a turn pushed down (grpo, wf < 0) keeps every token
                    cont, sp_, _, tf = target_item(z, anc, a, rt[6], args.dec_term == "own", ltg is None or bool(ltg[s[i]]) or wf < 0)
                    items_think.append((rt[0], cont, sp_, d0, mode) + ((ro,) if ro else ()))
                    w_think.append(float(wf) * (hw[mine[j]] if hw else 1.0))
                    if w_tt is not None:
                        if teacher_row(ri): tf = 0.0                    # --teacher_row: the own thought before the teacher's lines is not trained
                        w_tt.append(w_think[-1] * tf)
                        if w_think[-1] > 0: sw_lt[0] += w_think[-1]; sw_lt[1] += w_tt[-1]
        if critic and args.neg_thought > 0 and neg:                    # --neg_thought: the thoughts of those answers, negative weight
            n_nt = 0
            for h in mine:
                it_, w_ = neg_thought_items(neg[h], args.neg_thought, "blocks" if args.m_step == "blocks" else args.think, hw[h] if hw else 1.0)
                items_think += it_; w_think += w_; n_nt += len(it_)
                if w_tt is not None: w_tt += w_
            MSTEP["neg_thoughts"] = n_nt
        items_neg, w_neg = [], []                                     # --neg_answer: the rejected answers of the histories with no accepted turn
        if critic and args.neg_answer > 0 and neg:
            for j, h in enumerate(mine):
                rs = neg[h] or []
                for r in rs:
                    items_neg.append(neg_item(r)); w_neg.append(args.neg_answer / len(rs) * (hw[h] if hw else 1.0))
        N = max(1, nh)
        l_short = loss_items(model, tok, items_short, N, args.score_chunk, w_short) if items_short else 0.0
        if critic and args.m_step == "blocks":                     # two clipped steps: the thought block, then the decision block
            ppo = None
            if grpo and args.grpo_eps > 0 and items_think:            # --grpo_eps: each item under the phase prior pi_n, once per phase (grpo_old)
                assert not own_prior or prior["sd"] is not None, "--update grpo: the phase prior is not kept (the first update of a phase keeps it)"
                need = list({(it[0], it[1]): it for it in items_think if (it[0], it[1]) not in grpo_old}.values())
                if need:
                    model.eval()
                    with torch.no_grad(), prior_policy():
                        for c0 in range(0, len(need), args.think_chunk):
                            part = need[c0:c0 + args.think_chunk]
                            for it, x in zip(part, score(model, tok, part, args.think_chunk).double().cpu()): grpo_old[(it[0], it[1])] = x
                    model.train()
                old = torch.stack([grpo_old[(it[0], it[1])] for it in items_think])
                ntok = torch.tensor([block_counts(tok, it) for it in items_think], dtype=torch.float64)   # the blocks' counted tokens
                ppo = (old, ntok, args.grpo_eps)
            if grpo: MSTEP.update({"grpo_stop_t": 0, "grpo_stop_d": 0, "grpo_items": 0})     # this update's counts (a rank may have no item)
            lt, ld, gt, gd, ro = loss_items_blocks(model, tok, items_think, N, args.think_chunk, w_think, params, READOUT.get("head") is not None, w_tt, ppo) \
                if items_think else (0.0, 0.0, None, None, None)
            if grpo and world > 1:                                 # the counts of every rank (every rank runs this collective)
                t_ = torch.tensor([float(MSTEP[k]) for k in ("grpo_stop_t", "grpo_stop_d", "grpo_items")], device=model.device, dtype=torch.float64)
                dist.all_reduce(t_); MSTEP.update(dict(zip(("grpo_stop_t", "grpo_stop_d", "grpo_items"), t_.tolist())))
            n_items = len(items_think); ln, p_neg = 0.0, []
            if w_tt is not None and ltg is not None:               # --line_target: the share of the fitted weight on turns with a verified line (w_tt alone: --teacher_row)
                sw = sw_lt                                         # (those turns train their thought and both lines; the others their action only)
                if world > 1: t_ = torch.tensor(sw, device=model.device, dtype=torch.float64); dist.all_reduce(t_); sw = t_.tolist()
                MSTEP["w_line_target"] = sw[1] / sw[0] if sw[0] > 0 else 0.0
            if items_neg:                                          # --neg_answer: its gradient joins the decision block's step
                if gd is None: gd = [torch.zeros_like(p) for p in params]
                ln, p_neg = loss_neg(model, tok, items_neg, N, args.think_chunk, w_neg, gd, params)
            if world > 1:                                          # data parallel: each block's gradient summed over the ranks (each rank's loss is
                if gt is None: gt = [torch.zeros_like(p) for p in params]                                                   # over the global N)
                if gd is None: gd = [torch.zeros_like(p) for p in params]
                t = torch.tensor([lt, ld, float(n_items + len(items_neg)), ln, float(len(items_neg)), float(sum(p_neg))], device=model.device, dtype=torch.float64)
                dist.all_reduce(t); lt, ld, n_all, ln, n_neg, s_neg = t.tolist()
                for g in (gt, gd):
                    flat = torch.cat([x.reshape(-1).float() for x in g]); dist.all_reduce(flat); o = 0
                    for x in g: x.copy_(flat[o:o + x.numel()].view_as(x)); o += x.numel()
                if n_all == 0: gt = gd = None                      # no rank has an item: no step anywhere
            else:
                n_neg, s_neg = float(len(items_neg)), float(sum(p_neg))
            if args.neg_answer > 0:                                # --neg_answer: the pushed-down answers, their mean p(a) before this step, the loss
                MSTEP.update({"neg_items": n_neg, "neg_p": s_neg / n_neg if n_neg else 0.0, "loss_neg": ln})
            gns = []
            for g in (gt, gd):
                if g is None: gns.append(0.0); continue
                for gi, p in zip(g, params): p.grad = gi
                gns.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0))); opt.step(); opt.zero_grad()
            MSTEP["loss_dec"], MSTEP["grad_norm_think"], MSTEP["grad_norm_dec"], MSTEP["m_items"] = ld, gns[0], gns[1], n_items / max(1, len(mine))
            if ro is not None: MSTEP.update({**readout_stats(ro), "ro_fit": readout_fit(ro)})     # the readout: its own fitting on the buffer
            if kl_measured(args) and prior["probe"] is not None: MSTEP["kl_prior"] = probe_kl()        # --kl_stop: the phase's step so far
            stats = [(h, lp2[j].tolist(), w[j].tolist(), extra[j]) for j, h in enumerate(mine)]; model.eval()
            if world > 1:                                          # the ranks stay identical (a check), the E-step statistics of every history to rank 0
                cs = torch.tensor([float(sum(p.detach().double().sum() for p in params))], device=model.device, dtype=torch.float64)
                al = [torch.zeros_like(cs) for _ in range(world)]; dist.all_gather(al, cs)
                MSTEP["dp_param_spread"] = float(max(a.item() for a in al) - min(a.item() for a in al))
                parts = [None] * world; dist.all_gather_object(parts, stats); stats = sorted(s for part in parts for s in part)
            return l_short, lt, gns[0], stats
        l_think = loss_items(model, tok, items_think, N, args.think_chunk, w_think) if items_think else 0.0
        stats = [(h, lp2[j].tolist(), w[j].tolist(), extra[j]) for j, h in enumerate(mine)]
        if world > 1:
            for p in params:
                if p.grad is None: p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
            t = torch.tensor([l_short, l_think], device=model.device); dist.all_reduce(t); l_short, l_think = t.tolist()
            parts = [None] * world; dist.all_gather_object(parts, stats); stats = sorted(s for part in parts for s in part)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step(); model.eval()
        if kl_measured(args) and prior["probe"] is not None: MSTEP["kl_prior"] = probe_kl()            # --kl_stop: the phase's step so far
        return l_short, l_think, gn, stats

    def shard_score(sj, r):
        """Evaluation scoring on all ranks (the evaluation uses every GPU): this rank scores pairs
        r, r + world, ... with no gradient (under the base model if asked; with the lead's readout weights if asked)."""
        idx = list(range(r, len(sj["pairs"]), world)); sub = [sj["pairs"][i] for i in idx]; own_head = False
        if sj.get("readout_sd") is not None and READOUT.get("head") is None:
            head = Readout(model.get_output_embeddings().in_features).cuda(); head.load_state_dict(sj["readout_sd"]); head.eval()
            READOUT.update({"head": head, "norm": find_norm(model)}); own_head = True
        try:
            model.eval()
            with torch.no_grad(), (base_model(model) if sj.get("base") else contextlib.nullcontext()):
                if not sub: out = [] if sj.get("per_token") else {}
                elif sj.get("per_token"): out = [x.float().cpu() for x in score(model, tok, sub, sj["chunk"], per_token=True)]
                else:
                    ro = {}
                    for s_ in range(0, len(sub), sj["chunk"]):
                        _, ro_ = score(model, tok, sub[s_:s_ + sj["chunk"]], sj["chunk"], readout=True)
                        for k in ("ready", "rtgt", "ident", "act", "atgt", "agrp"):
                            ro.setdefault(k, []).extend(x.detach().float().cpu() if torch.is_tensor(x) else x for x in ro_[k])
                    out = ro
        finally:
            if own_head: READOUT.clear()
        return {"idx": idx, "out": out}

    def dist_eval_score(pairs, per_token=False, base=False, readout=False):
        """The lead's entry: the same results as score() on one GPU, from all ranks."""
        sj = {"pairs": pairs, "per_token": per_token, "base": base, "chunk": args.think_chunk,
              "readout_sd": {k: v.detach().cpu() for k, v in READOUT["head"].state_dict().items()} if readout else None}
        if world > 1: dist.broadcast_object_list([{"score": sj}], src=0, group=CPU_GROUP)
        res = shard_score(sj, 0)
        parts = [res]
        if world > 1: parts = [None] * world; dist.all_gather_object(parts, res, group=CPU_GROUP)
        if per_token:
            out = [None] * len(pairs)
            for pt in parts:
                for i, x in zip(pt["idx"], pt["out"]): out[i] = x
            return out
        st = {k: [] for k in ("ready", "rtgt", "ident", "act", "atgt", "agrp")}
        for pt in parts:
            for k in st: st[k] += list(pt["out"].get(k, []))
        return st

    def park_model(to):
        """--colocate: this rank's model and optimizer state to the CPU between the learner's phases (the picture server has the
        GPUs then), back to the GPU for a phase; Adam's step counters stay on the CPU. The server's processes give their GPU
        memory back only when they end (a learner that starts too early runs out of memory), so the move to the GPU
        waits until this GPU is free."""
        if to == "cuda":
            t_ = time.time()
            free = float(os.environ.get("PL_PARK_FREE", 0.9))          # the simulator: node 0's renderers keep their GPU memory
            while (m := torch.cuda.mem_get_info())[0] < free * m[1]:
                if time.time() - t_ > 900: raise RuntimeError(f"the GPU is not free after 900 s: {m[0] / 2**30:.1f} of {m[1] / 2**30:.1f} GiB free")
                time.sleep(2)
        model.to(to)
        for st_ in opt.state.values():
            for k, v in list(st_.items()):
                if torch.is_tensor(v) and k != "step": st_[k] = v.to(to)
        if to == "cpu": torch.cuda.synchronize(); torch.cuda.empty_cache()

    if not lead:                                                      # a worker rank: updates on the batches rank 0 sends
        while True:
            box = [None]; dist.broadcast_object_list(box, src=0, group=CPU_GROUP); job = box[0]
            if job is None: break
            if "park" in job:                                        # --colocate: rank 0 moves every rank's model (then all meet)
                park_model(job["park"]); dist.barrier(group=CPU_GROUP); continue
            if "score" in job:                                       # evaluation scoring shard (dist_eval_score)
                res = shard_score(job["score"], rank); parts = [None] * world; dist.all_gather_object(parts, res, group=CPU_GROUP); continue
            if job["images"]:
                from plearn.models.sampler import IMAGES; IMAGES.update(job["images"])
            em_update(job["rows"], job["nh"], job["sizes"], job.get("snap", False), job.get("hw"), job.get("kappa", 0.0), job.get("lrw", 0.0), job.get("okr"), job.get("neg"), job.get("ltg"))
            for k in job["images"]: IMAGES.pop(k, None)
        dist.destroy_process_group(); return
    assert not args.rho_exact or args.rho == "fidelity", "--rho_exact 1 needs --rho fidelity (the table's prior is then the exact posterior U(b))"
    assert not args.teacher_row or (args.objective == "critic" and args.update == "grpo" and args.credit == "own" and not args.neg_answer and not args.neg_thought), \
        "--teacher_row 1: the critic with --update grpo and --credit own, without --neg_answer / --neg_thought (their histories get the teacher row)"
    assert args.stall_cap <= 0 or args.stall_cap > math.log(args.accept), \
        "--stall_cap: above log(--accept) (else a stalling step stays accepted)"
    assert not args.stall_play or args.stall_cap > 0, "--stall_play 1 needs --stall_cap > 0 (the teacher plays by the capped scores)"
    rel = E.Reliability(args.kappa, args.ucb_z, fidelity=args.rho == "fidelity", value=args.value); rel.frozen = bool(args.rho_exact); planner = Planner(D, rel, args.tau, args.depth, args.tau_look, args.tau_look_play, args.stall_cap, args.stall_play)
    if args.value != "accuracy" and D is not None:                 # a domain's own lookahead must compute the value it is asked for
        for d in getattr(D, "subs", [D]):
            assert args.value in getattr(d, "value_modes", ("accuracy",)), f"{type(d).__module__} has no --value {args.value}"

    def ratio(Dm, B, y):
        """b(y) / U(b) of an answer, the agent's fidelity at b (None where the domain cannot say)."""
        if y is None or not hasattr(Dm, "answer_prob"): return None
        u = B.U(); return Dm.answer_prob(B, y) / u if u > 0 else 0.0
    os.makedirs(args.out, exist_ok=True); f_log = os.path.join(args.out, f"{tag}{'-actor' if actor else ''}.jsonl")
    old_rows = [json.loads(l) for l in open(f_log)] if args.resume_step else []   # --resume_step: the log goes on
    log = open(f_log, "a" if args.resume_step else "w")             # --role actor: its own log; the learner's is the run's
    # --resume_step R: when the sample set of the phase from R + 1 is in the queue (all its batches), the learner trains that phase
    # first and the actor's next set is the one from R + phase + 1; else the actor makes the set from R + 1 again. The actor goes on
    # with its answer-accuracy table and scene count as they were after the set before its next one.
    have_set = bool(args.resume_step) and all(os.path.exists(os.path.join(args.queue, f)) for f in [f"gen_done_{args.resume_step}"] +
                                              [f"batch_{u}.pkl" for u in range(args.resume_step + 1, min(args.steps, args.resume_step + args.phase) + 1)])
    last_set = None
    if args.resume_step and actor:
        n_prev = args.resume_step // args.phase - (0 if have_set else 1)       # the index of the set before the actor's next one
        last_set = ([r for r in old_rows if r.get("phase_set") == n_prev] or [None])[-1]
        assert last_set is not None, f"--resume_step: the actor log has no row of set {n_prev}"
        if args.rho_rb:
            import ast
            for k, (s_, n_) in json.load(open(os.path.join(args.out, f"{tag}-rho-{n_prev * args.phase}.json"))).items():
                rel.add(ast.literal_eval(k), s_, n_)
    board = Board(args, tag, world, enabled=not actor)                # W&B from the learner (or the one process) only
    if rank == 0: print(f"score: temperature {args.temperature}, top-p {args.top_p}, top-k {args.top_k}, presence penalty {pp} "
                        f"(the sampler's PL_PRESENCE_PENALTY: {os.environ.get('PL_PRESENCE_PENALTY') or 0}; --score_presence {args.score_presence})", flush=True)
    global CARRY, EXACT_LINE; CARRY = args.carry; EXACT_LINE = args.line_score in ("exact", "decision")
    import plearn.tasks as _tasks                                       # the teachers' switches: the module text
    _tasks.FL_VIEW_VALUE = bool(args.fl_view_value)                     # --fl_view_value: the module text
    _tasks.SK_GUESS_VALUE = bool(args.sk_guess_value)                   # --sk_guess_value: the module text
    _tasks.PLAN_PAST_LIMIT = bool(args.plan_past_limit)                 # --plan_past_limit: the module text
    assert args.line_score != "decision" or D is None or all(hasattr(d, "line_belief") for d in getattr(D, "subs", [D])), \
        "--line_score decision: every teacher reads a belief line as a belief (line_belief)"
    assert not args.line_wide or (args.line_score != "none" and (D is None or all(hasattr(d, "line_excess") or not hasattr(d, "belief_score")
                                                                                  for d in getattr(D, "subs", [D])))), \
        "--line_wide: the line is scored and every teacher with its own line score reads the share its exact line excludes (line_excess)"
    if EXACT_LINE: importlib.import_module(args.domain).EXACT = True
    sampler = None
    if actor and args.colocate:                                       # --colocate: the GPUs are the learner's until it parks its model
        while not os.path.exists(os.path.join(args.queue, f"learner_parked_{args.resume_step + args.phase if have_set else 0}")):
            if os.path.exists(os.path.join(args.queue, "learner_failed")): raise SystemExit("the learner failed before it parked its model")
            time.sleep(2)
    if not learner:                                                   # --role actor: until the learner's first adapter, the base model
        set_from = os.environ.get("PL_SET_FROM") if actor and args.phase_set and not args.resume_step else None
        sampler = ServerSampler(args.model, args.vllm_gpu, args.temperature, os.path.join(args.out, tag), args.lora_r, args.vllm_mem,
                                images=images,                       # (the same as the zero LoRA of the one-process run)
                                start=not (set_from and args.colocate and os.environ.get("PL_SKIP_EVAL0") == "1"))
        if not actor: sampler.sync(model)
    phase = args.phase or (1 if args.proposal == "policy" else 0); prior_step = used_prior = 0
    assert phase == 0 or args.eval_every % phase == 0, "evaluate the model at phase boundaries only (--eval_every a multiple of --phase)"

    def train_specs():
        while True: yield D.sample(rng)
    specs_it = train_specs()
    # held-out histories from the evaluation split (--eval_split), the same number from every task of a mixed domain
    held_eps, held, D_eval, P_eval, held_fut, held_rel = [], [], None, None, None, []
    eargs = json.loads(args.domain_args)
    if args.eval_split: eargs["split"] = args.eval_split
    if args.eval_domain_args: eargs.update(json.loads(args.eval_domain_args))   # the evaluation domain only (for example, a benchmark's rubric)
    if not learner:                                                   # --role learner: the actor evaluates
        D_eval = importlib.import_module(args.domain).Domain(**eargs)
        P_eval = Planner(D_eval, rel, args.tau, args.depth, args.tau_look, args.tau_look_play)
        def collect_held(P_held):
            if hasattr(D_eval, "families"):
                for i, fam in enumerate(D_eval.families):
                    specs = iter([sp for sp in D_eval.eval_specs(40 * args.held_n) if sp[0] == fam])
                    held.extend(collect_histories(D_eval, tok, np.random.default_rng(777 + i), specs, args.held_n // len(D_eval.families), P_held, args.mix, keep_eps=held_eps))
            else:
                held.extend(collect_histories(D_eval, tok, np.random.default_rng(777), iter(D_eval.eval_specs(4 * args.held_n)), args.held_n, P_held, args.mix, keep_eps=held_eps))
            held_rel.append(copy.deepcopy(P_held.rel))                 # the teacher's table as it made the probes (decision_regret)
        if os.environ.get("PL_HELD_BG") == "1":
            # speed: the held-out histories (first read at the first evaluation) are made beside the first set,
            # by a planner on a copy of the table as it is now: the same histories as before the set (a few minutes less)
            import concurrent.futures as _cfh
            held_fut = _cfh.ThreadPoolExecutor(1).submit(collect_held, Planner(D_eval, copy.deepcopy(rel), args.tau, args.depth, args.tau_look, args.tau_look_play))
        else:
            collect_held(P_eval)

    def own_thoughts(prompts, greedy=False, priority=1, max_think=0):
        """The policy's thought (the greedy flag is kept for the callers; the policy is always the sampler), at most max_think tokens
        (0: --max_think, the training cap)."""
        return [r[0] for r in sampler.sample(prompts, 1, max_think or args.max_think, stop_think=True, priority=priority)]

    def anchors(pres, priority=1):
        if NO_BELIEF_LINE: return [""] * len(pres)                    # --no_belief_line: no belief line is sampled
        return [first_line(t) for (t, _), in [r for r in sampler.sample(pres, 1, args.max_anchor, stop_think=False, priority=priority)]]

    # ---- the model acting on real episodes (training rollouts and evaluation)
    def act_scores(texts):
        """--act_pick: the action readout's score at the last token of each text (prompt + thought + </think> + belief line + the
        candidate action, no terminator), chunks of 8, no gradient."""
        out = []; model.eval()
        for s in range(0, len(texts), 8):
            part = texts[s:s + 8]; ids = [encode_prefix(tok, t)[0] for t in part]; L = max(len(x) for x in ids)
            inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in ids], device=model.device)
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device=model.device)
            cap = {}; hook = READOUT["norm"].register_forward_hook(lambda m, i_, o: cap.__setitem__("h", o))
            try:
                with torch.no_grad(): model(input_ids=inp, attention_mask=att, use_cache=False, logits_to_keep=1)
            finally:
                hook.remove()
            with torch.no_grad(): out += READOUT["head"](cap["h"][:, -1])[1].tolist()
        return out

    import threading
    ep_lock, plan_lock = threading.Lock(), threading.Lock()     # async episodes: results applied one at a time; the teacher plans one at a time
    plan_stream = []                                            # PL_PLAN_STREAM=1 (speed): the teacher's GPU work on its own CUDA stream.
                                                                # On the default stream it waited behind every M-step kernel (PL_PIPELINE):
    def plan_ctx():                                             # the episodes waited on plan_lock and vLLM had 4-9 requests per node
        if os.environ.get("PL_PLAN_STREAM") != "1" or not torch.cuda.is_available(): return contextlib.nullcontext()
        if not plan_stream: plan_stream.append(torch.cuda.Stream(device=torch.device("cuda")))   # the teacher's CUDA device
        return torch.cuda.stream(plan_stream[0])
    def plan_all(plan_fn, eps):
        with plan_lock, plan_ctx():
            plans = [plan_fn(e) for e in eps]
            if plan_stream: plan_stream[0].synchronize()        # the plans are complete before another thread reads them
        return plans
    import concurrent.futures as _cfp
    plan_pool = _cfp.ThreadPoolExecutor(1)                      # act(): the plans of a turn, made while the agent writes it

    def act(eps, beta_expert, Dm=None, plan_fn=None, greedy=False):
        """One decision per active episode: own thought, own anchor, restricted softmax over legal strings (or pi* with probability beta).
        Dm is the domain the episodes belong to (the evaluation domain keeps its own environment sessions). The policy is the same
        in training rollouts and at test: the model under the sampler."""
        Dm = Dm or D; plan_fn = plan_fn or planner
        fut = plan_pool.submit(plan_all, plan_fn, eps)                    # the teacher plans while the agent writes its turn (the turn does not
        prompts = [turn_prompt(Dm, tok, e) for e in eps]                 # use the plan; one lookahead at a time: its GPU work tensors)
        zs = own_thoughts(prompts, greedy, priority=0)                   # the episodes' turns go first (each waits for the one before)
        pres = [p + z + CLOSE for p, (z, _) in zip(prompts, zs)]
        anc = anchors(pres, priority=0); pre_act = [after_line(p, a) for p, a in zip(pres, anc)]
        n_pick = args.act_pick if args.act_pick > 1 and READOUT.get("head") is not None else 1
        cands = [[first_line(t) for t, _ in cs] for cs in sampler.sample(pre_act, n_pick, args.max_action, stop_think=False, priority=0)]
        plans = fut.result()
        picks = [None] * len(eps)
        if n_pick > 1:                                                  # --act_pick: the action readout chooses among the sampled candidates
            sc = act_scores([pa + c for pa, cs in zip(pre_act, cands) for c in cs]); k = 0; free = []
            for i, (cs, pl, a_) in enumerate(zip(cands, plans, anc)):
                s_ = sc[k:k + len(cs)]; k += len(cs); j = int(np.argmax(s_)); free.append(cs[j])
                picks[i] = {"cands": cs, "s": s_, "lp": [action_logp(Dm, ordered(a_, c)[1], pl) for c in cs], "pick": j}
        else:
            free = [cs[0] for cs in cands]
        recs = []
        for e, pl, p, (z, stopped), a, s, pk in zip(eps, plans, prompts, zs, anc, free, picks):
            a, s = ordered(a, s)                                             # the two lines as the test reads them
            acted = beta_expert > 0 and nrng.random() < beta_expert
            if acted:                                                        # the teacher acts: an action drawn from pi*
                ks = list(pl["tgt"]); ps = np.array([pl["tgt"][k] for k in ks]); s = ks[int(nrng.choice(len(ks), p=ps / ps.sum()))]
                inv = {v: k for k, v in pl["looks"].items()}
                kind, val = ("look", inv[s]) if s in inv else ("answer", next(y for y, c in pl["commits"].items() if c == s))
            else:
                kind, val = parse_action(Dm, e, s, pl)
            recs.append({"prompt": p, "plan": pl, "thought": z, "anchor": a, "anchor_ok": same(a, pl["anchor"], pl), "action": s,
                         "look": val if kind == "look" else None, "commit": kind == "answer", "answer_text": val if kind == "answer" else None,
                         "invalid": kind == "invalid", "teacher_acted": acted, "pick": pk})
        return recs

    qrng = np.random.default_rng(args.seed + 13)
    def act_q(eps, Dm, plan_fn, pick):
        """A check of the E-step target, not a policy: q = pi exp(R / alpha) / Z acts, pi the model under check (the base model,
        or --init_lora). K thoughts and M decisions after each from pi, scored by the teacher at the true state; pick 'max' takes the sample with the largest R,
        'sample' one draw from q."""
        plans = plan_all(plan_fn, eps)
        prompts = [turn_prompt(Dm, tok, e) for e in eps]; K, M = args.em_k, args.em_m; KM = K * M
        cands = sampler.sample(prompts, K, args.max_think, stop_think=True)
        zs = [z for cand in cands for z, _ in cand]; pres = [p + z + CLOSE for p, cand in zip(prompts, cands) for z, _ in cand]
        bs = [""] * (len(pres) * M) if NO_BELIEF_LINE else [first_line(t) for cs in sampler.sample(pres, M, args.max_anchor, stop_think=False) for t, _ in cs]
        pres = [p for p in pres for _ in range(M)]
        us = [first_line(t) for (t, _), in sampler.sample([after_line(p, b) for p, b in zip(pres, bs)], 1, args.max_action, stop_think=False)]
        bs, us = [list(x) for x in zip(*[ordered(b, u) for b, u in zip(bs, us)])]          # the two lines as the test reads them
        recs = []
        for i, (e, pl, p) in enumerate(zip(eps, plans, prompts)):
            Vmax = max(max(pl["values"].values(), default=pl["waste"]), pl["waste"])
            R = np.array([belief_score(Dm, pl["B"], bs[j], pl) + action_logp(Dm, us[j], pl) for j in range(i * KM, (i + 1) * KM)])
            w = np.exp((R - R.max()) / args.alpha); j = i * KM + (int(R.argmax()) if pick == "max" else int(qrng.choice(KM, p=w / w.sum())))
            kind, val = parse_action(Dm, e, us[j], pl)
            recs.append({"prompt": p, "plan": pl, "thought": zs[j // M], "anchor": bs[j], "anchor_ok": same(bs[j], pl["anchor"], pl), "action": us[j],
                         "look": val if kind == "look" else None, "commit": kind == "answer", "answer_text": val if kind == "answer" else None,
                         "invalid": kind == "invalid"})
        return recs


    def run_episodes(eps, beta_expert, update_rel, Dm=None, plan_fn=None, greedy=False, q="", cells=None):
        """cells (--rho_rb): the agent's answers go to cells[(scene key, belief class)] as b(y) / U(b), not to the table."""
        Dm = Dm or D; decs = []
        step_fn = lambda es: act_q(es, Dm, plan_fn or planner, q) if q else act(es, beta_expert, Dm, plan_fn, greedy)

        def apply(e, r):
                decs.append(r); B = r["plan"]["B"]; r["ep"] = id(e); r["key"] = getattr(e, "key", None); r["identified_before"] = B.identified(); e.note = r["anchor"]
                tgt = r["plan"]["tgt"]; r["teacher_p"] = tgt.get(r["action"], 0.0); r["teacher_best"] = r["action"] == max(tgt, key=tgt.get)
                if r["commit"]:                                   # scored by the agent's own answer text
                    y = r["answer_text"]; r["answer"] = y
                    e.correct = Dm.do_commit(e, y); e.done = True; e.commit_ident = B.identified(); e.commit_B = B
                    if update_rel and not r.get("teacher_acted"):     # the table measures the agent's answers only
                        if cells is not None: cells.setdefault((r["key"], B.cls()), []).append(ratio(Dm, B, y))
                        else: rel.update(B, e.correct, getattr(e, "key", None), ratio(Dm, B, y))
                elif r["look"] is not None:
                    if not r.pop("_looked", False): Dm.do_look(e, r["look"])   # PL_LOOK_UNLOCKED: already done, outside ep_lock
                    e.t += 1
                else:
                    e.t += 1; e.invalid = getattr(e, "invalid", 0) + 1      # an action the environment cannot carry out
                if getattr(e, "lab", None) is not None:           # a domain that also runs the benchmark's own episode: it decides the state and the score
                    th = r.get("thought") or ""; rep = r["action"] if NO_BELIEF_LINE else f"{r['anchor']}\n{r['action']}"
                    Dm.bench_turn(e, th, rep, len(tok.encode(th + CLOSE + rep, add_special_tokens=False)))
                if hasattr(Dm, "after_turn"): Dm.after_turn(e)    # a domain with its own turn limit for each episode

        if os.environ.get("PL_ASYNC_EPISODES") == "1" and len(eps) > 1:
            # speed: each episode asks for its next turn as soon as its own last turn returns, so the batch does
            # not wait for its slowest episode at every turn; the same policy and teacher, turn by turn (one thread per episode)
            from concurrent.futures import ThreadPoolExecutor
            unlocked = os.environ.get("PL_LOOK_UNLOCKED") == "1"
            def one(e):
                with plan_ctx():                                  # PL_PLAN_STREAM: the environment's GPU work too, off the M-step's stream
                    for _ in range(getattr(Dm, "turns", Dm.T + 2)):
                        if e.done: break
                        r = step_fn([e])[0]
                        if unlocked and not r["commit"] and r["look"] is not None and getattr(e, "lab", None) is None:
                            # speed: a look changes only its own episode (its own simulator session; each bridge
                            # has its own lock), so it runs outside ep_lock; inside it, one simulator step ran at a time
                            Dm.do_look(e, r["look"]); r["_looked"] = True
                        with ep_lock: apply(e, r)
            with ThreadPoolExecutor(max_workers=len(eps)) as ex: list(ex.map(one, eps))
        else:
            for _ in range(getattr(Dm, "turns", Dm.T + 2)):
                active = [e for e in eps if not e.done]
                if not active: break
                for e, r in zip(active, step_fn(active)): apply(e, r)
        for e in eps:
            if not e.done: e.correct = False; e.done = True; e.commit_B = Dm.belief(e); e.commit_ident = e.commit_B.identified()
        return decs

    def own_turns(hist_, K, M, own):
        """The critic's E-step samples: K own thoughts per history and M own belief lines and actions after each, scored by the
        teacher as the test reads the two lines. Per history, the rows (prompt, z, prompt + z + </think>, b, a, R, stopped, readout targets) and
        the scores."""
        cands = sampler.sample([p for p, _ in hist_], K, args.max_think, stop_think=True, adapter=own)
        zs = [(h, z, st) for h, cand in enumerate(cands) for z, st in cand for _ in range(M)]   # st: the thought ended by itself
        bs = [""] * len(zs) if NO_BELIEF_LINE else \
            [first_line(t) for cs in sampler.sample([hist_[h][0] + z + CLOSE for h, z, _ in zs[::M]], M, args.max_anchor,
                                                    stop_think=False, adapter=own) for t, _ in cs]   # --no_belief_line: no line
        pres = [hist_[h][0] + z + CLOSE for h, z, _ in zs]
        us = [first_line(t) for (t, _), in sampler.sample([after_line(p, b) for p, b in zip(pres, bs)], 1, args.max_action, stop_think=False, adapter=own)]
        rows, crit = [[] for _ in hist_], [[] for _ in hist_]
        for (h, z, st), pre, b, u in zip(zs, pres, bs, us):
            r, c = score_turn(hist_[h][0], hist_[h][1], z, st, pre, b, u); rows[h].append(r); crit[h].append(c)
        return rows, crit

    def score_turn(prompt, pl, z, st, pre, b, u):
        """One own turn scored by the teacher as the test reads its two lines: its M-step row and its scores."""
        bo, uo = ordered(b, u)                                         # a mixed run: the plan's own teacher scores the turn
        D = D_run.sub(pl["dom"]) if pl.get("dom") is not None and hasattr(D_run, "sub") else D_run
        S = belief_score(D, pl["B"], bo, pl, args.line_floor); St = belief_best(D, pl["B"], pl); A = action_logp(D, uo, pl)
        V = action_value(D, uo, pl); Vmax = max(max(pl["values"].values(), default=pl["waste"]), pl["waste"])
        Sd = None
        if args.line_score == "decision":                              # the act the line implies, once per line of a history
            dec = pl.setdefault("_dec", {}); k_ = norm(bo)
            if k_ not in dec: dec[k_] = decision_score(D, planner, pl, bo)
            Sd = dec[k_]
        W = line_wide(D, bo, pl, args.line_floor) if args.line_wide else 0.0   # --line_wide: the options the exact line excludes
        R = turn_score(args.line_score, args.line_mix, S, W, Sd, A, args.act_score)   # the teacher's log-likelihood of the turn (--act_score 0: no A)
        slots = D.slot_scores(pl["B"], bo, args.line_floor) if hasattr(D, "slot_scores") and args.line_score in ("exact", "decision") and bo == b else None   # the line's statements, scored apart (logged)
        c = {"S": S, "S_teacher": St, "S_dec": Sd, "W": W, "V": V, "V_best": Vmax, "A": A, "R": R, "exact": bool(D.read_exact(bo)) if hasattr(D, "read_exact") else None, "slots": slots,
             "read": D.reading(pl["B"], bo, pl.get("truth")) if hasattr(D, "reading") else None}   # the line against the pictures (tabletop)
        if "stall" in pl: c["stall"] = stall_rejected(A, pl, args.accept, action_logp(D, uo, uncapped(pl)))   # --stall_cap: rejected by the cap alone (logged)
        ro = {"U": float(pl["B"].U()), "identified": bool(pl["B"].identified()), "act_lp": float(A)} if args.readout > 0 else None   # --readout: the exact targets
        return (prompt, z, pre, b, u, R, st, ro, accepted(A, pl, args.accept)), c    # the model's own lines in its own order are trained

    def line_ok(c):
        """An own turn's belief line counts as correct: within --em_more_joint nats of the exact line's score (0: every line)."""
        return args.em_more_joint <= 0 or c["S"] >= c["S_teacher"] - args.em_more_joint

    def good_turn(c, pl):
        """An own turn whose action the teacher accepts (--em_more_joint: and whose line is within that of the best line)."""
        return accepted(c["A"], pl, args.accept) and line_ok(c)

    def line_ok_rows(g):
        """line_ok of every M-step row of a batch (the rows of the histories the E-step keeps, in their order); None without scores."""
        if not g.get("crit_h"): return None
        out = [line_ok(c) for ch, z in zip(g["crit_h"], g["sizes"]) if z for c in ch]
        assert len(out) == len(g["rows_"]), "line_ok_rows: the scores and the rows differ"
        return out

    def line_target_rows(g):
        """--line_target: per M-step row of a batch, whether its belief line is a positive target: within --line_target nats of the
        teacher's best line (the line's score against belief_best). None when the option is off or the batch has no scores."""
        if args.line_target < 0 or not g.get("crit_h"): return None
        out = [bool(c["S"] >= c["S_teacher"] - args.line_target) for ch, z in zip(g["crit_h"], g["sizes"]) if z for c in ch]
        assert len(out) == len(g["rows_"]), "line_target_rows: the scores and the rows differ"
        return out

    def own_chain(prompt, pl, K, M, own):
        """--role actor: the E-step of one history as its own chain of requests: K thoughts, M lines after each, an action after each
        line, and up to --em_more more rounds while no own turn is accepted. The same samples and scores as own_turns and the rounds
        of gen_update, without waiting for the other histories at any stage."""
        rows, crit = [], []
        for r_ in range(1 + args.em_more):
            if r_ and any(good_turn(c, pl) for c in crit): break
            cand = sampler.sample([prompt], K, args.max_think, stop_think=True, adapter=own)[0]
            pres1 = [prompt + z + CLOSE for z, _ in cand]
            zs = [(z, st) for z, st in cand for _ in range(M)]; pres = [p_ for p_ in pres1 for _ in range(M)]
            if NO_BELIEF_LINE: bs = [""] * len(zs)                    # --no_belief_line: M actions after each thought, no line
            else: bs = [first_line(t) for cs in sampler.sample(pres1, M, args.max_anchor, stop_think=False, adapter=own) for t, _ in cs]
            us = [first_line(t) for (t, _), in sampler.sample([after_line(p_, b) for p_, b in zip(pres, bs)], 1, args.max_action, stop_think=False, adapter=own)]
            for (z, st), pre, b, u in zip(zs, pres, bs, us):
                r, c = score_turn(prompt, pl, z, st, pre, b, u); rows.append(r); crit.append(c)
        return rows, crit

    held_dump = open(os.path.join(args.out, f"{tag}-held.jsonl"), "a")

    def read_metrics(greedy):
        """Held-out reading: anchor exactness, recall and implied answer after the model's own thought (greedy or sampled)."""
        if held_fut is not None: held_fut.result()                     # PL_HELD_BG: made beside the first set
        if not held: return {}, [], [], [], []                         # no held-out histories (HELD_N below the number of families)
        prompts = [p for p, _ in held]; plans = [pl for _, pl in held]
        zs = own_thoughts(prompts, greedy, max_think=args.eval_max_think)   # the evaluation's reasoning cap (--eval_max_think)
        pres = [p + z + CLOSE for p, (z, _) in zip(prompts, zs)]
        anc = anchors(pres)
        ok = [same(a, pl["anchor"], pl) for a, pl in zip(anc, plans)]
        labels = [D_eval.label(e) for e in held_eps]
        sup = [anchor_support(a, pl["answers"], D_eval, e) for a, pl, e in zip(anc, plans, held_eps)]
        nontriv = [o for o, pl in zip(ok, plans) if not pl["anchor"].lower().startswith("possible: any")]   # histories with evidence to read
        m = {"anchor_exact": statistics.fmean(ok), "anchor_exact_nontrivial": statistics.fmean(nontriv) if nontriv else None,
             "anchor_recall": statistics.fmean(lab in sp for lab, sp in zip(labels, sup)),
             "anchor_size": statistics.fmean(len(sp) / max(1, len(pl["answers"])) for sp, pl in zip(sup, plans)),
             "anchor_answer_acc": statistics.fmean(anchor_answer(a, pl["answers"], D_eval, e) == lab for a, pl, e, lab in zip(anc, plans, held_eps, labels)),
             "think_tokens": statistics.fmean(len(tok.encode(z, add_special_tokens=False)) for z, _ in zs), "think_stopped": statistics.fmean(st for _, st in zs)}
        if hasattr(D_eval, "belief_score"):              # the belief's meaning against the posterior, whatever its wording
            reg = [belief_best(D_eval, pl["B"], pl) - belief_score(D_eval, pl["B"], a, pl) for a, pl in zip(anc, plans)]
            m.update({"belief_ok": statistics.fmean(r_ <= 1e-6 for r_ in reg), "belief_regret": statistics.fmean(reg)})
        if held_rel and all(hasattr(teacher_of(D_eval, pl), "line_belief") for pl in plans):   # the line by the act it implies
            P_dec = Planner(D_eval, held_rel[0], args.tau, args.depth, args.tau_look, args.tau_look_play)
            dr = [decision_regret(D_eval, P_dec, pl, a) for a, pl in zip(anc, plans)]
            m.update({"decision_ok": statistics.fmean(r_ <= 1e-6 for r_ in dr), "decision_regret": statistics.fmean(dr)})
        if hasattr(D_eval, "family_of"):                 # per-family reading metrics for mixed training distributions
            fams = {}
            for e, o, (z, st), rc in zip(held_eps, ok, zs, [lab in sp for lab, sp in zip(labels, sup)]): fams.setdefault(D_eval.family_of(e), []).append((o, st, rc))
            m["by_family"] = {f: {"exact": statistics.fmean(o for o, _, _ in v), "stopped": statistics.fmean(st for _, st, _ in v),
                                  "recall": statistics.fmean(rc for _, _, rc in v), "n": len(v)} for f, v in sorted(fams.items())}
        return m, zs, pres, anc, plans

    def evaluate(step_id):
        if model is not None: model.eval()
        reg_eval = None
        if args.eval_tasks:                                           # registered tasks (plearn: python -m plearn tasks), this model, the
            from plearn import evaluator as PE                        # actor's server; the agent's turn as in training; a stop keeps the episodes
            edir = os.path.join(args.dump_rollouts or args.out, f"step{step_id}", "eval")
            reg_eval = _cf.ThreadPoolExecutor(1).submit(PE.evaluate, sampler, PROC if PROC is not None else tok, args.eval_tasks, edir, args.eval_n or None,
                                                        args.eval_split or None,         # the evaluation split (validation), not the tasks' test split
                                                        {"max_think": args.eval_max_think or args.max_think}, True, {"step": step_id}, None, None, lambda *a: None)
        watch = None
        if args.watch:                                                # --watch: a fixed sample of the benchmarks' test
            from plearn import evaluator as PE                        # lists, for the direction of training only: never reported
            def one_watch(item):                                      # the first n episodes of each task of name: a sample by task
                name, n = item
                return PE.evaluate(sampler, PROC if PROC is not None else tok, name, os.path.join(args.dump_rollouts or args.out,
                                   f"step{step_id}", "watch", name), int(n), None, {"max_think": args.eval_max_think or args.max_think},
                                   True, {"step": step_id, "watch": True}, None, None, lambda *a: None)
            def run_watch():                                          # the entries at the same time: the servers batch their turns
                items = list(json.loads(args.watch).items()); t0 = time.time(); res = {"tasks": {}, "groups": {}}
                with _cf.ThreadPoolExecutor(len(items)) as ex:
                    for r in ex.map(one_watch, items): res["tasks"].update(r["results"]); res["groups"].update(r["groups"])
                res["minutes"] = (time.time() - t0) / 60
                return res
            watch = _cf.ThreadPoolExecutor(1).submit(run_watch)
        out, zs, pres, anc, plans = read_metrics(greedy=True)
        for e, (z, _), a, pl in zip(held_eps, zs, anc, plans):      # every held-out thought, for inspection
            held_dump.write(json.dumps({"step": step_id, "family": D_eval.family_of(e) if hasattr(D_eval, "family_of") else None, "thought": z,
                                        "anchor": a, "target": pl["anchor"], "exact": same(a, pl["anchor"], pl)}) + "\n")
        held_dump.flush()
        ms, zs_s, _, anc_s, _ = read_metrics(greedy=False)
        out.update({k + "_sampled": v for k, v in ms.items() if k in ("anchor_exact", "anchor_recall", "anchor_answer_acc", "think_tokens", "think_stopped",
                                                                      "belief_ok", "belief_regret", "decision_ok", "decision_regret")})
        # how far the model has moved from the base model, on its own sampled thoughts and belief lines: the mean NLL per token
        # (an entropy estimate) and the mean of log pi - log pi_0 per token (a KL(pi || pi_0) estimate); no forced </think>
        if model is not None:                                       # --role actor: no trainer model, no token log-probabilities
            pairs = [(p, z + CLOSE + ("" if NO_BELIEF_LINE else a + TERM), None if st else (len(z), len(z) + len(CLOSE)), len(z) + len(CLOSE)) for (p, _), (z, st), a in zip(held, zs_s, anc_s)]
            lp = dist_eval_score(pairs, per_token=True); lp0 = dist_eval_score(pairs, per_token=True, base=True)
            n = max(1, sum(len(x) for x in lp))
            out.update({"tok_nll": -sum(x.sum().item() for x in lp) / n, "tok_kl_base": sum((x - y).sum().item() for x, y in zip(lp, lp0)) / n})
        if args.policy_metrics and model is not None:               # the model's policy over every legal string (slow when there are hundreds)
            pol = restricted_policy(model, tok, [after_line(p, a) for p, a in zip(pres, anc)], [pl["legal"] for pl in plans], args.score_chunk)
            act_ok, kl = [], []
            for pl, pi in zip(plans, pol):
                legal = pl["legal"]; best = legal[int(np.argmax(pi))]; tstar = max(pl["tgt"], key=pl["tgt"].get)
                act_ok.append(pl["tgt"].get(best, 0.0) >= 0.99 * pl["tgt"][tstar]); kl.append(E.kl_np(np.array([pl["tgt"].get(s, 0.0) for s in legal]), pi))
            out.update({"action_match": statistics.fmean(act_ok), "kl_act": statistics.fmean(kl)})
        out["n"] = len(held)
        ex = {"thought": zs[0][0][-500:], "anchor": anc[0], "target": plans[0]["anchor"]}
        if args.eval_episodes:
            specs = D_eval.eval_specs(args.eval_episodes); n_open = getattr(D_eval, "max_open", 0) or len(specs); eps, decs = [], []
            for s in range(0, len(specs), n_open):            # a simulator domain keeps at most max_open scenes open at a time
                part = D_eval.start(specs[s:s + n_open]); decs += run_episodes(part, 0.0, False, D_eval, P_eval, greedy=True, q=args.eval_q)
                D_eval.close(part); eps += part
            if args.dump_rollouts: dump_rollouts(args.dump_rollouts, step_id, D_eval, eps, decs)
            if images:
                from plearn.models.sampler import IMAGES, prompt_key
                for r in decs: IMAGES.pop(prompt_key(r["prompt"]), None)
            ident = [e for e in answered(eps) if e.commit_ident]
            looks_ = [r for r in decs if not r["commit"] and r["look"] is not None]          # seeking: how the chosen looks compare with the teacher's
            late = {r["ep"] for r in looks_ if r["identified_before"]}                          # stopping: looks taken after the belief had identified the answer
            out.update({"success": statistics.fmean(bool(e.correct) for e in eps), "premature": premature_rate(eps),
                        "mean_t": statistics.fmean(e.t for e in eps), "acc_when_identified": statistics.fmean(bool(e.correct) for e in ident) if ident else None,
                        "anchor_exact_episodes": statistics.fmean(r["anchor_ok"] for r in decs),
                        "seek_match": statistics.fmean(r["teacher_best"] for r in looks_) if looks_ else None,
                        "seek_mass": statistics.fmean(r["teacher_p"] for r in looks_) if looks_ else None,
                        "late_looks": sum(r["identified_before"] for r in looks_) / len(eps),
                        "stop_on_identified": statistics.fmean(id(e) not in late for e in ident) if ident else None,
                        "invalid_actions": sum(getattr(e, "invalid", 0) for e in eps) / len(eps), **D_eval.extra_metrics(eps)})
            if getattr(D_eval, "bench", False):                       # the benchmark's own numbers (from the benchmark's episode of each episode)
                bm, brows = D_eval.bench_metrics(eps); out.update(bm)
                if args.dump_rollouts:
                    os.makedirs(os.path.join(args.dump_rollouts, f"step{step_id}"), exist_ok=True)
                    with open(os.path.join(args.dump_rollouts, f"step{step_id}", "bench.jsonl"), "w") as f:
                        for r in brows: f.write(json.dumps(r) + "\n")
            if hasattr(D_eval, "slot_targets"):                     # the reading, statement by statement: exact, true but wider, or wrong
                n_ex = n_lo = n_wr = n_out = n_st = 0
                for r in decs:
                    tg = D_eval.slot_targets(r["plan"]["B"], r["anchor"], args.line_floor)
                    if tg is None: n_out += 1; continue
                    for _, _, _, Si, Sb, pin in tg:
                        n_st += 1
                        if abs(Si - Sb) < 1e-6: n_ex += 1
                        elif pin > 1 - 1e-9: n_lo += 1
                        else: n_wr += 1
                if n_st: out.update({"stmt_exact": n_ex / n_st, "stmt_loose": n_lo / n_st, "stmt_wrong": n_wr / n_st})
                out["line_outside"] = n_out / max(1, len(decs))
            try: out.update(monitor_eval_metrics(decs, eps, D_eval, args.line_floor))
            except Exception as ex_: out["metrics_err"] = 1; print("monitor_eval_metrics failed:", repr(ex_)[:300], flush=True)
            if READOUT.get("head") is not None and args.readout_eval: out.update(readout_eval(decs))
            pk = [r["pick"] for r in decs if r.get("pick") and len(r["pick"]["cands"]) > 1]
            if pk:                                                   # --act_pick: the teacher's mass on the readout's pick against the policy's own first sample
                gain, top, top0, kl, dA = [], [], [], [], []
                for x in pk:
                    lp = np.array(x["lp"]); ps = np.exp(lp - lp.max()); ps /= ps.sum(); s_ = np.array(x["s"]); qs = np.exp(s_ - s_.max()); qs /= qs.sum(); j = x["pick"]
                    gain.append(ps[j] - ps[0]); top.append(ps[j] >= ps.max() - 1e-9); top0.append(ps[0] >= ps.max() - 1e-9); dA.append(lp[j] - lp[0])
                    nz = ps > 1e-12; kl.append(float((ps[nz] * (np.log(ps[nz]) - np.log(np.maximum(qs[nz], 1e-12)))).sum()))
                out.update({"pick_n": len(pk), "pick_gain": float(np.mean(gain)), "pick_top1": float(np.mean(top)), "pick_top1_first": float(np.mean(top0)),
                            "pick_kl": float(np.mean(kl)), "pick_dlogp": float(np.mean(dA))})
        if reg_eval is not None:
            res = reg_eval.result()                                   # eval/tasks.<task>.<metric> and eval/groups.<group>.<metric> on W&B
            out.update({"tasks": res["results"], "groups": res["groups"], "eval_minutes": res["config"]["minutes"]})
        if watch is not None: out["watch"] = watch.result()            # eval/watch.tasks.<task>.<metric> on W&B
        return out, ex

    def readout_eval(decs):
        """The readiness readout on the evaluation turns (the agent alone) against U(b*): its mean absolute error and its AUROC
        for identified (readout_stats)."""
        items = []
        for r in decs:
            B = r["plan"]["B"]; z = r["thought"]; line = r["anchor"]; d0 = len(z) + len(CLOSE)
            items.append((r["prompt"], z + CLOSE + line + TERM + r["action"] + TERM, None, d0, "blocks", {"U": float(B.U()), "identified": bool(B.identified())}))
        return readout_stats(dist_eval_score(items, readout=True))                 # all ranks

    nxt = None
    if not args.episodes and args.prefetch and args.steps > 0:
        # warm-up histories do not depend on the model: a second environment session plays the next batch while this one
        # trains (in a simulator the weighings, not the GPUs, set the pace)
        import concurrent.futures as cf
        D2 = Dmod.Domain(**dargs); P2 = Planner(D2, rel, args.tau, args.depth, args.tau_look, args.tau_look_play, args.stall_cap, args.stall_play)
        rng2, nrng2 = random.Random(args.seed + 7), np.random.default_rng(args.seed + 7)
        def specs2():
            while True: yield D2.sample(rng2)
        it2 = specs2(); pool = cf.ThreadPoolExecutor(1)
        fetch = lambda: collect_histories(D2, tok, nrng2, it2, args.B, P2, args.mix)
        nxt = pool.submit(fetch)
    def rel_table():
        with ep_lock: return rel.table()                              # a copy while no other update writes (PL_PIPELINE_DEPTH)

    def rho_commit(cells):
        """--rho_rb: one measurement per scene and belief class, the mean of b(y) / U(b) over the agent's answers there: the episode's
        own answer and the E-step's own answer turns at the histories of that scene and class, all draws from the phase prior. The
        mean over the answer draws replaces one draw (the conditional mean, as in Rao-Blackwell). A cell weighs in proportion to its
        number of answers, so the class estimate stays the mean over the agent's answers; the cells of one update add one count
        each on average. Returns the fields of the log row, and the table just before and just after (for --dump_em)."""
        cells = {c: [x for x in xs if x is not None] for c, xs in cells.items()}; cells = {c: xs for c, xs in cells.items() if xs}
        if not cells: return {"rho_cells": 0, "rho_answers": 0}, None, None
        na = statistics.fmean(len(xs) for xs in cells.values())
        with ep_lock:
            before = rel.table()
            for (_, k), xs in cells.items(): rel.add(k, sum(xs) / na, len(xs) / na)
            after = rel.table()
        return {"rho_cells": len(cells), "rho_answers": sum(len(xs) for xs in cells.values())}, before, after

    def gen_update(step, B=None, worlds=None):
        """One update's data: its episodes, the history sample and the E-step's own turns, all from the vLLM servers (the phase prior)
        and the teacher; no trainer-model call, so it can run while the previous update trains (PL_PIPELINE). B and worlds (default
        --B and --worlds): the histories kept and the scenes played; --phase_set makes the phase's sample set with them."""
        B = B or args.B; worlds = worlds or args.worlds
        nonlocal nxt
        t1 = time.time(); roll = {}; crit_h = decs = eps = None
        cells = {} if args.rho_rb else None            # --rho_rb: the agent's answers of this update by (scene, belief class)
        if args.episodes:
            k_it = (step - 1) // phase if (phase and args.beta_per == "iteration") else step - 1   # alpha_k: k is the iteration
            beta = args.beta0 * args.beta_decay ** k_it
            specs = [D.sample(rng) for _ in range(worlds)]
            eps = D.start([s for s in specs for _ in range(args.G)])
            for k, e in enumerate(eps): e.key = (step, k // args.G)
            decs = run_episodes(eps, beta, True, cells=cells); D.close(eps)
            hist = [(r["prompt"], r["plan"]) for r in decs]; hkey = [r["key"] for r in decs]
            if args.excite:                               # random-look histories of the same worlds keep every belief class measured;
                rep = max(1, int(args.excite))            # --excite below 1: one random-look episode on that share of the worlds
                exe = D.start([s for s in specs for _ in range(rep)] if args.excite >= 1 else specs[:max(1, round(args.excite * len(specs)))])
                def walk(k, r_):
                    e = exe[k]; e.key = ("x", step, k // rep)
                    for _ in range(r_.randint(1, getattr(e, "_teacher", D).T)):
                        looks = list(D.look_strings(e))
                        if not looks or (D.belief(e).identified() and r_.random() < 0.5): break
                        D.do_look(e, r_.choice(looks)); e.t += 1
                        if hasattr(D, "after_turn"): D.after_turn(e)   # the domain's own turn count, as in an episode's turn (apply)
                        if e.done: break                          # a look that ended the game (a hole, a mine, a solved room)
                if os.environ.get("PL_EXCITE_PAR") == "1" and len(exe) > 1:
                    # speed: the random-look episodes walk at once (each its own simulator session), each with its own
                    # random stream from rng: the same distribution of looks, other draws than the one-by-one walk
                    rs = [random.Random(rng.getrandbits(64)) for _ in exe]
                    with _cfp.ThreadPoolExecutor(max_workers=len(exe)) as ex_: list(ex_.map(walk, range(len(exe)), rs))
                else:
                    for k in range(len(exe)): walk(k, rng)
                live = [e for e in exe if not e.done]          # a walk that ended its game has no next turn
                recs = act(live, 0.0) if live else []
                for e, r in zip(live, recs):
                    if r["commit"]:                       # the agent's own answer here (after a look there is no answer to measure)
                        y = r["answer_text"]
                        if cells is not None: cells.setdefault((e.key, r["plan"]["B"].cls()), []).append(ratio(D, r["plan"]["B"], y))
                        else:                             # the domain's own answer check (the episode closes after this turn)
                            ok = bool(D.do_commit(e, y))
                            with ep_lock: rel.update(r["plan"]["B"], ok, e.key, ratio(D, r["plan"]["B"], y))   # PL_PIPELINE_DEPTH: another update may write too
                    hist.append((r["prompt"], r["plan"])); hkey.append(e.key)
                D.close(exe)
            roll = {"success": statistics.fmean(bool(e.correct) for e in eps), "premature": premature_rate(eps),
                    "mean_t": statistics.fmean(e.t for e in eps), "beta": beta, "n_dec": len(decs), "n_hist_all": len(hist), "reliability": rel_table(),
                    "teacher_frac": statistics.fmean(bool(r.get("teacher_acted")) for r in decs) if decs else 0.0,
                    "answers_agent": sum(1 for r in decs if r["commit"] and not r.get("teacher_acted")),
                    "answers_teacher": sum(1 for r in decs if r["commit"] and r.get("teacher_acted"))}
            if len(hist) > B:                                            # the update's share of the on-policy histories (the same draw
                pick = rng.sample(range(len(hist)), B); hist = [hist[i] for i in pick]; hkey = [hkey[i] for i in pick]   # as rng.sample(hist, B))
        else:
            if nxt is not None: hist = nxt.result(); nxt = pool.submit(fetch)
            else: hist = collect_histories(D, tok, nrng, specs_it, args.B, planner, args.mix)
            hkey = [None] * len(hist)
        # the agent's own thoughts (EM): K per history from the current model, weighted by p(beta*, u* | h, z_k)
        K = args.em_k; own = args.objective != "critic" or phase > 0              # else from the base model
        M = args.em_m if args.objective == "critic" else 1; KM = K * M
        crit, more = [], {}
        if args.objective == "critic":                 # M own belief lines and actions after each own thought, scored by the teacher
            ok = lambda h: any(good_turn(c, hist[h][1]) for c in crit_h[h])
            if actor:                                  # every history its own chain, the more rounds included (own_chain)
                with _cfp.ThreadPoolExecutor(max_workers=max(1, len(hist))) as ex_:
                    res = list(ex_.map(lambda h: own_chain(hist[h][0], hist[h][1], K, M, own), range(len(hist))))
                rows_h, crit_h = [r for r, _ in res], [c for _, c in res]
            else:
                rows_h, crit_h = own_turns(hist, K, M, own)
                for _ in range(args.em_more):          # more own turns where no own turn takes an action the teacher accepts
                    miss = [h for h in range(len(hist)) if not ok(h)]
                    if not miss: break
                    for h, r, c in zip(miss, *own_turns([hist[h] for h in miss], K, M, own)): rows_h[h] += r; crit_h[h] += c
            n_trow = add_teacher_rows(rows_h, crit_h, [pl for _, pl in hist], args.accept,   # --teacher_row 1: the teacher's line and action
                                      lambda r0, pl, b, u: score_turn(r0[0], pl, r0[1], r0[6], r0[2], b, u)) if args.teacher_row else 0   # join a group with no accepted own turn
            why = [no_target(crit_h[h], hist[h][1], args.accept, args.line_floor if args.line_score != "none" else None)
                   if args.accept_only else None for h in range(len(hist))]
            left = {h for h in range(len(hist)) if args.em_more and args.leave_out and not ok(h)} | {h for h, y in enumerate(why) if y}
            # no estimate of q there: no M-step (--leave_out 0: a history that is not covered is fitted with its accepted turns;
            # --accept_only: one with no accepted turn, or whose accepted lines all have none of the posterior, gets no weight; the
            # dump keeps it)
            more = {"em_more_hist": sum(own_rows(r) > KM for r in rows_h), "em_left_out": len(left), "teacher_rows": n_trow,
                    "em_no_accept": why.count("no_accept"), "em_wrong_line": why.count("wrong_line"),
                    "em_accepted_0": statistics.fmean(any(accepted(c["A"], pl, args.accept) for c in ch[:KM]) for ch, (_, pl) in zip(crit_h, hist)),
                    "em_joint_0": statistics.fmean(any(accepted(c["A"], pl, args.accept) and c["S"] >= c["S_teacher"] - 1.0 for c in ch[:KM]) for ch, (_, pl) in zip(crit_h, hist)),
                    **look_stats([pl for _, pl in hist], args.accept)}
            if args.stall_cap > 0: more["stall_rejected"] = statistics.fmean([c.get("stall", False) for ch in crit_h for c in ch if not c.get("teacher")] or [0.0])   # --stall_cap: own turns rejected by the cap alone
            sizes = [0 if h in left else len(r) for h, r in enumerate(rows_h)]
            rows_ = [x for h, r in enumerate(rows_h) if h not in left for x in r]; crit = [c for ch in crit_h for c in ch if not c.get("teacher")]   # the logged means: own turns (--teacher_row: its turn is not one)
            if cells is not None:                      # --rho_rb: each own answer turn of the E-step, b(y) / U(b), into its history's cell
                for h, rr in enumerate(rows_h):
                    B = hist[h][1]["B"]
                    for row in rr:
                        if teacher_row(row): continue                    # the table measures the agent's answers only
                        y = answer_text(ordered(row[3], row[4])[1])      # the action line as the test reads it
                        if y is not None: cells.setdefault((hkey[h], B.cls()), []).append(ratio(D, B, y))
                fields, rho_before, rho_after = rho_commit(cells); more.update(fields)
        else:
            cands = sampler.sample([p for p, _ in hist], K, args.max_think, stop_think=True, adapter=own)
            rows_ = []
            for (prompt, pl), cand in zip(hist, cands):
                ks = list(pl["tgt"]); ps = np.array([pl["tgt"][k] for k in ks]); astar = ks[int(nrng.choice(len(ks), p=ps / ps.sum()))]
                for z, _ in cand: rows_.append((prompt, z, prompt + z + CLOSE, pl["anchor"], astar))
            sizes = [K] * len(hist); rows_h = [rows_[h * K:(h + 1) * K] for h in range(len(hist))]
        imgs = {}
        if images:                                                     # the worker ranks have no environment: send the images
            from plearn.models.sampler import IMAGES, prompt_key
            imgs = {prompt_key(p): IMAGES[prompt_key(p)] for p, _ in hist if prompt_key(p) in IMAGES}
        return {"roll": roll, "hist": hist, "rows_": rows_, "sizes": sizes, "rows_h": rows_h, "crit_h": crit_h, "crit": crit, "more": more,
                "K": K, "M": M, "KM": KM, "imgs": imgs, "decs": decs, "eps": eps, "t1": t1, "cells": cells, "hkey": hkey,
                "rho_tables": (rho_before, rho_after) if cells is not None and args.objective == "critic" else None}

    import concurrent.futures as _cf
    PIPE = os.environ.get("PL_PIPELINE") == "1" and os.environ.get("PL_VLLM_SLEEP") != "1"   # speed: next update's data during this update's M-step
    DEPTH = max(1, int(os.environ.get("PL_PIPELINE_DEPTH", "1")))   # speed: the data of up to DEPTH next updates at once
    gen_pool, ahead = _cf.ThreadPoolExecutor(DEPTH), {}             # step -> its gen_update future
    t0 = time.time()

    def record_update(step, g, l_short, l_think, gn, stats, used_prior, t_start, extra=None):
        """The log row of one update and, at --dump_em_steps, its E-step batch in full. With --role learner the batch came from the
        actor (slim: its plans without the belief objects, the agent-only success counted there)."""
        roll, hist, rows_, sizes, rows_h, crit_h, crit, more = g["roll"], g["hist"], g["rows_"], g["sizes"], g["rows_h"], g["crit_h"], g["crit"], g["more"]
        K, M, KM, imgs, decs, eps = g["K"], g["M"], g["KM"], g["imgs"], g["decs"], g["eps"]
        lp2 = [np.array(s[1]) for s in stats]; w = [np.array(s[2]) for s in stats]; fit = [h for h in range(len(hist)) if sizes[h]]
        ext = {s[0]: s[3] for s in stats if len(s) > 3 and s[3]}       # --temper, --credit share: lam_h and the credit moved per history
        if args.dump_em and step in {int(s) for s in args.dump_em_steps.split(",")}:     # one training batch in full, for inspection
            os.makedirs(args.dump_em, exist_ok=True); from plearn.models.sampler import untag
            wd = lambda h, k: float(w[h][k]) if sizes[h] else 0.0      # a history left out has no weights
            if crit:                                                   # own belief line and action, the teacher's scores and R
                th = lambda h, k: {"z": rows_h[h][k][1], "z_stopped": rows_h[h][k][6], "thought": k // M, "belief": rows_h[h][k][3],
                                   "action": rows_h[h][k][4], **crit_h[h][k], "w": wd(h, k)}
            else:
                th = lambda h, k: {"z": rows_h[h][k][1], "ll_belief": float(lp2[h][k, 0]), "ll_action": float(lp2[h][k, 1]), "w": wd(h, k)}
            json.dump([{"prompt": untag(prompt), "belief_line": pl["anchor"], "pi": pl["tgt"], "logp": pl.get("logp"), "values": pl["values"], "waste": pl["waste"],
                        "u_star": None if crit else rows_h[h][0][4], "trace": pl.get("exact"), "prior_step": used_prior, "left_out": not sizes[h], "e_step": ext.get(h),
                        "scene": str(g["hkey"][h]), "cls": pl["cls"] if "cls" in pl else str(pl["B"].cls()), "draw_w": g["hw"][h] if g.get("hw") else None,
                        "thoughts": [th(h, k) for k in range(len(rows_h[h]))]} for h, (prompt, pl) in enumerate(hist)],
                      open(os.path.join(args.dump_em, f"step{step}.json"), "w"), indent=1)
            if g.get("rho_tables") is not None:                        # --rho_rb: the answers of each cell, the table before and after them
                json.dump({"cells": {str(c): xs for c, xs in g["cells"].items()}, "table_before": g["rho_tables"][0], "table_after": g["rho_tables"][1]},
                          open(os.path.join(args.dump_em, f"step{step}_rho.json"), "w"), indent=1)
            if images:                                                 # the camera images of each history, small
                from plearn.models.sampler import IMAGES as _IM, prompt_key as _pk   # (the learner process has no enclosing import)
                for h, (prompt, _) in enumerate(hist):
                    for j, im in enumerate(_IM.get(_pk(prompt), [])):
                        im.convert("RGB").resize((320, 320)).save(os.path.join(args.dump_em, f"step{step}_h{h}_img{j}.jpg"), quality=82)
        if images:                                                     # the images of this batch are not used again
            from plearn.models.sampler import IMAGES as _IM, prompt_key as _pk
            for p, _ in hist: _IM.pop(_pk(p), None)
        N = max(1, len(hist))
        fm = lambda xs: float(statistics.fmean(xs)) if xs else 0.0   # over the histories the M-step fits
        em = {"em_w_max": fm([w[h].max() for h in fit]), "em_ess": fm([1.0 / (w[h] ** 2).sum() for h in fit]),
              "think_chars": fm([len(r[1]) for r in rows_]), **more}
        if crit:                                                       # how the samples score
            sl = [x for c in crit for x in (c.get("slots") or [])]      # the statements of the own lines in the language
            if sl: em.update({"slot_ok": fm([abs(x[3] - x[4]) < 1e-6 for x in sl]), "slot_gap": fm([x[4] - x[3] for x in sl]), "slot_n": len(sl) / max(1, len(crit))})
            em.update({"line_exact": statistics.fmean(bool(c["exact"]) for c in crit),
                       "crit_belief_ok": statistics.fmean(c["S"] >= c["S_teacher"] - 1e-6 for c in crit),
                       "crit_belief_gap": statistics.fmean(c["S_teacher"] - c["S"] for c in crit),
                       "crit_act_best": statistics.fmean(c["V"] >= c["V_best"] - 1e-9 for c in crit),
                       "crit_act_gap": statistics.fmean(c["V_best"] - c["V"] for c in crit),
                       "crit_R_max": statistics.fmean(max(c["R"] for c in ch) for ch in crit_h)})
            if args.line_wide: em["crit_line_wide"] = statistics.fmean(-c["W"] / args.line_floor for c in crit)   # the own lines' excluded share
            wp, wxp, wa, wx = [], [], [], []                           # where the weight of step 1 goes, per fitted history
            for h in fit:
                pr = xp = ac = xl = 0.0
                for k, c in enumerate(crit_h[h]):
                    if k >= len(w[h]): break
                    uo = ordered(rows_h[h][k][3], rows_h[h][k][4])[1]; wk = float(w[h][k])
                    prem = "SUBMIT" in uo.upper() and c["A"] < -5.0; exact = c["S"] >= c["S_teacher"] - 1e-6
                    pr += wk * prem; xp += wk * (prem and exact); ac += wk * accepted(c["A"], hist[h][1], args.accept); xl += wk * exact
                wp.append(pr); wxp.append(xp); wa.append(ac); wx.append(xl)
            em.update({"w_premature": fm(wp), "w_exact_premature": fm(wxp), "w_accepted": fm(wa), "w_exact_line": fm(wx)})
            ex_ = [ext[h] for h in fit if h in ext]
            if ex_: em.update({"em_lambda": fm([x["lam"] for x in ex_]), "em_lambda_lt1": fm([x["lam"] < 1.0 for x in ex_]),
                               "credit_moved": fm([x["moved"] for x in ex_]), "credit_n_share": fm([x["n_share"] for x in ex_]),
                               "credit_r_own": fm([x["r_own"] for x in ex_ if x["r_own"] is not None])})
            rd = [[r for r in (c.get("read") for c in ch) if r and r["evidence"]] for ch in crit_h]; rd = [rs for rs in rd if rs]
            if rd:                                                     # a domain with reading() (tabletop): the own samples of each
                kinds = ("exact", "wider", "misread", "reasoning", "unreadable")   # history by kind, and the histories where no sample
                em.update({f"estep_read_{k}": fm([sum(r["kind"] == k for r in rs) / len(rs) for rs in rd]) for k in kinds})   # reads
                em.update({"estep_read_no_exact": fm([not any(r["kind"] == "exact" for r in rs) for rs in rd]), "estep_read_hist": len(rd)})   # exactly
            if M > 1:                                                  # how many thoughts share the weight
                em["em_ess_think"] = fm([1.0 / (own_w(w[h], rows_h[h], M).reshape(-1, M).sum(1) ** 2).sum() for h in fit])
        else:
            em["em_target_ll"] = fm([lp2[h].sum(1).max() for h in fit])
        try: em.update(monitor_train_metrics(crit_h if crit else None, rows_h, w, fit, decs if args.episodes else None, eps if args.episodes else None))
        except Exception as ex_: em["metrics_err"] = 1; print("monitor_train_metrics failed:", repr(ex_)[:300], flush=True)
        em.update(g.get("agent_only") or {})                       # --role learner: counted by the actor
        row = {"step": step, "prior_step": used_prior, "loss_short": l_short, "loss_think": l_think, **em, "grad_norm": gn, "n_hist": N, **roll,
               **MSTEP, "sec": time.time() - t_start, "wall_min": (time.time() - t0) / 60, **(extra or {})}
        log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True); board.log(row, step, "train")

    def log_eval(s, ev, ex, w_step):
        """An evaluation row: in the log file at its own update s, and on W&B at the run's step w_step with eval/update = s (its x axis)."""
        ev = {**ev, "update": s}
        print("EVAL", json.dumps({"step": s, **ev}), json.dumps(ex), flush=True)
        log.write(json.dumps({"step": s, "eval": ev}) + "\n"); log.flush(); board.log(ev, w_step, "eval")

    def save_lora(step, with_opt=True):
        if args.save:                                   # every evaluated step is kept, so any point of the curve can be re-evaluated
            sd = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
            def put(obj, name):                         # write, then rename (several ranks of a node write the same file;
                f = os.path.join(args.out, name); t = f"{f}.{os.getpid()}.tmp"   # a reader or a stop never sees half a file)
                torch.save(obj, t); os.replace(t, f)
            put(sd, f"{tag}-lora.pt"); put(sd, f"{tag}-lora-{step}.pt")
            if opt is not None and with_opt: put(opt.state_dict(), f"{tag}-opt-{step}.pt")   # for --resume_step
            if READOUT.get("head") is not None: put(READOUT["head"].state_dict(), f"{tag}-readout-{step}.pt")

    # ---- --role actor / learner: two processes that share the folder --queue (batches, adapters, evaluations, progress)
    Q = args.queue
    def q_path(*a): return os.path.join(Q, *a)
    def q_write(name, data, binary=False):
        tmp = q_path(f".{name}.{os.getpid()}.tmp")                   # write, then rename: the reader never sees half a file
        with open(tmp, "wb" if binary else "w") as f: f.write(data)
        os.replace(tmp, q_path(name))
    def q_wait(name, what, poll=1.0):
        """Wait for a file of the other process (a line every 10 minutes); stop when the other process has failed."""
        t_ = last = time.time()
        while not os.path.exists(q_path(name)):
            if os.path.exists(q_path("actor_failed")) or os.path.exists(q_path("learner_failed")):
                raise SystemExit(f"the other process failed while this one waited for {what}")
            if time.time() - last > 600: print(f"waiting {int(time.time() - t_)} s for {what}", flush=True); last = time.time()
            time.sleep(poll)
        return time.time() - t_
    def learner_at():
        try: return int(open(q_path("learner_step")).read().strip() or 0)
        except Exception: return 0

    def slim(g):
        """An actor batch for the learner: the plans without their belief objects (only the fields the log and the dump read), no
        episode objects; the agent-only success counted here."""
        keep = ("anchor", "tgt", "logp", "values", "waste", "exact")
        hist = [(p, {**{k: pl[k] for k in keep if k in pl}, "cls": str(pl["B"].cls())}) for p, pl in g["hist"]]
        agent = {}
        if g.get("eps"):
            helped = {r["ep"] for r in g["decs"] if r.get("teacher_acted")}; alone = [e for e in g["eps"] if id(e) not in helped]
            agent = {"agent_only_success": float(np.mean([float(bool(e.correct)) for e in alone])) if alone else None, "agent_only_n": len(alone)}
        imgs = {k: [png_bytes(im) for im in ims] for k, ims in (g.get("imgs") or {}).items()}     # image domains: the learner needs them
        return {**{k: v for k, v in g.items() if k not in ("hist", "decs", "eps", "imgs")}, "hist": hist, "decs": None, "eps": None, "imgs": imgs,
                "agent_only": agent, "gen_sec": time.time() - g["t1"], "gen_end": time.time()}

    def actor_loop():
        """--role actor: every update's batch as soon as its phase prior is loaded, up to --gen_depth updates at one time and at most
        --ahead updates past the learner. At each phase end it loads the learner's adapter (the next prior: exactly the model after
        that update) and evaluates that model beside the next batches; the same model is the one the one-process run evaluates."""
        import pickle
        os.makedirs(Q, exist_ok=True)
        EVAL_DEFER = os.environ.get("PL_EVAL_DEFER") == "1"           # no evaluation in the run (the saved adapters are
                                                                       # evaluated after training): no switch waits for one
        ev_pool, gp, evs = _cf.ThreadPoolExecutor(1), _cf.ThreadPoolExecutor(max(1, args.gen_depth)), []
        def run_eval(s):
            ev, ex = evaluate(s); ev = {**ev, "update": s}
            q_write(f"eval_{s}.json", json.dumps({"step": s, "eval": ev, "ex": ex}))
            log.write(json.dumps({"step": s, "eval": ev}) + "\n"); log.flush(); print("EVAL", json.dumps({"step": s, **ev}), flush=True)
        def gen_one(s): return slim(gen_update(s))
        def load_prior(b):
            nonlocal loaded
            wt = q_wait(f"adapter_{b}.ready", f"the adapter of update {b}"); sampler.load(q_path(f"adapter_{b}")); loaded = b
            log.write(json.dumps({"actor_wait_adapter": wt, "update": b, "wall_min": (time.time() - t0) / 60}) + "\n"); log.flush()
            if b % args.eval_every == 0 and not os.path.exists(q_path(f"eval_{b}.json")) and not EVAL_DEFER: evs.append(ev_pool.submit(run_eval, b))   # (--resume_step: done)
        def subset(s, idx, first):
            """A batch of the phase's sample set: the histories idx with their own turns from the set's E-step. The set's episode
            statistics, answer accuracy tables and generation time go with the phase's first batch only; the later batches reuse the set."""
            from plearn.models.sampler import prompt_key
            pick = lambda xs: [xs[i] for i in idx]
            hist, sizes, rows_h = pick(s["hist"]), pick(s["sizes"]), pick(s["rows_h"]); crit_h = pick(s["crit_h"]) if s.get("crit_h") else None
            more = {**(s["more"] if first else {}), "em_more_hist": sum(own_rows(r) > s["KM"] for r in rows_h), "em_left_out": sum(z == 0 for z in sizes),
                    "teacher_rows": sum(any(teacher_row(x) for x in r) for r in rows_h),   # --teacher_row: the histories of this batch with a teacher row
                    "set_size": len(s["hist"]), "set_new": int(first)}
            keys = {prompt_key(p) for p, _ in hist}
            return {**s, "hist": hist, "hkey": pick(s["hkey"]), "sizes": sizes, "rows_h": rows_h, "crit_h": crit_h,
                    "rows_": [x for r, z in zip(rows_h, sizes) if z for x in r], "crit": [c for ch in crit_h for c in ch if not c.get("teacher")] if crit_h else s["crit"],   # --teacher_row: own turns only
                    "more": more, "roll": s["roll"] if first else {}, "cells": s["cells"] if first else None,
                    "rho_tables": s["rho_tables"] if first else None, "agent_only": s["agent_only"] if first else {},
                    "imgs": {k: v for k, v in s["imgs"].items() if k in keys}, "gen_sec": s["gen_sec"] if first else 0.0}
        if os.environ.get("PL_SKIP_EVAL0") != "1" and not args.resume_step: evs.append(ev_pool.submit(run_eval, 0))   # beside the first batches (the same base model)
        step, loaded = 1, 0
        if args.resume_step: step = args.resume_step + 1 + (phase if have_set else 0)   # the first update whose set this actor makes
        if args.phase_set and phase:                                   # one sample set per phase: q_n is fixed for the phase
            draw, worlds_set = random.Random(rseed + 1), max(1, math.ceil(args.phase_set * args.worlds / max(1, args.B)))
            if last_set: worlds_set = max(1, math.ceil(1.3 * args.phase_set * last_set["set_worlds"] / last_set["set_pool"]))   # as below
            while step <= args.steps:
                b = ((step - 1) // phase) * phase
                if b > loaded: load_prior(b)
                end = min(args.steps, b + phase)
                if b == 0 and set_from:                                # speed: PL_SET_FROM=<dir>, the first sample set of an
                    # earlier run with the same model and settings (set.json): its batches 1 to --phase and its answer-accuracy table
                    # (as --resume_step reads it), so the first update needs no hour of plays; later sets are made as before
                    meta = json.load(open(os.path.join(set_from, "set.json")))
                    free = {"role", "tag", "out", "queue", "colocate", "resume_step", "vllm_gpu", "vllm_mem", "ahead", "gen_depth", "steps",
                            "eval_every", "eval_episodes", "eval_max_think", "watch", "policy_metrics", "dump_em", "dump_em_steps", "dump_rollouts", "domain_args",
                            "think_cost", "length_reward", "length_budget", "em_j", "kl_stop", "kl_probe", "kl_log", "lora_all_linear", "score_presence", "line_target",
                            "update", "grpo_clip", "grpo_eps", "grpo_norm", "attn"}   # run and speed only (lora_all_linear: LoRA B = 0 at the start, so the first set's policy is the same), and the learner's E-step weights, M-step
                                                                       # draws and step limit (no sample depends on them); --update grpo: the same sets as em, its weights the learner's
                    free.add("teacher_row_cap")                       # --teacher_row_cap: the learner's weights only, as grpo_clip
                    mine = vars(args)
                    for k, v in meta["args"].items():
                        if k.startswith("wandb_") or k in free: continue
                        assert mine.get(k) == v, f"PL_SET_FROM: --{k} {mine.get(k)!r}, the saved set's {v!r}"
                    da = lambda s: {k: v for k, v in json.loads(s).items() if k != "bridges"}
                    assert da(args.domain_args) == da(meta["args"]["domain_args"]), "PL_SET_FROM: --domain_args differ (bridges aside)"
                    assert meta["args"].get("set_keep", 0) == args.set_keep, "PL_SET_FROM: the saved set has another --set_keep"
                    assert (meta["args"].get("teacher_row", 0), meta["args"].get("rho_exact", 0)) == (args.teacher_row, args.rho_exact), \
                        "PL_SET_FROM: the saved set has another --teacher_row or --rho_exact (a set saved before these switches has neither)"
                    assert (float(meta["args"].get("stall_cap", 0.0)), int(meta["args"].get("stall_play", 0))) == (args.stall_cap, args.stall_play), \
                        "PL_SET_FROM: the saved set has another --stall_cap or --stall_play (its plans hold the capped scores, and stall_play the teacher's moves; a set saved before them: 0)"
                    assert (meta["args"].get("tau_look", 0.0), meta["args"].get("tau_look_play", 0)) == (args.tau_look, args.tau_look_play), \
                        "PL_SET_FROM: the saved set has another --tau_look / --tau_look_play (its batches hold the own turns' scores and accepts; a set.json without them: 0)"
                    assert not any(getattr(d, "name", None) == "frozenlake" for d in getattr(D, "subs", [D])) or meta["args"].get("fl_view_value", 1) == args.fl_view_value, \
                        "PL_SET_FROM: the saved set's FrozenLake teacher has another --fl_view_value (a set made before the fix: 1; the module text)"
                    assert not any(getattr(d, "name", None) == "sokoban" for d in getattr(D, "subs", [D])) or meta["args"].get("sk_guess_value", 1) == args.sk_guess_value, \
                        "PL_SET_FROM: the saved set's Sokoban teacher has another --sk_guess_value (a set made before the fix: 1; the module text)"
                    assert not any(getattr(d, "name", None) in ("minigrid", "minesweeper", "tabletop", "roboquest", "roboquest_img") for d in getattr(D, "subs", [D])) or meta["args"].get("plan_past_limit", 1) == args.plan_past_limit, \
                        "PL_SET_FROM: the saved set's MiniGrid / MineSweeper / TableTop / RoboQuest teacher has another --plan_past_limit (a set made before the fix: 1; the module text)"
                    assert int(meta["args"].get("no_belief_line", 0)) == args.no_belief_line, \
                        "PL_SET_FROM: the saved set has another --no_belief_line (its prompts and own turns have or lack the belief line; a set saved before it: 0)"
                    for k, v in meta["env"].items():
                        assert os.environ.get(k) == v, f"PL_SET_FROM: {k}={os.environ.get(k)!r}, the saved set's {v!r}"
                    assert all(os.path.exists(os.path.join(set_from, f"batch_{u}.pkl")) for u in range(step, end + 1)), "PL_SET_FROM: a batch is missing"
                    if args.rho_rb:
                        import ast, shutil
                        shutil.copy(os.path.join(set_from, "rho-0.json"), os.path.join(args.out, f"{tag}-rho-0.json"))
                        for k, (s_, n_) in json.load(open(os.path.join(set_from, "rho-0.json"))).items(): rel.add(ast.literal_eval(k), s_, n_)
                    for u in range(step, end + 1):
                        q_write(f"batch_{u}.pkl", open(os.path.join(set_from, f"batch_{u}.pkl"), "rb").read(), binary=True)
                    log.write(json.dumps({"phase_set": 0, "set_size": meta["set_size"], "set_pool": meta["set_pool"], "set_worlds": meta["set_worlds"],
                                          "set_sec": meta["set_sec"], "set_from": set_from, "wall_min": (time.time() - t0) / 60}) + "\n"); log.flush()
                    worlds_set = max(1, math.ceil(1.3 * args.phase_set * meta["set_worlds"] / meta["set_pool"]))   # as below
                else:
                    s = slim(gen_update(step, B=args.phase_set, worlds=worlds_set)); H = len(s["hist"])
                    pool = (s["roll"] or {}).get("n_hist_all", H); kept = {}
                    if args.set_keep:                                  # dynamic sampling: the histories with a gradient, skipped ones not redrawn
                        keep, n_ok = keep_set(s["sizes"], s["rows_h"], args.set_keep, args.neg_answer > 0 or args.neg_thought > 0)
                        if not keep: raise SystemExit(f"--set_keep: no history of the sample set from update {step} has an accepted own turn")
                        plans = [pl for _, pl in s["hist"]]; drawn, kt = by_task(plans, range(H)), by_task(plans, keep)
                        kept = {"set_drawn": H, "set_accepted": n_ok, "set_kept": len(keep), "set_by_task": {k: [kt.get(k, 0), n] for k, n in drawn.items()}}
                        s = subset(s, keep, True); s["more"]["set_accept_share"] = n_ok / H; H = len(keep)
                    log.write(json.dumps({"phase_set": b // phase, "set_size": H, "set_pool": pool, "set_worlds": worlds_set, "set_sec": s["gen_sec"],
                                          **kept, "wall_min": (time.time() - t0) / 60}) + "\n"); log.flush()
                    if s.get("rho_tables"):                                # the answer-accuracy table after this set (for --resume_step)
                        with open(os.path.join(args.out, f"{tag}-rho-{b}.json"), "w") as f: json.dump(s["rho_tables"][1], f)
                    if pool: worlds_set = max(1, math.ceil(1.3 * args.phase_set * worlds_set / pool))   # next phase: about 30% more histories than the set needs
                                                                       # (the episodes get shorter as the agent learns; extra turns cost only episode thoughts)
                    order = []
                    if args.set_draw == "is":                              # importance sampling of the histories (--set_draw)
                        sig = [set_signal([x[5] for x in r], args.alpha) if z else None for r, z in zip(s["rows_h"], s["sizes"])]
                        incl = set_inclusion(sig, args.B, args.set_prio); groups = [str(k) for k in s["hkey"]]
                    for u in range(step, end + 1):                         # passes over the set without replacement (or --set_draw is)
                        hw = None
                        if args.set_draw == "is":                          # each drawn history's loss times 1 / (H p_h): the phase's
                            idx = pps_draw(incl, groups, draw); hw = [len(idx) / (H * incl[i]) for i in idx]   # objective in expectation
                        else:
                            idx = []
                            while len(idx) < min(args.B, H):
                                if not order: order = [i for i in draw.sample(range(H), H) if i not in idx]
                                idx.append(order.pop())
                        g = subset(s, idx, u == step); g["hw"] = hw
                        q_write(f"batch_{u}.pkl", pickle.dumps(g), binary=True)
                        log.write(json.dumps({"actor_step": u, "gen_sec": g["gen_sec"], "set_new": int(u == step), "learner_at": learner_at(),
                                              "draw_w_max": max(hw) if hw else None, "wall_min": (time.time() - t0) / 60}) + "\n"); log.flush()
                if args.colocate:                                      # all the GPUs to the learner for the phase's updates, then back
                    for f in evs: f.result()                           # the evaluations use the server: they end first
                    evs.clear(); t_sw = time.time()
                    sampler.stop(); q_write(f"gen_done_{b}", "1")                 # (PL_SET_FROM: no server ran: nothing to stop)
                    q_wait(f"learner_parked_{end}", f"the learner's updates {step} to {end}")
                    log.write(json.dumps({"colocate_learner_min": (time.time() - t_sw) / 60, "update": end, "wall_min": (time.time() - t0) / 60}) + "\n"); log.flush()
                    sampler.start()                                    # the next phase's prior is loaded at the loop's next turn
                else: q_write(f"gen_done_{b}", "1")                    # the phase's batches are all written (--resume_step reads it)
                step = end + 1
        while step <= args.steps:
            b = ((step - 1) // phase) * phase if phase else step - 1       # the update whose model is this step's prior (0: the base model)
            if b > loaded: load_prior(b)
            end = min(args.steps, b + phase) if phase else step
            futs, s_next = {}, step
            while step <= end:
                while s_next <= end and len(futs) < max(1, args.gen_depth) and s_next - learner_at() <= max(1, args.ahead):
                    futs[s_next] = gp.submit(gen_one, s_next); s_next += 1
                if step in futs and futs[step].done():
                    g = futs.pop(step).result()
                    q_write(f"batch_{step}.pkl", pickle.dumps(g), binary=True)
                    log.write(json.dumps({"actor_step": step, "gen_sec": g["gen_sec"], "learner_at": learner_at(), "wall_min": (time.time() - t0) / 60}) + "\n"); log.flush()
                    step += 1
                else:
                    time.sleep(0.2)
        if args.steps > loaded:                                        # the last model: its adapter, then its evaluation
            q_wait(f"adapter_{args.steps}.ready", f"the adapter of update {args.steps}"); sampler.load(q_path(f"adapter_{args.steps}"))
            if not EVAL_DEFER: evs.append(ev_pool.submit(run_eval, args.steps))
        for f in evs: f.result()
        q_write("actor_done", "1")

    def learner_loop():
        """--role learner: the M-step of every update on the actor's batch (it waits only while that batch is not there yet); the adapter
        after every phase end (the actor's next prior) and after the last update; the actor's evaluations logged as they arrive."""
        import pickle
        nonlocal prior_step
        os.makedirs(Q, exist_ok=True); t_prev = time.time(); logged = {r["step"] for r in old_rows if "eval" in r}
        SAVE_STEPS = int(os.environ.get("PL_SAVE_STEPS") or 25)      # an adapter every 25 updates, not every update
        assert SAVE_STEPS >= 1, "PL_SAVE_STEPS must be >= 1"
        KEEP_LAST = int(os.environ.get("PL_KEEP_LAST", "2"))           # + the full state of the last 2 updates
        assert KEEP_LAST >= 0, "PL_KEEP_LAST must be >= 0 (0 = the old saves only)"
        def kept_step(n): return n % args.eval_every == 0 or n % SAVE_STEPS == 0 or n == args.steps   # never removed by keep_last_saves
        if os.environ.get("PL_SAVE_EVERY"):                            # the old flag (an adapter every update) is not read
            print(f"PL_SAVE_EVERY={os.environ['PL_SAVE_EVERY']} IGNORED: adapters every PL_SAVE_STEPS={SAVE_STEPS} updates "
                  f"(+ every --eval_every={args.eval_every} with the optimizer state)", flush=True)
        prior_step = args.resume_step                                  # --resume_step R: the prior of the phase from R + 1 is the model after R
        def park(to):
            """--colocate: every rank's model to the CPU (between phases) or to the GPU (for a phase): rank 0 sends the job to the
            worker ranks, moves its own model, and all ranks meet before the GPUs change hands."""
            if world > 1: dist.broadcast_object_list([{"park": to}], src=0, group=CPU_GROUP)
            park_model(to)
            if world > 1: dist.barrier(group=CPU_GROUP)
        if args.colocate:
            park("cpu"); q_write("learner_parked_0", "1")             # every rank has loaded its model into CPU memory: the server can start
        def eval_rows(w_step):
            for f in sorted(os.listdir(Q)):
                m = re.fullmatch(r"eval_(\d+)\.json", f)
                if m and int(m.group(1)) not in logged:
                    r = json.load(open(q_path(f))); log_eval(r["step"], r["eval"], r.get("ex"), w_step); logged.add(int(m.group(1)))
        skip_to = 0                                                    # --kl_stop: the phase's updates up to this one are skipped
        for step in range(args.resume_step + 1, args.steps + 1):
            if args.colocate and phase and (step - 1) % phase == 0:  # the phase's batches are written and the server is stopped
                q_wait(f"gen_done_{step - 1}", f"the sample set of the phase from update {step}"); park("cuda")
            waited = q_wait(f"batch_{step}.pkl", f"the batch of update {step}", poll=0.2)
            skip = step <= skip_to                                     # (the batch is there: the phase's set is made at its start)
            if not skip:
                g = pickle.load(open(q_path(f"batch_{step}.pkl"), "rb"))
                imgs = {k: [from_png(b) for b in bs] for k, bs in (g.get("imgs") or {}).items()}
                if imgs:
                    from plearn.models.sampler import IMAGES; IMAGES.update(imgs)
                model.eval(); snap = step == 1 or bool(phase and (step - 1) % phase == 0)      # --credit share: the first update of a phase keeps its prior
                kp, lw, okr = think_cost_at(args.think_cost, step, phase), think_cost_at(args.length_reward, step, phase), line_ok_rows(g)
                neg = neg_rows(g["sizes"], g["rows_h"]) if (args.neg_answer > 0 or args.neg_thought > 0) and g.get("rows_h") else None   # --neg_answer, --neg_thought
                ltg = line_target_rows(g)                              # --line_target: the rows whose belief line is verified
                if world > 1: dist.broadcast_object_list([{"rows": g["rows_"], "nh": len(g["hist"]), "sizes": g["sizes"], "images": imgs, "snap": snap,
                                                           "hw": g.get("hw"), "kappa": kp, "lrw": lw, "okr": okr, "neg": neg, "ltg": ltg}], src=0, group=CPU_GROUP)
                l_short, l_think, gn, stats = em_update(g["rows_"], len(g["hist"]), g["sizes"], snap, g.get("hw"), kp, lw, okr, neg, ltg); used = prior_step
                kl, stop = MSTEP.get("kl_prior") if args.kl_stop else None, {}
                if kl is not None and kl > args.kl_stop and step < phase_end(step, phase, args.steps):   # the phase has gone its limit: the
                    skip_to = phase_end(step, phase, args.steps); stop = {"kl_stop_left": skip_to - step}   # model after this update is the next prior
                    print(f"KL STOP at update {step}: KL {kl:.4f} > {args.kl_stop} nats per token; updates {step + 1}-{skip_to} skipped", flush=True)
            if phase == 0 or step % phase == 0 or step == args.steps:  # the actor's next prior, and the last model for its evaluation
                from plearn.models.sampler import export_lora
                export_lora(model, q_path(f"adapter_{step}"), args.model, args.lora_r); q_write(f"adapter_{step}.ready", "1")
                if phase == 0 or step % phase == 0: prior_step = step if phase else 0
            q_write("learner_step", str(step))
            if not skip:
                record_update(step, g, l_short, l_think, gn, stats, used, t_prev,
                              {"queue_wait": waited, "gen_sec": g.get("gen_sec"), "batch_age": time.time() - g.get("gen_end", time.time()), **stop})
                if imgs:                                               # after the dump of the batch's images (--dump_em)
                    from plearn.models.sampler import IMAGES
                    for k in imgs: IMAGES.pop(k, None)
            t_prev = time.time()
            if step % args.eval_every == 0 or step == args.steps or KEEP_LAST: save_lora(step)   # PL_KEEP_LAST > 0 = the full state every update
            elif step % SAVE_STEPS == 0: save_lora(step, with_opt=False)   # one adapter every 25 updates (PL_SAVE_STEPS)
            if KEEP_LAST and args.save:                                     # only the last KEEP_LAST of those stay (+ the kept steps)
                gone = keep_last_saves(args.out, tag, step, KEEP_LAST, kept_step)
                if gone and lead: print(f"KEEP LAST {KEEP_LAST}: removed the saves of updates {gone}", flush=True)
            if lead: keep_batch(q_path(f"batch_{step}.pkl"), args.dump_em, step, log)   # rank 0 only: every rank has read the batch (the
                                                                       # update's collectives); another rank's delete would race the move
            if args.colocate and (phase and step % phase == 0 or step == args.steps):   # the phase is trained and its adapter written:
                park("cpu"); q_write(f"learner_parked_{step}", "1")               # the GPUs go back to the actor's server
            eval_rows(step); board.commit(step)
        q_wait("actor_done", "the actor's last evaluations"); eval_rows(args.steps); board.commit(args.steps)

    try:
        if actor: actor_loop()
        elif learner: learner_loop()
        else:
            if os.environ.get("PL_SKIP_EVAL0") != "1":                   # PL_SKIP_EVAL0=1: no update-0 evaluation (the
                ev, ex = evaluate(0); log_eval(0, ev, ex, 0); board.commit(0)   # untrained model), straight to training
            for step in range(1, args.steps + 1):
                g = ahead.pop(step).result() if step in ahead else gen_update(step)
                if PIPE:                                              # the next updates' data, all under the same phase prior: never across
                    for s_ in range(step + 1, min(step + DEPTH, args.steps) + 1):   # a phase boundary (sync) or an evaluation (DEPTH 1 = one update ahead)
                        if any(phase == 0 or u % phase == 0 or u % args.eval_every == 0 for u in range(step, s_)): break
                        if s_ not in ahead: ahead[s_] = gen_pool.submit(gen_update, s_)
                model.eval()
                sampler.sleep()                                       # PL_VLLM_SLEEP=1: before any rank starts its M-step
                snap = step == 1 or bool(phase and (step - 1) % phase == 0)      # --credit share: the first update of a phase keeps its prior
                kp, lw, okr = think_cost_at(args.think_cost, step, phase), think_cost_at(args.length_reward, step, phase), line_ok_rows(g)
                neg = neg_rows(g["sizes"], g["rows_h"]) if (args.neg_answer > 0 or args.neg_thought > 0) and g.get("rows_h") else None   # --neg_answer, --neg_thought
                ltg = line_target_rows(g)                              # --line_target: the rows whose belief line is verified
                if world > 1: dist.broadcast_object_list([{"rows": g["rows_"], "nh": len(g["hist"]), "sizes": g["sizes"], "images": g["imgs"], "snap": snap,
                                                           "kappa": kp, "lrw": lw, "okr": okr, "neg": neg, "ltg": ltg}], src=0, group=CPU_GROUP)
                l_short, l_think, gn, stats = em_update(g["rows_"], len(g["hist"]), g["sizes"], snap, None, kp, lw, okr, neg, ltg); used_prior = prior_step
                sampler.wake()
                if phase == 0 or step % phase == 0: sampler.sync(model); prior_step = step if phase else 0    # the next phase's prior
                record_update(step, g, l_short, l_think, gn, stats, used_prior, g["t1"])
                if step % args.eval_every == 0 or step == args.steps:
                    ev, ex = evaluate(step); log_eval(step, ev, ex, step); save_lora(step)
                board.commit(step)                              # the train row (and the evaluation) of this update on W&B now
    except BaseException:
        if actor or learner:                                         # the other process stops instead of waiting for ever
            import traceback
            try: q_write("actor_failed" if actor else "learner_failed", traceback.format_exc())
            except Exception: pass
        raise
    finally:
        if world > 1: dist.broadcast_object_list([None], src=0, group=CPU_GROUP); dist.destroy_process_group()
        if sampler is not None: sampler.stop()
        board.finish()
    print("DONE", tag, time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True); ap.add_argument("--domain_args", default="{}"); ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=os.environ.get("EVIDENCE_MODEL", "Qwen/Qwen3.5-4B"))
    ap.add_argument("--vllm_gpu", default="0", help="GPU ids for the vLLM servers, comma-separated: one server per GPU, requests routed by history"); ap.add_argument("--vllm_mem", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=0, help="0: warm-up on teacher histories; 1: the agent's own episodes")
    ap.add_argument("--worlds", type=int, default=4); ap.add_argument("--G", type=int, default=4); ap.add_argument("--excite", type=float, default=2)
    ap.add_argument("--beta_per", choices=("iteration", "update"), default="iteration", help="alpha_k decays per iteration (phase), as the method states; 'update': the runs before 2026-10-24")
    ap.add_argument("--beta0", type=float, default=0.5); ap.add_argument("--beta_decay", type=float, default=0.93)
    ap.add_argument("--value", default="accuracy", choices=E.VALUES, help="the teacher's value of a belief: accuracy (the average answer accuracy, "
                    "c rho) or log (the average log accuracy, log c + log rho; the scale of --tau is then nats)")
    ap.add_argument("--eval_split", default="", help="the split of the evaluation and the held-out histories (Domain(split=...)), "
                    "e.g. validation or test; empty: the domain's default")
    ap.add_argument("--eval_episodes", type=int, default=0); ap.add_argument("--dump_rollouts", default="", help="a directory: write every evaluation episode in full")
    ap.add_argument("--init_lora", default="")
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_all_linear", type=int, default=0, help="1: LoRA on every Linear layer of the language model (+ the Gated DeltaNet in_proj_a, in_proj_b; plearn/method/estimator.py add_lora)")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="AdamW's weight decay of the LoRA (0: none; plearn/train.py METHOD: one LoRA adapter for every method, lr 1e-5 and weight decay 0.01)")
    ap.add_argument("--mix", type=float, default=0.5); ap.add_argument("--tau", type=float, default=0.02); ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--tau_look", type=float, default=0.0, help="the temperature of pi*'s Boltzmann among the looks only (0: --tau); "
                    "answer-or-look, the stop rule and the answers keep --tau; a domain with look_scale = False (mastermind) keeps --tau")
    ap.add_argument("--tau_look_play", type=int, default=0, choices=[0, 1], help="1: the teacher's played moves (plan tgt) use --tau_look "
                    "too (other histories); 0: only the scores of own turns and the accept rule")
    ap.add_argument("--kappa", type=float, default=4.0); ap.add_argument("--ucb_z", type=float, default=1.0)
    ap.add_argument("--rho", choices=["class", "fidelity"], default="class")   # the answer accuracy: per class, or U(b) * fidelity
    ap.add_argument("--role", choices=("both", "actor", "learner"), default="both",
                    help="both: one process makes each update's data and then fits it (the order of the method); actor: only the episodes, "
                         "the E-step and the evaluations (vLLM and the teacher, no trainer model); learner: only the M-steps (torchrun). The "
                         "actor writes each batch to --queue and the learner writes the adapter of each phase end there; inside a phase every "
                         "sample comes from the fixed phase prior, so the actor may run ahead without changing any sample's distribution")
    ap.add_argument("--queue", default="", help="--role actor/learner: the folder both processes share")
    ap.add_argument("--ahead", type=int, default=3, help="--role actor: at most this many updates past the learner (never past a phase end)")
    ap.add_argument("--gen_depth", type=int, default=2, help="--role actor: the updates whose data is made at one time")
    ap.add_argument("--rho_rb", type=int, default=0, help="1: the fidelity table gets, per scene and belief class, the mean b(y) / U(b) "
                    "over all the agent's answers there (its episode answer and the E-step's own answer turns), not one answer")
    ap.add_argument("--dump_em", default="", help="a directory: write the EM batch (own thoughts, weights, targets) at --dump_em_steps")
    ap.add_argument("--dump_em_steps", default="1,25,50,100,150")
    ap.add_argument("--em_k", type=int, default=4, help="own thoughts per history (EM); 1 trains only the belief line and the action")
    ap.add_argument("--objective", choices=("strings", "critic"), default="strings",
                    help="strings: the teacher's belief line and action are the targets; critic: the teacher scores the agent's own")
    ap.add_argument("--update", choices=("em", "grpo"), default="em",
                    help="critic: em (the method: the E-step's weights softmax(R / alpha) on the accepted own turns) or grpo (GRPO-posterior, "
                         "the ablation: every own turn with its group-relative advantage, the same histories, samples and scores)")
    ap.add_argument("--attn", choices=("sdpa", "flash_attention_3", "flash_attention_2", "auto"), default="sdpa",
                    help="the learner's attention: sdpa, or a flash attention (PL_ATTN overrides it, e.g. "
                         "flash_attention_2 on A100), checked once per rank against sdpa at the first score (attn_check; sdpa if it fails)")
    ap.add_argument("--grpo_clip", type=float, default=20.0,
                    help="--update grpo: scores raised to at least the group's best minus C, advantages in [-C, C] nats (0: neither)")
    ap.add_argument("--grpo_eps", type=float, default=0.2,
                    help="--update grpo: PPO's clip epsilon, a block's gradient stops beyond log(1 -/+ eps) per token from the phase prior (0: no stop)")
    ap.add_argument("--grpo_norm", type=int, choices=(0, 1), default=1,
                    help="--update grpo: 1 = every batch's weights get --update em's total absolute size (one factor per batch); 0 = raw")
    ap.add_argument("--alpha", type=float, default=1.0, help="critic: the likelihood is exp(R / alpha)")
    ap.add_argument("--em_m", type=int, default=1, help="critic: own belief lines and actions after each own thought")
    ap.add_argument("--em_j", type=int, default=0, help="critic: draws from q per history for the M-step (0: every sample, weighted)")
    ap.add_argument("--em_more", type=int, default=0,
                    help="critic: up to R more rounds of K M own turns at a history with no turn the teacher accepts; a history still "
                         "without one is left out of the M-step (0: the K M turns only, every history fitted)")
    ap.add_argument("--temper", type=float, default=0.0,
                    help="critic: > 0 is the KL bound EPS of the tempered E-step step (lam_h per history, the settled step where the bound holds "
                         "at lam = 1); replaces --em_more. 0: lam = 1 everywhere (the settled E-step)")
    ap.add_argument("--credit", choices=("own", "share"), default="own",
                    help="critic: own (a decision's weight on its own thought) or share (each accepted own decision's weight over the K own "
                         "thoughts in proportion to pi_k(d | h, z_j), the balance-heuristic estimate)")
    ap.add_argument("--share_max", type=int, default=8, help="--credit share: at most this many distinct accepted own decisions per history, the highest R")
    ap.add_argument("--act_score", type=int, choices=(0, 1), default=1,
                    help="critic: 1 = the turn score R has the action's log pi* (the method); 0 = R is the belief line's part only (the "
                         "ablation without the action reward; needs a scored line, and --accept_rule 0 to close the acceptance rule too)")
    ap.add_argument("--accept_rule", type=int, choices=(0, 1), default=1,
                    help="critic: 1 = the teacher's acceptance of the action counts (--update grpo: a rejected turn never gets a positive "
                         "advantage; --length_reward: a correct turn has an accepted action); 0 = it does not count anywhere (the ablation "
                         "without the action reward: the acceptance rule would give an action reward by another way); needs "
                         "--accept_only 0, --set_keep 0, --em_more 0")
    ap.add_argument("--accept", type=float, default=20.0, help="critic: the teacher accepts an action with at least 1/A of its best action's probability")
    ap.add_argument("--leave_out", type=int, default=1,
                    help="critic, with --em_more: 1 = a history whose own turns never meet the rule after the more rounds is left out of the "
                         "M-step; 0 = it is fitted with its accepted turns (no history left out); with --accept_only a "
                         "history with no positive target (no_target) is left out all the same")
    ap.add_argument("--accept_only", type=int, default=0,
                    help="critic: 1 = the E-step's weights go only to the own turns whose action the teacher accepts (--accept); a turn with a "
                         "rejected action gets weight 0. A history with no accepted turn, or whose accepted lines all have none of the "
                         "posterior (after the --em_more rounds), gets no weight: the E-step leaves it out of the M-step, whatever "
                         "--leave_out")
    ap.add_argument("--m_step", choices=("joint", "blocks"), default="joint",
                    help="critic: the M-step's step: joint (one clipped step on the whole own sample) or blocks (one clipped step on the thought "
                         "tokens, then one on the decision tokens, the belief line and the action: each block of KL(q || pi) at its own pace)")
    ap.add_argument("--readout", type=float, default=0.0,
                    help="critic: > 0 turns the belief readout on (--m_step blocks, --line_score exact): one linear head on detached final hidden "
                         "states, the readiness of the turn (target U(b*)) and the listwise preference among a history's action candidates (target "
                         "pi* on them), fitted on a buffer of recent updates; no gradient into the LoRA")
    ap.add_argument("--readout_lr", type=float, default=1e-3, help="the learning rate of the readout's own parameters")
    ap.add_argument("--readout_inner", type=int, default=20, help="the readout's own Adam steps per update on the buffer of recent states")
    ap.add_argument("--readout_buffer", type=int, default=4000, help="reading points (per readout) the buffer keeps, the most recent updates")
    ap.add_argument("--readout_batch", type=int, default=512, help="reading points per mini-batch of the readout's own steps")
    ap.add_argument("--readout_eval", type=int, default=1, help="1: the readouts' AUROC and errors on the evaluation turns at every evaluation")
    ap.add_argument("--init_readout", default="", help="a saved readout (*-readout-STEP.pt) to start from, with --init_lora")
    ap.add_argument("--act_pick", type=int, default=1,
                    help="with --readout: M > 1 samples M action lines per turn and takes the one the action readout prefers (its softmax over the "
                         "candidates is the head's distribution; evaluation rows log the teacher's mass on the pick against the first sample)")
    ap.add_argument("--wandb_project", default="", help="W&B project (or WANDB_PROJECT); empty: no W&B. WANDB_MODE=offline writes to disk")
    ap.add_argument("--wandb_entity", default="", help="W&B entity (or WANDB_ENTITY)")
    ap.add_argument("--wandb_name", default="", help="W&B run name (default: the run's tag)")
    ap.add_argument("--wandb_group", default="", help="W&B group (or WANDB_GROUP)")
    ap.add_argument("--em_more_joint", type=float, default=0.0,
                    help="critic: with --em_more, a history is covered only when some own turn has an accepted action AND a belief line within this "
                         "many nats of the exact line's score (the E-step's mode, exact line + right action, must be among the samples; else more "
                         "own turns, then the history is left out). 0: the action alone decides, as before. Fixes the least-bad attractor "
                         "('exact line + premature answer' takes the weight when no sample has both) without changing the target")
    ap.add_argument("--dec_term", choices=["term", "own"], default="term",
                    help="critic M-step decision target (fix A; METHOD: own): term = belief line TERM action TERM; own = the "
                         "turn as the model wrote it: no TERM after the action, the TERM between the lines not counted (also in the --kl_stop probe)")
    ap.add_argument("--neg_thought", type=float, default=0.0,
                    help="critic, --m_step blocks, --accept_only 1 (push down the thought): L > 0 = at a history with no own "
                         "turn the teacher accepts, each distinct own thought that led to a rejected answer enters the thought block with weight "
                         "-L / (their number); the trust region is --kl_stop. 0: off (METHOD)")
    ap.add_argument("--neg_answer", type=float, default=0.0,
                    help="critic, --m_step blocks, --accept_only 1: W > 0 = at a history with no own turn the teacher "
                         "accepts (now left out), push down each own answer (all rejected there) with total weight W per history: the bounded "
                         "unlikelihood -log(1 - p(answer | own thought, own belief line)), answer tokens only, its factor p / (1 - p) capped at 1; "
                         "the thought and the belief line get no term. --set_keep keeps these histories. 0: off (METHOD)")
    ap.add_argument("--think", choices=("sum", "mean"), default="sum",
                    help="critic: the thought term of the M-step, the summed log-likelihood of the thought tokens or their mean per token")
    ap.add_argument("--line_score", choices=("reader", "exact", "decision", "none"), default="reader",
                    help="critic: the belief line in the turn score: read by the prose reader; exact (read in the environment's belief "
                         "language only, else the minimum score; the prompt states the two-line protocol and the language); decision (the "
                         "same language, scored by the act it implies, log pi*(a_b), plus --line_mix times the exact score); none (not scored)")
    ap.add_argument("--line_mix", type=float, default=0.1,
                    help="--line_score decision: the share of the line's log score in R, which ranks the lines that imply the same act")
    ap.add_argument("--line_floor", type=float, default=200.0,
                    help="critic: the floor m of the belief line's log score, S = -b*(B) log|B| - m b*(outside B); the line's optimum is the "
                         "posterior's support for every m, m sets the cost of a wrong exclusion (m per unit of excluded mass)")
    ap.add_argument("--line_wide", action="store_true",
                    help="critic: the line's log score also costs --line_floor times the share of the line's options that "
                         "the teacher's exact line excludes, so a line too wide pays as a line too narrow (line_wide)")
    ap.add_argument("--carry", action="store_true", help="the agent's own belief line of its last turn enters its next prompt")
    ap.add_argument("--no_belief_line", type=int, choices=(0, 1), default=0,
                    help="1 (24 Nov 2026, the user): no belief line at all: the action-only rules (plearn/tasks/domain.py "
                         "action_only, the one-line turn of the tests), no carried line (--carry has no effect), and each own turn is the thought, </think> and the action line "
                         "(M actions after each thought). Needs --line_score none (no belief reward). 0: no change")
    ap.add_argument("--fl_view_value", type=int, default=0,
                    help="1: FrozenLake's teacher values a move by the view it opens, scored as an answer, as before the fix (with the goal "
                         "known, a walk to fog is worth as much as reaching the goal: the module text); 0 (the fix): by the return it leads to")
    ap.add_argument("--sk_guess_value", type=int, default=0,
                    help="1: Sokoban's teacher values heading for a target as a guess among the candidates, as before the fix (pushes "
                         "undervalued: the module text); 0 (the fix): a move by the return it leads to, an exact expectimax to the game's end")
    ap.add_argument("--plan_past_limit", type=int, default=0,
                    help="1: the MiniGrid and MineSweeper lookaheads also value plans that end after the game's action limit, as before "
                         "the fix (the module text); 0 (the fix): a plan must end within the limit, with a turn left to answer")
    ap.add_argument("--proposal", choices=("base", "policy"), default="base",
                    help="critic: the prior that gives the samples, the base model (q ~ pi_0 exp(R / alpha)) or the current model (--phase 1)")
    ap.add_argument("--phase", type=int, default=0,
                    help="critic: policy mirror descent, the prior of phase n is the model after n * PHASE updates, frozen for the phase "
                         "(0: the base model throughout; 1: the current model, a target that moves every update)")
    ap.add_argument("--phase_set", type=int, default=0,
                    help="--role actor with --phase: one sample set per phase. At the phase start the actor plays episodes for this many "
                         "histories, runs the E-step at all of them, and writes the phase's --phase batches as draws of --B histories from "
                         "the set (passes without replacement). The target q_n of a phase is fixed, so its samples stay valid for the "
                         "whole phase (0: new episodes and samples for every update)")
    ap.add_argument("--colocate", type=int, default=0,
                    help="--role actor/learner with --phase_set, on the same GPUs: the actor's server has all the GPUs while it makes a phase's "
                         "sample set and evaluations, then it stops and the learner moves its model from CPU memory to the GPUs for the "
                         "phase's updates, then back (no GPU has a fixed role)")
    ap.add_argument("--eval_domain_args", default="", help="JSON added to --domain_args for the evaluation domain only, for example {\"bench\": 1}")
    ap.add_argument("--eval_tasks", default="", help="registered tasks and groups (plearn; e.g. train_suite) that every evaluation "
                                                       "also plays through the actor's server, in the agent's turn format")
    ap.add_argument("--eval_n", type=int, default=0, help="--eval_tasks: episodes of each task (0: the task's num_episodes)")
    ap.add_argument("--watch", default="", help="JSON {task or group: n}: at every evaluation, the first n episodes of each task's own test "
                                                "list (the benchmarks; a sample by task), to watch the direction of training; never reported")
    ap.add_argument("--resume_step", type=int, default=0,
                    help="--phase_set: go on after update R, a phase end (same --out, --tag and --queue): the learner loads the adapter and "
                         "Adam's state saved at R, the logs go on, the actor loads its answer-accuracy table after its last set; when the "
                         "set of the phase from R + 1 is in the queue (gen_done_R and all its batches), the learner trains that phase first")
    ap.add_argument("--set_draw", choices=("passes", "is"), default="passes",
                    help="--phase_set: the histories of each batch. passes: draws without replacement, in passes over the set. is: "
                         "importance sampling (as prioritized replay): history h is drawn with p_h = (1 - P) / N + P sig_h / sum(sig), "
                         "sig_h its E-step signal (half the L1 distance between its weights and equal weights; 0: the target is the prior "
                         "there), by systematic sampling spread over the episodes, and its loss is multiplied by 1 / (N p_h), so the "
                         "phase's objective is the same in expectation; P = --set_prio")
    ap.add_argument("--set_prio", type=float, default=0.5,
                    help="--set_draw is: the share P of the draw probability that follows the signal; the rest is uniform, so no weight "
                         "exceeds 1 / (1 - P)")
    ap.add_argument("--set_keep", type=int, default=0,
                    help="critic with --phase_set (dynamic sampling, as DAPO's): the batches come only from the histories of the sample set "
                         "that give a gradient (the E-step fits them, an own turn takes an accepted action), the first L of them in the set's "
                         "random order; the others are skipped, not sampled again (0: every history of the set)")
    ap.add_argument("--kl_stop", type=float, default=0.0,
                    help="critic with --phase_set: the limit of a phase's step in nats per token. After each update the learner measures "
                         "KL(pi_n || pi_phi) on own turns of the phase prior pi_n (the probe, --kl_probe turns of the phase's first batch) and, "
                         "above the limit, skips the phase's other updates: the model after that update is the next prior (0: off)")
    ap.add_argument("--kl_probe", type=int, default=32, help="--kl_stop: the own turns of the probe, one per thought, spread over the histories")
    ap.add_argument("--kl_log", type=int, default=0, help="1: kl_prior is measured after every update as with --kl_stop, and only logged "
                                                           "(a phase is never stopped): the training dynamics of a run without the limit")
    ap.add_argument("--prefetch", type=int, default=1, help="warm-up: play the next batch of histories while this one trains")
    ap.add_argument("--eval_temp", type=float, default=0.0, help="ignored: the policy is the same sampler in training and at test")
    ap.add_argument("--eval_q", choices=("", "max", "sample"), default="",
                    help="critic check: the evaluation episodes are played by the E-step target q (base samples, teacher scores)")
    ap.add_argument("--max_think", type=int, default=2048); ap.add_argument("--max_anchor", type=int, default=160); ap.add_argument("--max_action", type=int, default=200)
    ap.add_argument("--eval_max_think", type=int, default=0,
                    help="the reasoning cap of the evaluation during training (the registered tasks, the reading metrics); 0: --max_think")
    ap.add_argument("--think_cost", default="",
                    help="critic: the price of a thought in nats per 1,000 tokens, a comma list by phase (the last value for later "
                         "phases): among the E-step samples that the teacher scores alike, the shorter thoughts get the weight")
    ap.add_argument("--length_budget", type=int, default=0,
                    help="critic, with --length_reward: 0 (default) the term over all own turns as Kimi k1.5; B > 0: a correct turn "
                         "whose thought fits B tokens gets no length term and one over B a penalty up to -0.5 (the term only "
                         "where the test would cut the thought; one rule for every model; B = the test budget, 8192)")
    ap.add_argument("--rho_exact", type=int, choices=(0, 1), default=0,
                    help="1 (needs --rho fidelity): the answer accuracy of the teacher is the exact posterior U(b) for the whole "
                         "run; the table measured from the agent's own answers keeps its prior (no update, no --rho_rb cell, no table "
                         "read on a resume). 0 (default): the table is updated as before")
    ap.add_argument("--teacher_row", type=int, choices=(0, 1), default=0,
                    help="critic, 1: at a history where no own turn takes an action the teacher accepts, the teacher's exact "
                         "belief line and its most likely action are added to the group as one more turn, after the history's first own "
                         "thought; the history is kept, and that turn trains its two lines only (the thought's weight is 0). Needs --update grpo, "
                         "--credit own, no --neg_answer / --neg_thought. 0 (default): "
                         "such a history is left out as before")
    ap.add_argument("--teacher_row_cap", type=float, default=0.0,
                    help="--teacher_row 1, --update grpo: the largest advantage of the teacher's turn in nats (0, the default: no limit, "
                         "so the own turns of the group lose in total what the teacher's turn gains)")
    ap.add_argument("--stall_cap", type=float, default=0.0,
                    help="critic, C > 0 (above log(--accept)): a look that is neither one of the "
                         "teacher's best looks nor one with real expected evidence (some outcome of it changes the posterior of the answer: "
                         "Planner look_info) gets a log pi* of at most the best action's minus C, and so does an action that is not legal; "
                         "such a step is then rejected (MineSweeper: a cursor move off the way to the best cell; MiniGrid: turning on the "
                         "spot). A value is only lowered. Log: stall_rejected. 0 (default): off, the scores as before")
    ap.add_argument("--stall_play", type=int, choices=(0, 1), default=0,
                    help="1 (needs --stall_cap): the teacher's own moves (plan tgt: its moves in the training games, "
                         "--beta_expert) come from the capped scores too: a look the cap lowered keeps exp(new - old) of its probability "
                         "(capped_tgt). At tau 0.1 the teacher otherwise draws a move the cap rejects at about a third of its steps in "
                         "MineSweeper, MiniGrid, FrozenLake and Sokoban (offline check). 0 (default): its moves as before")
    ap.add_argument("--length_reward", default="",
                    help="critic: C, a comma list by phase like --think_cost: the E-step's R of each own turn gets C times its length term "
                         "(length_reward, Kimi k1.5's form over the history's own turns: the shorter correct thought gets more, a wrong "
                         "turn never gets a positive term and a long one a negative term); empty or 0: no term")
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top_p", type=float, default=0.95); ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--line_target", type=float, default=-1.0,
                    help="critic, --m_step blocks: a belief line is a positive target only when the teacher verifies it, that is when its score is "
                         "within this many nats of the teacher's best line (0.1: the exact line). A turn with an accepted action and a line that "
                         "is not verified trains its ACTION only: its belief line is not counted and its thought gets no weight (target_item). "
                         "-1: off, every fitted turn trains its thought and both lines. Why: where the model seldom writes the exact "
                         "line, the fully open line 'Possible: any' is its best sample (a wrong line pays the floor, an open line a few nats), so "
                         "the M-step taught the open line: 9% of the base model's turns, 38% of the E-step weight, 28% of the turns after 7 updates")
    ap.add_argument("--score_presence", type=int, default=0,
                    help="1: the learner scores the own turns under the sampler's presence penalty too (PL_PRESENCE_PENALTY; the logit of every "
                         "token its request has generated before, lowered by the penalty before the temperature, as vLLM does). 0: the score has "
                         "no penalty. With a penalty in the sampler and none in the score, the samples do not come from the scored policy: the "
                         "likelihood of own samples then has a gradient even at equal weights, toward the tokens the penalty favours")
    ap.add_argument("--score_chunk", type=int, default=8); ap.add_argument("--think_chunk", type=int, default=2)
    ap.add_argument("--policy_metrics", type=int, default=1, help="0: skip action_match/kl_act (they score every legal string)")
    ap.add_argument("--held_n", type=int, default=60); ap.add_argument("--eval_every", type=int, default=10); ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="runs/evidence")
    args = ap.parse_args()
    assert args.temper == 0 or args.em_more == 0, "--temper replaces --em_more"
    assert args.credit == "own" or args.objective == "critic", "--credit share: the critic objective"
    train(args)
