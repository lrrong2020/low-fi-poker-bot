<a name="zh-cn"></a>

# 单挑德州扑克机器人

[简体中文](#zh-cn) · [English](#english)

这是一个可在本机训练、评测和对局的命令行扑克机器人。它用 K-means 按摊牌胜率分桶，再用表格型外部抽样 MCCFR 学习下注策略。项目自带训练好的 100 BB 模型，解压后即可对局。

## 思路来源与致谢

本项目采用 K-means 缩小扑克抽象、再做约半小时自我对弈的思路，来自 **Noam Brown** 在红杉资本《Training Data》No Limit Finals 现场访谈中的建议。Alfred Lin 问，如果只有一小时，他会如何构建扑克机器人。Brown 在回顾 CFR 自我对弈后，提出用 K-means 把抽象压到很小，为训练留出约 30 分钟。他随即补充，针对现场多人赛的计分方式，他自己更倾向于观察并利用对手的弱点。[官方逐字稿](https://sequoiacap.com/podcast/openai%E2%80%99s-noam-brown-why-game-theory-didn%E2%80%99t-lead-to-general-intelligence-%28live-at-no-limit-finals%29) · [访谈视频，约 2:59](https://www.youtube.com/watch?v=tRaFbDFBl_A&t=179s)

我们选取了他谈到的短时自我对弈方案，具体实现为 100 BB 单挑、按摊牌胜率在每个翻牌后下注轮聚成 16 个桶，并使用表格型外部抽样 MCCFR。分桶参数、换池机制以及下文的评测结果均来自本项目。

## 快速开始

在 macOS 或 Linux 上准备 Python 3.10 或更高版本，以及支持 C99 的 Clang 或 GCC。在项目目录运行以下命令。首次计算牌力时，程序会自动编译随附的七张牌评估器，缓存到 `__pycache__`。无需安装第三方 Python 包。

```sh
python main.py play                            # 与自带的新版模型对局
python main.py eval --hands 10000              # 对每个基线对手各评测一万手
python -m unittest discover -v                 # 运行测试
```

用浏览器打开本地的 [strategy.html](strategy.html)，可以离线查看策略。选择当前行动者的起手牌或翻牌后胜率桶，即可看到该玩家持有这手牌时各动作的频率。点击一个动作，会切换到下一位行动者的决策。训练后可运行 `python export_strategy.py` 更新 HTML。

## 训练和保存模型

`python main.py train --seconds 1800 --seed 42` 会在默认目录 `model-v2` 续训并更新其中的 checkpoint。若想保留随附模型、从零开始训练，请给新实验使用空目录。

```sh
python main.py train --model-dir my-run --seconds 1800 --seed 42
python main.py eval --model-dir my-run --hands 10000 --output my-run/evaluation.json
python main.py train --model-dir my-run --resume --seconds 60
python main.py play --model-dir my-run
python export_strategy.py --model-dir my-run --output my-strategy.html
```

每次调用的计时预算包含本次预处理、换池和自我对弈。程序默认每 120 秒及退出时原子保存。在自我对弈中按 Ctrl-C，会完成当前小批次后保存。续训会恢复累计遗憾、策略和、下一次更新的玩家、当前牌池、正在编码的替换牌池和随机数状态。末次写盘可能使实际耗时略超设定值。运行 `python main.py train --help` 可查看牌池大小、采样次数和保存间隔等选项。

训练、对局、评测和导出时都会校验下注树、筹码配置、牌力特征版本、采样次数及聚类中心。新版模型的分桶参数保存在 [model-v2/abstraction.json](model-v2/abstraction.json)。随附的旧版模型按文件哈希及原下注树指纹识别，传入 `--model-dir model` 可用旧分桶算法对局、评测或导出。继续训练请使用新版 checkpoint，或指定空目录。

## 游戏规则与求解方法

- **下注规则。** 每手重置为每人 200 筹码（100 BB），小盲 1、大盲 2，无抽水。按钮位在翻牌前先行动，翻牌后由大盲先行动。小盲平跟后，大盲仍可过牌或加注。合法动作有弃牌、过牌或跟注、半池、满池和全压。面对下注时，半池和满池都以跟注后的底池为基数，半筹码向上取整，并满足最小加注额。相同下注额只保留一个动作，全压允许不足最小加注额。每街双方合计最多主动下注或加注两次，盲注不计。全压获跟注后发完公共牌，未匹配筹码退回，平局平分底池。
- **牌力抽象。** 翻牌前使用 169 种标准起手牌。翻牌、转牌和河牌分别拟合 16 个一维 K-means 桶。特征只读取本方底牌及已公开的牌。新版在翻牌和转牌对均匀随机的合法对手底牌做 256 次确定性模拟，并将等价花色规范化。河牌精确枚举全部 990 种合法对手底牌。平局计半胜。
- **发牌池。** 初始默认采样 5,000 手牌并编码双方各街牌桶，预处理接近五分钟上限时提前停止采样。训练从当前牌池有放回抽样，默认每 20,000 次遍历更换整池。新牌用同一组中心编码，编码完毕后才切换，递归遍历直接查桶。新实验可用 `--refresh-every` 调整间隔。
- **MCCFR。** 两个座位轮流更新。每次遍历抽取一次发牌结果，按策略抽取对手动作，并枚举更新玩家所有合法动作。以动作收益减去策略加权平均收益来累计遗憾，再按正遗憾匹配。没有正遗憾时均匀选取。每次访问对手节点，将其完整动作分布累加一次。在这套双人外部抽样中，按策略访问节点已经包含对手自己的到达概率。稀疏表的键由公开下注历史、位置、街和当前牌桶决定，先前街的牌桶会被遗忘。
- **实战策略。** 对局和评测从累计平均策略中抽样。尚无平均策略的节点使用胜率与底池赔率决策。胜率低于跟注所需赔率时弃牌，达到 85% 时优先满池，达到 65% 时优先半池，其余过牌或跟注。随机基线从不同合法动作中均匀抽样。胜率基线固定使用原版每街 64 次模拟和同一套决策阈值，便于跨版本比较。

## 验证与实测结果

测试覆盖整棵下注树，并用独立的最佳五张牌实现核对七张牌评估器。将同一求解器用于 Kuhn 扑克，100,000 次遍历后平均策略的先手收益为 −0.05542 筹码/手，理论均衡值为 −1/18，按精确最佳响应计算的可剥削度为 0.00645 筹码/手。

在翻牌和转牌各 512 个独立局面上（seed 921848），按 64、128、256 次模拟估计胜率，相对穷举结果的均方根误差分别为翻牌 5.61、3.95、2.73 个百分点，转牌 5.44、3.83、2.71 个百分点。即使使用 256 次模拟，分桶与精确值一致的比例也只有翻牌 59.2%、转牌 65.6%。例如持 `Js 9d`，河牌面为 `9s 5s 6c 5c Kh` 时，旧版估计为 53.91%，精确胜率为 70.10%。

新版用 seed 42 训练 **1,800.09 秒**，完成 **8,266,904 次遍历**。初始牌池加 413 次换池，共启用 2,070,000 手牌样本。初始拟合与编码用时 0.35 秒，换池 147.44 秒，自我对弈 1,651.06 秒。相同预算下，旧版在固定的 5,000 手牌上遍历了 9,709,304 次。

下表使用训练之外的新发牌，每个对手轮换座位，两版模型使用相同的发牌序列。收益单位为每 100 手赢得的大盲数（bb/100），方括号是按座位分别估计方差、再等权合并得到的近似 95% 置信区间，反映给定 checkpoint 的发牌与对局抽样波动。

| 评测 seed | 每个对手的手数 | 对手 | 旧版 bb/100 [95% 区间] | 新版 bb/100 [95% 区间] |
|---|---:|---|---:|---:|
| 730031 | 10,000 | 随机合法动作 | +200.17 [+140.09, +260.25] | +252.23 [+199.78, +304.68] |
| 730031 | 10,000 | 胜率基线 | +5.32 [−41.83, +52.46] | +106.03 [+64.62, +147.44] |
| 20261126 | 100,000 | 随机合法动作 | +202.53 [+183.37, +221.69] | +234.52 [+218.03, +251.02] |
| 20261126 | 100,000 | 胜率基线 | −0.33 [−15.53, +14.87] | +101.83 [+88.66, +115.00] |
| 7261409 | 100,000 | 随机合法动作 | +205.07 [+185.97, +224.16] | +246.54 [+229.96, +263.12] |
| 7261409 | 100,000 | 胜率基线 | +5.66 [−9.48, +20.80] | +111.56 [+98.29, +124.83] |

复现十万手评测可运行 `python main.py eval --hands 100000 --seed 20261126`。加上 `--first-pool` 会从最初用于训练的牌池有放回抽样，置信区间也相应地以该固定牌池为条件。面对胜率基线时，旧版在首池的收益为 +122.26 bb/100，新版为 +100.43 bb/100。首池收益减去新牌收益，差距由旧版的 +122.59 缩小至新版的 −1.40 bb/100。新版在两组十万手新牌上均击败了这两个对手。由于只训练了一次新版模型，各项改动的单独贡献及不同训练 seed 的差异仍待检验。[完整评测数据](model-v2/comparison.json)

---

<a name="english"></a>

[简体中文](#zh-cn) · [English](#english)

# Local heads-up Hold’em bot

A CLI-only poker bot with 169 preflop classes, per-street equity K-means, and tabular external-sampling MCCFR. Requires Python 3.10+ and a C99 compiler (Clang or GCC) on macOS or Linux. The first card evaluation compiles the bundled seven-card evaluator into `__pycache__`. There are no pip dependencies.

## Inspiration and credit

**Noam Brown** inspired this project's combination of K-means abstraction and a short self-play run. At Sequoia Capital’s No Limit Finals interview for *Training Data*, Alfred Lin asked how he would build a poker bot in an hour. After describing CFR self-play, Brown suggested using “k-means clustering to reduce the abstraction size” and spending about 30 minutes on self-play. For the multiplayer tournament itself, he said he would favor exploiting observed weaknesses in the field. [Official transcript](https://sequoiacap.com/podcast/openai%E2%80%99s-noam-brown-why-game-theory-didn%E2%80%99t-lead-to-general-intelligence-%28live-at-no-limit-finals%29) · [Interview at 2:59](https://www.youtube.com/watch?v=tRaFbDFBl_A&t=179s)

We implemented the short self-play idea for 100 BB heads-up Hold’em. The scalar showdown-equity feature, 16 buckets per postflop street, tabular external-sampling MCCFR, rotating deal pools and reported benchmarks are specific to this project.

Open [strategy.html](strategy.html) in a browser to explore the checkpoint offline. Select a preflop hand or a postflop equity bucket to see its action frequencies. Click an action to inspect the next player's decision in that betting line. After training, refresh the snapshot with `python export_strategy.py`. Use `--model-dir my-run --output my-strategy.html` to export another run.

```sh
python main.py play
python main.py eval --hands 10000
python main.py train --seconds 1800 --seed 42
python -m unittest discover -v
```

The default checkpoint is `model-v2/checkpoint.pkl`. Training continues its solver, alternating player, current deal pool, partially encoded replacement pool and both random streams. The original 100 BB checkpoint is available for play, evaluation and export with `--model-dir model`. To reproduce training from scratch, use an empty directory. Each invocation gets its own wall-clock budget, including preprocessing on a fresh run. The final checkpoint write can add a fraction of a second.

```sh
python main.py train --model-dir my-run --seconds 1800 --seed 42
python main.py eval --model-dir my-run --hands 10000 --output my-run/evaluation.json
python main.py train --model-dir my-run --resume --seconds 60
python main.py play --model-dir my-run
```

For the 100,000-hand fresh-deal benchmark, run `python main.py eval --hands 100000 --seed 20261126`. Add `--model-dir model` for the original checkpoint. To diagnose memorization, add `--first-pool` to either command. This resamples the initial training pool with replacement. Its confidence interval is conditional on that fixed pool.

Use `train --help` for preprocessing counts and checkpoint frequency. Checkpoints are saved atomically every 120 seconds and on exit. Ctrl-C during self-play finishes the current batch and saves. Each pickle stores the regret and strategy sums, card model, encoded chance pool, and sampler state. The three fitted centroid vectors, rollout count and card-feature version are also readable in `model-v2/abstraction.json`.

Training, play, evaluation and export all verify the saved game configuration and public-tree fingerprint. The fingerprint covers node IDs, legal action ordering, sizing and leaf payoffs. The loader also verifies the card-feature version, rollout count and centroid digest. The original shipped format-1 checkpoint is recognized by its exact file hash and original game fingerprint and uses its original bucketing code. Format-1 files lack configuration metadata. New training requires format 2 or an empty directory.

## Game and training

- Each hand resets to 200 chips per seat. Blinds are 1/2 chips (0.5/1 BB). The button acts first preflop, the big blind acts first postflop, and the big blind retains its option after a limp. There is no rake.
- Fold is available when facing a bet. A single check/call action matches the amount owed. Half-pot and pot raises first call, then add their fraction of the resulting pot, rounding half chips upward and enforcing the minimum full raise. Equal sizes collapse to one action. Short raises are legal only all-in. Two voluntary bets/raises close raising for the street. A called all-in runs out the board, a fold refunds uncalled chips, and ties split the pot.
- A fresh run samples 5,000 complete deals. For each player, the flop and turn use 256 deterministic Monte Carlo trials against a uniform legal opponent. Canonical suit labels make equivalent suit permutations share the estimate. The river enumerates all 990 legal opposing hands exactly. Features use only the player's hole cards and visible board. Ties contribute half. Separate 1-D Lloyd K-means fits each street's 16 centers. Preprocessing stops collecting deals early if it approaches its five-minute cap.
- Training draws with replacement from the current pool. After every four pool-lengths of traversals (20,000 by default), it draws a new pool from an independent random stream and encodes it with the original fitted centers. The solver switches pools between traversals when the entire replacement is ready. Set the interval for a new run with `--refresh-every`. A traversal samples one deal and the opponent’s actions, enumerates every legal updating-player action, and adds its payoff difference from the policy-weighted mean to cumulative regret. Positive regrets determine the next policy, with uniform play when none are positive. At each visited opponent node, the full current action distribution is added once to the strategy sum. On-policy sampling supplies the opponent’s reach weighting. This is the two-player averaging rule used in [OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/external_sampling_mccfr.py).
- A sparse row combines a public betting-tree node ID (street, position, full action history) with the actor's current bucket. Previous card buckets are forgotten. Sampling and bucket approximation still introduce error, and the abstraction has imperfect recall. Benchmark wins alone do not establish equilibrium play.
- Evaluation and interactive play sample the normalized accumulated policy. With no accumulated action mass, the bot folds below pot odds, prefers a pot-sized raise at equity ≥85%, prefers half-pot at ≥65%, and otherwise checks/calls. The equity benchmark uses the same thresholds and is pinned to the original suit-dependent 64-rollout estimator on every street, so changing the bot’s features preserves the opponent. Random-legal samples uniformly from the distinct legal actions.

## Verification and benchmarks

The tests cover the entire betting tree, compare the native hand scorer to an independent best-five reference, and reproduce the precomputed buckets from visible cards. After 100,000 traversals of Kuhn poker, the same solver's average-policy value was −0.05542 chips/hand (equilibrium −1/18), with exact best-response exploitability of 0.00645 chips/hand.

On seed 921848, we compared suit-canonical estimates against exhaustive showdown equity for 512 independent flop positions and 512 turn positions. With 64, 128 and 256 rollouts, flop RMSE was 5.61, 3.95 and 2.73 percentage points. Turn RMSE was 5.44, 3.83 and 2.71 points. Even at 256 rollouts, the estimated and exact equity mapped to the same fitted bucket on only 59.2% of flops and 65.6% of turns. For Js 9d on 9s 5s 6c 5c Kh, version 1 estimates 53.91% equity and exact river enumeration gives 70.10%.

For each benchmark below, both models face the same fresh deals, alternate seats, and sample their own accumulated average strategy. Intervals are 95% normal approximations using within-seat sample variances and equal seat weights. They describe hand-sampling uncertainty for the two saved policies. The original 10,000-hand results remain in [the baseline evaluation](model/evaluation.json).

The seed-42 version-2 run finished in 1,800.09 seconds including the final save. It made 8,266,904 traversals and activated 2,070,000 deal samples across the initial pool and 413 replacements. Initial fitting and encoding took 0.35 seconds, pool refreshes took 147.44 seconds, and self-play took 1,651.06 seconds. The original model made 9,709,304 traversals on one pool of 5,000 deals in the same time budget.

| Fresh-deal seed | Hands per opponent | Opponent | Original bb/100 [95% CI] | Version 2 bb/100 [95% CI] |
|---|---:|---|---:|---:|
| 730031 | 10,000 | Random legal | +200.17 [+140.09, +260.25] | +252.23 [+199.78, +304.68] |
| 730031 | 10,000 | Equity | +5.32 [−41.83, +52.46] | +106.03 [+64.62, +147.44] |
| 20261126 | 100,000 | Random legal | +202.53 [+183.37, +221.69] | +234.52 [+218.03, +251.02] |
| 20261126 | 100,000 | Equity | −0.33 [−15.53, +14.87] | +101.83 [+88.66, +115.00] |
| 7261409 | 100,000 | Random legal | +205.07 [+185.97, +224.16] | +246.54 [+229.96, +263.12] |
| 7261409 | 100,000 | Equity | +5.66 [−9.48, +20.80] | +111.56 [+98.29, +124.83] |

For the first-pool diagnostic against the equity opponent (seed 20261126, 100,000 hands), version 1 earned +122.26 [+107.03, +137.49] bb/100 and version 2 earned +100.43 [+87.03, +113.83]. The pool-minus-fresh gap was +122.59 bb/100 in version 1 and −1.40 bb/100 in version 2. The fresh-deal gain held up on the second seed for this fixed opponent. The combined changes were trained once, so the experiment cannot separate the effect of refreshing the pool from the improved equity features. One version-2 decision against equity on the primary fresh benchmark used the fallback. The full [comparison](model-v2/comparison.json) includes exact policy-hit counts, benchmark seeds, and checkpoint hashes.
