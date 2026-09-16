# Phase 3B expanded strict retrospective replay report

Date: 2026-09-16. Outcome: **largest defensible audited window is 195 games at 14 origins in the 2025 season**: continuous Tuesday origins for Weeks 6–18 plus a separately frozen Super Bowl origin (Week 22). Weeks 19–21 fail closed because no exact pre-Tuesday archive schedule contains the resolved matchups. This is retrospective historical-source replay development evidence, not prospective evidence or proof of historical forecast existence.

## Review publication

The approved initial implementation is published as draft [PR #23](https://github.com/EMERKOR/BK_v2/pull/23), head `5ade4f640b49f297f8615dd56605e937d1285481`. Its GitHub Actions `test` job passed. The PR contains the original immutable 30-row table and 103-test result. Expansion work is isolated on local branch `review/phase3b-expanded-replay`; it does not mutate PR #23.

## Search and selection

Sixteen exact dated releases from October 2, 2025 through January 29, 2026 were inventoried. All 32 requested PBP/schedule assets were downloaded by their first-party URLs, checked for exact size, GitHub provider digest and SHA-256, and rechecked against public, non-draft post-download release metadata. The conservative availability bound is max(release published time, individual asset updated time). Original bytes, embedded attributes, full schema identities and retrieval metadata are preserved.

The frozen selection rule is the **first jointly complete exact PBP plus schedule/result artifact for each competition week**. It requires identical PBP/schedule/terminal game sets and complete score evidence. Later cumulative files never replace a selected week's EPA. The forecast schedule is separately the latest exact games asset strictly before the Tuesday 16:00 UTC cutoff, with concrete teams/game IDs, missing outcomes and all kickoffs after cutoff. Completed observation evidence and pre-origin schedules are separate tables and version IDs.

Weeks 1–4 first become jointly source-proven in the October 2 release and enter the Week 6 fit together. Ordinarily the newest completed week is delayed one target origin. The December 11 PBP asset is provider-hash-valid but is actually a 54,894-byte GitHub HTML document, and no December 18 archive release survived. It is classified **invalid exact bytes**, never repaired from a current file. Consequently Weeks 14–15 remain unavailable longer; Weeks 14–16 first enter together from December 25 at the Week 18 origin. No fake observations are inserted; competition-week transitions span every gap.

## Replay inventory and diagnostics

| Target | Forecast cutoff | Rows | Eligible competition weeks | Unavailable prior weeks | Selected scale/df | Prefix obs | 90% coverage | PIT outside 1–99% | Innovation mean | Innovation second moment | Mean offense SD | Intercept SD |
|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | 2025-10-07T16:00:00+00:00 | 15 | 1,2,3,4 | 5 | 1.6/5.0 | 7728 | 0.9655 | 0.0065 | 0.0027 | 0.4253 | 0.0953 | 0.0183 |
| 7 | 2025-10-14T16:00:00+00:00 | 15 | 1,2,3,4,5 | 6 | 1.6/5.0 | 9433 | 0.9651 | 0.0071 | 0.0085 | 0.4335 | 0.0881 | 0.0167 |
| 8 | 2025-10-21T16:00:00+00:00 | 13 | 1,2,3,4,5,6 | 7 | 1.6/5.0 | 11201 | 0.9657 | 0.0071 | 0.0067 | 0.4288 | 0.0827 | 0.0153 |
| 9 | 2025-10-28T16:00:00+00:00 | 14 | 1,2,3,4,5,6,7 | 8 | 1.6/5.0 | 13055 | 0.9652 | 0.0073 | 0.0027 | 0.4351 | 0.0786 | 0.0142 |
| 10 | 2025-11-04T16:00:00+00:00 | 14 | 1,2,3,4,5,6,7,8 | 9 | 1.6/5.0 | 14597 | 0.9645 | 0.0073 | 0.0042 | 0.4366 | 0.0762 | 0.0135 |
| 11 | 2025-11-11T16:00:00+00:00 | 15 | 1,2,3,4,5,6,7,8,9 | 10 | 1.6/5.0 | 16308 | 0.9648 | 0.0072 | 0.0058 | 0.4353 | 0.0740 | 0.0127 |
| 12 | 2025-11-18T16:00:00+00:00 | 14 | 1,2,3,4,5,6,7,8,9,10 | 11 | 1.6/5.0 | 18030 | 0.9641 | 0.0075 | 0.0041 | 0.4377 | 0.0724 | 0.0121 |
| 13 | 2025-11-25T16:00:00+00:00 | 16 | 1,2,3,4,5,6,7,8,9,10,11 | 12 | 1.6/5.0 | 19861 | 0.9642 | 0.0075 | 0.0020 | 0.4374 | 0.0710 | 0.0115 |
| 14 | 2025-12-02T16:00:00+00:00 | 14 | 1,2,3,4,5,6,7,8,9,10,11,12 | 13 | 1.6/5.0 | 21597 | 0.9634 | 0.0077 | -0.0019 | 0.4408 | 0.0700 | 0.0111 |
| 15 | 2025-12-09T16:00:00+00:00 | 16 | 1,2,3,4,5,6,7,8,9,10,11,12,13 | 14 | 1.6/5.0 | 23548 | 0.9633 | 0.0076 | -0.0027 | 0.4407 | 0.0689 | 0.0106 |
| 16 | 2025-12-16T16:00:00+00:00 | 16 | 1,2,3,4,5,6,7,8,9,10,11,12,13 | 14,15 | 1.6/5.0 | 23548 | 0.9633 | 0.0076 | -0.0027 | 0.4407 | 0.0706 | 0.0106 |
| 17 | 2025-12-23T16:00:00+00:00 | 16 | 1,2,3,4,5,6,7,8,9,10,11,12,13 | 14,15,16 | 1.6/5.0 | 23548 | 0.9633 | 0.0076 | -0.0027 | 0.4407 | 0.0721 | 0.0106 |
| 18 | 2025-12-30T16:00:00+00:00 | 16 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 | 17 | 1.6/5.0 | 29140 | 0.9628 | 0.0072 | 0.0010 | 0.4411 | 0.0672 | 0.0095 |
| 22 | 2026-02-03T16:00:00+00:00 | 1 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 |  | 1.6/5.0 | 34490 | 0.9627 | 0.0074 | -0.0029 | 0.4435 | 0.0686 | 0.0088 |

All 14 origins select the already declared 1.6/df 5 candidate. This is a configuration-selection history, not a promotion. Nominal 90% prefix coverage ranges roughly 96.3%–96.6%, and innovation second moments roughly 0.425–0.441, indicating conservative dispersion on tuning prefixes. The state uncertainty contracts from mean offense/defense SD about 0.095 at Week 6 to 0.067 by Week 18; missing late-season evidence produces visible transition-driven uncertainty. The Week 22 extension is separately frozen after regular-run mechanics were known, uses the unchanged family/policy, and consumes no target outcomes.

These are eligible-prefix tuning and state-mechanics diagnostics only. They are not held-out game-level calibration, predictive validation, an untouched promotion gate, or a production Bayesian posterior. Target outcomes remain absent. The robust state filter remains an approximation. Neither 1.0 nor 1.38 is promoted; pooled residual SD is never substituted for Student-t observation scale.

## Revisions and schedule changes

One adjacent PBP revision was found: archive October 23 to October 30 changes Week 1 game `2025_01_CAR_JAX`, play 1282, from EPA -2.179037803784013 to -0.0. It is a lightning-suspension row with missing play type and is excluded by the pass/run observation policy. The selected October 2 Week 1 bytes remain unchanged regardless. No other adjacent shared EPA key changed in the audited valid sequence.

The December 25 Week 18 schedule still carries placeholder 13:00 Eastern times for ten later-flexed games; subsequent evidence changes those kickoff times. The December 30 forecast correctly preserves the exact schedule version actually available under this archive policy. Matchup teams/game IDs are unchanged, and structural state does not use the later kickoff update.

## 2024, 2023 and earlier

A systematic probe covered 48 Thursday archive tags across the 2024 and 2023 seasons; none exposes `games.rds`. Representative first-party release inventories (October 3, 2024 and October 5, 2023) contain dated cumulative PBP assets but no game/schedule/result asset under any schedule/game/result name. This agrees with the earlier exact 2024/2023 PBP samples and prior audit. Without corresponding exact pre-cutoff schedule/result versions, they cannot extend the strict table. Current schedules, Git client dates, kickoff/final status or later downloads were not substituted. Earlier periods remain excluded for the same or stronger evidence gaps.

## Source/version inventory

The two schema identities are stable across all valid assets: games `0d46bca4...d2ef4`; PBP `22397698...9e61f`. Embedded nflfastR versions progress through 5.1.0.9003, 5.1.0.9007, 5.1.0.9008 and 5.1.0.9009.

| Release | Artifact | Asset ID | Availability UTC | SHA-256 | Provider digest | Rows | nflfastR | Schema SHA-256 | Class |
|---|---|---:|---|---|---|---:|---|---|---|
| archive-2025-10-02 | games.rds | 299769091 | 2025-10-02T15:30:53Z | `97689d88414fed363c017c849dd8e43c3ffe9f3db1ce09ea4984f92e391b0a90` | `sha256:97689d88414fed363c017c849dd8e43c3ffe9f3db1ce09ea4984f92e391b0a90` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-10-02 | play_by_play_2025.rds | 299770475 | 2025-10-02T15:33:20Z | `0b04ffcee75f07c5a1cec292d7b485ec3222bbbcb65f7d5866fcd81b802b9d9b` | `sha256:0b04ffcee75f07c5a1cec292d7b485ec3222bbbcb65f7d5866fcd81b802b9d9b` | 11034 | 5.1.0.9003 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-10-09 | games.rds | 302373781 | 2025-10-09T15:44:46Z | `e4520f43c14c79657dff904a5f3a56e467fb3aea0de933db087da8de84f767b2` | `sha256:e4520f43c14c79657dff904a5f3a56e467fb3aea0de933db087da8de84f767b2` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-10-09 | play_by_play_2025.rds | 302374075 | 2025-10-09T15:45:49Z | `14a594b86b207508b89a42755f624c241e6ceef516d859a4b6bd2c4a5a7b472c` | `sha256:14a594b86b207508b89a42755f624c241e6ceef516d859a4b6bd2c4a5a7b472c` | 13488 | 5.1.0.9003 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-10-16 | games.rds | 305098532 | 2025-10-16T15:45:33Z | `a8420a6949b782c450e92ed57ce19bcdde751e9bec627f86c1a58e5051170083` | `sha256:a8420a6949b782c450e92ed57ce19bcdde751e9bec627f86c1a58e5051170083` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-10-16 | play_by_play_2025.rds | 305097630 | 2025-10-16T15:42:35Z | `d4c020717d85af1950259a7a7d79b5db9ebca8d70701c2fcc7a5da0d29ea9d15` | `sha256:d4c020717d85af1950259a7a7d79b5db9ebca8d70701c2fcc7a5da0d29ea9d15` | 16011 | 5.1.0.9003 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-10-23 | games.rds | 307812009 | 2025-10-23T15:32:34Z | `7e0cca2e97a6ca6a6de54f402905a251e084236f8e90a7a674d7f42196f856be` | `sha256:7e0cca2e97a6ca6a6de54f402905a251e084236f8e90a7a674d7f42196f856be` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-10-23 | play_by_play_2025.rds | 307813872 | 2025-10-23T15:36:48Z | `6d8296b5f9e02ac2a51b35a96ade7f214d3c4fc4483036820efa46e642fd800b` | `sha256:6d8296b5f9e02ac2a51b35a96ade7f214d3c4fc4483036820efa46e642fd800b` | 18625 | 5.1.0.9007 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-10-30 | games.rds | 310570721 | 2025-10-30T15:34:50Z | `ab96f6641c891d46a73cd33baeb05b9ec5cd4e1dec5b03dccee343b1769070e7` | `sha256:ab96f6641c891d46a73cd33baeb05b9ec5cd4e1dec5b03dccee343b1769070e7` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-10-30 | play_by_play_2025.rds | 310572709 | 2025-10-30T15:39:36Z | `09d4cbb60ab73ae5329ac5776b00c41dc3b603a21a1dd001f2b8ad43cd48f57c` | `sha256:09d4cbb60ab73ae5329ac5776b00c41dc3b603a21a1dd001f2b8ad43cd48f57c` | 20767 | 5.1.0.9007 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-11-06 | games.rds | 313354781 | 2025-11-06T15:34:54Z | `33e44a0c02934558ee79649b4de326a2b046169ba491a244da662f9f7eb6439a` | `sha256:33e44a0c02934558ee79649b4de326a2b046169ba491a244da662f9f7eb6439a` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-11-06 | play_by_play_2025.rds | 313356862 | 2025-11-06T15:40:00Z | `fcfd0ca94a4121a36eb2aa561f28f2d32fffd199c4844e0a3da4cde2932c9a53` | `sha256:fcfd0ca94a4121a36eb2aa561f28f2d32fffd199c4844e0a3da4cde2932c9a53` | 23152 | 5.1.0.9007 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-11-13 | games.rds | 316009177 | 2025-11-13T15:33:07Z | `73f6bce17fa6876ead1ee07522afdd4a3627fc218fc543b22de97ec97b335905` | `sha256:73f6bce17fa6876ead1ee07522afdd4a3627fc218fc543b22de97ec97b335905` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-11-13 | play_by_play_2025.rds | 316011371 | 2025-11-13T15:38:04Z | `bdb4fba0965dfacc093c4f71e12ebc2dc180baf63b96e5f8a889e3f844d5adec` | `sha256:bdb4fba0965dfacc093c4f71e12ebc2dc180baf63b96e5f8a889e3f844d5adec` | 25552 | 5.1.0.9007 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-11-20 | games.rds | 318837553 | 2025-11-20T15:34:43Z | `0c3cc4cb9fdff4d1991c25a9edfa6e77fcff0f0c395d55981f1791da85cd5d5a` | `sha256:0c3cc4cb9fdff4d1991c25a9edfa6e77fcff0f0c395d55981f1791da85cd5d5a` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-11-20 | play_by_play_2025.rds | 318839645 | 2025-11-20T15:38:58Z | `421313e98803fec97d321b6a289a4645a38246871ee12b957197e54e3eac28ae` | `sha256:421313e98803fec97d321b6a289a4645a38246871ee12b957197e54e3eac28ae` | 28127 | 5.1.0.9007 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-11-27 | games.rds | 321531075 | 2025-11-27T15:33:15Z | `329873c6f615a0c3aecac8b59a323c695af8c33435b396f2ef1c2b1c24977a22` | `sha256:329873c6f615a0c3aecac8b59a323c695af8c33435b396f2ef1c2b1c24977a22` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-11-27 | play_by_play_2025.rds | 321532841 | 2025-11-27T15:37:31Z | `7aa6d2e9d52e22101e988c6fe56aa2c19c87937ffc728e6e8bbaebeb81f77482` | `sha256:7aa6d2e9d52e22101e988c6fe56aa2c19c87937ffc728e6e8bbaebeb81f77482` | 30572 | 5.1.0.9007 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-12-04 | games.rds | 324362504 | 2025-12-04T15:37:22Z | `899a366f22db4cb561fb4d88d209476a769ea9bee0cd8beab34020389fba5016` | `sha256:899a366f22db4cb561fb4d88d209476a769ea9bee0cd8beab34020389fba5016` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-12-04 | play_by_play_2025.rds | 324364390 | 2025-12-04T15:42:12Z | `a23a110d335ef5361e70166976cca8f13392b97dcf98dfe783e4066c1df224bd` | `sha256:a23a110d335ef5361e70166976cca8f13392b97dcf98dfe783e4066c1df224bd` | 33292 | 5.1.0.9007 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2025-12-11 | games.rds | 327433996 | 2025-12-11T15:39:24Z | `fa7666574968fa23b6f3fefd1bfaf1e7a3e971d5be509730925e203b43ee9068` | `sha256:fa7666574968fa23b6f3fefd1bfaf1e7a3e971d5be509730925e203b43ee9068` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-12-11 | play_by_play_2025.rds | 327435650 | 2025-12-11T15:44:21Z | `3590bcb90a75c32ba8b10d692d26838caedbc267a57db23931694abc9598c873` | `sha256:3590bcb90a75c32ba8b10d692d26838caedbc267a57db23931694abc9598c873` | invalid | n/a | `invalid` | invalid exact bytes |
| archive-2025-12-25 | games.rds | 332861918 | 2025-12-25T15:32:52Z | `493dc1dc5a36242e6329e6c421f09447e5938c1681ffda92ad61165c0f8b512f` | `sha256:493dc1dc5a36242e6329e6c421f09447e5938c1681ffda92ad61165c0f8b512f` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2025-12-25 | play_by_play_2025.rds | 332863213 | 2025-12-25T15:37:03Z | `fd8b789271b0dd3f3e5e4a2ec9ae3a680b7b2387b715d9796f494c8da06b3867` | `sha256:fd8b789271b0dd3f3e5e4a2ec9ae3a680b7b2387b715d9796f494c8da06b3867` | 41152 | 5.1.0.9008 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2026-01-01 | games.rds | 335211517 | 2026-01-01T15:33:25Z | `7e8045a638a2bfb41769bb824ea689c7009321227082d44307f341752e6d7fce` | `sha256:7e8045a638a2bfb41769bb824ea689c7009321227082d44307f341752e6d7fce` | 7263 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2026-01-01 | play_by_play_2025.rds | 335212696 | 2026-01-01T15:37:18Z | `e912cd312aca03724079fe1d13a7efe5e2d335b27b1000cbc486623c2aeadfd5` | `sha256:e912cd312aca03724079fe1d13a7efe5e2d335b27b1000cbc486623c2aeadfd5` | 43816 | 5.1.0.9008 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2026-01-08 | games.rds | 337932927 | 2026-01-08T15:40:24Z | `157c82dd3b6a18032e36f7ec2554360d1174928f75918995b242bbff0c432fac` | `sha256:157c82dd3b6a18032e36f7ec2554360d1174928f75918995b242bbff0c432fac` | 7269 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2026-01-08 | play_by_play_2025.rds | 337934967 | 2026-01-08T15:44:49Z | `59ac112922db4531327908d7f18f0cfed7c4e37f4382ab2f27d98fb832a0e8c6` | `sha256:59ac112922db4531327908d7f18f0cfed7c4e37f4382ab2f27d98fb832a0e8c6` | 46452 | 5.1.0.9008 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2026-01-15 | games.rds | 341004025 | 2026-01-15T15:40:46Z | `5af4150fbf79fb8905a5a436b84a948878a420ecd7d9136729b8c2bbf97fb259` | `sha256:5af4150fbf79fb8905a5a436b84a948878a420ecd7d9136729b8c2bbf97fb259` | 7273 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2026-01-15 | play_by_play_2025.rds | 344356279 | 2026-01-22T16:28:41Z | `fb8b7221797516c062aaba0c10ca1e347acf5a2029f0d58a33261498744309b1` | `sha256:fb8b7221797516c062aaba0c10ca1e347acf5a2029f0d58a33261498744309b1` | 48238 | 5.1.0.9008 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |
| archive-2026-01-29 | games.rds | 347739173 | 2026-01-29T15:53:07Z | `1a9ae09e03df4c53e6b5d29ff5ac4259a198fc21dffccdf10f5bc78a48e41ca8` | `sha256:1a9ae09e03df4c53e6b5d29ff5ac4259a198fc21dffccdf10f5bc78a48e41ca8` | 7276 | n/a | `0d46bca4f0f0da778d1de53ea602ad49b8e29fa23f3b0c705e2bb8dc010d2ef4` | proven |
| archive-2026-01-29 | play_by_play_2025.rds | 347741288 | 2026-01-29T15:57:51Z | `4cf54e6c29bb6e9eb2d3e3480985713d4dae93d16deed059470f29d4e12a653d` | `sha256:4cf54e6c29bb6e9eb2d3e3480985713d4dae93d16deed059470f29d4e12a653d` | 48578 | 5.1.0.9009 | `2239769889e0dcad7a1c0351b56c9332ffbdf16aa064e9fde6338421c029e61f` | proven |

Every valid row is `historical_source_proven`; the broken December 11 PBP is retained as exact but invalid and never eligible.

## Stop-condition assessment

The table now has 195 games, 14 independent weekly origins, all 32 teams, and early/mid/late regular-season plus Super Bowl state regimes. It contains meaningful uncertainty variation and genuine publication gaps. That is large enough to demonstrate and stress the structural replay mechanism.

It is **not yet scientifically sufficient to recommend beginning the direct margin/total model**. All forecasts come from one season and one provider/model era; only 14 origin-level state snapshots are independent; Weeks 1–5 supply no forecast rows; playoff Weeks 19–21 are absent; and there is no cross-season variation. A chronological scoreboard bridge would have too little separation for model-family choice, calibration, sensitivity and a genuinely untouched evaluation gate. The conservative state candidate also remains visibly overdispersed on tuning prefixes. These limitations are structural, not an arbitrary numeric threshold.

Phase 3B should keep this table as retrospective development/mechanics evidence, continue exact-source expansion if a defensible schedule source is found, and establish future prospective validation. Direct margin/total modeling has **not** begun.

## Artifacts and validation

Combined table SHA-256: `0fab56f5a145ad5d9fcbbc52b85cf02267337ab31747a546b42bda4bc826103e`. It contains 195 rows, no target scores, exact config/state hashes and the `retrospective_historical_source_replay` label. Bundle manifest SHA-256: computed in the accompanying completion receipt. All manifest hashes are verified. The expansion implementation adds explicit pre-origin schedule/result-table separation. Focused suite: **104 passed in 15.01s** (the original 103 plus one schedule-separation regression).
