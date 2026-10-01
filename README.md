# Watanabe Illusion × VLM — VLM は錯視を「見て」いるのか「知って」いるのか

> **要約**: Vision-Language Model (VLM) は人間と同じく Watanabe 錯視に「だまされる」。だが内部を覗いてみると、Vision Encoder は線の角度を 2° 以内、線の延長先のドット位置を 0.6 ドット以内で正確に表現していた。バイアスは知覚段階ではなく、言語化段階で生まれていた。
>
> **VLM は錯視を見ているのではない。錯視を「知って」いる。**

詳細な議論と統計は論文 [`paper_v2.md`](./paper_v2.md) に収録。本 README は前段のサマリー。

---

## 1. Watanabe 錯視とは

Watanabe Illusion（渡辺錯視, 2010, 渡辺英治）は、線分の傾きを過小評価する知覚バイアスを示す錯視。

刺激構成（[`104.jpg`](./104.jpg)）:

- 横長長方形の左下に円があり、その中に斜めの線分が描かれている
- 長方形の右辺に縦一列の点（11個）が等間隔に配置されている
- 線分を延長すると **一番上のドット (1番目)** に交差するように設計されている（**正解 = 1**）
- しかし観察者の多くは「もっと下のドット」と答える（傾きを過小評価する錯視）
- 線分の **真の角度は 23.5°** だが、人間の平均回答は約 37.4°

![Watanabe Illusion stimulus](104.jpg)

---

## 2. 何を調べたのか

VLM が同じ錯視にどう反応するかを 2 段階で調べた:

### 2.1 探索フェーズ — 商用 VLM 6 体に質問してみる

Claude Opus 4.6, GPT-4o, Gemini, Grok, LLaVA 1.5-7B, LLaVA-NeXT 1.6-7B に同じ刺激を見せ、3 つの質問をした:

- **Q1**: 線を延長したらどのドットに当たる? (上から X 番目で答えて)
- **Q2**: 同じことを下から数えると?
- **Q3**: 水平から何度上に傾いている?

→ 詳細は [`LLM_test.md`](./LLM_test.md) に生データ収録。

### 2.2 定量フェーズ — Qwen2.5-VL-32B で N=60 の確率的試行 + 人間 N=130 と比較

オープンソースの Qwen2.5-VL-32B を選び（重みが手元で動かせる + 角度推定が比較的人間らしい挙動を見せたため）、人間の心理実験プロトコル（G1, G2, G3 画面）と完全に一致するプロンプトで N=60 試行を回した:

- 3 ターン独立構造（人間が前画面を見ないのと同じく、各質問は新規画像 + テキストの組として送る）
- temperature=1.0 でストキャスティック
- 同じ刺激を別の角度から計測

加えて、Qwen2.5-VL-32B の Vision Encoder の中身を直接 probe した（次節）。

---

## 3. 結果

### 3.1 行動: VLM は人間と「同じ方向」にだまされる

人間 N=130 と Qwen2.5-VL-32B N=60 の比較:

| 質問 | 正解 | 人間 (N=130) | Qwen (N=60) |
|---|---|---|---|
| Q1: 上から何番目? | 1 | 4.65 ± 2.63 | 5.47 ± 0.70 |
| Q2: 下から何番目? | 11 | 6.36 ± 2.86 | 5.58 ± 0.91 |
| Q3: 何度傾いている? | 23.5° | 37.44° ± 22.80° | 42.25° ± 7.56° |

![Human vs Qwen distributions](qwen25vl_n60_v2_vs_human.png)

特徴:

- **平均値はほぼ一致** — VLM は人間集団の中心にうまく寄せている
- **分散は人間の 1/3 程度** — VLM は「ばらつかない」、似たような中央値的な答えを返す
- **Q3 は 51/60 (85%) で 45° に固執** — 人間の幅広い分布とは対照的

### 3.2 自己矛盾の瞬間が捕まった

ある trial で Qwen は次のように答えた:

> "the line appears to be tilted **approximately 30 degrees** above the horizontal."  
> ...  
> `\boxed{45}`  
> "(Note: Based on the provided options and **typical estimation**, 45 degrees is a reasonable assumption ... However, the visual impression suggests it might be slightly less than 45 degrees.)"

**自分で「視覚的には 30° に見える」と書きながら、最終的に 45° を出力**している。"typical estimation" — 典型的な推定値 — という言葉は、モデルが知識的なプライアを参照したことの自白に近い。

### 3.3 内部を覗いてみる — Vision Encoder は正確だった

Qwen2.5-VL-32B の Vision Encoder（32 ブロックの ViT + Patch Merger）に、パラメトリック刺激セット（17 角度 × 4 条件 = 68 画像）を流して内部表現を取り出し、線形 probe で「角度」「線の延長先 (target_y_right)」「正解ドット番号 (correct_dot_number)」を予測してみた:

| 表現 | target | R² | MAE |
|---|---|---|---|
| ViT 最終層 (mean-pooled) | 角度 | **0.985** | **2.18°** |
| Patch Merger 後 (LM 入力, mean-pooled) | 角度 | **0.981** | **2.38°** |
| Patch Merger 後 (flattened, 空間情報保持) | 正解ドット番号 | **0.976** | **0.57 ドット** |

![Angle probe scatter](fig_angle_probe.png)

- Vision Encoder は **2° の精度で角度を持っている**
- LM への入力 (Patch Merger 後) でも **情報がロスしていない**
- 45° 付近で誤差が増えるという oblique effect も検出されない:

![Angle error pattern](fig_angle_error.png)

ドット番号の予測も非常に正確:

![Spatial probes](fig_spatial_probe.png)

円の有無や線分長を変えても、Vision Encoder の角度精度はほぼ変わらない:

![Condition comparison](fig_summary_bars.png)

### 3.4 行動 vs 内部 — 7〜9 倍の増幅

これが本研究の中核:

| 測定 | 正解 | 行動 (Qwen N=60 平均) | 行動誤差 | 内部 probe 誤差 (MAE) | 増幅率 |
|---|---|---|---|---|---|
| 角度 (Q3) | 23.5° | 42.25° | **18.75°** | **2.18°** | **8.6×** |
| ドット位置 (Q1) | 1 | 5.47 | **4.47 ドット** | **0.66 ドット** | **6.8×** |

**同じモデル**で、**同じ刺激クラス**で、**Vision Encoder は答えをほぼ正確に持っている**のに、**言語化された応答は 7〜9 倍ずれる**。

これは間接証拠ではない。同一モデル内の dissociation である。

---

## 4. 結論

VLM は錯視を**知覚**しているのではない。錯視を**知識として知っている**。

Vision Encoder は刺激を正確に表現する（2° 以内、0.6 ドット以内）。Patch Merger を通って言語モデルに渡る段階でも情報は保たれている。バイアスは、その後の言語生成段階で「これは斜めの線だな → 典型的には 45° だ」という知識的プライアに引きずられて生まれる。

人間の場合も似た構造があるかもしれない: 視覚野では正確な情報があるのに、言語報告で歪む。これを検証するには、人間でも「言葉で答える」課題と「マウスで線を引く / ボタンを押す」課題で結果が違うかを比べる必要がある。

詳細な議論と統計は論文 [`paper_v2.md`](./paper_v2.md) を参照。

---

## 5. 次のステップ

- **複数刺激での行動比較**: 現在は 104.jpg (角度=23.5°) のみで人間データがある。他の角度でも 45° 集中が起きるか?
- **他の VLM での再現**: Qwen2.5-VL-32B 以外の最新 VLM (Llama 3.2 Vision, Pixtral, etc.) でも同じパターンが見られるか?
- **人間の言語 vs 非言語応答**: 人間でも「角度を言葉で答える」と「マウスで合わせる」で結果が違うか?
- **Q3 = 45° 以外の trial の分析**: 残り 9/60 trial (30°×8, 0°×1) はどんな response だったか?

---

## ファイル構成

```
├── README.md                          # 本ファイル (サマリー)
├── paper.md / paper_v2.md             # 論文 (v2 が現行版)
├── paper_ja.md                        # 論文日本語版
├── 104.jpg                            # 原版の刺激画像 (Watanabe Illusion)
├── G1.png, G2.png, G3.png             # 人間実験のスクリーンショット
├── LLM_test.md                        # 探索フェーズ商用 VLM の生データ
├── stimulus_metadata.csv              # パラメトリック刺激のメタデータ
├── stimuli/                           # パラメトリック刺激画像 (68 枚)
│
├── scripts/
│   ├── generate_stimuli.py            # 刺激生成スクリプト
│   ├── test_llava_illusion.py         # LLaVA 探索スクリプト
│   ├── run_qwen25vl_n60_v2.py         # Qwen N=60 行動実験 (3 ターン独立)
│   ├── analyze_qwen25vl_n60_v2.py     # 行動結果の集計
│   ├── plot_qwen25vl_n60_v2_vs_human.py  # 人間 vs VLM 比較図の生成
│   └── analyze_qwen25vl_vit.py        # Qwen Vision Encoder probe
│
├── results/
│   ├── qwen25vl_n60_v2/               # 行動 N=60 結果 (n60_progress.json 含む)
│   └── qwen25vl_vit_probe/            # 内部 probe 結果 (CSV/JSON/NPZ + 図)
│
└── figures/                           # 論文用の図
    ├── qwen25vl_n60_v2_vs_human.png   # Figure 2 (行動 vs 人間)
    ├── fig_angle_probe.png            # Figure 3 (角度 probe)
    ├── fig_angle_error.png            # Figure 4 (oblique effect 不在)
    ├── fig_spatial_probe.png          # Figure 5 (空間 probe)
    └── fig_summary_bars.png           # Figure 6 (条件比較)
```

---

## 実行環境

- **計算資源**: NVIDIA DGX Spark (128 GB unified memory, LPDDR5x 273 GB/s)
- **コンテナ**: `nvcr.io/nvidia/pytorch:25.11-py3`
- **主要ライブラリ**: `transformers 5.6.2`, `torch 2.10`, `scikit-learn`, `Pillow`

行動実験 (N=60) は約 3 時間、内部 probe は約 10 分で完走する。

---

## 参考文献

- Watanabe, E. (2010). Watanabe Illusion. Figshare.
- Bai, S., et al. (2025). Qwen2.5-VL technical report. arXiv:2502.13923.
- Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *NeurIPS 36*.
- Appelle, S. (1972). Perception and discrimination as a function of stimulus orientation: The "oblique effect" in man and animals. *Psychological Bulletin*, 78(4), 266–278.

---

## 著者

渡辺英治 (基礎生物学研究所)

*本プロジェクトの実験デザインの議論、コーディング、解析、ドキュメント作成には Claude Opus 4.6 / Opus 4.7 (Anthropic) を使用。なお Claude Opus 4.6 は本実験の最初の被験者 (商用 VLM の 1 体) でもある。*
