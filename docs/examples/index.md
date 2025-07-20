# 実用例

anim_nnモジュールの実用的な例を紹介します。

## プロジェクト例

### 1. ニューラルネットワークの学習過程

ニューラルネットワークの学習過程を段階的に可視化する例です。

```python
from anim_nn import NeuralNetwork
from manim import *

class LearningProcess(Scene):
    def construct(self):
        # ニューラルネットワークを作成
        nn = NeuralNetwork(layer_sizes=[3, 4, 3])
        
        # 学習過程をシミュレート
        for epoch in range(5):
            # 値をランダム化（学習のシミュレーション）
            self.play(nn.randomize_layer_values())
            self.wait(1)
```

**実行方法:**
```bash
manim -pql LearningProcess.py LearningProcess
```

### 2. トークン化の可視化

テキストのトークン化過程を可視化する例です。

```python
from anim_nn import tokenize_text, create_token_rectangles
from manim import *

class TokenizationDemo(Scene):
    def construct(self):
        # テキストをトークン化
        text = "Hello world! This is a test."
        tokens = tokenize_text(text)
        
        # トークンを矩形で可視化
        token_rects = create_token_rectangles(tokens)
        
        # 段階的に表示
        for i, rect in enumerate(token_rects):
            self.play(Create(rect))
            self.wait(0.5)
```

### 3. 畳み込みフィルターの可視化

畳み込みフィルターの動作を可視化する例です。

```python
from anim_nn import Convolution
from manim import *

class ConvolutionDemo(Scene):
    def construct(self):
        # 畳み込みフィルターを作成
        conv = Convolution(kernel_size=3)
        
        # フィルターの動作を可視化
        self.play(Create(conv))
        self.wait(2)
```

## カスタマイズのヒント

### 色の変更

```python
# カスタム色を使用
nn = NeuralNetwork(layer_sizes=[3, 4, 3])
nn.set_color(BLUE)
```

### アニメーション速度の調整

```python
# アニメーション速度を調整
self.play(Create(nn), run_time=3)
self.wait(1)
```

### 複数のオブジェクトの組み合わせ

```python
# 複数の可視化を組み合わせ
nn = NeuralNetwork(layer_sizes=[3, 4, 3])
tokens = create_token_rectangles(["Hello", "world"])

# 並べて表示
nn.shift(LEFT * 3)
tokens.shift(RIGHT * 3)

self.play(Create(nn), Create(tokens))
```

## パフォーマンス最適化

### 大きなデータセットの処理

```python
# 大きなネットワークの場合
nn = NeuralNetwork(
    layer_sizes=[10, 20, 10],
    neuron_radius=0.05,  # 小さくする
    v_buff_ratio=0.1     # 間隔を狭くする
)
```

### レンダリング品質の調整

```bash
# 低品質でレンダリング（高速）
manim -l -pql scene.py SceneName

# 高品質でレンダリング（低速）
manim -qh -pqh scene.py SceneName
```

## トラブルシューティング

### よくある問題と解決方法

1. **メモリ不足**
   - ニューロン数を減らす
   - レンダリング品質を下げる

2. **表示が崩れる**
   - 画面サイズを調整
   - オブジェクトのサイズを小さくする

3. **アニメーションが重い**
   - アニメーション時間を短くする
   - オブジェクト数を減らす

## 貢献

新しい例や改善案を歓迎します。

### 例の追加方法

1. コード例を作成
2. 実行方法を記載
3. カスタマイズのヒントを追加
4. トラブルシューティング情報を記載

### ガイドライン

- 実用的で分かりやすい例
- 完全な実行可能なコード
- 適切なコメントと説明 