# ユーザーガイド

anim_nnモジュールの高度な使用方法を説明します。

## ベストプラクティス

### アニメーション設計

1. **段階的な表示** - 複雑な概念は段階的に表示する
2. **適切なタイミング** - 視聴者が理解できる速度でアニメーションする
3. **一貫性** - 色やスタイルを統一する
4. **説明テキスト** - 重要な概念には説明を追加する

### パフォーマンス最適化

```python
# 効率的なアニメーション作成
class OptimizedScene(Scene):
    def construct(self):
        # 事前にオブジェクトを作成
        nn = NeuralNetwork(layer_sizes=[3, 4, 3])
        
        # 一括でアニメーション
        self.play(Create(nn), run_time=2)
        
        # 個別のアニメーションは最小限に
        self.play(nn.randomize_layer_values(), run_time=1)
```

## カスタムアニメーション作成

### 独自のMobjectクラス

```python
from manim import *

class CustomNeuron(Circle):
    def __init__(self, value=0.5, **kwargs):
        super().__init__(radius=0.1, **kwargs)
        self.value = value
        self.set_fill(opacity=value)
        self.set_stroke(width=2)
    
    def update_value(self, new_value):
        self.value = new_value
        self.set_fill(opacity=new_value)
```

### カスタムアニメーション

```python
class ValueChange(Animation):
    def __init__(self, neuron, new_value, **kwargs):
        super().__init__(neuron, **kwargs)
        self.new_value = new_value
        self.start_value = neuron.value
    
    def interpolate_mobject(self, alpha):
        current_value = self.start_value + (self.new_value - self.start_value) * alpha
        self.mobject.update_value(current_value)
```

## 教育コンテンツ作成

### 段階的な説明

```python
class EducationalScene(Scene):
    def construct(self):
        # 1. タイトル
        title = Text("ニューラルネットワークの構造", font_size=36)
        self.play(Write(title))
        self.wait(1)
        
        # 2. 入力層の説明
        input_text = Text("入力層", font_size=24).next_to(title, DOWN)
        self.play(Write(input_text))
        
        # 3. ニューラルネットワークの表示
        nn = NeuralNetwork(layer_sizes=[3, 4, 3])
        self.play(Create(nn))
        
        # 4. 各層の説明
        for i, layer_name in enumerate(["入力層", "隠れ層", "出力層"]):
            explanation = Text(layer_name, font_size=20)
            explanation.next_to(nn.layers[i], UP)
            self.play(Write(explanation))
            self.wait(1)
```

### インタラクティブ要素

```python
class InteractiveScene(Scene):
    def construct(self):
        # ボタン風のオブジェクト
        button = Rectangle(width=2, height=0.5)
        button_text = Text("クリック", font_size=20)
        button_group = VGroup(button, button_text)
        
        self.play(Create(button_group))
        
        # クリック効果
        self.play(
            button.animate.set_fill(YELLOW, opacity=0.3),
            run_time=0.2
        )
        self.play(
            button.animate.set_fill(opacity=0),
            run_time=0.2
        )
```

## パフォーマンス最適化

### メモリ管理

```python
# 大きなオブジェクトの効率的な管理
class MemoryEfficientScene(Scene):
    def construct(self):
        # 必要な時だけ作成
        nn = None
        
        for i in range(5):
            if nn is not None:
                self.play(FadeOut(nn))
            
            nn = NeuralNetwork(layer_sizes=[3, 4, 3])
            self.play(Create(nn))
            self.wait(1)
```

### レンダリング最適化

```python
# レンダリング設定の最適化
config.frame_rate = 30  # フレームレートを下げる
config.pixel_height = 720  # 解像度を調整
config.pixel_width = 1280
```

## トラブルシューティング

### よくある問題

1. **アニメーションが重い**
   - オブジェクト数を減らす
   - アニメーション時間を短くする
   - レンダリング品質を下げる

2. **メモリ不足**
   - 大きなオブジェクトを段階的に作成
   - 不要なオブジェクトを削除

3. **表示の問題**
   - フォントの確認
   - 画面サイズの調整

### デバッグ方法

```python
# デバッグ用の設定
config.verbosity = "DEBUG"
config.log_to_file = True

# オブジェクトの状態確認
print(f"NeuralNetwork layers: {len(nn.layers)}")
print(f"Connections: {len(nn.connections)}")
```

## 貢献ガイドライン

### コード品質

- PEP 8に準拠したコード
- 適切なドキュメント文字列
- エラーハンドリングの実装

### テスト

```python
# 基本的なテスト
def test_neural_network():
    nn = NeuralNetwork(layer_sizes=[3, 4, 3])
    assert len(nn.layers) == 3
    assert len(nn.connections) == 2
```

### ドキュメント

- 新しい機能には必ずドキュメントを追加
- 使用例を含める
- パラメータの説明を詳細に記載 