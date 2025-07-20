# NeuralNetwork

ニューラルネットワークの構造を可視化するためのクラスです。

## 概要

`NeuralNetwork`は、多層パーセプトロンの構造を美しいアニメーションで表現します。各層のニューロンを円で表現し、層間の接続を線で表現することで、ニューラルネットワークの構造を直感的に理解できます。

## クラス定義

```python
class NeuralNetwork(VGroup):
    def __init__(
        self,
        layer_sizes=[6, 12, 6],
        neuron_radius=0.1,
        v_buff_ratio=0.2,
        h_buff_ratio=7.0,
        max_stroke_width=2.0,
        stroke_decay=2.0,
        **kwargs
    ):
```

## パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|-----------|------|
| `layer_sizes` | `list[int]` | `[6, 12, 6]` | 各層のニューロン数 |
| `neuron_radius` | `float` | `0.1` | ニューロンの半径 |
| `v_buff_ratio` | `float` | `0.2` | 垂直方向の間隔比率 |
| `h_buff_ratio` | `float` | `7.0` | 水平方向の間隔比率 |
| `max_stroke_width` | `float` | `2.0` | 接続線の最大太さ |
| `stroke_decay` | `float` | `2.0` | 接続線の太さの減衰率 |
| `**kwargs` | - | - | VGroupのその他のパラメータ |

## メソッド

### randomize_layer_values()

各層のニューロンの値をランダムに設定します。

```python
def randomize_layer_values(self):
    """各層のニューロンの値をランダムに設定する。"""
```

**使用例:**
```python
nn = NeuralNetwork(layer_sizes=[3, 4, 3])
nn.randomize_layer_values()
```

### randomize_line_style()

接続線のスタイルをランダムに設定します。

```python
def randomize_line_style(self):
    """接続線のスタイルをランダムに設定する。"""
```

**使用例:**
```python
nn = NeuralNetwork(layer_sizes=[3, 4, 3])
nn.randomize_line_style()
```

## 属性

| 属性 | 型 | 説明 |
|------|----|------|
| `layers` | `list[VGroup]` | 各層のニューロングループ |
| `connections` | `list[VGroup]` | 層間の接続線グループ |
| `layer_sizes` | `list[int]` | 各層のニューロン数 |

## 使用例

### 基本的な使用例

```python
from anim_nn import NeuralNetwork
from manim import *

class NeuralNetworkExample(Scene):
    def construct(self):
        # 3層のニューラルネットワークを作成
        nn = NeuralNetwork(layer_sizes=[3, 4, 3])
        
        # アニメーションで表示
        self.play(Create(nn))
        self.wait(2)
        
        # 値をランダム化
        self.play(nn.randomize_layer_values())
        self.wait(2)
```

### カスタマイズ例

```python
from anim_nn import NeuralNetwork
from manim import *

class CustomNeuralNetwork(Scene):
    def construct(self):
        # カスタム設定でニューラルネットワークを作成
        nn = NeuralNetwork(
            layer_sizes=[5, 8, 6, 4],
            neuron_radius=0.15,
            v_buff_ratio=0.3,
            h_buff_ratio=5.0,
            max_stroke_width=3.0
        )
        
        self.play(Create(nn))
        self.wait(1)
        
        # 接続線のスタイルをランダム化
        self.play(nn.randomize_line_style())
        self.wait(2)
```

## アニメーション

### 推奨アニメーション

- `Create()` - ニューラルネットワークの作成
- `FadeIn()` - フェードイン表示
- `Transform()` - 他のオブジェクトからの変換
- `randomize_layer_values()` - 値のランダム化
- `randomize_line_style()` - 接続線スタイルのランダム化

### アニメーション例

```python
from anim_nn import NeuralNetwork
from manim import *

class NeuralNetworkAnimation(Scene):
    def construct(self):
        # ニューラルネットワークを作成
        nn = NeuralNetwork(layer_sizes=[3, 5, 3])
        
        # 段階的に表示
        self.play(Create(nn.layers[0]))  # 入力層
        self.wait(0.5)
        self.play(Create(nn.layers[1]))  # 隠れ層
        self.wait(0.5)
        self.play(Create(nn.layers[2]))  # 出力層
        self.wait(0.5)
        
        # 接続を表示
        self.play(Create(nn.connections[0]))
        self.play(Create(nn.connections[1]))
        
        # 値をランダム化
        self.play(nn.randomize_layer_values())
        self.wait(2)
```

## パフォーマンス

- 大きなネットワーク（1000以上のニューロン）の場合は、レンダリング時間が長くなる可能性があります
- 複雑なネットワークの場合は、`neuron_radius`を小さくして、`v_buff_ratio`と`h_buff_ratio`を調整してください

## 注意事項

- ニューロン数が多い場合、画面に収まらない可能性があります
- 接続線の数は層のニューロン数の積に比例するため、大きなネットワークでは線が密集する可能性があります
- 日本語フォントを使用する場合は、適切なフォントがシステムにインストールされていることを確認してください 