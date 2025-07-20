# anim_nn

数学的アニメーションとディープラーニング可視化のためのPythonモジュール

## 概要

anim_nnは、ManimCEを使用してニューラルネットワーク、トランスフォーマー、畳み込みネットワークなどのディープラーニング概念を美しく可視化するためのPythonモジュールです。教育目的に最適化されており、複雑な概念を直感的に理解できるアニメーションを提供します。

## インストール

```bash
pip install -r docs/requirements.txt
```

## クイックスタート

```python
from anim_nn import NeuralNetwork
from manim import *

class ExampleScene(Scene):
    def construct(self):
        # ニューラルネットワークを作成
        nn = NeuralNetwork(layer_sizes=[3, 4, 3])
        self.play(Create(nn))
        self.wait(2)
```

## モジュール一覧

### NeuralNetwork
多層パーセプトロンの構造を可視化するクラス

### Transformer
トークン化、埋め込み、アテンション機構の可視化機能

### Convolution
畳み込み演算とフィルターの可視化

### helpers
行列演算、色生成、ユーティリティ関数

### utils
設定管理、ディレクトリ操作

## ドキュメント

詳細なドキュメントは以下のディレクトリを参照してください：

- `docs/api/` - APIリファレンス
- `docs/tutorials/` - チュートリアル
- `docs/examples/` - 実用例
- `docs/user_guide/` - ユーザーガイド

## ライセンス

MIT License 