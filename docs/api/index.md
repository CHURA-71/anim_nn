# API リファレンス

anim_nnモジュールのAPIリファレンスです。

## モジュール一覧

### NeuralNetwork
ニューラルネットワークの構造を可視化するクラス

### Transformer
トークン化、埋め込み、アテンション機構の可視化機能

### Convolution
畳み込み演算とフィルターの可視化

### helpers
行列演算、色生成、ユーティリティ関数

### utils
設定管理、ディレクトリ操作

## クラス階層

```
VGroup
├── NeuralNetwork
└── (その他のManim Mobjects)
```

## 関数一覧

### トークン化関数
- `tokenize_text()`
- `detect_language()`

### 可視化関数
- `create_token_rectangles()`
- `create_embedding_visualization()`
- `create_attention_map()`

### 数学関数
- `sigmoid()`
- `relu()`
- `softmax()`

### ユーティリティ関数
- `generate_colors()`
- `get_directories()`
- `create_matrix()` 