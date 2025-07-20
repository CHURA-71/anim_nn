# Transformer

トークン化、埋め込み、アテンション機構の可視化機能を提供するモジュールです。

## 概要

`Transformer`モジュールは、自然言語処理におけるトランスフォーマーモデルの主要コンポーネントを可視化するための関数群を提供します。テキストのトークン化から埋め込み、アテンション機構まで、段階的に理解できるアニメーションを作成できます。

## 関数一覧

### トークン化関数

#### tokenize_text()

テキストをトークンに分割します。

```python
def tokenize_text(text, tokenizer_name="cl100k_base", max_tokens=None):
    """
    テキストをトークンに分割する
    
    Parameters:
        text (str): トークン化するテキスト
        tokenizer_name (str): 使用するトークナイザー名
        max_tokens (int, optional): 最大トークン数
    
    Returns:
        list: トークンのリスト
    """
```

**使用例:**
```python
from anim_nn import tokenize_text

tokens = tokenize_text("Hello world!", max_tokens=10)
print(tokens)  # ['Hello', ' world', '!']
```

#### detect_language()

テキストの言語を検出します。

```python
def detect_language(text):
    """
    テキストの言語を検出する
    
    Parameters:
        text (str): 検出対象のテキスト
    
    Returns:
        str: 言語コード（'en', 'ja', 'zh', 'ko'など）
    """
```

### 可視化関数

#### create_token_rectangles()

トークンを矩形で可視化します。

```python
def create_token_rectangles(tokens, width=0.8, height=0.4, buff=0.1):
    """
    トークンを矩形で可視化する
    
    Parameters:
        tokens (list): トークンのリスト
        width (float): 矩形の幅
        height (float): 矩形の高さ
        buff (float): 矩形間の間隔
    
    Returns:
        VGroup: トークン矩形のグループ
    """
```

#### create_embedding_visualization()

埋め込みベクトルを可視化します。

```python
def create_embedding_visualization(embeddings, token_rectangles):
    """
    埋め込みベクトルを可視化する
    
    Parameters:
        embeddings (np.ndarray): 埋め込み行列
        token_rectangles (VGroup): トークン矩形のグループ
    
    Returns:
        VGroup: 埋め込み可視化のグループ
    """
```

#### create_attention_map()

アテンション行列を可視化します。

```python
def create_attention_map(attention_matrix, token_rectangles):
    """
    アテンション行列を可視化する
    
    Parameters:
        attention_matrix (np.ndarray): アテンション行列
        token_rectangles (VGroup): トークン矩形のグループ
    
    Returns:
        VGroup: アテンションマップのグループ
    """
```

### 単語ベクトル関数

#### get_word_vectors()

単語ベクトルを取得します。

```python
def get_word_vectors(tokens, model_name="word2vec-google-news-300"):
    """
    単語ベクトルを取得する
    
    Parameters:
        tokens (list): トークンのリスト
        model_name (str): 使用するモデル名
    
    Returns:
        np.ndarray: 単語ベクトル行列
    """
```

### 数学関数

#### sigmoid()

シグモイド関数を計算します。

```python
def sigmoid(x):
    """シグモイド関数: 1 / (1 + exp(-x))"""
```

#### relu()

ReLU関数を計算します。

```python
def relu(x):
    """ReLU関数: max(0, x)"""
```

#### softmax()

ソフトマックス関数を計算します。

```python
def softmax(x):
    """ソフトマックス関数: exp(x) / sum(exp(x))"""
```

## 使用例

### 基本的なトークン化

```python
from anim_nn import tokenize_text, create_token_rectangles
from manim import *

class TokenizationExample(Scene):
    def construct(self):
        # テキストをトークン化
        text = "Hello world! This is a test."
        tokens = tokenize_text(text)
        
        # トークンを矩形で可視化
        token_rects = create_token_rectangles(tokens)
        
        # アニメーションで表示
        self.play(Create(token_rects))
        self.wait(2)
```

### 埋め込み可視化

```python
from anim_nn import tokenize_text, create_token_rectangles, create_embedding_visualization
from manim import *

class EmbeddingExample(Scene):
    def construct(self):
        # トークン化
        tokens = tokenize_text("Hello world")
        token_rects = create_token_rectangles(tokens)
        
        # 埋め込み作成（ダミーデータ）
        import numpy as np
        embeddings = np.random.rand(len(tokens), 10)
        
        # 埋め込み可視化
        embedding_viz = create_embedding_visualization(embeddings, token_rects)
        
        self.play(Create(token_rects))
        self.wait(1)
        self.play(Create(embedding_viz))
        self.wait(2)
```

## エラーハンドリング

### よくあるエラー

1. **トークナイザーエラー**
   ```python
   try:
       tokens = tokenize_text(text)
   except Exception as e:
       print(f"トークン化エラー: {e}")
   ```

2. **埋め込みエラー**
   ```python
   try:
       vectors = get_word_vectors(tokens)
   except Exception as e:
       print(f"単語ベクトル取得エラー: {e}")
   ```

## パフォーマンス

- 大きなテキスト（1000トークン以上）の場合は、処理時間が長くなる可能性があります
- 単語ベクトルの取得には、初回時にモデルのダウンロードが必要です
- アテンションマップの可視化は、行列サイズに応じてレンダリング時間が増加します

## 注意事項

- 日本語テキストの場合は、適切なトークナイザーを使用してください
- 単語ベクトルモデルは初回使用時にダウンロードされます
- 大きなアテンション行列の可視化は、メモリ使用量が増加する可能性があります 