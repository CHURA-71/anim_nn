from manim import *
import numpy as np
import re
from typing import Optional
import random
import tiktoken
import gensim
import gensim.downloader
import os
import math

from .helpers import *

# =========================
# トークン分割・可視化ユーティリティ
# =========================
def get_token_encoding():
    """tiktokenのエンコーダを取得する。
    
    最新のモデル名を使用し、エラーハンドリングを含む。
    
    Returns:
        tiktoken.Encoding: トークナイザーエンコーダー
        
    Raises:
        Exception: トークナイザーの取得に失敗した場合
    """
    try:
        # 最新のモデル名を使用
        print("Using tiktoken encoding for 'chatgpt-4o-' model")
        return tiktoken.encoding_for_model("chatgpt-4o-")
    except Exception as e:
        print(f"Warning: Could not get tiktoken encoder: {e}")
        print("Falling back to cl100k_base encoding")
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception as e2:
            print(f"Error: Could not get any tiktoken encoder: {e2}")
            raise e2


def get_principle_components(data: np.ndarray, n_components: int = 3) -> np.ndarray:
    """データの主成分分析を行い、上位n_components個の主成分を返す。

    Args:
        data: 入力データ配列
        n_components: 取得する主成分の数（デフォルト: 3）
        
    Returns:
        np.ndarray: 主成分ベクトルの配列
    """
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns
    return sorted_eigenvectors[:, :n_components]

def find_nearest_words(model, vector, n=20):
    """ベクトルに最も近い単語をn個返す。

    Args:
        model: 単語ベクトルモデル
        vector: 比較対象のベクトル
        n: 返す単語の数（デフォルト: 20）

    Returns:
        list: 最も近いn個の単語のリスト
    """
    data = model.vectors
    indices = np.argsort(((data - vector)**2).sum(1))
    return [model.index_to_key[i] for i in indices[:n]]

def is_japanese_text(text: str) -> bool:
    """テキストが日本語を含むかどうかを判別する。

    Args:
        text: 判定対象のテキスト

    Returns:
        bool: 日本語を含む場合はTrue、そうでなければFalse
    """

    # ひらがな、カタカナ、漢字、日本語句読点の範囲をチェック
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3000-\u303F\uFF00-\uFFEF]')
    return bool(japanese_pattern.search(text))

def has_descender(text: str) -> bool:
    """テキストにdescender（下にはみ出す文字）が含まれているかチェック。

    Args:
        text: 判定対象のテキスト

    Returns:
        bool: descender文字を含む場合はTrue、そうでなければFalse
    """
    descender_chars = "gjpqy"  
    return any(char in descender_chars for char in text.lower())

def get_appropriate_font(text: str) -> Optional[str]:
    """テキストの言語に応じて適切なフォントを返す。

    Args:
        text: フォントを決定するテキスト

    Returns:
        Optional[str]: 日本語の場合は"Yu Gothic"、英語の場合はNone（デフォルトフォント）
    """
    if is_japanese_text(text):
        return "Yu Gothic"  # 日本語用
    else:
        return None  # Manimのデフォルトフォントを使用

def break_into_pieces(phrase_mob: Text, offsets: list[int], auto_font: Optional[bool] = True) -> VGroup:
    """文字列とオフセットリストから、各部分をTextとしてVGroupで返す。

    ManimCE対応版：substr_to_path_countの代わりに新しいTextオブジェクトを作成。
    言語に応じてフォントを自動選択し、descender文字を適切に調整する。

    Args:
        phrase_mob: 元のTextオブジェクト
        offsets: 分割位置のオフセットリスト
        auto_font: フォント自動選択を有効にするかどうか（デフォルト: True）

    Returns:
        VGroup: 分割されたTextオブジェクトのグループ
    """
    phrase = phrase_mob.original_text
    lhs = offsets
    rhs = [*offsets[1:], len(phrase)]
    result = []
    
    for lh, rh in zip(lhs, rhs):
        substr = phrase[lh:rh]
        # 空文字列はスキップ
        if substr.strip() == "":
            continue
        
        if auto_font:
            # 言語に応じてフォントを自動選択
            font = get_appropriate_font(substr)
            if font is not None:
                token_text = Text(
                    substr, 
                    font_size=phrase_mob.font_size,
                    font=font
                )
            else:
                # Manimのデフォルトフォントを使用
                token_text = Text(substr, font_size=phrase_mob.font_size)
        else:
            # 元のフォントを使用
            token_text = Text(substr, font_size=phrase_mob.font_size)
        
        result.append(token_text)
    
    # VGroupとして配置
    group = VGroup(*result)
    if len(group) > 0:
        group[0].move_to(ORIGIN)
        for i in range(1, len(group)):
            group[i].next_to(group[i-1], RIGHT, buff=0.1, aligned_edge=DOWN)
    
    # descender文字だけを個別に調整
    for i in range(len(group)):
        if has_descender(group[i].original_text):
            descender_offset = phrase_mob.height * 0.2  
            group[i].shift(DOWN * descender_offset)
    
    return group

def break_into_words(phrase_mob: Text) -> VGroup:
    """テキストを空白で区切って単語単位で分割する。

    Args:
        phrase_mob: 分割対象のTextオブジェクト

    Returns:
        VGroup: 単語単位に分割されたTextオブジェクトのグループ
    """
    offsets = [m.start() for m in re.finditer(" ", phrase_mob.original_text)]
    return break_into_pieces(phrase_mob, [0, *offsets])

def break_into_tokens(phrase_mob: Text, auto_font: Optional[bool] = True) -> VGroup:
    """テキストをトークナイザーを使用してトークン単位で分割する。

    Args:
        phrase_mob: 分割対象のTextオブジェクト
        auto_font: フォント自動選択を有効にするかどうか（デフォルト: True）

    Returns:
        VGroup: トークン単位に分割されたTextオブジェクトのグループ
    """
    tokenizer = get_token_encoding()
    tokens = tokenizer.encode(phrase_mob.original_text)
    _, offsets = tokenizer.decode_with_offsets(tokens)
    return break_into_pieces(phrase_mob, offsets, auto_font)

# =========================
# 矩形で囲むユーティリティ
# =========================
def get_piece_rectangles(
    phrase_pieces: VGroup,
    h_buff=0.05,
    v_buff=0.1,
    fill_opacity=0.15,
    fill_color=None,
    stroke_width=1,
    stroke_color=None,
    hue_range=(0.5,0.6),
    leading_spaces=False,
):
    """テキストピースを囲む矩形を作成する。

    Args:
        phrase_pieces: 囲む対象のVGroup
        h_buff: 水平方向の余白（デフォルト: 0.05）
        v_buff: 垂直方向の余白（デフォルト: 0.1）
        fill_opacity: 塗りつぶしの透明度（デフォルト: 0.15）
        fill_color: 塗りつぶし色（デフォルト: None、ランダム色）
        stroke_width: 枠線の太さ（デフォルト: 1）
        stroke_color: 枠線色（デフォルト: None、塗りつぶし色と同じ）
        hue_range: 色相範囲（デフォルト: (0.5, 0.6)）
        leading_spaces: 先頭スペースを含めるかどうか（デフォルト: False）

    Returns:
        VGroup: 矩形のグループ
    """
    rects = VGroup()
    height = phrase_pieces.height + 2 * v_buff
    last_right_x = phrase_pieces.get_x(LEFT)
    for piece in phrase_pieces:
        left_x = last_right_x if leading_spaces else piece.get_x(LEFT)
        right_x = piece.get_x(RIGHT)
        fill = random_bright_color_with_hue(hue_range) if fill_color is None else fill_color
        stroke = fill if stroke_color is None else stroke_color
        rect = Rectangle(
            width=right_x - left_x + 2 * h_buff,
            height=height,
            fill_color=fill,
            fill_opacity=fill_opacity,
            stroke_color=stroke,
            stroke_width=stroke_width
        )
        if leading_spaces:
            rect.set_x(left_x, LEFT)
        else:
            rect.move_to(piece)
        rect.set_y(0)
        rects.add(rect)

        last_right_x = right_x

    rects.match_y(phrase_pieces)
    return rects

def get_word_to_vec_model(model_name: str = "glove-wiki-gigaword-50"):
    """単語ベクトルモデルを取得またはダウンロードする。
    
    指定されたモデル名の単語ベクトルモデルをローカルから読み込むか、
    存在しない場合はgensimのダウンローダーを使用してダウンロードします。
    ダウンロードされたモデルは自動的にローカルに保存されます。
    
    Args:
        model_name (str): 取得するモデル名。
                         利用可能なモデル:
                         - "glove-wiki-gigaword-50" (推奨、軽量)
                         - "glove-wiki-gigaword-100"
                         - "glove-wiki-gigaword-200" 
                         - "glove-wiki-gigaword-300"
                         - "word2vec-google-news-300" (大容量)
    
    Returns:
        gensim.models.keyedvectors.KeyedVectors: 読み込まれた単語ベクトルモデル
    
    Raises:
        ValueError: 指定されたモデル名が無効な場合
        ConnectionError: ダウンロードに失敗した場合
        
    Note:
        初回実行時はインターネット接続が必要です。
        大きなモデル（word2vec-google-news-300等）は1GB以上になる場合があります。
        
    Examples:
        >>> # 軽量モデルを使用
        >>> model = get_word_to_vec_model("glove-wiki-gigaword-50")
        >>> vector = model["king"]
        >>> 
        >>> # 高精度モデルを使用
        >>> model = get_word_to_vec_model("word2vec-google-news-300")
    """
    # ローカル保存用のファイルパスを構築
    filename = str(Path(DATA_DIR, model_name))
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # ローカルファイルが存在する場合は読み込み
    if os.path.exists(filename):
        try:
            return gensim.models.keyedvectors.KeyedVectors.load(filename)
        except Exception as e:
            print(f"Warning: ローカルモデルの読み込みに失敗: {e}")
            print("新しいモデルをダウンロードします...")
    
    try:
        # gensimのダウンローダーでモデルを取得
        print(f"モデル '{model_name}' をダウンロード中...")
        model = gensim.downloader.load(model_name)
        
        # モデルがKeyedVectorsインスタンスの場合のみローカル保存
        if isinstance(model, gensim.models.keyedvectors.KeyedVectors):
            try:
                model.save(filename)
                print(f"モデルを {filename} に保存しました")
            except Exception as e:
                print(f"Warning: モデルの保存に失敗: {e}")
        
        return model
        
    except Exception as e:
        raise ValueError(f"モデル '{model_name}' の取得に失敗しました: {e}")

def get_direction_lines(
    axes: ThreeDAxes, 
    direction: np.ndarray, 
    n_lines: int = 500, 
    color=YELLOW, 
    line_length: float = 1.0, 
    stroke_width: float = 3
) -> VGroup:
    """3D空間内に方向を示すランダムな線のグループを作成する。
    
    指定された方向ベクトルに沿って、3D軸の範囲内にランダムに配置された
    線のグループを生成します。各線は指定された色と太さで描画され、
    ベクトル場や方向性の可視化に使用できます。
    
    Args:
        axes (ThreeDAxes): 線を配置する3D軸オブジェクト
        direction (np.ndarray): 線の方向を決定するベクトル（正規化される）
        n_lines (int, optional): 生成する線の数。デフォルトは500
        color: 線の色。Manimの色定数またはカラーコード。デフォルトはYELLOW
        line_length (float, optional): 各線の長さ。デフォルトは1.0
        stroke_width (float, optional): 線の太さ。デフォルトは3
        
    Returns:
        VGroup: 生成された線オブジェクトのグループ
        
    Note:
        - 各線は軸の範囲内にランダムに配置される
        - 線にはカーブが追加され、滑らかな外観になる
        - 線の両端は細くなるグラデーション効果が適用される
    """
    # 正規化された方向ベクトルに基づいて基本線を作成
    line = Line(ORIGIN, line_length * normalize(direction))
    
    # 線に滑らかなカーブを追加し、グラデーション効果を設定
    line.insert_n_curves(20).set_stroke(
        width=(0, stroke_width, stroke_width, stroke_width, 0)
    )
    
    # 指定された数の線をコピーして作成
    lines = VGroup(*(line.copy() for _ in range(n_lines)))
    
    # 全ての線に同じ色を設定
    lines.set_color(color)
    
    # 各線を軸の範囲内のランダムな位置に配置
    for line in lines:
        line.move_to(axes.c2p(
            random.uniform(*axes.x_range),      # X軸範囲内のランダムな値
            random.uniform(*axes.y_range),      # Y軸範囲内のランダムな値
            random.uniform(*axes.z_range),      # Z軸範囲内のランダムな値
        ))
    
    return lines


class Word2VecScene(ThreeDScene):
    """Word2Vecの単語埋め込みを3D空間で可視化するManimシーン。
    
    このクラスは、単語ベクトルを3次元空間内の矢印として表現し、
    単語間の意味的関係を視覚的に理解できるようにします。
    主成分分析（PCA）を使用して高次元ベクトルを3次元に投影します。
    
    Attributes:
        axes_config (dict): 3D軸の設定（範囲、長さなど）
        label_rotation (float): ラベルの回転角度（度単位）
        embedding_model (str): 使用する単語埋め込みモデル名
        default_frame_orientation (tuple): デフォルトのカメラ向き（phi, theta）
        model: 読み込まれた単語ベクトルモデル
        basis (np.ndarray): PCAによる基底ベクトル
        axes (ThreeDAxes): 3D軸オブジェクト
    
    Examples:
        >>> class MyWordVecScene(Word2VecScene):
        ...     def construct(self):
        ...         word_arrow = self.get_labeled_vector("king")
        ...         self.play(FadeIn(word_arrow))
    """
    
    def __init__(self, **kwargs):
        """Word2VecSceneの初期化。
        
        Args:
            **kwargs: ThreeDSceneの引数（camera_config等）
        """
        super().__init__(**kwargs)
        # デフォルトのカメラ向き（phi=-30度, theta=70度）
        self.default_frame_orientation = (-30, 70)

    # 3D軸の設定辞書
    axes_config = dict(
        x_range=(-5, 5, 1),    # X軸: -5から5まで、刻み1
        y_range=(-5, 5, 1),    # Y軸: -5から5まで、刻み1
        z_range=(-4, 4, 1),    # Z軸: -4から4まで、刻み1
        x_length=8,            # X軸の表示長さ
        y_length=8,            # Y軸の表示長さ
        z_length=6.4,          # Z軸の表示長さ
    )

    # ラベルの回転角度（度単位）
    label_rotation = 0

    # 使用する埋め込みモデル
    # より大きなモデル: "word2vec-google-news-300"
    embedding_model = "glove-wiki-gigaword-50"

    def setup(self):
        """シーンの初期設定を行う。
        
        以下の処理を順次実行します：
        1. 親クラスのsetupメソッドを呼び出し
        2. 指定された単語埋め込みモデルを読み込み
        3. PCAによる3次元基底を計算
        4. 3D軸オブジェクトを作成・追加
        5. カメラの向きを設定
        
        Note:
            このメソッドはシーン開始時に自動的に呼び出されます。
            モデルの読み込みには時間がかかる場合があります。
        """
        super().setup()

        # 指定されたモデル名で単語ベクトルモデルを読み込み
        # 初回実行時はダウンロードが発生する可能性がある
        self.model = get_word_to_vec_model(self.embedding_model)

        # モデルから3次元の主成分基底を計算
        # 高次元ベクトルを3次元空間に投影するために使用
        self.basis = self._get_basis(self.model)

        # 設定に基づいて3D軸オブジェクトを作成
        self.axes = ThreeDAxes(**self.axes_config)
        self.add(self.axes)
        
        # カメラの向きを設定（phi: 仰角, theta: 方位角）
        self.set_camera_orientation(
            phi=self.default_frame_orientation[0], 
            theta=self.default_frame_orientation[1]
        )

    def _get_basis(self, model) -> np.ndarray:
        """単語ベクトルモデルから3次元基底を計算する。
        
        主成分分析（PCA）を使用して、モデルの全ての単語ベクトルから
        最も分散の大きい3つの方向を基底として選択します。
        
        Args:
            model: 単語ベクトルモデル（gensim.models.keyedvectors.KeyedVectors）
        
        Returns:
            np.ndarray: 3×モデル次元の基底行列（転置済み）
                        各行が一つの主成分ベクトルを表す
        
        Note:
            戻り値は転置されているため、ベクトル変換時は 
            `basis @ vector` の形で使用します。
        """
        return get_principle_components(model.vectors, 3).T

    def add_plane(self, color=GREY, stroke_width: float = 1.0) -> NumberPlane:
        """3D軸に対応する2D平面（グリッド）を追加する。
        
        XY平面上にグリッド線を描画し、空間の奥行き感を向上させます。
        背景線とフェード線の2種類の線で構成されます。
        
        Args:
            color: グリッド線の色。デフォルトはGREY
            stroke_width (float): グリッド線の太さ。デフォルトは1.0
        
        Returns:
            NumberPlane: 作成された平面オブジェクト
        
        Note:
            作成された平面はself.planeに保存され、シーンに自動追加されます。
        """
        axes = self.axes
        
        # 軸の範囲と寸法に合わせて平面を作成
        plane = NumberPlane(
            x_range=axes.x_range,           # X軸の範囲を継承
            y_range=axes.y_range,           # Y軸の範囲を継承
            width=axes.get_width(),         # 軸の幅を継承
            height=axes.get_height(),       # 軸の高さを継承
            background_line_style=dict(
                stroke_color=color,         # 主要グリッド線の色
                stroke_width=stroke_width,  # 主要グリッド線の太さ
            ),
            faded_line_style=dict(
                stroke_opacity=0.25,        # フェード線の透明度
                stroke_width=0.5 * stroke_width,  # フェード線の太さ（半分）
            ),
            faded_line_ratio=1,             # フェード線の比率
        )
        
        # インスタンス変数として保存
        self.plane = plane
        self.add(plane)
        
        return plane

    def get_labeled_vector(
        self,
        word: str,
        coords: Optional[np.ndarray] = None,
        thickness: float = 5,
        color=YELLOW,
        func_name: Optional[str] = "E",
        buff: float = 0.05,
        direction: Optional[np.ndarray] = None,
        label_config: Optional[dict] = None
    ) -> TextLabeledArrow:
        """指定された単語のラベル付きベクトル矢印を作成する。
        
        単語を3D空間内のベクトルとして可視化し、適切なラベルを付与します。
        座標が指定されない場合は、モデルから自動的に計算されます。
        
        Args:
            word (str): 表示する単語
            coords (Optional[np.ndarray]): ベクトルの3D座標。
                                            Noneの場合はモデルから自動計算
            thickness (float): 矢印の太さ。デフォルトは5
            color: 矢印の色。Manimの色定数またはカラーコード。デフォルトはYELLOW
            func_name (Optional[str]): 関数名の表示。Noneの場合は単語のみ表示。
                                        デフォルトは"E"（Embedding関数）
            buff (float): ラベルと矢印の間隔。デフォルトは0.05
            direction (Optional[np.ndarray]): ラベルの配置方向。
                                            Noneの場合は自動決定
            label_config (Optional[dict]): ラベルの追加設定。
                                            font_size等のパラメータを指定可能
        
        Returns:
            TextLabeledArrow: ラベル付きの矢印オブジェクト
        
        Raises:
            KeyError: 指定された単語がモデルの語彙に存在しない場合
        
        Examples:
            >>> # 基本的な使用法
            >>> arrow = scene.get_labeled_vector("king")
            >>> 
            >>> # カスタマイズされた矢印
            >>> arrow = scene.get_labeled_vector(
            ...     "queen", 
            ...     thickness=8, 
            ...     color=RED,
            ...     label_config={"font_size": 36}
            ... )
        """
        # デフォルト値の設定
        if label_config is None:
            label_config = {}
        
        axes = self.axes
        
        # 座標が指定されていない場合は、モデルから計算
        if coords is None:
            try:
                # 単語ベクトルを基底で変換して3D座標を取得
                coords = self.basis @ self.model[word.lower()]
            except KeyError:
                raise KeyError(f"単語 '{word}' がモデルの語彙に存在しません")
        
        # 3D座標をシーン座標に変換
        point = axes.c2p(*coords)
        
        # ラベル設定の更新
        label_config.update(label_buff=buff)
        if "label_rotation" not in label_config:
            label_config.update(label_rotation=self.label_rotation)
        
        # ラベルテキストの決定
        label_text = word if func_name is None else f"{func_name}({word})"
        
        # ラベル付き矢印の作成
        arrow = TextLabeledArrow(
            axes.get_origin(),      # 矢印の開始点（原点）
            point,                  # 矢印の終点
            stroke_width=thickness, # 矢印の太さ
            stroke_color=color,     # 矢印の色
            label_text=label_text,  # ラベルテキスト
            direction=direction,    # ラベルの方向
            scene=self,            # 親シーン
            **label_config,        # その他の設定
        )
        
        return arrow

    def add_fixed_in_frame_mobjects(self, *mobjects):
        """オブジェクトをカメラフレームに固定して追加する。
        
        3Dシーン内でカメラが回転しても位置が変わらない
        固定オブジェクト（UIエレメント、ラベルなど）を追加します。
        
        Args:
            *mobjects: フレームに固定するMobjectオブジェクト群
        
        Note:
            - 固定されたオブジェクトは3D空間の回転に影響されません
            - タイトル、説明文、UIコントロールなどに使用します
            - オブジェクトはシーンからは除去され、フレームにのみ表示されます
        
        Examples:
            >>> title = Text("Word Embeddings")
            >>> scene.add_fixed_in_frame_mobjects(title)
        """
        # 親クラスのメソッドでフレーム固定オブジェクトとして追加
        super().add_fixed_in_frame_mobjects(*mobjects)
        
        # シーンの3D空間からは除去（フレーム固定のみ）
        self.remove(*mobjects)

    def get_word_vectors(self, words: list[str]) -> np.ndarray:
        """複数の単語のベクトル座標を一括取得し、正規化する。
        
        Args:
            words (list[str]): 処理する単語のリスト
            
        Returns:
            np.ndarray: 正規化された3D座標の配列（単語数×3）
            
        Note:
            座標は中心化され、表示範囲に合わせてスケール調整されます。
        """
        # 各単語のベクトルを基底で変換
        coords = np.array([
            self.basis @ self.model[word.lower()]
            for word in words if word.lower() in self.model
        ])
        
        # 中心化（平均を0にする）
        coords -= coords.mean(0)
        
        # スケール正規化
        max_coord = max(coords.max(), -coords.min())
        if max_coord > 0:
            coords *= 4.0 / max_coord
            
        return coords

    def create_word_embeddings(
        self, 
        words: list[str], 
        thickness: float = 2,
        base_color=BLUE_D,
        target_color=BLUE_A,
        label_font_size: float = 30
    ) -> VGroup:
        """複数の単語の埋め込みベクトルを一括作成する。
        
        Args:
            words (list[str]): 可視化する単語のリスト
            thickness (float): 矢印の太さ
            base_color: グラデーションの開始色
            target_color: グラデーションの終了色
            label_font_size (float): ラベルのフォントサイズ
            
        Returns:
            VGroup: 埋め込みベクトルのグループ
        """
        coords = self.get_word_vectors(words)
        
        embeddings = VGroup(*(
            self.get_labeled_vector(
                word,
                coord,
                thickness=thickness,
                color=interpolate_color(base_color, target_color, random.random()),
                func_name=None,
                label_config=dict(font_size=label_font_size)
            )
            for word, coord in zip(words, coords)
        ))
        
        return embeddings

    def begin_ambient_rotation(self, rate: float = 0.02):
        """環境回転アニメーションを開始する。
        
        Args:
            rate (float): 回転速度（rad/frame）
        """
        self.begin_ambient_camera_rotation(rate=rate)

class PatchedImage(Group):
    """画像をパッチに分割できるMobjectクラス。"""
    
    def __init__(
        self, 
        image_path: str, 
        n_divisions: int = 64, 
        height: float = 5,
        **kwargs
    ):
        """PatchedImageの初期化。"""
        super().__init__(**kwargs)
        
        # 設定値の保存
        self.n_divisions = n_divisions
        self.is_patched = False
        
        # 元画像の作成
        self.original_image = ImageMobject(image_path)
        self.original_image.set_height(height)
        
        # 初期状態では元画像のみを表示
        self.add(self.original_image)
        
        # パッチ関連の属性を初期化
        self.pixels = None
        self.patches = None
        
        # アニメーターの初期化
        self._animator = None
    
    @property
    def animate(self):
        """アニメーション機能にアクセスするためのプロパティ。
        
        Returns:
            PatchedImageAnimator: アニメーション機能を提供するオブジェクト
        """
        if self._animator is None:
            self._animator = PatchedImageAnimator(self)
        return self._animator
    
    def space_out_patches(
        self, 
        factor: float = 2.0, 
        scale: float = 0.75
    ) -> VGroup:
        """パッチ間の間隔を広げる（アニメーションなし）。
        
        Args:
            factor (float): 間隔の倍率（デフォルト: 2.0）
            scale (float): 全体のスケール（デフォルト: 0.75）
        
        Returns:
            VGroup: 調整されたパッチのグループ
        
        Raises:
            ValueError: パッチ化されていない場合
        """
        if not self.is_patched or self.patches is None:
            raise ValueError("convert_to_patches()を先に実行してください")
        
        # 即座にパッチ間隔を調整
        self.patches.space_out_submobjects(factor).scale(scale)
        return self.patches
    
    def space_out_pixels(
        self,
        factor: float = 1.2,
        scale: float = 0.9
    ) -> VGroup:
        """PatchedImage内の全ピクセル間隔を広げる（アニメーションなし）。
        
        Args:
            factor (float): ピクセル間隔の倍率（デフォルト: 1.2）
            scale (float): 全体のスケール（デフォルト: 0.9）
        
        Returns:
            VGroup: 調整されたピクセルのグループ
        """
        # ピクセルが存在しない場合は作成する
        if self.pixels is None:
            pixel_width = self.original_image.width / self.n_divisions
            self.pixels = create_pixels(self.original_image, pixel_width=pixel_width)
        
        # 即座にピクセル間隔を調整
        self.pixels.space_out_submobjects(factor).scale(scale)
        return self.pixels
    
    # ...existing code...（他のメソッドはそのまま）
    
    def _create_patches(self) -> VGroup:
        """内部メソッド：パッチを準備する。"""
        # ピクセルを作成
        pixel_width = self.original_image.width / self.n_divisions
        big_pixel_width = self.original_image.width / (self.n_divisions / 4)
        
        self.pixels = create_pixels(self.original_image, pixel_width=pixel_width)
        big_pixels = create_pixels(self.original_image, pixel_width=big_pixel_width)
        
        # パッチを準備（透明な大きなピクセルのコピー）
        patches = big_pixels.copy().set_fill(opacity=0)
        
        # 各ピクセルを最も近いパッチに割り当て
        p_points = np.array([p.get_center() for p in self.pixels])
        bp_points = np.array([bp.get_center() for bp in big_pixels])
        
        for pixel in self.pixels:
            # 各ピクセルから全パッチまでの距離を計算
            dists = np.linalg.norm(bp_points - pixel.get_center(), axis=1)
            # 最も近いパッチにピクセルを追加
            patches[np.argmin(dists)].add(pixel)
        
        # パッチを画像の中心に配置
        patches.move_to(self.original_image.get_center())
        
        return patches
    
    def convert_to_patches(self, **anim_kwargs) -> Animation:
        """画像をパッチに変換するアニメーションを返す。"""
        if self.is_patched:
            return Wait(0)  # 既にパッチ化されている場合は何もしない
        
        # パッチを作成
        self.patches = self._create_patches()
        
        # ピクセルをGroupに追加（パッチ化後もspace_out_pixelsが使えるように）
        self.add(self.pixels)
        
        # パッチ化フラグを設定
        self.is_patched = True
        
        # アニメーション設定のデフォルト値
        anim_config = {"run_time": 1}
        anim_config.update(anim_kwargs)
        
        # パッチをフェードインし、元の画像をフェードアウト
        return AnimationGroup(
            FadeIn(self.patches, **anim_config),
            FadeOut(self.original_image, **anim_config),
        )
    
    def get_patch_count(self) -> int:
        """パッチの総数を返す。"""
        if self.is_patched and self.patches is not None:
            return len(self.patches)
        return 0
    
    def get_individual_patch(self, index: int) -> VGroup:
        """指定されたインデックスのパッチを取得する。"""
        if not self.is_patched or self.patches is None:
            raise ValueError("convert_to_patches()を先に実行してください")
        
        if not 0 <= index < len(self.patches):
            raise IndexError(f"パッチインデックス {index} は範囲外です（0-{len(self.patches)-1}）")
        
        return self.patches[index]


class PatchedImageAnimator:
    """PatchedImageのアニメーション機能を提供するクラス"""
    
    def __init__(self, patched_image: PatchedImage):
        self.patched_image = patched_image
    
    def space_out_patches(
        self,
        factor: float = 2.0,
        scale: float = 0.75,
        **anim_kwargs
    ) -> Animation:
        """パッチ間の間隔を広げるアニメーションを返す。
        
        Args:
            factor (float): 間隔の倍率（デフォルト: 2.0）
            scale (float): 全体のスケール（デフォルト: 0.75）
            **anim_kwargs: アニメーションの追加引数
        
        Returns:
            Animation: 間隔調整アニメーション
        
        Raises:
            ValueError: パッチ化されていない場合
        """
        if not self.patched_image.is_patched or self.patched_image.patches is None:
            raise ValueError("convert_to_patches()を先に実行してください")
        
        # アニメーション設定のデフォルト値
        anim_config = {"run_time": 1}
        anim_config.update(anim_kwargs)
        
        return self.patched_image.patches.animate(**anim_config).space_out_submobjects(factor).scale(scale)
    
    def space_out_pixels(
        self,
        factor: float = 1.2,
        scale: float = 0.9,
        **anim_kwargs
    ) -> Animation:
        """PatchedImage内の全ピクセル間隔を広げるアニメーションを返す。
        
        Args:
            factor (float): ピクセル間隔の倍率（デフォルト: 1.2）
            scale (float): 全体のスケール（デフォルト: 0.9）
            **anim_kwargs: アニメーションの追加引数
        
        Returns:
            Animation: ピクセル間隔調整アニメーション
        """
        # ピクセルが存在しない場合は作成する
        if self.patched_image.pixels is None:
            pixel_width = self.patched_image.original_image.width / self.patched_image.n_divisions
            self.patched_image.pixels = create_pixels(self.patched_image.original_image, pixel_width=pixel_width)
        
        # アニメーション設定のデフォルト値
        anim_config = {"run_time": 1.5}
        anim_config.update(anim_kwargs)
        
        return self.patched_image.pixels.animate(**anim_config).space_out_submobjects(factor).scale(scale)
    
    def patch_flash(
        self, 
        stroke_color=TEAL, 
        stroke_width: float = 3, 
        lag_ratio: Optional[float] = None,
        **anim_kwargs
    ) -> Animation:
        """各パッチを順次フラッシュするアニメーションを返す。
        
        Args:
            stroke_color: フラッシュ時の枠線色（デフォルト: TEAL）
            stroke_width (float): 枠線の太さ（デフォルト: 3）
            lag_ratio (Optional[float]): パッチ間の遅延比率
            **anim_kwargs: アニメーションの追加引数
        
        Returns:
            Animation: パッチフラッシュアニメーション
        """
        if not self.patched_image.is_patched or self.patched_image.patches is None:
            raise ValueError("convert_to_patches()を先に実行してください")
        
        # lag_ratioの自動計算
        if lag_ratio is None:
            lag_ratio = 5.0 / len(self.patched_image.patches)
        
        # アニメーション設定のデフォルト値
        anim_config = {"run_time": 2}
        anim_config.update(anim_kwargs)
        
        # 各パッチのフラッシュアニメーション
        patch_animations = (
            patch.animate(rate_func=there_and_back).set_stroke(stroke_color, stroke_width)
            for patch in self.patched_image.patches
        )
        
        return LaggedStart(
            *patch_animations,
            lag_ratio=lag_ratio,
            **anim_config
        )

class TokenizedWaveform(VGroup):
    """音声波形をトークン化できるMobjectクラス。
    
    指定された関数に基づいて波形を生成し、チャンクに分割してトークン化を
    可視化する機能を提供します。音声処理でのトークン化概念の理解に使用されます。
    
    Attributes:
        n_lines (int): 波形を構成する線の数
        chunk_size (int): 各チャンクの線の数
        wave_function (callable): 波形生成関数
        waveform_lines (VGroup): 波形を構成する線のグループ
        chunks (VGroup): チャンクのグループ
        is_chunked (bool): チャンク化されているかどうかのフラグ
        
    Examples:
        >>> # 基本的な使用法
        >>> def my_wave_func(x):
        ...     return math.sin(x) + 0.3 * math.sin(3 * x)
        >>> 
        >>> waveform = TokenizedWaveform(
        ...     wave_function=my_wave_func,
        ...     n_lines=120,
        ...     chunk_size=6,
        ...     width=6
        ... )
        >>> scene.add(waveform)
        >>> scene.play(waveform.tokenize())
        >>> scene.play(waveform.space_out_chunks(factor=1.8))
        >>> scene.play(waveform.chunk_flash())
    """
    
    def __init__(
        self,
        wave_function: Optional[callable] = None,
        n_lines: int = 100,
        chunk_size: int = 5,
        width: float = 5.0,
        line_color: str = WHITE,
        **kwargs
    ):
        """TokenizedWaveformの初期化。
        
        Args:
            wave_function (Optional[callable]): 波形生成関数 f(x) -> float
                                              Noneの場合はデフォルトの複合波形
            n_lines (int): 波形を構成する線の数（デフォルト: 100）
            chunk_size (int): 各チャンクの線の数（デフォルト: 5）
            width (float): 波形全体の幅（デフォルト: 5.0）
            line_color: 線の色（デフォルト: WHITE）
            **kwargs: VGroupの追加引数
        """
        super().__init__(**kwargs)
        
        # 設定値の保存
        self.n_lines = n_lines
        self.chunk_size = chunk_size
        self.wave_function = wave_function or self._default_wave_function
        self.is_chunked = False
        
        # 波形の線を作成
        self.waveform_lines = self._create_waveform_lines(width, line_color)
        self.add(self.waveform_lines)
        
        # チャンク関連の属性を初期化
        self.chunks = None
    
    def _default_wave_function(self, x: float) -> float:
        """デフォルトの波形生成関数。
        
        複数の正弦波を合成した複雑な波形を生成します。
        
        Args:
            x (float): 入力値
            
        Returns:
            float: 波形の振幅値
        """
        x *= 1.7
        return sum([
            math.sin(x),
            0.5 * math.sin(2 * x),
            0.3 * math.sin(3 * x),
            0.2 * math.sin(4 * x),
            0.1 * math.sin(5 * x),
            0.15 * math.sin(6 * x),
        ])
    
    def _create_waveform_lines(self, width: float, color: str) -> VGroup:
        """波形を構成する線のグループを作成する。
        
        Args:
            width (float): 波形全体の幅
            color: 線の色
            
        Returns:
            VGroup: 波形線のグループ
        """
        # 基本線を作成
        base_line = Line(UP, DOWN)
        lines = VGroup(*[base_line.copy() for _ in range(self.n_lines)])
        
        # 線を水平に配置
        lines.arrange(RIGHT)
        lines.set_width(width)
        lines.set_color(color)
        
        # 各線の高さを波形関数に基づいて設定
        for line in lines:
            amplitude = abs(self.wave_function(line.get_x()))
            line.set_height(amplitude)
        
        # 波形を中央に配置
        lines.center()
        
        return lines
    
    def tokenize(self, **anim_kwargs) -> Animation:
        """波形をチャンクに分割（トークン化）するアニメーションを返す。
        
        Args:
            **anim_kwargs: アニメーションの追加引数（run_time等）
            
        Returns:
            Animation: トークン化アニメーション
            
        Note:
            このメソッド実行後、is_chunkedがTrueになります。
        """
        if self.is_chunked:
            return Wait(0)  # 既にチャンク化されている場合
        
        # チャンクを作成
        self.chunks = VGroup(
            self.waveform_lines[i:i + self.chunk_size] 
            for i in range(0, len(self.waveform_lines), self.chunk_size)
        )
        
        # チャンク化フラグを設定
        self.is_chunked = True
        
        # アニメーション設定
        anim_config = {"run_time": 1}
        anim_config.update(anim_kwargs)
        
        # 視覚的な変化を示すため、チャンクを少し強調
        return AnimationGroup(
            *[
                chunk.animate(**anim_config).set_stroke(opacity=0.8)
                for chunk in self.chunks
            ]
        )
    
    def space_out_chunks(
        self,
        factor: float = 2.0,
        scale: float = 0.75,
        **anim_kwargs
    ) -> Animation:
        """チャンク間の間隔を広げるアニメーションを返す。
        
        Args:
            factor (float): 間隔の倍率（デフォルト: 2.0）
            scale (float): 全体のスケール（デフォルト: 0.75）
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: 間隔調整アニメーション
            
        Raises:
            ValueError: トークン化されていない場合
        """
        if not self.is_chunked or self.chunks is None:
            raise ValueError("tokenize()を先に実行してください")
        
        # アニメーション設定
        anim_config = {"run_time": 1.5}
        anim_config.update(anim_kwargs)
        
        return self.chunks.animate(**anim_config).space_out_submobjects(factor).scale(scale)
    
    def chunk_flash(
        self,
        stroke_color=TEAL,
        stroke_width: float = 3,
        scale_factor: float = 1.5,
        lag_ratio: Optional[float] = None,
        **anim_kwargs
    ) -> Animation:
        """各チャンクを順次フラッシュするアニメーションを返す。
        
        Args:
            stroke_color: フラッシュ時の線の色（デフォルト: TEAL）
            stroke_width (float): フラッシュ時の線の太さ（デフォルト: 3）
            scale_factor (float): フラッシュ時のスケール倍率（デフォルト: 1.5）
            lag_ratio (Optional[float]): チャンク間の遅延比率。
                                        Noneの場合は自動計算
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: チャンクフラッシュアニメーション
            
        Raises:
            ValueError: トークン化されていない場合
            
        Examples:
            >>> # 基本的なフラッシュ
            >>> scene.play(waveform.chunk_flash())
            >>> 
            >>> # カスタマイズされたフラッシュ
            >>> scene.play(waveform.chunk_flash(
            ...     stroke_color=RED,
            ...     stroke_width=4,
            ...     scale_factor=2.0,
            ...     run_time=3
            ... ))
        """
        if not self.is_chunked or self.chunks is None:
            raise ValueError("tokenize()を先に実行してください")
        
        # lag_ratioの自動計算
        if lag_ratio is None:
            lag_ratio = 2.0 / len(self.chunks)
        
        # アニメーション設定
        anim_config = {"run_time": 2}
        anim_config.update(anim_kwargs)
        
        # 各チャンクのフラッシュアニメーション
        chunk_animations = (
            chunk.animate(rate_func=there_and_back)
            .set_stroke(stroke_color, stroke_width)
            .scale(scale_factor)
            for chunk in self.chunks
        )
        
        return LaggedStart(
            *chunk_animations,
            lag_ratio=lag_ratio,
            **anim_config
        )
    
    def get_chunk_count(self) -> int:
        """チャンクの総数を返す。
        
        Returns:
            int: チャンク数（トークン化されていない場合は0）
        """
        if self.is_chunked and self.chunks is not None:
            return len(self.chunks)
        return 0
    
    def get_individual_chunk(self, index: int) -> VGroup:
        """指定されたインデックスのチャンクを取得する。
        
        Args:
            index (int): チャンクのインデックス
            
        Returns:
            VGroup: 指定されたチャンク
            
        Raises:
            ValueError: トークン化されていない場合
            IndexError: インデックスが範囲外の場合
        """
        if not self.is_chunked or self.chunks is None:
            raise ValueError("tokenize()を先に実行してください")
        
        if not 0 <= index < len(self.chunks):
            raise IndexError(f"チャンクインデックス {index} は範囲外です（0-{len(self.chunks)-1}）")
        
        return self.chunks[index]
    
    def update_wave_function(
        self,
        new_function: callable,
        **anim_kwargs
    ) -> Animation:
        """波形関数を変更するアニメーションを返す。
        
        Args:
            new_function (callable): 新しい波形生成関数
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: 波形変更アニメーション
        """
        self.wave_function = new_function
        
        # 新しい波形の高さを計算
        new_heights = []
        for line in self.waveform_lines:
            amplitude = abs(self.wave_function(line.get_x()))
            new_heights.append(amplitude)
        
        # アニメーション設定
        anim_config = {"run_time": 1.5}
        anim_config.update(anim_kwargs)
        
        # 各線の高さを新しい値にアニメーション
        animations = []
        for line, new_height in zip(self.waveform_lines, new_heights):
            animations.append(
                line.animate(**anim_config).set_height(new_height)
            )
        
        return AnimationGroup(*animations)
    
    def reset_to_waveform(self, **anim_kwargs) -> Animation:
        """チャンク化された状態から元の波形に戻すアニメーションを返す。
        
        Args:
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: 波形復元アニメーション
        """
        if not self.is_chunked:
            return Wait(0)  # チャンク化されていない場合
        
        # フラグをリセット
        self.is_chunked = False
        
        # アニメーション設定
        anim_config = {"run_time": 1}
        anim_config.update(anim_kwargs)
        
        # 元の波形に戻す
        return self.waveform_lines.animate(**anim_config).arrange(RIGHT).center()


class TokenizedText(VGroup):
    """テキストをトークン化できるMobjectクラス。
    
    指定されたテキスト文字列をTextオブジェクトとして保存し、
    トークン化してVGroupに変換する機能を提供します。
    自然言語処理でのトークン化概念の可視化に使用されます。
    
    Attributes:
        original_text (str): 元のテキスト文字列
        text_object (Text): 元のTextオブジェクト
        tokens (VGroup): トークン化されたTextオブジェクトのグループ
        rects (VGroup): トークンを囲む矩形のグループ
        is_tokenized (bool): トークン化されているかどうかのフラグ
        font_size (float): フォントサイズ
        auto_font (bool): 自動フォント選択の有効/無効
        
    Examples:
        >>> # 基本的な使用法
        >>> tokenized_text = TokenizedText(
        ...     "私はAIアシスタントです。",
        ...     font_size=36
        ... )
        >>> scene.add(tokenized_text)
        >>> scene.play(tokenized_text.tokenize())
        >>> scene.play(tokenized_text.get_rect())
    """
    
    def __init__(
        self,
        text: str,
        font_size: float = 36,
        auto_font: bool = True,
        **kwargs
    ):
        """TokenizedTextの初期化。
        
        Args:
            text (str): トークン化対象のテキスト
            font_size (float): フォントサイズ（デフォルト: 36）
            auto_font (bool): 自動フォント選択を有効にするかどうか（デフォルト: True）
            **kwargs: VGroupの追加引数
        """
        super().__init__(**kwargs)
        
        # 設定値の保存
        self.original_text = text
        self.font_size = font_size
        self.auto_font = auto_font
        self.is_tokenized = False
        
        # テキストオブジェクトの作成
        if auto_font:
            font = get_appropriate_font(text)
            if font is not None:
                self.text_object = Text(text, font_size=font_size, font=font)
            else:
                self.text_object = Text(text, font_size=font_size)
        else:
            self.text_object = Text(text, font_size=font_size)
        
        # 初期状態では元のテキストのみを表示
        self.add(self.text_object)
        
        # トークン関連の属性を初期化
        self.tokens = None
        self.rects = None
    
    def tokenize(self, **anim_kwargs) -> Animation:
        # TODO: Text->Tokenへの滑らかな変化を実装したい
        """テキストをトークンに分割するアニメーションを返す。
        
        元のTextオブジェクトをトークン化されたVGroupに変換します。
        
        Args:
            **anim_kwargs: アニメーションの追加引数（run_time等）
            
        Returns:
            Animation: トークン化アニメーション
            
        Note:
            このメソッド実行後、is_tokenizedがTrueになります。
        """
        if self.is_tokenized:
            return Wait(0)  # 既にトークン化されている場合
        
        # トークン化を実行
        self.tokens = break_into_tokens(self.text_object, self.auto_font)
        
        # トークンを元のテキストと同じ位置に配置
        self.tokens.move_to(self.text_object.get_center())
        
        # トークン化フラグを設定
        self.is_tokenized = True
        
        # アニメーション設定
        anim_config = {"run_time": 1.5}
        anim_config.update(anim_kwargs)
        
        # 元のテキストからトークンVGroupへの変換アニメーション
        return AnimationGroup(
            FadeOut(self.text_object, **anim_config),
            FadeIn(self.tokens, **anim_config),
        )
    
    def get_rect(
        self,
        h_buff: float = 0.05,
        v_buff: float = 0.1,
        fill_opacity: float = 0.15,
        fill_color=None,
        stroke_width: float = 1,
        stroke_color=None,
        hue_range: tuple = (0.5, 0.7),
        **anim_kwargs
    ) -> Animation:
        """トークンを矩形で囲むアニメーションを返す。
        
        Args:
            h_buff (float): 水平方向の余白（デフォルト: 0.05）
            v_buff (float): 垂直方向の余白（デフォルト: 0.1）
            fill_opacity (float): 塗りつぶしの透明度（デフォルト: 0.15）
            fill_color: 塗りつぶし色（デフォルト: None、ランダム色）
            stroke_width (float): 枠線の太さ（デフォルト: 1）
            stroke_color: 枠線色（デフォルト: None、塗りつぶし色と同じ）
            hue_range (tuple): 色相範囲（デフォルト: (0.5, 0.7)、青系）
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: 矩形描画アニメーション
            
        Raises:
            ValueError: トークン化されていない場合
            
        Examples:
            >>> # 基本的な矩形描画
            >>> scene.play(tokenized_text.get_rect())
            >>> 
            >>> # カスタマイズされた矩形
            >>> scene.play(tokenized_text.get_rect(
            ...     fill_opacity=0.3,
            ...     stroke_width=2,
            ...     hue_range=(0.8, 1.0),  # 赤系
            ...     run_time=2
            ... ))
        """
        if not self.is_tokenized or self.tokens is None:
            raise ValueError("tokenize()を先に実行してください")
        
        # 矩形を作成
        self.rects = get_piece_rectangles(
            self.tokens,
            h_buff=h_buff,
            v_buff=v_buff,
            fill_opacity=fill_opacity,
            fill_color=fill_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            hue_range=hue_range
        )
        
        # アニメーション設定
        anim_config = {"run_time": 1}
        anim_config.update(anim_kwargs)
        
        return FadeIn(self.rects, **anim_config)
    
    def reset_to_text(self, **anim_kwargs) -> Animation:
        """トークン化された状態から元のテキストに戻すアニメーションを返す。
        
        Args:
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: テキスト復元アニメーション
        """
        if not self.is_tokenized:
            return Wait(0)  # トークン化されていない場合
        
        # フラグをリセット
        self.is_tokenized = False
        
        # アニメーション設定
        anim_config = {"run_time": 1}
        anim_config.update(anim_kwargs)
        
        # 矩形も存在する場合は一緒にフェードアウト
        fade_out_objects = [self.tokens]
        if self.rects is not None:
            fade_out_objects.append(self.rects)
            self.rects = None  # 矩形をリセット
        
        return AnimationGroup(
            *[FadeOut(obj, **anim_config) for obj in fade_out_objects],
            FadeIn(self.text_object, **anim_config),
        )
    
    def get_token_count(self) -> int:
        """トークンの総数を返す。
        
        Returns:
            int: トークン数（トークン化されていない場合は0）
        """
        if self.is_tokenized and self.tokens is not None:
            return len(self.tokens)
        return 0
    
    def get_individual_token(self, index: int) -> Text:
        """指定されたインデックスのトークンを取得する。
        
        Args:
            index (int): トークンのインデックス
            
        Returns:
            Text: 指定されたトークン
            
        Raises:
            ValueError: トークン化されていない場合
            IndexError: インデックスが範囲外の場合
        """
        if not self.is_tokenized or self.tokens is None:
            raise ValueError("tokenize()を先に実行してください")
        
        if not 0 <= index < len(self.tokens):
            raise IndexError(f"トークンインデックス {index} は範囲外です（0-{len(self.tokens)-1}）")
        
        return self.tokens[index]
    
    def highlight_token(
        self,
        index: int,
        color=RED,
        scale_factor: float = 1.2,
        **anim_kwargs
    ) -> Animation:
        """指定されたインデックスのトークンをハイライトするアニメーションを返す。
        
        Args:
            index (int): ハイライトするトークンのインデックス
            color: ハイライト色（デフォルト: RED）
            scale_factor (float): スケール倍率（デフォルト: 1.2）
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: ハイライトアニメーション
            
        Raises:
            ValueError: トークン化されていない場合
            IndexError: インデックスが範囲外の場合
        """
        token = self.get_individual_token(index)  # バリデーション含む
        
        # アニメーション設定
        anim_config = {"run_time": 0.8}
        anim_config.update(anim_kwargs)
        
        return token.animate(**anim_config).set_color(color).scale(scale_factor)
    
    def flash_tokens(
        self,
        color=YELLOW,
        scale_factor: float = 1.1,
        lag_ratio: Optional[float] = None,
        **anim_kwargs
    ) -> Animation:
        """全トークンを順次フラッシュするアニメーションを返す。
        
        Args:
            color: フラッシュ色（デフォルト: YELLOW）
            scale_factor (float): スケール倍率（デフォルト: 1.1）
            lag_ratio (Optional[float]): トークン間の遅延比率。
                                        Noneの場合は自動計算
            **anim_kwargs: アニメーションの追加引数
            
        Returns:
            Animation: フラッシュアニメーション
            
        Raises:
            ValueError: トークン化されていない場合
        """
        if not self.is_tokenized or self.tokens is None:
            raise ValueError("tokenize()を先に実行してください")
        
        # lag_ratioの自動計算
        if lag_ratio is None:
            lag_ratio = 3.0 / len(self.tokens)
        
        # アニメーション設定
        anim_config = {"run_time": 2}
        anim_config.update(anim_kwargs)
        
        # 各トークンのフラッシュアニメーション
        token_animations = (
            token.animate(rate_func=there_and_back)
            .set_color(color)
            .scale(scale_factor)
            for token in self.tokens
        )
        
        return LaggedStart(
            *token_animations,
            lag_ratio=lag_ratio,
            **anim_config
        )


# =========================
# ここからテスト
# =========================
class TestTokenRectScene(Scene):
    """トークン分割と矩形囲みのテストシーン。

    様々な言語のテキストに対してトークン分割を行い、
    各トークンを矩形で囲んで可視化する。
    """
    def construct(self):
        # タイトル
        title = Text("トークン分割と矩形囲みのテスト", font_size=36).to_edge(UP)
        self.play(Write(title))
        
        # テスト用の文章
        phrases = [
            "私はAIアシスタントです。",
            "Hello world!",
            "こんにちは、世界！",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        for i, phrase in enumerate(phrases):
            # 説明テキスト
            phrase_text = Text(f"文章 {i+1}: {phrase}", font_size=24).next_to(title, DOWN, buff=0.5)
            self.play(Write(phrase_text))
            
            # トークン分割
            phrase_mob = Text(phrase, font_size=36, font="Yu Gothic")
            tokens = break_into_tokens(phrase_mob, auto_font=True)
            tokens.next_to(phrase_text, DOWN, buff=0.5)
            self.play(FadeIn(tokens))
            
            # 矩形で囲む（青系の色相範囲を使用）
            rects = get_piece_rectangles(
                tokens, 
                stroke_width=2, 
                fill_opacity=0.2, 
                h_buff=0.05,
                v_buff=0.1,
                hue_range=(0.5, 0.7)  # 青系の色相範囲
            )
            self.play(FadeIn(rects))
            
            # 少し待機
            self.wait(2)
            
            # 次のテストの前にクリア
            if i < len(phrases) - 1:
                self.play(
                    FadeOut(phrase_text),
                    FadeOut(tokens),
                    FadeOut(rects)
                )
        
        # 最終的な説明
        final_text = Text("各トークンが青系の色で囲まれています", font_size=28).next_to(tokens, DOWN)
        self.play(Write(final_text))
        self.wait(3)




class AmbientWordEmbedding(Word2VecScene):
    """
    Word2Vecの埋め込みを可視化するシーン。
    """
    def construct(self):
        # Setup - ManimCEではカメラの向きを直接設定
        self.begin_ambient_camera_rotation()
        # self.wait()
        
        axes = self.axes
        axes.set_stroke(width=2)
        axes.scale_to_fit_height(7)
        axes.move_to(2*LEFT + 1.0 * IN)

        # Add titles
        titles = VGroup(Text("Words"), Text("Vectors"))
        colors = [YELLOW, BLUE]
        titles.scale_to_fit_height(0.5)
        xs = [-5.0, 5.0]
        for title, x, color in zip(titles, xs, colors):
            title.move_to(x * RIGHT)
            title.to_edge(UP, buff=1)
            title.add(Underline(title))
            title.set_color(color)

        arrow = Arrow(titles[0].get_right(), titles[1].get_left(), buff=0.3)

        # ManimCEではTexTextの代わりにTextを使用
        arrow_label = Text("Embedding")
        # グラデーション効果は手動で実装
        arrow_label.set_color_by_gradient(YELLOW, BLUE)
        arrow_label.next_to(arrow, UP, SMALL_BUFF)
        
        # ManimCEではadd_fixed_in_frame_mobjectsを使用
        self.add_fixed_in_frame_mobjects(titles, arrow, arrow_label)

        self.add(titles)
        self.add(arrow)

        # Add words
        words = "All data in deep learning must be represented as vectors".split(" ")
        pre_labels = VGroup(*(Text(word) for word in words))
        pre_labels.arrange(DOWN, aligned_edge=LEFT)
        pre_labels.next_to(titles[0], DOWN, buff=0)
        pre_labels.align_to(titles[0][0], LEFT)
        pre_labels.scale_to_fit_height(config.frame_height*0.65)
        # ManimCEでは背景ストロークの設定方法が変更されている
        for label in pre_labels:
            label.set_stroke(width=2, color=BLACK, background=True)
        
        # フレーム固定オブジェクトを追加
        self.add_fixed_in_frame_mobjects(pre_labels) 

        coords = np.array([
            self.basis @ self.model[word.lower()]
            for word in words
        ])
        coords -= coords.mean(0)
        max_coord = max(coords.max(), -coords.min())
        coords *= 4.0 / max_coord

        embeddings = VGroup(*(
            self.get_labeled_vector(
                word,
                coord,
                thickness=2,
                color=interpolate_color(BLUE_D, BLUE_A, random.random()),
                func_name=None,
                label_config=dict(font_size=30)
            )
            for word, coord in zip(words, coords)
        ))

        self.play(LaggedStartMap(FadeIn, pre_labels, shift=0.2 * UP, lag_ratio=0.1, run_time=1))

        self.play(Write(arrow_label), run_time=2)
        
        for label, vect in zip(pre_labels, embeddings):
            self.play(
                TransformFromCopy(label, vect.label, run_time=2),
                FadeIn(vect, run_time=1)
            )
            self.wait(0.5)
        self.wait()
