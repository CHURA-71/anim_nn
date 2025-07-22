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

from helpers import *

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

def get_word_to_vec_model(model_name="glove-wiki-gigaword-50"):
    """単語ベクトルモデルを取得またはダウンロードする。

    Args:
        model_name: モデル名（デフォルト: "glove-wiki-gigaword-50"）

    Returns:
        gensim.models.keyedvectors.KeyedVectors: 単語ベクトルモデル
    """
    filename = str(Path(DATA_DIR, model_name))
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if os.path.exists(filename):
        return gensim.models.keyedvectors.KeyedVectors.load(filename)
    model = gensim.downloader.load(model_name)
    # モデルがKeyedVectorsインスタンスの場合のみ保存
    if isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        model.save(filename)
    return model

def get_direction_lines(axes:ThreeDAxes, direction, n_lines=500, color=YELLOW, line_length=1.0, stroke_width=3):
    line = Line(ORIGIN, line_length * normalize(direction))
    line.insert_n_curves(20).set_stroke(width=(0, stroke_width, stroke_width, stroke_width, 0))
    lines = VGroup(*(line.copy() for _ in range(n_lines)))
    # lines = line.replicate(n_lines)
    lines.set_color(color)
    for line in lines:
        line.move_to(axes.c2p(
            random.uniform(*axes.x_range),
            random.uniform(*axes.y_range),
            random.uniform(*axes.z_range),  # pyright: ignore
        ))
    return lines


class Word2VecScene(ThreeDScene):
    """
        Word2Vecの可視化シーン。
        このシーンでは、単語ベクトルを3D空間にプロットし、
        各単語を矢印で表現します。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_frame_orientation = (-30, 70)

    axes_config = dict(
        x_range=(-5, 5, 1),
        y_range=(-5, 5, 1),
        z_range=(-4, 4, 1),
        x_length=8,
        y_length=8,
        z_length=6.4,
    )

    label_rotation = 0

    # embedding_model = "word2vec-google-news-300"
    embedding_model = "glove-wiki-gigaword-50"

    def setup(self):
        super().setup()

        # Load model
        self.model = get_word_to_vec_model(self.embedding_model)

        # Decide on basis
        self.basis = self.get_basis(self.model)

        # Add axes
        self.axes = ThreeDAxes(**self.axes_config)
        self.add(self.axes)
        
        # Set camera orientation
        self.set_camera_orientation(phi=self.default_frame_orientation[0], theta=self.default_frame_orientation[1])

    def get_basis(self, model):
        return get_principle_components(model.vectors, 3).T

    def add_plane(self, color=GREY, stroke_width=1.0):
        axes = self.axes
        plane = NumberPlane(
            x_range=axes.x_range,
            y_range=axes.y_range,
            width=axes.get_width(),
            height=axes.get_height(),
            background_line_style=dict(
                stroke_color=color,
                stroke_width=stroke_width,
            ),
            faded_line_style=dict(
                stroke_opacity=0.25,
                stroke_width=0.5 * stroke_width,
            ),
            faded_line_ratio=1,
        )
        self.plane = plane
        self.add(plane)
        return plane

    def get_labeled_vector(
        self,
        word,
        coords=None,
        thickness=5,
        color=YELLOW,
        func_name: str | None = "E",
        buff=0.05,
        direction=None,
        label_config: dict = dict()
    )-> TextLabeledArrow:

        axes = self.axes
        if coords is None:
            coords = self.basis @ self.model[word.lower()]
        point = axes.c2p(*coords)
        label_config.update(label_buff=buff)
        if "label_rotation" not in label_config:
            label_config.update(label_rotation=self.label_rotation)
        arrow = TextLabeledArrow(
            axes.get_origin(),
            point,
            stroke_width=thickness,
            stroke_color=color,
            label_text=word if func_name is None else f"{func_name}({word})",
            direction=direction,
            scene=self,
            **label_config,
        )
        return arrow

    def add_fixed_in_frame_mobjects(self, *mobjects):
        super().add_fixed_in_frame_mobjects(*mobjects)
        self.remove(*mobjects)  # シーンには追加せず、フレームに固定

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
        self.wait()
        
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
        pre_labels.scale_to_fit_height(config.frame_height*0.7)
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
