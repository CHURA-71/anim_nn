import os
from pathlib import Path
import hashlib
from manim import *
import random
from typing import Optional


def get_output_dir() -> str:
    """
    ManimCEでの出力ディレクトリを取得し、存在を保証する。
    """
    return config.media_dir



def random_bright_color_with_hue(hue_range: Optional[tuple[float, float]] = None) -> ManimColor:
    """指定された色相範囲内でランダムな明るい色を返す。

    HSV色空間で色を生成し、高彩度・高明度の明るい色を返す。
    hue_rangeがNoneの場合は全色相範囲から選択する。

    Args:
        hue_range: 色相の範囲 (0.0-1.0)。Noneの場合は全範囲から選択（デフォルト: None）

    Returns:
        ManimColor: ランダムな明るい色
    """
    if hue_range is None:
        # 元の動作：全範囲からランダム選択
        curr_rgb = color_to_rgb(random_color())
        new_rgb = 0.5 * (curr_rgb + np.ones(3))
        return ManimColor(new_rgb)
    else:
        # 指定された色相範囲内から選択
        min_hue, max_hue = hue_range
        hue = random.uniform(min_hue, max_hue)
        saturation = random.uniform(0.7, 1.0)  # 高彩度
        value = random.uniform(0.8, 1.0)       # 高明度
        
        # HSVからRGBに変換
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return ManimColor(rgb)


def random_bright_color_morewhite() -> ManimColor:
    """ランダムな明るい色を生成する（白に比重を置く）。

    random_bright_colorよりも白に近い色を生成する。
    RGB値を0.25倍して0.75の白を加えることで、
    より明るく白に近い色を作成する。

    Returns:
        ManimColor: 白に近いランダムな明るい色
    """
    curr_rgb = color_to_rgb(random_color())
    new_rgb = 0.25 * curr_rgb + 0.75 * np.ones(3)
    return ManimColor(new_rgb)