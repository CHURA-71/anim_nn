# PixelsAsSquare.py

from manim import *
import numpy as np
from sklearn.datasets import fetch_openml
import os
from PIL import Image


# ----------------------------------------------------------------------------
# 画像ファイルをNumPy配列に変換する関数
# ----------------------------------------------------------------------------
def image_to_array(image_path: str) -> np.ndarray:
    """
    画像ファイルを読み込み、NumPy配列に変換する。

    - 画像がグレースケールの場合、2D配列(height, width)を返す。
    - 画像がカラー(RGB/RGBA)の場合、3D配列(height, width, channels)を返す。

    Args:
        image_path (str): 画像ファイルのパス。

    Returns:
        np.ndarray: 画像データを格納したNumPy配列。

    Raises:
        FileNotFoundError: 指定されたパスにファイルが存在しない場合。
        IOError: ファイルが有効な画像でない場合。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"エラー: ファイル '{image_path}' が見つかりません。")

    try:
        with Image.open(image_path) as img:
            # 画像モードが'L'(グレースケール)または'1'(二値)の場合
            if img.mode in ['L', '1']:
                # 2D配列として返す
                return np.array(img)
            else:
                # その他のモード(RGB, RGBAなど)は3D配列として返す
                return np.array(img)
    except Exception as e:
        raise IOError(f"エラー: '{image_path}' を画像として読み込めませんでした。詳細: {e}")


# ----------------------------------------------------------------------------
#  NumPy配列をManimのMatrixに変換する関数
# ----------------------------------------------------------------------------

def array_to_matrix(
    array: np.ndarray,
    label: str,
    color: ManimColor,
    max_size: int = 16
) -> VGroup:
    """
    2D NumPy配列をManimのMatrixに変換する。
    サイズが大きい場合は、最後から2番目の行/列を省略記号で置き換える。
    """
    if array.ndim != 2:
        raise TypeError("array_to_matrixには2DのNumPy配列が必要です。")

    h, w = array.shape
    elements_str = array.astype(str)

    if h > max_size or w > max_size:
        new_size = max_size + 1  # 17
        dot_index = new_size - 2 # 15 (最後から2番目のインデックス)
        
        # 17x17の空の配列を作成
        final_elements = np.full((new_size, new_size), "", dtype=object)
        
        # --- 変更点：先に全ての数値を埋める ---
        # 左上 (15x15)
        final_elements[:dot_index, :dot_index] = elements_str[:dot_index, :dot_index]
        # 右上 (15x1)
        final_elements[:dot_index, -1] = elements_str[:dot_index, -1]
        # 左下 (1x15)
        final_elements[-1, :dot_index] = elements_str[-1, :dot_index]
        # 右下 (1x1)
        final_elements[-1, -1] = elements_str[-1, -1]
        
        # --- 変更点：最後の行/列の交差しない部分の数値を維持する ---
        # 最後から2番目の行の、最後の列の要素
        final_elements[dot_index, -1] = elements_str[dot_index, -1]
        # 最後の行の、最後から2番目の列の要素
        final_elements[-1, dot_index] = elements_str[-1, dot_index]

        # --- 変更点：その後、内側の部分だけを省略記号で上書きする ---
        # 最後から2番目の行を \vdots で埋める (最後の列を除く)
        final_elements[dot_index, :dot_index] = r'\vdots'
        # 最後から2番目の列を \cdots で埋める (最後の行を除く)
        final_elements[:dot_index, dot_index] = r'\cdots'
        # 交点を \ddots で置き換える
        final_elements[dot_index, dot_index] = r'\ddots'
        
        matrix_entries = final_elements.tolist()
    else:
        matrix_entries = elements_str.tolist()

    matrix = Matrix(
        matrix_entries,
        h_buff=1.2,
        v_buff=0.7,
        bracket_h_buff=0.4,
        bracket_v_buff=0.4,
        element_to_mobject_config={"font_size": 20}
    )

    matrix_label = Text(label, color=color, weight=BOLD).scale(0.8)
    matrix_label.next_to(matrix, UP, buff=0.3)

    return VGroup(matrix, matrix_label)

# ----------------------------------------------------------------------------
# グレースケール画像用クラス
# ----------------------------------------------------------------------------

class PixelsAsSquare(VGroup):
    """
    画像のピクセルデータ（2D NumPy配列）を受け取り、
    各ピクセルを色付けされたSquareとオプションのテキストで表現するVGroupを生成するクラス。

    使用例:
    image_array = np.random.rand(10, 10) * 255  # 10x10のランダムな画像データ
    pixel_grid = PixelsAsSquare(image_array)
    self.play(Create(pixel_grid))
    """
    def __init__(
        self,
        image_data: np.ndarray,
        show_values: bool = True,
        value_format_func=lambda v: f"{v/255.0:.1f}",
        font_size: int = 14,
        square_side_length: float = 1.0,
        stroke_width: float = 0.1,
        stroke_color: ManimColor = GRAY,
        contrast_threshold: float = 0.6,
        **kwargs
    ):
        """
        PixelsAsSquareのコンストラクタ

        Args:
            image_data (np.ndarray): グレースケール画像のピクセルデータ（2D配列）。
                                     値の範囲は0-255を想定。
            show_values (bool, optional): ピクセル値のテキストを表示するかどうか。デフォルトはTrue。
            value_format_func (function, optional): ピクセル値をフォーマットする関数。
                                                    デフォルトは0-1の範囲で小数点以下1桁に丸める。
            font_size (int, optional): ピクセル値テキストのフォントサイズ。デフォルトは14。
            square_side_length (float, optional): 各ピクセルを表す正方形の一辺の長さ。デフォルトは1.0。
            stroke_width (float, optional): 正方形の境界線の幅。デフォルトは0.1。
            stroke_color (ManimColor, optional): 正方形の境界線の色。デフォルトはGRAY。
            contrast_threshold (float, optional): テキストの色を白か黒かを切り替える輝度の閾値。
                                                  デフォルトは0.6。
        """
        super().__init__(**kwargs)
        self.font_size = font_size
        self.contrast_threshold = contrast_threshold


        if not isinstance(image_data, np.ndarray) or image_data.ndim != 2:
            raise TypeError("image_data must be a 2D NumPy array.")

        height, width = image_data.shape
        pixel_mobjects = []

        # Pixel×Pixelsのグリッドを作成
        for row_data in image_data:
            for pixel_value in row_data:
                # ピクセル値（0-255）を輝度（0.0-1.0）に正規化
                normalized_value = pixel_value / 255.0

                # Squareを作成し、輝度に応じたグレースケールで塗りつぶす
                square = Square(
                    side_length=square_side_length,
                    fill_opacity=1.0,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color
                )
                square.set_color(rgb_to_color([normalized_value] * 3))

                pixel_group = VGroup(square)

                # show_valuesがTrueの場合、ピクセル値を表示するTextを作成
                if show_values:
                    # 背景色に応じて文字色を白か黒に変更し、視認性を確保
                    text_color = BLACK if normalized_value > contrast_threshold else WHITE
                    text = Text(
                        value_format_func(pixel_value),
                        font_size=font_size,
                        color=text_color
                    )

                    # テキストがSquareに収まるようにスケーリング
                    if text.width > square.width * 0.9:
                        text.scale_to_fit_width(square.width * 0.9)

                    pixel_group.add(text)

                pixel_mobjects.append(pixel_group)

        # 作成したピクセルオブジェクトのリストからVGroupを作成し、グリッド状に配置
        self.add(*pixel_mobjects)
        self.arrange_in_grid(rows=height, cols=width, buff=0)


# ----------------------------------------------------------------------------
# カラー画像用クラス
# ----------------------------------------------------------------------------
class PixelsAsSquareColor(VGroup):
    """
    カラー画像のピクセルデータ（3D NumPy配列）を受け取り、
    各ピクセルを色付けされたSquareで表現するVGroupを生成するクラス。
    """
    def __init__(
        self,
        image_data: np.ndarray,
        square_side_length: float = 1.0,
        stroke_width: float = 0.1,
        stroke_color: ManimColor = GRAY,
        **kwargs
    ):
        """
        PixelsAsSquareColorのコンストラクタ

        Args:
            image_data (np.ndarray): カラー画像のピクセルデータ（3D配列, shape=(h, w, 3)）。
                                     値の範囲は0-255の整数を想定。
            square_side_length (float, optional): 各ピクセルを表す正方形の一辺の長さ。デフォルトは1.0。
            stroke_width (float, optional): 正方形の境界線の幅。デフォルトは0.1。
            stroke_color (ManimColor, optional): 正方形の境界線の色。デフォルトはGRAY。
        """
        super().__init__(**kwargs)

        # 入力データの形式を検証
        if not isinstance(image_data, np.ndarray) or image_data.ndim != 3:
            raise TypeError("image_data must be a 3D NumPy array (height, width, channels).")
        
        height, width, channels = image_data.shape
        if channels < 3:
            raise ValueError("image_data must have at least 3 channels (R, G, B).")

        pixel_mobjects = []

        # 画像データをループ処理
        for row_data in image_data:
            for pixel_rgb in row_data:
                # ピクセル値（0-255）をManimが使う色（0.0-1.0）に正規化
                normalized_rgb = pixel_rgb[:3] / 255.0
                
                # Squareを作成し、ピクセル色で塗りつぶす
                square = Square(
                    side_length=square_side_length,
                    fill_opacity=1.0,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color
                )
                square.set_color(rgb_to_color(normalized_rgb))
                
                pixel_mobjects.append(square)
        
        # 作成したピクセルオブジェクトのリストからVGroupを作成し、グリッド状に配置
        self.add(*pixel_mobjects)
        self.arrange_in_grid(rows=height, cols=width, buff=0)

# ----------------------------------------------------------------------------
# 畳み込み可視化クラス 
# ----------------------------------------------------------------------------
class Convolution(VGroup):
    """
    PixelsAsSquareオブジェクト上の畳み込み操作を可視化するクラス。
    """
    def __init__(
        self,
        pixel_grid: PixelsAsSquare,
        image_shape: tuple,
        filter_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        kernel_color: ManimColor = YELLOW,
        kernel_stroke_width: float = 3.0,
        kernel_fill_opacity: float = 0.3,
        **kwargs
    ):
        """
        Convolutionのコンストラクタ

        Args:
            pixel_grid (PixelsAsSquare): 畳み込み対象のピクセルグリッド。
            image_shape (tuple): 元の画像の形状 (height, width)。
            filter_size (tuple, optional): カーネル（フィルター）のサイズ (rows, cols)。デフォルトは (3, 3)。
            stride (tuple, optional): ストライド (row_stride, col_stride)。デフォルトは (1, 1)。
            kernel_color (ManimColor, optional): カーネル矩形の色。デフォルトは YELLOW。
            kernel_stroke_width (float, optional): カーネル矩形の線の太さ。デフォルトは 3.0。
            kernel_fill_opacity (float, optional): カーネル矩形の塗りつぶしの透明度。デフォルトは 0.3。
        """
        super().__init__(**kwargs)
        self.pixel_grid = pixel_grid
        self.image_h, self.image_w = image_shape
        self.filter_h, self.filter_w = filter_size
        self.stride_h, self.stride_w = stride

        if not self.pixel_grid:
            raise ValueError("pixel_grid cannot be empty.")

        # ピクセルグリッド内の最初のSquareから一辺の長さを取得
        # pixel_gridはVGroupで、その要素もVGroup(Square+Text)なので、[0][0]でSquareにアクセス
        first_square = self.pixel_grid[0][0]
        square_side = first_square.width

        # カーネルのサイズを計算
        kernel_height = self.filter_h * square_side
        kernel_width = self.filter_w * square_side

        # カーネルを表現するRectangleを作成
        self.kernel = Rectangle(
            height=kernel_height,
            width=kernel_width,
            color=kernel_color,
            stroke_width=kernel_stroke_width,
            fill_opacity=kernel_fill_opacity
        )
        
        # VGroupにカーネルを追加
        self.add(self.kernel)
        
        # カーネルの初期位置を計算
        self.set_kernel_to_starting_position()

    def set_kernel_to_starting_position(self):
        """カーネルを開始位置（左上）に配置する。"""
        # 最初のカーネル適用領域の中心に移動
        target_center = self.get_kernel_target_center(0, 0)
        self.kernel.move_to(target_center)

    def get_kernel_target_center(self, step_row: int, step_col: int) -> np.ndarray:
        """
        指定されたステップにおけるカーネルの中心位置を計算する。

        Args:
            step_row (int): 垂直方向のステップインデックス。
            step_col (int): 水平方向のステップインデックス。

        Returns:
            np.ndarray: カーネルが移動すべき中心座標。
        """
        # カーネルがカバーする領域の左上のピクセルインデックス
        start_row = step_row * self.stride_h
        start_col = step_col * self.stride_w

        # カーネルがカバーする領域の右下のピクセルインデックス
        end_row = start_row + self.filter_h - 1
        end_col = start_col + self.filter_w - 1
        
        # インデックスが画像サイズを超えないようにクリップ
        end_row = min(end_row, self.image_h - 1)
        end_col = min(end_col, self.image_w - 1)

        # 対応するピクセルMobjectを取得
        top_left_pixel = self.pixel_grid[start_row * self.image_w + start_col]
        bottom_right_pixel = self.pixel_grid[end_row * self.image_w + end_col]

        # 2つのピクセルの中心の中点を計算
        target_center = (top_left_pixel.get_center() + bottom_right_pixel.get_center()) / 2
        return target_center
    
    def get_convolution_animation(
            self,
            run_time_per_step: float = 0.2,
            move_proportion: float = 0.7,
            transition_run_time: float = 0.3
        ) -> Animation:
        """
        畳み込みの全経路をたどるアニメーションを生成する。
        行内はステップ感のある動き、行の切り返しは滑らかな動きになるように、
        複数のアニメーションをSuccessionで結合する。

        Args:
            run_time_per_step (float): 各移動ステップにかける平均時間。
            move_proportion (float): 各ステップ時間のうち、移動に使う時間の割合。
            transition_run_time (float): 行を切り返す際の滑らかな移動にかける時間。

        Returns:
            Succession: 複数のアニメーションを結合した単一のアニメーションオブジェクト。
        """
        path_points = []
        output_h = (self.image_h - self.filter_h) // self.stride_h + 1
        output_w = (self.image_w - self.filter_w) // self.stride_w + 1

        # ステップがない場合は何もしない
        if output_w <= 0 or output_h <= 0:
            return Wait(0)

        # 全ての経由点を計算
        for i in range(output_h):
            for j in range(output_w):
                target_center = self.get_kernel_target_center(i, j)
                path_points.append(target_center)

        # アニメーションのパーツを格納するリスト
        animation_parts = []

        # 行ごとにアニメーションを作成
        for i in range(output_h):
            # 現在の行の経由点をスライス
            start_index = i * output_w
            end_index = start_index + output_w
            row_points = path_points[start_index:end_index]

            # 1. 行内のステップ移動アニメーション
            if len(row_points) > 1:
                row_path = VMobject().set_points_as_corners(row_points)
                num_row_steps = len(row_points) - 1
                row_run_time = num_row_steps * run_time_per_step
                
                stepped_rate = create_stepped_rate_func(
                    num_steps=num_row_steps,
                    move_proportion=move_proportion
                )
                
                row_animation = MoveAlongPath(
                    self.kernel,
                    row_path,
                    run_time=row_run_time,
                    rate_func=stepped_rate
                )
                animation_parts.append(row_animation)

            # 2. 次の行への滑らかな切り返しアニメーション (最後の行を除く)
            if i < output_h - 1:
                current_row_end_point = row_points[-1]
                next_row_start_point = path_points[end_index]
                
                transition_path = Line(current_row_end_point, next_row_start_point)
                
                transition_animation = MoveAlongPath(
                    self.kernel, 
                    transition_path,
                    run_time=transition_run_time,
                    rate_func=smooth # 切り返しは滑らかに
                )
                animation_parts.append(transition_animation)

        if not animation_parts:
            return Wait(0)

        # 全てのアニメーションパーツをSuccessionで結合して返す
        return Succession(*animation_parts)

# ----------------------------------------------------------------------------
# 畳み込み計算と可視化を行うクラス (新規作成)
# ----------------------------------------------------------------------------
# immersive/convolution_calculator の CalcConv クラスを以下に置き換えてください

# class CalcConv(Convolution):
#     """
#     畳み込み計算を実行し、その過程をアニメーション化するクラス。
#     ValueTrackerとUpdaterを使い、動的な状態変化を堅牢に扱う。
#     """
#     def __init__(
#         self,
#         pixel_grid: PixelsAsSquare,
#         image_data: np.ndarray,
#         kernel_weights: np.ndarray,
#         feature_map_grid: PixelsAsSquare,
#         **kwargs
#     ):
#         # 親クラスの初期化
#         super().__init__(
#             pixel_grid=pixel_grid,
#             image_shape=image_data.shape,
#             filter_size=kernel_weights.shape,
#             **kwargs
#         )
        
#         self.image_data = image_data
#         self.kernel_weights = kernel_weights
#         self.feature_map_grid = feature_map_grid

#         # 出力サイズの計算
#         self.output_h = (self.image_h - self.filter_h) // self.stride_h + 1
#         self.output_w = (self.image_w - self.filter_w) // self.stride_w + 1

#         # アニメーションの前に全計算を実行し、値の範囲を把握
#         self.full_output_data = self._calculate_all()
#         self.min_val = self.full_output_data.min()
#         self.max_val = self.full_output_data.max()
#         if self.min_val == self.max_val:
#             self.max_val += 1e-6 # ゼロ除算を避ける

#         # --- ValueTrackerとUpdaterの設定 ---
#         # 各ピクセルの値を追跡するValueTrackerを作成
#         self.value_trackers = [ValueTracker(0) for _ in range(self.output_h * self.output_w)]
        
#         # 各ピクセルに、対応するValueTrackerを監視するUpdaterを追加
#         for i in range(self.output_h * self.output_w):
#             # この関数で、pixel_groupが自身のValueTrackerを監視し続けるようになる
#             self._add_updater_to_pixel(self.feature_map_grid[i], self.value_trackers[i])

#     def _add_updater_to_pixel(self, pixel_group, value_tracker):
#         """指定されたpixel_groupに、ValueTrackerを監視するUpdaterを追加する"""
#         square = pixel_group[0]
#         text = pixel_group[1] if len(pixel_group) > 1 else None

#         def updater(mob):
#             # ValueTrackerから現在の値を取得
#             value = value_tracker.get_value()
            
#             # 値を0-1に正規化して色を決定
#             normalized_value = (value - self.min_val) / (self.max_val - self.min_val)
#             normalized_value = np.clip(normalized_value, 0, 1)

#             # Squareの色を更新
#             square.set_color(interpolate_color(BLACK, WHITE, normalized_value))

#             # テキストがあれば、テキストの内容と色を更新
#             if text:
#                 # 新しい状態を持つテキストオブジェクトを作成
#                 new_text_state = Text(
#                     f"{value:.0f}",
#                     font_size=self.feature_map_grid.font_size,
#                     color=BLACK if normalized_value > self.feature_map_grid.contrast_threshold else WHITE
#                 ).move_to(text.get_center())
                
#                 # テキストがはみ出ないように調整
#                 if new_text_state.width > square.width * 0.9:
#                     new_text_state.scale_to_fit_width(square.width * 0.9)
                
#                 # becomeを使って、既存のテキストを新しい状態に滑らかに変化させる
#                 text.become(new_text_state)

#         # 作成したupdaterをpixel_groupに永続的に追加
#         pixel_group.add_updater(updater)

#     def _calculate_all(self) -> np.ndarray:
#         """畳み込み計算をすべて実行し、結果の配列を返す。（変更なし）"""
#         output = np.zeros((self.output_h, self.output_w))
#         for i in range(self.output_h):
#             for j in range(self.output_w):
#                 start_row = i * self.stride_h
#                 start_col = j * self.stride_w
#                 image_slice = self.image_data[
#                     start_row : start_row + self.filter_h,
#                     start_col : start_col + self.filter_w
#                 ]
#                 output[i, j] = np.sum(image_slice * self.kernel_weights)
#         return output

#     def get_calculation_animation(self, run_time_per_step: float = 0.5) -> Animation:
#         """
#         畳み込み計算の全ステップをアニメーション化する。
#         カーネル移動とValueTrackerの更新を同期させる。
#         """
#         animations = []
        
#         for i in range(self.output_h):
#             for j in range(self.output_w):
#                 # 1. カーネルの移動アニメーション
#                 target_center = self.get_kernel_target_center(i, j)
#                 move_anim = self.kernel.animate.move_to(target_center)
                
#                 # 2. 対応するValueTrackerの値を更新するアニメーション
#                 feature_map_index = i * self.output_w + j
#                 result_value = self.full_output_data[i, j]
                
#                 value_tracker = self.value_trackers[feature_map_index]
#                 update_anim = value_tracker.animate.set_value(result_value)
                
#                 # 3. 更新されたピクセルをハイライトするアニメーション
#                 indicate_anim = Indicate(self.feature_map_grid[feature_map_index], scale_factor=1.2, color=YELLOW)

#                 # 移動、値の更新、ハイライトを同時に実行
#                 animations.append(AnimationGroup(move_anim, update_anim, indicate_anim, lag_ratio=0.1))

#         return Succession(*animations, lag_ratio=1.0)
# immersive/convolution_calculator の CalcConv クラスを以下に置き換えてください

# immersive/convolution_calculator の CalcConv クラスを以下に置き換えてください

# immersive/convolution_calculator の CalcConv クラスを以下に置き換えてください

class CalcConv(Convolution):
    """
    畳み込み計算を実行し、その過程をアニメーション化するクラス。
    ValueTrackerと効率的なUpdaterを使い、パフォーマンスと確実性を両立する。
    """
    def __init__(
        self,
        pixel_grid: PixelsAsSquare,
        image_data: np.ndarray,
        kernel_weights: np.ndarray,
        feature_map_grid: PixelsAsSquare,
        **kwargs
    ):
        super().__init__(
            pixel_grid=pixel_grid, image_shape=image_data.shape,
            filter_size=kernel_weights.shape, **kwargs
        )
        self.image_data = image_data
        self.kernel_weights = kernel_weights
        self.feature_map_grid = feature_map_grid
        self.output_h = (self.image_h - self.filter_h) // self.stride_h + 1
        self.output_w = (self.image_w - self.filter_w) // self.stride_w + 1
        self.full_output_data = self._calculate_all()
        self.min_val = self.full_output_data.min()
        self.max_val = self.full_output_data.max()
        if self.min_val == self.max_val:
            self.max_val += 1e-6
        
        # --- ValueTrackerとUpdaterの設定 ---
        self.value_trackers = [ValueTracker(0) for _ in range(self.output_h * self.output_w)]
        for i in range(self.output_h * self.output_w):
            self._add_updater_to_pixel(self.feature_map_grid[i], self.value_trackers[i])

    def _add_updater_to_pixel(self, pixel_group, value_tracker):
        """指定されたpixel_groupに、ValueTrackerを監視する効率的なUpdaterを追加する"""
        square = pixel_group[0]
        text = pixel_group[1] if len(pixel_group) > 1 else None
        current_text_value = float('-inf')

        def updater(mob):
            nonlocal current_text_value
            value = value_tracker.get_value()
            normalized_value = np.clip((value - self.min_val) / (self.max_val - self.min_val), 0, 1)
            square.set_color(interpolate_color(BLACK, WHITE, normalized_value))
            if text and value != current_text_value:
                current_text_value = value
                new_text_state = Text(
                    f"{value:.0f}", font_size=self.feature_map_grid.font_size,
                    color=BLACK if normalized_value > self.feature_map_grid.contrast_threshold else WHITE
                ).move_to(text.get_center())
                if new_text_state.width > square.width * 0.9:
                    new_text_state.scale_to_fit_width(square.width * 0.9)
                text.become(new_text_state)
        pixel_group.add_updater(updater)

    def _calculate_all(self) -> np.ndarray:
        """畳み込み計算をすべて実行し、結果の配列を返す。"""
        output = np.zeros((self.output_h, self.output_w))
        for i in range(self.output_h):
            for j in range(self.output_w):
                start_row, start_col = i * self.stride_h, j * self.stride_w
                image_slice = self.image_data[
                    start_row : start_row + self.filter_h,
                    start_col : start_col + self.filter_w
                ]
                output[i, j] = np.sum(image_slice * self.kernel_weights)
        return output

    def get_calculation_animation(self, run_time_per_step: float = 0.3) -> Animation:
        """
        畳み込み計算の全ステップをアニメーション化する。
        カーネル移動とValueTrackerの更新を同期させる。
        """
        animations = []
        for i in range(self.output_h):
            for j in range(self.output_w):
                move_anim = self.kernel.animate.move_to(self.get_kernel_target_center(i, j))
                index = i * self.output_w + j
                value = self.full_output_data[i, j]
                tracker_anim = self.value_trackers[index].animate.set_value(value)
                indicate_anim = Indicate(self.feature_map_grid[index], scale_factor=1.2, color=YELLOW)
                animations.append(AnimationGroup(move_anim, tracker_anim, indicate_anim, lag_ratio=0.1))
        return Succession(*animations, lag_ratio=1.0)

# ステップ感のある動きを実現するためのカスタムレート関数を生成するヘルパー関数
def create_stepped_rate_func(num_steps: int, move_proportion: float = 0.7):
    """
    移動と静止を繰り返す、階段状のカスタムレート関数を生成する。

    Args:
        num_steps (int): アニメーション全体のステップ数。
        move_proportion (float): 各ステップ時間のうち、移動に使う時間の割合 (0から1)。
                                 残りの時間は静止に使われる。

    Returns:
        function: Manimのrate_funcとして使用できる関数。
    """
    def stepped_rate_func(t: float) -> float:
        if t >= 1.0:
            return 1.0
        if t < 0:
            return 0.0

        # 現在どのステップ区間にいるかを計算 (0, 1, ..., num_steps-1)
        step_index = np.floor(t * num_steps)
        
        # 現在のステップ区間内での経過時間 (0から1/num_stepsの範囲)
        time_in_segment = t - (step_index / num_steps)
        
        # 1ステップあたりの時間
        segment_duration = 1.0 / num_steps
        
        # 1ステップ内の「移動」に使われる時間
        move_duration = segment_duration * move_proportion
        
        # 前のステップまでに完了した進行度
        base_progress = step_index / num_steps
        
        # 現在のステップ区間内での進行度を計算
        if time_in_segment < move_duration and move_duration > 0:
            # 「移動」期間の場合：線形に進行度を増加させる
            progress_in_segment = (time_in_segment / move_duration) * (1.0 / num_steps)
        else:
            # 「静止」期間の場合：進行度をそのステップの最大値で固定
            progress_in_segment = 1.0 / num_steps
            
        return base_progress + progress_in_segment

    return stepped_rate_func



#============================================================================
# デモ用のシーン集
#============================================================================

class MNISTExampleScene(Scene):
    """
    PixelsAsSquareクラスを使用してMNIST画像を可視化するデモシーン。
    """
    def construct(self):
        # --- 1. MNISTデータの準備 ---
        self.camera.background_color = "#2d3c4c"
        loading_text = Text("Loading MNIST dataset...", font_size=36).to_edge(UP)
        self.play(Write(loading_text))

        try:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        except TypeError:
            # 古いバージョンのscikit-learn用
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)

        image_index = 4  # 表示する画像のインデックス（例: 4は'3'の画像）
        image_data = mnist.data[image_index].reshape(28, 28)
        image_label = mnist.target[image_index]

        self.play(FadeOut(loading_text))

        # --- 2. PixelsAsSquareクラスを使ってグリッドを作成 ---
        title = Text(f"MNIST Image (Label: '{image_label}') as Pixel Grid").to_edge(UP)
        
        # PixelsAsSquareクラスのインスタンスを作成
        pixel_grid = PixelsAsSquare(image_data)
        
        # グリッド全体を画面の高さに合わせてスケーリング
        pixel_grid.scale_to_fit_height(config.frame_height * 0.85)

        # --- 3. アニメーションの実行 ---
        self.play(Write(title))
        
        # グリッドをフェードインで表示
        self.play(FadeIn(pixel_grid, lag_ratio=0.005, run_time=4))
        self.wait(2)
        
        # グリッドを拡大して詳細を表示
        self.play(pixel_grid.animate.scale(2.5).move_to(ORIGIN), run_time=3)
        self.wait(3)
        
        # 元のサイズに戻す
        self.play(pixel_grid.animate.scale(1/2.5).center(), run_time=3)
        self.wait(2)

        # 全てをフェードアウト
        self.play(FadeOut(pixel_grid), FadeOut(title))
        self.wait(1)


class CatImageScene(Scene):
    """
    'cat.png'を読み込み、ピクセルグリッドとしてアニメーション表示するシーン。
    """
    def construct(self):
        self.camera.background_color = "#000000"
        image_path = "patch_0_0.png"

        try:
            # image_to_array関数で画像を読み込む
            image_array = image_to_array(image_path)
            
            h, w = image_array.shape[:2]
            title = Text(f"'{image_path}' ({w}x{h})").scale(0.8).to_edge(UP)

            # 画像の次元数に応じて使用するクラスを切り替える
            if image_array.ndim == 3:
                # 3D配列 -> カラー画像
                pixel_grid = PixelsAsSquareColor(image_array)
            elif image_array.ndim == 2:
                # 2D配列 -> グレースケール画像
                pixel_grid = PixelsAsSquare(image_array)
            else:
                # 予期せぬ形式の場合
                error_text = Text("対応していない画像形式です。", color=RED)
                self.play(Write(error_text))
                self.wait(2)
                return

            # グリッドを画面サイズに合わせて調整
            pixel_grid.scale_to_fit_height(config.frame_height*0.8)

            # アニメーション実行
            self.play(Write(title))
            self.play(Create(pixel_grid, lag_ratio=5 / (w*h), run_time=4))
            self.wait(3)
            self.play(FadeOut(pixel_grid), FadeOut(title))
            self.wait()

        except FileNotFoundError as e:
            # ファイルが見つからない場合のエラーメッセージを表示
            error_text = Text(
                f"エラー: '{image_path}' が見つかりません。\n同じディレクトリに配置してください。",
                color=YELLOW,
                font_size=36
            )
            self.play(Write(error_text))
            self.wait(3)
        except Exception as e:
            # その他のエラー
            error_text = Text(f"エラーが発生しました: {e}", color=RED, font_size=24)
            self.play(Write(error_text))
            self.wait(3)

class PixelsToMatricesScene(Scene):
    """
    ピクセルグリッドをRGBの3つの行列に分解して表示するシーン。
    """
    def construct(self):
        self.camera.background_color = "#1E2127"
        image_path = "patch_0_0.png"

        try:
            # 画像を読み込み、配列に変換
            image_array = image_to_array(image_path)
            if image_array.ndim != 3:
                self.play(Write(Text("このシーンはカラー画像専用です。", color=RED)))
                self.wait(2)
                return

            # --- ステップ1: ピクセルグリッドとして画像を表示 ---
            pixel_grid = PixelsAsSquareColor(image_array)
            pixel_grid.scale_to_fit_height(config.frame_height * 0.7)
            
            title = Text("Image as Pixel Grid", font_size=36).to_edge(UP)
            self.play(Write(title))
            self.play(Create(pixel_grid, lag_ratio=0.005, run_time=3))
            self.wait(2)

            # --- ステップ2: RGB行列に変換 ---
            new_title = Text("Decomposed into R, G, B Matrices", font_size=36).to_edge(UP)
            self.play(ReplacementTransform(title, new_title))

            # RGBチャンネルに分割
            r_channel = image_array[:, :, 0]
            g_channel = image_array[:, :, 1]
            b_channel = image_array[:, :, 2]

            # 各チャンネルをMatrixに変換
            r_matrix_group = array_to_matrix(r_channel, "R", RED)
            g_matrix_group = array_to_matrix(g_channel, "G", GREEN)
            b_matrix_group = array_to_matrix(b_channel, "B", BLUE)
            
            # 3つの行列をグループ化し、ずらして配置
            all_matrices = VGroup(b_matrix_group, g_matrix_group, r_matrix_group)
            all_matrices.arrange(RIGHT, buff=1.0) # 最初は横に並べて作成
            all_matrices.scale_to_fit_width(config.frame_width*1.8)

            # 最終的な重ね配置を定義
            b_matrix_group.move_to(ORIGIN)
            g_matrix_group.move_to(b_matrix_group.get_center() + UR * 0.3)
            r_matrix_group.move_to(g_matrix_group.get_center() + UR * 0.3)
            
            # --- ステップ3: アニメーション実行 ---
            self.play(
                AnimationGroup(
                FadeOut(pixel_grid),
                Write(all_matrices, run_time=2)
                )
            )
            self.wait(4)

            # 全てをフェードアウト
            self.play(FadeOut(all_matrices), FadeOut(new_title))
            self.wait()

        except FileNotFoundError as e:
            self.play(Write(Text(f"エラー: '{image_path}' が見つかりません。", color=YELLOW, font_size=36)))
            self.wait(3)
        except Exception as e:
            self.play(Write(Text(f"エラーが発生しました: {e}", color=RED, font_size=24)))
            self.wait(3)


class ConvolutionScene(Scene):
    """
    Convolutionクラスを使用して畳み込みを可視化するデモシーン。
    """
    def construct(self):
        self.camera.background_color = "#2d3c4c"

        # --- 1. データとピクセルグリッドの準備 ---
        # 小さなダミーデータを作成 (例: 10x10)
        image_data = np.random.randint(50, 200, size=(10, 10))
        image_shape = image_data.shape

        # PixelsAsSquareクラスのインスタンスを作成
        pixel_grid = PixelsAsSquare(
            image_data,
            show_values=False, # 値を非表示にして見やすくする
            square_side_length=0.5
        )
        pixel_grid.scale(0.9).center()
        
        title = Text("Convolution Visualization").to_edge(UP)
        self.play(Write(title))
        self.play(Create(pixel_grid, lag_ratio=0.01, run_time=2))
        self.wait(1)

        # --- 2. Convolutionクラスのインスタンスを作成 (Stride=1) ---
        convolution_s1 = Convolution(
            pixel_grid,
            image_shape=image_shape,
            filter_size=(3, 3),
            stride=(1, 1),
            kernel_color=YELLOW
        )
        
        # カーネルをシーンに追加してフェードイン
        self.play(FadeIn(convolution_s1.kernel))
        self.wait(1)

        # --- 3. 畳み込みアニメーションの実行 (Stride=1) ---
        explanation_s1 = Text(
            "Filter: 3x3, Stride: 1x1",
            font_size=28
        ).next_to(pixel_grid, DOWN, buff=0.5)
        self.play(Write(explanation_s1))

        path_animation_s1 = convolution_s1.get_convolution_animation(run_time_per_step=0.3)
        self.play(path_animation_s1)
        self.wait(2)
        
        # --- 4. ストライドを変更して再度アニメーション (Stride=2) ---
        self.play(FadeOut(explanation_s1))
        
        # 新しいConvolutionインスタンスを作成
        convolution_s2 = Convolution(
            pixel_grid,
            image_shape=image_shape,
            filter_size=(3, 3),
            stride=(2, 2), # ストライドを変更
            kernel_color=GREEN
        )
        
        explanation_s2 = Text(
            "Filter: 3x3, Stride: 2x2",
            font_size=28
        ).next_to(pixel_grid, DOWN, buff=0.5)

        # 古いカーネルを新しいカーネルに入れ替え
        self.play(
            ReplacementTransform(convolution_s1.kernel, convolution_s2.kernel),
            Write(explanation_s2)
        )
        self.wait(1)
        
        path_animation_s2 = convolution_s2.get_convolution_animation(run_time_per_step=0.3)
        self.play(path_animation_s2)
        self.wait(2)

        # --- 5. 終了 ---
        self.play(
            FadeOut(pixel_grid),
            FadeOut(convolution_s2.kernel),
            FadeOut(explanation_s2),
            FadeOut(title)
        )
        self.wait(1)

class ConvolutionCalculationScene(Scene):
    def construct(self):
        # 変更点：シーン全体のタイトルを追加
        title = Text("Convolution Operation Visualization").scale(0.8).to_edge(UP)
        self.add(title)

        # 1. 入力データとカーネルの準備
        image_data = np.random.randint(50, 200, size=(10, 10))
        kernel_weights = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) # Sobel Filter
        
        # 2. Manimオブジェクトの作成
        input_grid = PixelsAsSquare(
            image_data, square_side_length=0.6, show_values=True,
            value_format_func=lambda v: f"{v/2.55:.0f}"
        )
        input_label = Text("Input Image").scale(0.7).next_to(input_grid, UP)
        input_vgroup = VGroup(input_grid, input_label)
        
        kernel_matrix = array_to_matrix(kernel_weights, "Kernel (Sobel)", BLUE)
        
        output_h = (image_data.shape[0] - kernel_weights.shape[0]) // 1 + 1
        output_w = (image_data.shape[1] - kernel_weights.shape[1]) // 1 + 1
        feature_map_data = np.zeros((output_h, output_w))
        feature_map_grid = PixelsAsSquare(
            feature_map_data, square_side_length=0.6, show_values=True,
            value_format_func=lambda v: f"{v:.0f}"
        )
        feature_map_label = Text("Feature Map").scale(0.7).next_to(feature_map_grid, UP)
        feature_map_vgroup = VGroup(feature_map_grid, feature_map_label)
        
        # 3. オブジェクトの配置 (変更点)
        # input_vgroup.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.5)
        # feature_map_vgroup.next_to(title, DOWN, buff=0.5).to_edge(RIGHT, buff=0.5)
        # kernel_matrix.next_to(input_vgroup, RIGHT, buff=1.5)
        all_mob = VGroup(input_vgroup, kernel_matrix, feature_map_vgroup).arrange(RIGHT,buff=1).scale_to_fit_width(config.frame_width)

        # 4. 畳み込み計算オブジェクトの初期化
        conv_calculator = CalcConv(
            pixel_grid=input_grid, image_data=image_data,
            kernel_weights=kernel_weights, feature_map_grid=feature_map_grid,
            kernel_color=YELLOW
        )
        conv_calculator.kernel.move_to(input_grid.get_center())

        # 5. アニメーションの実行
        self.play(
            FadeIn(input_vgroup), FadeIn(feature_map_vgroup), FadeIn(kernel_matrix)
        )
        self.wait(1)
        calculation_animation = conv_calculator.get_calculation_animation(run_time_per_step=0.3)
        self.play(
            conv_calculator.kernel.animate.move_to(conv_calculator.get_kernel_target_center(0, 0))
        )
        self.play(calculation_animation)
        self.wait(2)
        self.play(FadeOut(conv_calculator.kernel))
        self.wait(1)