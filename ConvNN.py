import torch
import torchvision
import numpy as np
from manim import *

from NeuralNetwork import NeuralNetworkMobject

# --- 設定 ---
MNIST_IMAGE_SCALE = 0.12
KERNEL_SIZE = 3
STRIDE = 2 # ストライドを2にして、出力が小さくなる様子を見せる

class ConvolutionalNeuralNetworkMobject(Group):
    """
    畳み込み層と全結合層からなるCNN全体を可視化するMobject。
    
    - 畳み込み層は四角い特徴マップとして表現。
    - 全結合層はインポートしたNeuralNetworkクラスを利用。
    """
    def __init__(
        self,
        input_array: np.ndarray,
        conv_layer_specs: list, # [(filters, color), (filters, color), ...]
        fc_layer_sizes: list,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_array = input_array
        self.conv_layer_specs = conv_layer_specs
        self.fc_layer_sizes = fc_layer_sizes

        # CNNの各パーツを生成
        self._create_input_layer()
        self._create_conv_layers()
        self._create_flatten_arrow()
        self._create_fc_layers()
        
        # 全体を配置
        self.arrange(RIGHT, buff=0.7)
        self.center()

    def _create_input_layer(self):
        """入力画像レイヤーを生成"""
        input_rgb = np.stack([self.input_array * 255] * 3, axis=-1).astype(np.uint8)
        self.input_image = ImageMobject(input_rgb).scale(MNIST_IMAGE_SCALE)
        self.input_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        input_label = Text("Input", font_size=24).next_to(self.input_image, DOWN)
        self.input_layer = Group(self.input_image, input_label)
        self.add(self.input_layer)

    def _create_conv_layers(self):
        """畳み込み/プーリング層のブロックを生成"""
        self.conv_layers = VGroup().arrange(RIGHT, buff=0.5)
        last_mobject = self.input_image
        
        for i, (num_filters, color) in enumerate(self.conv_layer_specs):
            # 複数フィルターを奥行きがあるように見せる
            layer_block = VGroup(*[
                Rectangle(height=last_mobject.height * 0.85, width=last_mobject.width * 0.85, color=color, fill_opacity=0.2)
                for _ in range(num_filters)
            ]).arrange(DR, buff=-0.15) # DR(右下)にずらして重ねる
            
            # ラベルを追加
            label = Text(f"Conv/Pool {i+1}", font_size=20).next_to(layer_block, DOWN)
            
            # グループ化してリストに追加
            full_block = VGroup(layer_block, label)
            self.conv_layers.add(full_block)
            
            # 次のレイヤーサイズのために更新
            last_mobject = layer_block

        # 矢印を追加
        conv_arrows = VGroup(*[
            Arrow(self.conv_layers[i].get_right(), self.conv_layers[i+1].get_left(), buff=0.1, stroke_width=3)
            for i in range(len(self.conv_layers) - 1)
        ])
        
        # 最初の矢印（入力 -> Conv1）
        first_arrow = Arrow(self.input_layer.get_right(), self.conv_layers[0].get_left(), buff=0.2, stroke_width=3)
        
        self.conv_part = Group(self.conv_layers, first_arrow, conv_arrows)
        self.add(self.conv_part)

    def _create_flatten_arrow(self):
        """畳み込み層から全結合層への平坦化を示す矢印"""
        self.flatten_arrow = Arrow(
            self.conv_part.get_right(),
            self.conv_part.get_right() + RIGHT * 0.8,
            buff=0.1, stroke_width=5
        )
        flatten_label = Text("Flatten", font_size=24).next_to(self.flatten_arrow, UP, buff=0.1)
        self.flatten_group = Group(self.flatten_arrow, flatten_label)
        self.add(self.flatten_group)

    def _create_fc_layers(self):
        """提供されたNeuralNetworkクラスを使って全結合層を生成"""
        self.fc_layers = NeuralNetworkMobject(self.fc_layer_sizes, layer_to_layer_buff=0.8)
        self.add(self.fc_layers)

    def get_convolution_animation(self) -> Animation:
        """畳み込み演算の可視化アニメーションを生成"""
        input_img = self.input_layer[0]
        
        # 出力特徴マップを仮に作成
        output_h = (input_img.height - (input_img.height / 28 * KERNEL_SIZE)) / (input_img.height / 28 * STRIDE) + 1
        output_w = (input_img.width - (input_img.width / 28 * KERNEL_SIZE)) / (input_img.width / 28 * STRIDE) + 1
        
        output_grid = VGroup(*[
            Square(side_length=0.1, color=GREY, stroke_width=0.5)
            for _ in range(int(output_h * output_w))
        ]).arrange_in_grid(rows=int(output_h), cols=int(output_w), buff=0.05)
        
        output_group = Group(output_grid).next_to(input_img, RIGHT, buff=0.5)
        
        # フィルターを作成
        pixel_h = input_img.height / 28
        filter_size = pixel_h * KERNEL_SIZE
        filter_mobj = Square(side_length=filter_size, color=YELLOW, fill_opacity=0.3)
        filter_mobj.move_to(input_img.get_corner(UL), aligned_edge=UL)
        
        # アニメーションを作成
        animations = [Create(output_group), Create(filter_mobj)]
        
        # スライド処理
        for r in range(int(output_h)):
            for c in range(int(output_w)):
                target_pos = input_img.get_corner(UL) + \
                             DOWN * r * pixel_h * STRIDE + \
                             RIGHT * c * pixel_h * STRIDE
                
                output_cell = output_grid[r * int(output_w) + c]
                
                anim_group = AnimationGroup(
                    filter_mobj.animate.move_to(target_pos, aligned_edge=UL),
                    Indicate(output_cell, color=YELLOW, scale_factor=1.2),
                    run_time=0.1
                )
                animations.append(anim_group)

        animations.append(FadeOut(output_group, filter_mobj))
        return Succession(*animations, lag_ratio=0.1)

    def get_forward_pass_animation(self) -> Animation:
        """CNN全体の順伝播アニメーション"""
        animations = []
        
        # 畳み込み部分の伝播
        all_arrows = VGroup(self.conv_part[1], *self.conv_part[2])
        for arrow in all_arrows:
             animations.append(ShowPassingFlash(arrow.copy().set_color(YELLOW), time_width=0.3, run_time=0.5))
        
        # Flatten部分の伝播
        animations.append(ShowPassingFlash(self.flatten_arrow.copy().set_color(YELLOW), time_width=0.3, run_time=0.5))

        # 全結合層の伝播（インポートしたクラスのメソッドを利用）
        animations.append(self.fc_layers.forward_pass_animation())
        
        return Succession(*animations, lag_ratio=0.3)

# ----------------------------------------------------------------------------
# CNN可視化シーン
# ----------------------------------------------------------------------------
class CNNVisualizationScene(Scene):
    def construct(self):
        # 1. MNISTデータを準備
        try:
            mnist_data = torchvision.datasets.MNIST(root=".", download=True, train=True)
            image_tensor, _ = mnist_data[7] # 数字の7の画像を使用
            image_numpy = np.array(image_tensor) / 255.0 # 0-1に正規化
        except Exception:
            image_numpy = np.random.rand(28, 28)

        # 2. タイトル表示
        title = Title("Convolutional Neural Network (CNN)")
        self.play(Write(title))

        # 3. CNNアーキテクチャを定義してMobjectを生成
        cnn_arch = ConvolutionalNeuralNetworkMobject(
            input_array=image_numpy,
            conv_layer_specs=[(3, BLUE), (5, GREEN)], # (フィルター数, 色)
            fc_layer_sizes=[10, 8, 4] # 全結合層のニューロン数
        ).scale(0.85).to_edge(DOWN, buff=0.8)
        
        self.play(Create(cnn_arch))
        self.wait(2)
        
        # 4. 畳み込み演算の可視化
        conv_title = Text("Convolution Operation Example", font_size=36).to_edge(UP, buff=1.5)
        self.play(Transform(title, conv_title))
        self.play(cnn_arch.get_convolution_animation(), run_time=8)
        self.wait(2)
        
        # 5. 順伝播の可視化
        fp_title = Text("Full Forward Propagation", font_size=36).to_edge(UP, buff=1.5)
        self.play(Transform(title, fp_title))
        self.play(cnn_arch.get_forward_pass_animation())
        self.wait(2)

        # 6. 終了
        self.play(FadeOut(cnn_arch), FadeOut(title))
        self.wait(1)