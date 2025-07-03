# manim_mnist_pixels.py

from manim import *
import numpy as np
from sklearn.datasets import fetch_openml

class MNISTPixelGrid(Scene):
    """
    MNIST画像の一枚を読み込み、各ピクセルを色付けされたSquareと
    ピクセル値のテキストで可視化するシーン。
    """
    def construct(self):
        # --- 1. MNISTデータの準備 ---
        # scikit-learnを使ってMNISTデータセットをダウンロードします。
        # 初回実行時はデータのダウンロードに数分かかる場合があります。
        self.camera.background_color = "#2d3c4c" # 背景色を少し明るく設定
        
        # データのロード状況をテキストで表示
        loading_text = Text("Loading MNIST dataset...").to_edge(UP)
        self.play(Write(loading_text))

        try:
            # scikit-learn 1.2以降で推奨される引数
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        except TypeError:
            # 古いバージョンのscikit-learn用のフォールバック
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        
        # 4番目の画像データ（数字の'3'）を28x28のNumpy配列に変換
        image_index = 4
        image_data = mnist.data[image_index].reshape(28, 28)
        image_label = mnist.target[image_index]
        
        self.play(FadeOut(loading_text))

        # --- 2. ピクセルグリッドの作成 ---
        pixel_mobjects = []
        
        # 28x28の画像データをループ処理
        for row in image_data:
            for pixel_value in row:
                # ピクセル値（0-255）を輝度（0.0-1.0）に正規化
                normalized_value = pixel_value / 255.0
                
                # Squareを作成し、輝度に応じたグレースケールで塗りつぶす
                square = Square(side_length=1.0, fill_opacity=1.0)
                square.set_color(rgb_to_color([normalized_value, normalized_value, normalized_value]))
                
                # 境界線を薄く描画
                square.set_stroke(color=GRAY, width=0.1)

                # ピクセル値を表示するTextを作成
                # 背景色に応じて文字色を白か黒に変更し、視認性を確保
                text_color = BLACK if normalized_value > 0.6 else WHITE
                text = Text(round((pixel_value)/255.0,1), font_size=14, color=text_color)
                
                # テキストがSquareに収まるようにスケーリング
                if text.width > square.width * 0.9:
                    text.scale_to_fit_width(square.width * 0.9)

                # SquareとTextをグループ化してリストに追加
                pixel_group = VGroup(square, text)
                pixel_mobjects.append(pixel_group)
        
        # 作成したピクセルオブジェクトのリストからVGroupを作成し、グリッド状に配置
        pixel_grid = VGroup(*pixel_mobjects).arrange_in_grid(rows=28, cols=28, buff=0)
        
        # グリッド全体を画面の高さに合わせてスケーリング
        pixel_grid.scale_to_fit_height(config.frame_height * 0.85)

        # --- 3. アニメーションの実行 ---
        title = Text(f"MNIST Image (Label: '{image_label}') as Pixel Grid").to_edge(UP)
        
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