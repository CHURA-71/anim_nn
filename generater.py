from manim import *
from typing import Tuple, Optional

try:
    from .gpt_2 import GPT2
except ImportError:
    # 直接実行時は絶対インポートを試行
    try:
        from gpt_2 import GPT2
    except ImportError:
        # 最後の手段：パッケージパスを調整
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from gpt_2 import GPT2

class GPT2Generator(VGroup):
    def __init__(
            self,
            model_color: ManimColor = GRAY,
            prob_color_range: Tuple[ManimColor, ManimColor] = (YELLOW, BLUE_C),
            max_prob_width: float = 2.0,
            top_k: int = 6,
            input_text: Optional[str] = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.gpt2 = GPT2(top_k=top_k)
        if input_text is not None:
            self.input_text = input_text
        self.model_color = model_color
        self.prob_color_range = prob_color_range
        self.max_prob_width = max_prob_width
        self.top_k = top_k
        self.model = self._create_model()
        self.arrow = self._create_arrow()

        self.add(self.model, self.arrow)

        self._animator = None
    
    @property
    def animate(self):
        """アニメーション機能にアクセスするためのプロパティ。
        
        Returns:
            GPT2Animator: アニメーション機能を提供するオブジェクト
        """
        if self._animator is None:
            self._animator = GPT2Animator(self)
        return self._animator


    def _create_model(self):
        """
        モデルの矩形とテキストを生成し、VGroupにまとめる。
        """
        model_box = Rectangle(
            color=self.model_color,
            width=3,
            height=2,
            fill_opacity=0.4
            )
        model_name = Text(
            "GPT-2",
            color=WHITE,
            font_size=32
        )
        model_name.move_to(model_box.get_center())
        model = VGroup(model_box, model_name)
        return model
    
    def _create_arrow(self):
        """
        矢印を生成し、モデルの右側に配置する。
        """
        arrow = Arrow(
            start=self.model.get_right(),
            end=self.model.get_right() + RIGHT * 2,
            color=WHITE
        )
        return arrow

    def generate_token(self, input_text: str):
        _, token_list, selected_index = self.gpt2(input_text)

        tokens = VGroup()
        probablities = VGroup()
        for token in token_list:
            token_mob = Text(
                token[0],
                color=WHITE,
                font_size=24
            )
            prob_box = Rectangle(
                width=self._prob_to_width(token[1], self.max_prob_width),
                height=0.5,
                fill_color=self._prob_to_color(token[1], self.prob_color_range),
                fill_opacity=0.8,
                stroke_color=WHITE
            )
            prob = MathTex(
                f"{token[1]*100:0.0f}\\%",
                color=WHITE,
                font_size=24
            )

            prob_row = VGroup(prob_box, prob).arrange(RIGHT, buff=0.1)
            probablities.add(prob_row)
            tokens.add(token_mob)

        probablities.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        for token_mob, prob in zip(tokens, probablities):
            token_mob.next_to(prob, LEFT,buff=0.3)
        output_tokens = VGroup(tokens, probablities)
        if self.top_k > 10:
            output_tokens.scale_to_fit_height(config.frame_height*0.8)
        output_tokens.next_to(self.arrow, RIGHT, buff=0.5)
        vdots = MathTex(r"\vdots", color=WHITE, font_size=24)
        vdots.next_to(tokens[-1], DOWN, buff=0.2)
        output_tokens.add(vdots)

        selected_token = VGroup(tokens[selected_index-1], probablities[selected_index-1])
        pred_token = SurroundingRectangle(
            selected_token,  # インデックスは1から始まるため-1
            color=YELLOW,
            fill_opacity=0.2,
            buff=0.1
        )

        return output_tokens, pred_token, selected_index
    
    def repeat_generate_animation(self, input_text: str, max_length: int = 20, wait_time: float = 0.7):
        """
        トークン生成のアニメーションを繰り返し実行
        Args:
            input_text (str): 初期入力テキスト
            max_length (int): 最大生成トークン数
            wait_time (float): 各アニメーション間の待機時間
        Returns:
            animations (list): アニメーションのリスト
        
        Example:
        class Test(Scene):
            def construct(self):
                gpt = GPT2Generator(top_k=10)
                self.add(gpt)
                self.wait(0.5)
                animations = gpt.repeat_generate_animation("Once upon a time", max_length=10)
                for anim in animations:
                    self.play(anim)
        """
        input_text_len = len(input_text.replace(" ", ""))
        text2mob = input_text
        output_token_list = []
        pred_token_list = []
        token_size_list = []
        selected_index_list = []
        animations = []
        for i in range(max_length):
            output_tokens, pred_token, selected_index = self.generate_token(input_text)
            output_token_list.append(output_tokens)
            pred_token_list.append(pred_token)
            selected_index_list.append(selected_index)
            input_text += " " + output_tokens[0][selected_index-1].text # 新しいトークンを入力テキストに追加
            if i % 4 == 0:
                text2mob += " " + output_tokens[0][selected_index-1].text + "\n"
            else:
                text2mob += " " + output_tokens[0][selected_index-1].text
            
            token_size_list.append(len(output_tokens[0][selected_index-1].text))

        input_mob = Text(text2mob, font_size=32, color=WHITE)
        input_mob.next_to(self.model,LEFT)
        animations.append(FadeIn(input_mob[:input_text_len], run_time=1/config.frame_rate))
        token_index = input_text_len # 初期のトークン数を設定
        for i in range(max_length):
            if i == 0:
                animations.append(
                AnimationGroup(
                    FadeIn(output_token_list[i]),
                    FadeIn(pred_token_list[i]),
                    run_time=1/config.frame_rate
                )
            )
            else:
                animations.append(
                    FadeIn(
                        input_mob[token_index : token_index+token_size_list[i-1]],
                        run_time=1/config.frame_rate
                    )
                )
                animations.append(
                    AnimationGroup(
                        FadeIn(output_token_list[i]),
                        FadeIn(pred_token_list[i]),
                        run_time=1/config.frame_rate
                    )
                )
            animations.append(Wait(wait_time))
            if i < len(output_token_list) - 1:
                animations.append(
                    AnimationGroup(
                        FadeOut(output_token_list[i]),
                        FadeOut(pred_token_list[i]),
                        run_time=1/config.frame_rate
                    )
                )
            animations.append(Wait(wait_time))
            if not i == 0:
                token_index += token_size_list[i-1]
        return animations
        

    def _prob_to_width(self, prob: float, max_width: float = 2.0):
        """
        確率を幅に変換
        
        Args:
            prob (float): 確率値 (0-1)
            max_width (float): 最大幅
        
        Returns:
            float: 幅
        """
        return prob * max_width
    
    def _prob_to_color(self, prob: float, prob_color_range: tuple):
        """
        確率を色に変換
        
        Args:
            prob (float): 確率値 (0-1)
        
        Returns:
            ManimColor: 色
        """
        return interpolate_color(prob_color_range[0], prob_color_range[1], prob)

class GPT2Animator:
    """
    GPT-2トークン生成のアニメーションを管理するクラス。
    """
    def __init__(self, gpt2_generator: GPT2Generator):
        self.gpt2_generator = gpt2_generator
    
    def generate_token(self, input_text: str):
        """
        トークンを生成し、アニメーションを実行
        
        Args:
            input_text (str): 入力テキスト
        
        Returns:
            Animation: トークン生成のアニメーション
        """
        output_tokens, pred_token, _ = self.gpt2_generator.generate_token(input_text)

        return Add(
            output_tokens,
            pred_token,
            run_time=0.01
        )


class Test(Scene):
    def construct(self):
        gpt = GPT2Generator(top_k=10)
        self.add(gpt)
        self.wait(0.5)
        # self.play(
        #     gpt.animate.generate_token("Once upon a time")
        # )

        # gpt.repeat_generate_animation(self, "Once upon a time")
        animations = gpt.repeat_generate_animation("Once upon a time", max_length=4)
        for anim in animations:
            self.play(anim)