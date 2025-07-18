from manim import *
import random
from typing import Optional

# ----------------------------------------------------------------------------
# ニューラルネットワーク Mobject クラス
# ----------------------------------------------------------------------------
class NeuralNetworkMobject(VGroup):
    """
    ニューラルネットワークを可視化するためのManim Mobject。

    レイヤー数と各レイヤーのニューロン数を指定してネットワークを生成します。
    ニューロン数が多すぎる場合は自動的に省略されます。
    順伝播や逆伝播のアニメーションを簡単に作成できるメソッドを提供します。
    """
    """
    --- Example ---
    class TestNeuralNetworkScene(Scene):
    def construct(self):
        # 1. タイトル表示
        title = Tex("Neural Network Mobject Demo").to_edge(UP)
        self.play(Write(title))
        
        # 2. ネットワーク生成
        # 中間層を20ニューロンにし、省略表示をテスト
        nn = NeuralNetworkMobject([5, 20, 14, 8],edge_color_random=True).scale(0.7)
        self.play(Create(nn))
        self.wait(3)
        
        # 3. 順伝播アニメーション
        status_text = Tex("Forward Propagation", font_size=36).next_to(nn, DOWN)
        self.play(Write(status_text))
        self.play(nn.forward_pass_animation())
        self.wait(3)
        self.play(FadeOut(status_text))
        
        # 4. 色をリセット
        self.play(nn.reset_colors())
        self.wait(2)
        
        # 5. 逆伝播アニメーション
        status_text.become(Tex("Backward Propagation", font_size=36).next_to(nn, DOWN))
        self.play(Write(status_text))
        self.play(nn.backprop_animation())
        self.wait(2)
        self.play(FadeOut(status_text))
        
        # 6. 終了
        self.play(FadeOut(nn), FadeOut(title))
        self.wait(1)
    """
    def __init__(
        self,
        layer_sizes:list[int],
        ellipse_layer_sizes:Optional[list[bool]]=None,
        neuron_radius:float=0.15,
        neuron_stroke_color:ManimColor=BLUE,
        neuron_fill_color:ManimColor=BLACK,
        neuron_to_neuron_buff:float=MED_SMALL_BUFF,
        layer_to_layer_buff:float=LARGE_BUFF,
        edge_color:ManimColor=WHITE,
        edge_color_random:Optional[bool]=False,
        edge_stroke_width:float=1.5,
        max_shown_neurons:int=16,
        activation_color:ManimColor=YELLOW,
        backprop_color:ManimColor=RED,
        **kwargs,
    ):
        if ellipse_layer_sizes is None:
            ellipse_layer_sizes = [False] * len(layer_sizes)

        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.ellipse_layer_sizes = ellipse_layer_sizes
        self.neuron_radius = neuron_radius
        self.neuron_stroke_color = neuron_stroke_color
        self.neuron_fill_color = neuron_fill_color
        self.neuron_to_neuron_buff = neuron_to_neuron_buff
        self.layer_to_layer_buff = layer_to_layer_buff
        self.edge_color = edge_color
        self.edge_color_random = edge_color_random
        self.edge_stroke_width = edge_stroke_width
        self.max_shown_neurons = max_shown_neurons
        self.activation_color = activation_color
        self.backprop_color = backprop_color

        self.neuron_layers = VGroup()
        self.edge_layers = VGroup()
        self._neuron_mobjects_list = []

        self._construct_network()

        self.add(self.neuron_layers, self.edge_layers)
        self.center()

    def _construct_network(self):
        """ネットワークのMobjectを生成する内部メソッド。"""
        self._create_neuron_layers()
        self.neuron_layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self._create_edge_layers()

    def _create_neuron_layers(self):
        """ニューロン層を生成する。"""
        for i, num_neurons in enumerate(self.layer_sizes):
            ellipse = self.ellipse_layer_sizes[i] or num_neurons > self.max_shown_neurons
            layer, neurons = self._create_one_layer(num_neurons, ellipse)
            self.neuron_layers.add(layer)
            self._neuron_mobjects_list.append(neurons)

    def _create_one_layer(self, num_neurons, ellipse:bool=False):
        """指定された数のニューロンを持つ単一の層を生成する。"""
        layer_vgroup = VGroup()
        neurons_vgroup = VGroup()
    
        if ellipse:
            # 省略表示の場合
            num_top = num_neurons // 2
            num_bottom = num_neurons - num_top
            
            for _ in range(num_top):
                neuron = self._create_neuron()
                layer_vgroup.add(neuron)
                neurons_vgroup.add(neuron)

            dots = Tex(r"\vdots", font_size=32)
            layer_vgroup.add(dots)

            for _ in range(num_bottom):
                neuron = self._create_neuron()
                layer_vgroup.add(neuron)
                neurons_vgroup.add(neuron)
        else:
            # 全てのニューロンを表示する場合
            for _ in range(num_neurons):
                neuron = self._create_neuron()
                layer_vgroup.add(neuron)
                neurons_vgroup.add(neuron)
        
        layer_vgroup.arrange(DOWN, buff=self.neuron_to_neuron_buff)
        return layer_vgroup, neurons_vgroup

    def _create_neuron(self):
        """単一のニューロン（Circle）を生成する。"""
        if self.edge_color_random == True:
            return Circle(
            radius=self.neuron_radius,
            stroke_color=WHITE,
            fill_color=self.neuron_fill_color,
            fill_opacity=1,
            )
        else:
            return Circle(
                radius=self.neuron_radius,
                stroke_color=self.neuron_stroke_color,
                fill_color=self.neuron_fill_color,
                fill_opacity=1,
            )

    def _create_edge_layers(self,):
        """
        ニューロン間のエッジを生成する。
        デフォルトで白色。edge_color_random=Trueにすると赤青ランダムに設定。
        """
        if self.edge_color_random == True:
            for i in range(len(self._neuron_mobjects_list) - 1):
                source_layer = self._neuron_mobjects_list[i]
                target_layer = self._neuron_mobjects_list[i+1]
                
                edges = VGroup()
                
                # source_layerの全ニューロンとtarget_layerの全ニューロンを
                # 総当たりで接続するための二重ループ（全結合層）。
                for source_neuron in source_layer:
                    for target_neuron in target_layer:
                        dete_num = random.random()
                        if dete_num >= 0.5:
                            edge = Line(
                                source_neuron.get_center(),
                                target_neuron.get_center(),
                                stroke_color=BLUE,
                                stroke_width=self.edge_stroke_width,
                                z_index=-1
                            )
                        else:
                            edge = Line(
                                source_neuron.get_center(),
                                target_neuron.get_center(),
                                stroke_color=RED,
                                stroke_width=self.edge_stroke_width,
                                z_index=-1
                            ) 
                        edges.add(edge)
                self.edge_layers.add(edges)
        else:
            for i in range(len(self._neuron_mobjects_list) - 1):
                source_layer = self._neuron_mobjects_list[i]
                target_layer = self._neuron_mobjects_list[i+1]
                
                edges = VGroup()
                
                # source_layerの全ニューロンとtarget_layerの全ニューロンを
                # 総当たりで接続するための二重ループ（全結合層）。
                for source_neuron in source_layer:
                    for target_neuron in target_layer:
                        edge = Line(
                            source_neuron.get_center(),
                            target_neuron.get_center(),
                            stroke_color=self.edge_color,
                            stroke_width=self.edge_stroke_width,
                            z_index=-1
                        )
                        edges.add(edge)
                self.edge_layers.add(edges)

    def activate_layer(self, layer_index, color=None, animation_kwargs=None):
        """指定された層をハイライトするアニメーションを返す。"""
        if animation_kwargs is None: animation_kwargs = {}
        if color is None: color = self.activation_color
        return self._neuron_mobjects_list[layer_index].animate(**animation_kwargs).set_color(color)
    
    def deactivate_layer(self, layer_index, animation_kwargs=None):
        """指定された層をハイライトを解除(元のスタイルに戻す)するアニメーションを返す。"""
        return self.reset_colors(layer_index=layer_index, animation_kwargs=animation_kwargs)



    def forward_pass_animation(self, animation_kwargs=None):
        """順伝播のアニメーションを生成する。"""
        if animation_kwargs is None: animation_kwargs = {"run_time": 0.4, "lag_ratio": 0.25}
        
        animations = [self.activate_layer(0, animation_kwargs={"run_time": 0.3})]
        
        for i in range(len(self.edge_layers)):
            edge_flash = ShowPassingFlash(
                self.edge_layers[i].copy().set_stroke(color=self.activation_color, width=self.edge_stroke_width * 1.5),
                time_width=0.4,
                run_time=animation_kwargs.get("run_time", 0.4)
            )
            neuron_activation   = self.activate_layer(i + 1, animation_kwargs=animation_kwargs)
            neuron_deactivation = self.deactivate_layer(i, animation_kwargs=animation_kwargs)

            animations.append(AnimationGroup(edge_flash, neuron_activation, neuron_deactivation))
        
        last_layer_index = len(self.layer_sizes) - 1
        animations.append(self.deactivate_layer(last_layer_index,animation_kwargs=animation_kwargs))
            
        return Succession(*animations, lag_ratio=0.8)

    def backprop_animation(self, animation_kwargs=None):
        """逆伝播のアニメーションを生成する。"""
        if animation_kwargs is None: animation_kwargs = {"run_time": 0.4, "lag_ratio": 0.25}
        
        num_layers = len(self.layer_sizes)
        animations = [self.activate_layer(num_layers - 1, color=self.backprop_color, animation_kwargs={"run_time": 0.3})]
        
        for i in range(num_layers - 2, -1, -1):
            #逆向きにアニメーションするため、逆向きのエッジを生成
            reversed_edges = VGroup()
            for edge in self.edge_layers[i]:
                reversed_edge = Line(
                    edge.get_end(),
                    edge.get_start(),
                )
                reversed_edges.add(reversed_edge)
            
            edge_flash = ShowPassingFlash(
                reversed_edges.set_stroke(color=self.backprop_color, width=self.edge_stroke_width * 1.5),
                time_width=0.4,
                run_time=animation_kwargs.get("run_time", 0.4)
            )
            neuron_activation = self.activate_layer(i, color=self.backprop_color, animation_kwargs=animation_kwargs)
            neuron_deactivation = self.deactivate_layer(i + 1, animation_kwargs=animation_kwargs)

            animations.append(AnimationGroup(edge_flash, neuron_activation, neuron_deactivation))

        animations.append(self.deactivate_layer(0,animation_kwargs=animation_kwargs))
            
        return Succession(*animations, lag_ratio=0.8)

    def reset_colors(self, layer_index=None, animation_kwargs=None):
        """
        ニューロンとエッジの色を初期状態に戻すアニメーションを返す。
        layer_indexが指定されればその層のみ、なければ全体をリセットする。
        """
        if animation_kwargs is None: animation_kwargs = {"run_time": 0.5}

        if layer_index is not None:
            layer_neurons = self._neuron_mobjects_list[layer_index]
            anim_stroke = layer_neurons.animate.set_stroke(color=self.neuron_stroke_color)
            anim_fill = layer_neurons.animate.set_fill(self.neuron_fill_color, opacity=1)
            return AnimationGroup(anim_stroke, anim_fill, **animation_kwargs) 
        
        else:            
            neuron_anim = self.neuron_layers.animate(**animation_kwargs).set_color(self.neuron_stroke_color) #pyright: ignore
            anim_fill = self.neuron_layers.animate.set_fill(self.neuron_fill_color, opacity=1)
            return AnimationGroup(neuron_anim, anim_fill, **animation_kwargs)


# ----------------------------------------------------------------------------
# ニューラルネットワーク Mobject クラス(Model対応)
# ----------------------------------------------------------------------------
class NeuralNetworkWithActivation(NeuralNetworkMobject):
    """
    NeuralNetworkMobjectを継承し、モデルの実際の活性化に基づいて
    ニューロンの発火を可視化する機能を追加したクラス。
    """
    def __init__(self, layer_sizes, **kwargs):
        # 親クラスの__init__を呼び出し、ネットワークの基本的な構造を構築する
        super().__init__(layer_sizes, **kwargs)

    def activate_layer(self, layer_index, activations=None, animation_kwargs=None):
        """
        指定された層をハイライトするアニメーションを返す。(オーバーライド)
        activationsが与えられた場合、その値に応じてニューロンの色を変化させる。
        """
        if animation_kwargs is None: animation_kwargs = {}
        
        layer_neurons = self._neuron_mobjects_list[layer_index]
        animations = []

        if activations is not None:
            # アクティベーションを正規化 (0以上で最大値が1になるように)
            activations_flat = activations.flatten()
            max_val = activations_flat.max().item() 
            activations_norm = activations_flat / (max_val + 1e-6) # ゼロ除算を防止
            
            
            for i, neuron in enumerate(layer_neurons):
                if i < len(activations_norm):
                    # alpha値（色の混合率）を計算
                    alpha = np.clip(activations_norm[i].item(), 0, 1)
                    # 黒 (非発火) から指定色 (最大発火) へ補間
                    new_color = interpolate_color(self.neuron_fill_color, self.activation_color, alpha)
                    # フィルカラーのみを変化させる
                    animations.append(neuron.animate(**animation_kwargs).set_fill(color=new_color, opacity=1))
        else:
            # activationsがなければ、層全体を単色でハイライトする
            # 親クラスとは異なり、フィルカラーのみを変更する
            animations = [
                neuron.animate(**animation_kwargs).set_fill(self.activation_color, opacity=1)
                for neuron in layer_neurons
            ]
        
        return AnimationGroup(*animations, lag_ratio=0)

    def forward_pass_animation(self, model, input_tensor, animation_kwargs=None):
        """
        順伝播のアニメーションを生成する。(オーバーライド)
        PyTorchモデルと入力テンソルを受け取り、各層の活性化を視覚化する。
        """
        if animation_kwargs is None: animation_kwargs = {"run_time": 0.4}

        # モデルを実行して中間層の活性化リストを取得する
        # (モデル側にactivationsを保存するフックなどの実装が前提)
        output = model(input_tensor.unsqueeze(0))
        activations_list = model.activations
        
        animations = []
        # 入力層の活性化
        # 入力自体を最初の活性化と見なす
        animations.append(self.activate_layer(0, activations=input_tensor, animation_kwargs=animation_kwargs))
        
        for i in range(len(self.edge_layers)):
            edge_flash = ShowPassingFlash(
                self.edge_layers[i].copy().set_stroke(color=self.activation_color, width=self.edge_stroke_width * 1.5),
                time_width=0.4, run_time=animation_kwargs.get("run_time", 0.4)
            )
            # 次の層を、モデルから得た活性化情報でハイライト
            neuron_activation = self.activate_layer(i + 1, activations=activations_list[i], animation_kwargs=animation_kwargs)
            # 前の層を非アクティブ化（親クラスのdeactivate_layerメソッドを利用）
            neuron_deactivation = self.deactivate_layer(i, animation_kwargs=animation_kwargs)
            
            animations.append(AnimationGroup(edge_flash, neuron_activation, neuron_deactivation))

        # 最終層を非アクティブ化
        animations.append(self.deactivate_layer(len(self.layer_sizes) - 1, animation_kwargs=animation_kwargs))
        
        return Succession(*animations, lag_ratio=0.8)
