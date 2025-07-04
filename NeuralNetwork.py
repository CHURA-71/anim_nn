from manim import *

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
    def __init__(
        self,
        layer_sizes,
        neuron_radius=0.15,
        neuron_stroke_color=BLUE,
        neuron_fill_color=BLACK,
        neuron_to_neuron_buff=MED_SMALL_BUFF,
        layer_to_layer_buff=LARGE_BUFF,
        edge_color=WHITE,
        edge_stroke_width=1.5,
        max_shown_neurons=16,
        activation_color=YELLOW,
        backprop_color=RED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.neuron_radius = neuron_radius
        self.neuron_stroke_color = neuron_stroke_color
        self.neuron_fill_color = neuron_fill_color
        self.neuron_to_neuron_buff = neuron_to_neuron_buff
        self.layer_to_layer_buff = layer_to_layer_buff
        self.edge_color = edge_color
        self.edge_stroke_width = edge_stroke_width
        self.max_shown_neurons = max_shown_neurons
        self.activation_color = activation_color
        self.backprop_color = backprop_color

        self.neuron_layers = VGroup()
        self.edge_layers = VGroup()
        # Circleオブジェクトのみを保持するリスト（エッジ接続・アニメーション用）
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
        for num_neurons in self.layer_sizes:
            layer, neurons = self._create_one_layer(num_neurons)
            self.neuron_layers.add(layer)
            self._neuron_mobjects_list.append(neurons)

    def _create_one_layer(self, num_neurons):
        """指定された数のニューロンを持つ単一の層を生成する。"""
        layer_vgroup = VGroup()
        neurons_vgroup = VGroup()

        if num_neurons > self.max_shown_neurons:
            # 省略表示の場合
            num_top = self.max_shown_neurons // 2
            num_bottom = self.max_shown_neurons - num_top
            
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
        return Circle(
            radius=self.neuron_radius,
            stroke_color=self.neuron_stroke_color,
            fill_color=self.neuron_fill_color,
            fill_opacity=1,
        )

    def _create_edge_layers(self):
        """ニューロン間のエッジを生成する。"""
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
            neuron_anim = self.neuron_layers.animate(**animation_kwargs).set_color(self.neuron_stroke_color)
            anim_fill = self.neuron_layers.animate.set_fill(self.neuron_fill_color, opacity=1)
            return AnimationGroup(neuron_anim, anim_fill, **animation_kwargs)

# ----------------------------------------------------------------------------
# テスト用アニメーションシーン
# ----------------------------------------------------------------------------
class TestNeuralNetworkScene(Scene):
    def construct(self):
        # 1. タイトル表示
        title = Tex("Neural Network Mobject Demo").to_edge(UP)
        self.play(Write(title))
        
        # 2. ネットワーク生成
        # 中間層を20ニューロンにし、省略表示をテスト
        nn = NeuralNetworkMobject([5, 20, 14, 8]).scale(0.7)
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