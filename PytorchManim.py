# visualize_pytorch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

from manim import *

# (PyTorchモデルとヘルパー関数の部分は変更ありません)
# ----------------------------------------------------------------------------
# PyTorchモデルとヘルパー関数
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        self.activations = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU) or layer == self.layers[-1]:
                 self.activations.append(x.detach().cpu().numpy())
        return x

def get_layer_sizes_from_model(model):
    sizes = []
    for module in model.layers:
        if isinstance(module, nn.Linear):
            if not sizes:
                sizes.append(module.in_features)
            sizes.append(module.out_features)
    return sizes

def train_model_if_needed(model, file_path="mnist_model.pth"):
    if os.path.exists(file_path):
        print(f"Loading pre-trained model from {file_path}")
        model.load_state_dict(torch.load(file_path))
        return
    print("Training a new model...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(2):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), file_path)


# ----------------------------------------------------------------------------
# ニューラルネットワーク Mobject クラス (★★ ここを修正 ★★)
# ----------------------------------------------------------------------------
class NeuralNetworkMobject(VGroup):
    def __init__(
        self, layer_sizes, neuron_radius=0.15, neuron_stroke_color=BLUE,
        neuron_fill_color=BLACK, neuron_to_neuron_buff=MED_SMALL_BUFF,
        layer_to_layer_buff=LARGE_BUFF, edge_color=WHITE, edge_stroke_width=1.5,
        max_shown_neurons=16, activation_color=YELLOW, backprop_color=RED, **kwargs,
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
        self._neuron_mobjects_list = []
        self._construct_network()
        self.add(self.neuron_layers, self.edge_layers)
        self.center()

    def _construct_network(self):
        self._create_neuron_layers()
        self.neuron_layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self._create_edge_layers()

    def _create_neuron_layers(self):
        for num_neurons in self.layer_sizes:
            layer, neurons = self._create_one_layer(num_neurons)
            self.neuron_layers.add(layer)
            self._neuron_mobjects_list.append(neurons)

    def _create_one_layer(self, num_neurons):
        layer_vgroup = VGroup()
        neurons_vgroup = VGroup()
        if num_neurons > self.max_shown_neurons:
            num_top = self.max_shown_neurons // 2
            num_bottom = self.max_shown_neurons - num_top
            for i in range(num_top):
                neuron = self._create_neuron()
                layer_vgroup.add(neuron)
                neurons_vgroup.add(neuron)
            dots = Tex(r"\vdots", font_size=32)
            layer_vgroup.add(dots)
            for i in range(num_bottom):
                neuron = self._create_neuron()
                layer_vgroup.add(neuron)
                neurons_vgroup.add(neuron)
        else:
            for i in range(num_neurons):
                neuron = self._create_neuron()
                layer_vgroup.add(neuron)
                neurons_vgroup.add(neuron)
        layer_vgroup.arrange(DOWN, buff=self.neuron_to_neuron_buff)
        return layer_vgroup, neurons_vgroup

    def _create_neuron(self):
        return Circle(
            radius=self.neuron_radius, stroke_color=self.neuron_stroke_color,
            fill_color=self.neuron_fill_color, fill_opacity=1,
        )

    def _create_edge_layers(self):
        for i in range(len(self._neuron_mobjects_list) - 1):
            source_layer = self._neuron_mobjects_list[i]
            target_layer = self._neuron_mobjects_list[i+1]
            edges = VGroup()
            for source_neuron in source_layer:
                for target_neuron in target_layer:
                    edge = Line(
                        source_neuron.get_center(), target_neuron.get_center(),
                        stroke_color=self.edge_color, stroke_width=self.edge_stroke_width, z_index=-1
                    )
                    edges.add(edge)
            self.edge_layers.add(edges)

    def activate_layer(self, layer_index, activations=None, animation_kwargs=None):
        if animation_kwargs is None: animation_kwargs = {}
        layer_neurons = self._neuron_mobjects_list[layer_index]
        animations = []

        if activations is not None:
            # アクティベーションを正規化 (ReLUを想定し、0以上で最大値が1になるように)
            activations_norm = activations / (np.max(activations) + 1e-6)
            
            for i, neuron in enumerate(layer_neurons):
                if i < len(activations_norm.flatten()):
                    # alpha値（色の混合率）を計算
                    alpha = np.clip(activations_norm.flatten()[i], 0, 1)
                    # 黒 (非発火) から黄 (最大発火) へ補間
                    new_color = interpolate_color(self.neuron_fill_color, self.activation_color, alpha)
                    # 色を変化させ、opacityは1に固定
                    animations.append(neuron.animate(**animation_kwargs).set_fill(color=new_color, opacity=1))
        else:
            animations = [neuron.animate(**animation_kwargs).set_fill(self.activation_color, opacity=1) for neuron in layer_neurons]
        
        return AnimationGroup(*animations, lag_ratio=0)

    def forward_pass_animation(self, model, input_tensor, animation_kwargs=None):
        if animation_kwargs is None: animation_kwargs = {"run_time": 0.4, "lag_ratio": 0.25}
        
        output = model(input_tensor.unsqueeze(0))
        activations_list = model.activations
        
        animations = []
        for i in range(len(self.edge_layers)):
            edge_flash = ShowPassingFlash(
                self.edge_layers[i].copy().set_stroke(color=self.activation_color, width=self.edge_stroke_width * 1.5),
                time_width=0.4, run_time=animation_kwargs.get("run_time", 0.4)
            )
            neuron_activation = self.activate_layer(i + 1, activations=activations_list[i], animation_kwargs=animation_kwargs)
            neuron_deactivation = self._neuron_mobjects_list[i].animate(lag_ratio=0).set_fill(self.neuron_fill_color, opacity=1)
            
            animations.append(AnimationGroup(edge_flash, neuron_activation, neuron_deactivation))

        animations.append(self._neuron_mobjects_list[-1].animate(lag_ratio=0).set_fill(self.neuron_fill_color, opacity=1))
        
        return Succession(*animations, lag_ratio=0.8)

# ----------------------------------------------------------------------------
# Manim Sceneの実装 (変更なし)
# ----------------------------------------------------------------------------
class PyTorchToManim(Scene):
    def construct(self):
        title = Text("PyTorch MNIST Classification", font_size=36).to_edge(UP,buff=0)
        self.play(Write(title))

        model = MLP()
        train_model_if_needed(model, "mnist_model.pth")
        model.eval()

        layer_sizes = get_layer_sizes_from_model(model)
        network = NeuralNetworkMobject(
            layer_sizes,
            layer_to_layer_buff=1.5,
            edge_stroke_width=1.0,
            neuron_fill_color=BLACK, # ニューロンの基本色を黒に設定
            activation_color=YELLOW # 発火色を黄色に設定
        )
        network.scale(0.8).center()
        
        output_labels = VGroup()
        output_layer_neurons = network._neuron_mobjects_list[-1]
        for i, neuron in enumerate(output_layer_neurons):
            label = Text(str(i), font_size=24).next_to(neuron, RIGHT, buff=MED_SMALL_BUFF)
            output_labels.add(label)

        data_index = 5
        model_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor_dataset = datasets.MNIST(root='./data', train=False, transform=model_transform)
        image_for_model, label = tensor_dataset[data_index]

        vis_transform = transforms.ToTensor()
        vis_dataset = datasets.MNIST(root='./data', train=False, transform=vis_transform)
        image_for_vis, _ = vis_dataset[data_index]

        pil_dataset = datasets.MNIST(root='./data', train=False, transform=None)
        image_pil, _ = pil_dataset[data_index]
        image_np_gray = np.array(image_pil)
        image_np_rgb = np.stack([image_np_gray] * 3, axis=-1)
        image_mobj = ImageMobject(image_np_rgb)
        image_mobj.invert(True)
        image_mobj.set_height(1.5)
        image_mobj.to_corner(UL)
        sor_rec = SurroundingRectangle(image_mobj,color=BLUE)
        img = Group(image_mobj,sor_rec)

        input_label_text = Text(f"Input Digit: {label}", font_size=28).next_to(image_mobj, DOWN, buff=SMALL_BUFF)

        self.play(
            Create(network),
            FadeIn(img),
            Write(input_label_text),
            Write(output_labels)
        )
        self.wait(1)
        
        input_layer_neurons = network._neuron_mobjects_list[0]
        pixel_grid = VGroup(*[
            Square(side_length=0.05, fill_opacity=1, stroke_width=0)
            .set_color(interpolate_color(BLACK, WHITE, val.item()))
            for val in image_for_vis.flatten()
        ]).arrange_in_grid(28, 28, buff=0).move_to(image_mobj)

        self.play(FadeOut(img), FadeIn(pixel_grid))
        self.play(
            pixel_grid.animate.set_height(input_layer_neurons.get_height())
            .move_to(input_layer_neurons).set_opacity(0),
            run_time=1.5
        )

        input_activations = image_for_vis.flatten().numpy()
        self.play(network.activate_layer(0, activations=input_activations))

        self.play(network.forward_pass_animation(model, image_for_model, animation_kwargs={"run_time": 0.5}))
        
        with torch.no_grad():
            output = model(image_for_model.unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()

        result_text = Text(f"Model Prediction: {prediction}", font_size=32).move_to(DOWN).to_edge(RIGHT)
        predicted_neuron = output_layer_neurons[prediction]
        
        self.play(Write(result_text))
        self.play(
            predicted_neuron.animate.set_color(GREEN_C).scale(1.5),
            output_labels[prediction].animate.set_color(GREEN_C).scale(1.5),
            Circumscribe(predicted_neuron, color=GREEN_C, fade_out=True, run_time=2)
        )

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])