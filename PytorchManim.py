# visualize_pytorch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

from manim import *

from NeuralNetwork import *

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
# ニューラルネットワーク Mobject クラス 
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

# ----------------------------------------------------------------------------
# Manim Sceneの実装 (変更なし)
# ----------------------------------------------------------------------------
class PyTorchToManim(Scene):
    def MNISTforwardAnim(self, model, network_mob, data_index, output_labels):
        """
        PytorchのモデルとMNISTデータを受け取って、順伝播を可視化
        """
        data_index = data_index
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

        input_label_text = Tex(f"Input Digit: {label}", font_size=28).next_to(image_mobj, DOWN, buff=SMALL_BUFF)

        self.play(
            FadeIn(img),
            Write(input_label_text)
        )
        
        input_layer_neurons = network_mob._neuron_mobjects_list[0]
        Inputs=VGroup()
        for i in range(0,16):
            input_circ = Circle(radius=0.15,color=WHITE,fill_opacity=1).move_to(image_mobj.get_center())
            Inputs.add(input_circ)

        self.play(
            AnimationGroup(
                Inputs[i].animate.move_to(input_layer_neurons[i].get_center())
                for i in range(len(Inputs))
            )
        )
        self.play(FadeOut(Inputs))
        

        input_activations = image_for_vis.flatten().numpy()
        self.play(network_mob.activate_layer(0, activations=input_activations))

        self.play(network_mob.forward_pass_animation(model, image_for_model, animation_kwargs={"run_time": 0.5}))
        
        with torch.no_grad():
            output = model(image_for_model.unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()

        result_text = Tex(f"Model Prediction: {prediction}", font_size=40).to_edge(RIGHT)
        output_layer_neurons = network_mob._neuron_mobjects_list[-1]
        predicted_neuron = output_layer_neurons[prediction]
        
        self.play(Write(result_text))
        self.play(
            predicted_neuron.animate.set_color(GREEN_C).scale(1.5),
            output_labels[prediction].animate.set_color(GREEN_C).scale(1.5),
            Circumscribe(predicted_neuron, color=GREEN_C, fade_out=True, run_time=2)
        )

        self.wait(1.5)
        self.Reset_forwad_Anim(network_mob,img,input_label_text, output_labels, prediction, result_text)

    def Reset_forwad_Anim(self, network_mob, img_mob, input_label, output_labels, pred_index, pred_text):
        """
        順伝播後のMojectsの状態をリセット。
        """
        self.play(
            AnimationGroup(
                FadeOut(img_mob),
                FadeOut(input_label),
                network_mob._neuron_mobjects_list[-1][pred_index].animate.set_color(BLUE).set_fill(BLACK).scale(1/1.5),
                output_labels[pred_index].animate.set_color(WHITE).scale(1/1.5),
                FadeOut(pred_text)
            )
        )

    def construct(self):
        title = Text("PyTorch MNIST Classification", font_size=36).to_edge(UP,buff=0)
        self.play(Write(title))

        model = MLP()
        train_model_if_needed(model, "mnist_model.pth")
        model.eval()
        layer_sizes = get_layer_sizes_from_model(model)
        network = NeuralNetworkWithActivation(
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
        self.play(
            Create(network),
            Write(output_labels)
        )
        self.wait(1)

        self.MNISTforwardAnim(model,network,4,output_labels)
        self.MNISTforwardAnim(model,network,12,output_labels)
        self.MNISTforwardAnim(model,network,29,output_labels)

        

    
        