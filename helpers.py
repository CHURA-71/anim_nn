from __future__ import annotations

from manim import *

from typing import TYPE_CHECKING

import warnings
import os
import sys
from pathlib import Path
import itertools as it
import random
# import datasets

# DATA_DIR = Path(get_output_dir(), "2024/transformers/data/")
# WORD_FILE = Path(DATA_DIR, "OWL3_Dictionary.txt")


if TYPE_CHECKING:
    from typing import Optional
    from manim.typing import Vector3D

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from .Convolution import PixelsAsSquareColor

def get_paragraph(words, line_len=40, font_size=48):
    """
    Handle word wrapping
    """
    words = list(map(str.strip, words))
    word_lens = list(map(len, words))
    lines = []
    lh, rh = 0, 0
    while rh < len(words):
        rh += 1
        if sum(word_lens[lh:rh]) > line_len:
            rh -= 1
            lines.append(words[lh:rh])
            lh = rh
    lines.append(words[lh:])
    text = "\n".join([" ".join(line).strip() for line in lines])
    return Text(text, alignment="LEFT", font_size=font_size)


def softmax(logits, temperature=1.0):
    logits = np.array(logits)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # Ignore all warnings within this block
        logits = logits - np.max(logits)  # For numerical stability
        exps = np.exp(np.divide(logits, temperature, where=temperature != 0))
    
    if np.isinf(exps).any() or np.isnan(exps).any() or temperature == 0:
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1
        return result
    return exps / np.sum(exps)

#ここまでデバッグ済み

def value_to_color(
    value,
    low_positive_color=BLUE_E,
    high_positive_color=BLUE_B,
    low_negative_color=RED_E,
    high_negative_color=RED_B,
    min_value=0.0,
    max_value=10.0
):
    alpha = clip(float(inverse_interpolate(min_value, max_value, abs(value))), 0, 1)
    if value >= 0:
        colors = (low_positive_color, high_positive_color)
    else:
        colors = (low_negative_color, high_negative_color)
    return interpolate_color(*colors, alpha)


# def read_in_book(name="tale_of_two_cities"):
#     return Path(DATA_DIR, name).with_suffix(".txt").read_text()

# def load_image_net_data(dataset_name="image_net_1k"):
#     data_path = Path(Path.home(), "Documents", dataset_name)
#     image_dir = Path(data_path, "images")
#     label_category_path = Path(DATA_DIR, "image_categories.txt")
#     image_label_path = Path(data_path, "image_labels.txt")

#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)
#         image_data = datasets.load_from_disk(str(data_path))
#         indices = range(len(image_data))
#         categories = label_category_path.read_text().split("\n")
#         labels = [categories[image_data[index]['label']] for index in indices]
#         image_label_path.write_text("\n".join(labels))
#         for index in ProgressDisplay(indices):
#             image = image_data[index]['image']
#             image.save(str(Path(image_dir, f"{index}.jpeg")))


#     labels = image_label_path.read_text().split("\n")
#     return [
#         (Path(image_dir, f"{index}.jpeg"), label)
#         for index, label in enumerate(labels)
#     ]


def show_matrix_vector_product(scene, matrix, vector, buff=0.25, x_max=999, fix_in_frame=False):
    """
    show_matrix_vector_productの使用例
        class ShowMatrixProductTest(Scene):
            def construct(self):
                matrix = WeightMatrix(shape=(8,6),ellipses_col=None,ellipses_row=None)
                vector = WeightMatrix(shape=(6,1))
                matrix.to_edge(UP)
                vector.next_to(matrix, RIGHT)
                group = VGroup(matrix, vector)
                group.move_to(ORIGIN).shift(LEFT*1.5)
                self.add(group)
                show_matrix_vector_product(self, matrix, vector)
                self.wait(2)
    """
    # "=" 記号
    eq = Tex("=")
    eq.set_width(0.5 * vector.width)

    # rhsの作成
    shape = (len(matrix.get_rows()), 1)
    rhs = NumericEmbedding(
        values=x_max * np.ones(shape),
        value_range=(-x_max, x_max),
        decimal_config=dict(include_sign=True, edge_to_fix=ORIGIN),
        ellipses_row=getattr(matrix, "ellipses_row", None),
    )
    rhs.scale(vector[0][0].height / rhs[0][0].height)
    eq.next_to(vector, RIGHT, buff=buff)
    rhs.next_to(eq, RIGHT, buff=buff)

    if fix_in_frame and config.renderer == "opengl":
        eq.fix_in_frame()
        rhs.fix_in_frame()

    scene.play(FadeIn(eq), FadeIn(rhs.get_brackets()))

    n_rows = len(matrix.get_rows())
    for n, row, entry in zip(it.count(), matrix.get_rows(), rhs.get_rows()):
        if hasattr(matrix, "ellipses_row") and matrix.ellipses_row is not None and n == (matrix.ellipses_row % n_rows):
            scene.add(entry)
        else:
            matrix_row_vector_product(
                scene, row, vector, entry[0], fix_in_frame=fix_in_frame
            )

    return eq, rhs

def matrix_row_vector_product(scene:Scene, row:WeightMatrix, vector:WeightMatrix, entry, fix_in_frame=False):
    def get_rect(elem):
        sur_rect = SurroundingRectangle(elem, buff=0.1).set_stroke(YELLOW, 2)
        if fix_in_frame and config.renderer == "opengl":
            sur_rect.fix_in_frame()
        return sur_rect
    row_rects = VGroup(*map(get_rect, row))
    vect_rects = VGroup(*map(get_rect, *vector[:-2]))
    partial_values = [0]
    for e1, e2 in zip(row, vector.get_entries()):
        if isinstance(e1, DecimalNumber) and not isinstance(e2, DecimalNumber): 
            increment=0
        else:
            val1 = round(e1.get_value(), int(e1.num_decimal_places))
            val2 = round(e2.get_value(), int(e2.num_decimal_places))
            increment = val1 * val2
        partial_values.append(partial_values[-1] + increment)
    n_values = len(partial_values)

    
    scene.play(
        AnimationGroup(
            AnimationGroup(
                LaggedStart(*[FadeIn(m) for m in row_rects], lag_ratio=0.1),
                LaggedStart(*[FadeIn(m) for m in vect_rects], lag_ratio=0.1),
                UpdateFromAlphaFunc(entry, lambda m, a: m.set_value(
                    partial_values[min(int(np.round(a * n_values)), n_values - 1)]
                )),
            ),
            AnimationGroup(
                LaggedStart(*[FadeOut(m) for m in row_rects], lag_ratio=0.1),
                LaggedStart(*[FadeOut(m) for m in vect_rects], lag_ratio=0.1),
            ),
            rate_func=linear,
            lag_ratio=1.5,
        ),
        lag_ratio=0.1,
        run_time=1,
    )



def get_full_matrix_vector_product(
    mat_sym="w",
    vect_sym="x",
    n_rows=5,
    n_cols=5,
    mat_sym_color=BLUE,
    height=3.0,
    ellipses_row: int = -2,
    ellipses_col: int = -2,
    ):
    """
    get_full_matrix_vector_productの使用例
        class FullMatVecProductTest(Scene):
            def construct(self):
                matrix, vector, equals, rhs = get_full_matrix_vector_product(
                    mat_sym="w",
                    vect_sym="x",
                    mat_sym_color=BLUE,
                    height=3.5,
                    ellipses_row=2,
                    ellipses_col=2,
                )

                expr = VGroup(matrix, vector, equals, rhs).arrange(RIGHT, buff=0.8).scale_to_fit_width(config.frame_width*0.95)
                self.play(Write(expr))
                self.wait()

    """
    m_indices = list(range(1, n_rows + 1))
    n_indices = list(range(1, n_cols + 1))

    # Matrix 左辺
    matrix_entries = []
    for m_i, m in enumerate(m_indices):
        row = []
        for n_j, n in enumerate(n_indices):
            if m_i == ellipses_row and n_j == ellipses_col:
                row.append(R"\ddots")
            elif m_i == ellipses_row:
                row.append(R"\vdots")
            elif n_j == ellipses_col:
                row.append(R"\cdots")
            else:
                row.append(Rf"{mat_sym}_{{{m},{n}}}")
        matrix_entries.append(row)

    matrix = Matrix(matrix_entries,element_alignment_corner=tuple(ORIGIN))
    matrix.set_height(height)
    matrix.get_entries().set_color(mat_sym_color)

    # Vector 左辺
    vector_entries = [
        [Rf"{vect_sym}_{{{n}}}"] if i != ellipses_row else [R"\vdots"]
        for i, n in enumerate(m_indices)
    ]
    vector = Matrix(vector_entries,element_alignment_corner=tuple(ORIGIN))
    vector.match_height(matrix)
    vector.next_to(matrix, RIGHT)

    # 等号
    equals = Tex("=", font_size=72)
    equals.next_to(vector, RIGHT)

    # 右辺（結果の各行）
    rhs_entries = []
    for m_i, m in enumerate(m_indices):
        row = []
        for n_j, n in enumerate(n_indices):
            if m_i == ellipses_row and n_j == ellipses_col:
                row.append(R"\ddots")
            elif m_i == ellipses_row:
                row.append(R"\vdots")
            elif n_j == ellipses_col:
                row.append(R"\cdots")
            else:
                row.append(Rf"{mat_sym}_{{{m},{n}}} {vect_sym}_{{{n}}}")
        rhs_entries.append(row)

    rhs = Matrix(rhs_entries,h_buff=1.8,element_alignment_corner=tuple(ORIGIN))
    rhs.match_height(matrix)
    rhs.next_to(equals, RIGHT)

    def is_dot(tex_mob):
        text = tex_mob.get_tex_string()
        return text in [r"\vdots", r"\ddots"]

    for row in rhs.get_rows():
        for i in range(len(row) - 1):
            e1, e2 = row[i], row[i + 1]
            if is_dot(e1) and is_dot(e2):
                continue
            plus = Tex("+")
            plus.match_height(e1)
            plus.move_to((e1.get_right() + e2.get_left()) / 2)
            plus.align_to(e1, UP)
            e2.add(plus)
    return matrix, vector, equals, rhs


def show_symbolic_matrix_vector_product(
        scene:Scene,
        matrix:Matrix,
        vector:Vector,
        rhs:Vector,
        run_time_per_row=0.75,
        show_rhs_later=False
    ):
    """
    show_symbolic_matrix_vector_productの使用例
        class MatVecProductTest(Scene):
            def construct(self):
                # 左辺の行列 A（2x2）
                matrix = Matrix([
                    ["a_{11}", "a_{12}"],
                    ["a_{21}", "a_{22}"],
                ])
                # # 掛けるベクトル x（2x1）
                vector = Matrix([
                    ["x_1"],
                    ["x_2"],
                ])
                # 右辺（2x1）
                rhs = Matrix([
                    ["a_{11} x_1 + a_{12} x_2"],
                    ["a_{21} x_1 + a_{22} x_2"],
                ])
                # 左から右に並べて表示
                group = VGroup(matrix, vector, rhs).arrange(RIGHT, buff=1)
                self.play(Write(matrix),Write(vector))
                self.wait(0.5)
                # アニメーション実行
                show_symbolic_matrix_vector_product(self, matrix, vector, rhs,show_rhs_later=True)

                self.wait(1)
    """
    last_rects = VGroup()
    if show_rhs_later:
        # 非表示で配置（透明）
        for row in rhs.get_rows():
            row.set_opacity(0)
        scene.add(rhs)
    else:
        scene.play(Write(rhs))
    for mat_row, rhs_row in zip(matrix.get_rows(), rhs.get_rows()):
        mat_rects = VGroup(*map(SurroundingRectangle, mat_row))
        vect_rects = VGroup(*map(SurroundingRectangle, vector.get_columns()[0]))
        rect_group = VGroup(mat_rects, vect_rects)
        rect_group.set_stroke(YELLOW, 2)
        scene.play(
            FadeOut(last_rects),
            *(
                ShowIncreasingSubsets(group, rate_func=linear)
                for group in [mat_rects, vect_rects, rhs_row]
            ),
            run_time=run_time_per_row,
        )
        last_rects = rect_group
    scene.play(FadeOut(last_rects))



def data_flying_animation(
    point,
    vect=2 * DOWN + RIGHT,
    color=GREY_C,
    max_opacity=0.75,
    font_size=48,
    fix_in_frame=False
    ):
    word = Text("Data", color=color, font_size=font_size)
    if fix_in_frame:
        word.fix_in_frame()
    return UpdateFromAlphaFunc(
        word, lambda m, a: m.move_to(
            interpolate(point, point + vect, a)
        ).set_opacity(there_and_back(a) * max_opacity)
    )


def get_data_modifying_matrix_anims(
    matrix,
    word_shape=(5, 10),
    alpha_maxes=(0.7, 0.9),
    shift_vect=2 * DOWN + RIGHT,
    run_time=3,
    fix_in_frame=False,
    font_size=48,
    ):
    x_min, x_max = [matrix.get_x(LEFT), matrix.get_x(RIGHT)]
    y_min, y_max = [matrix.get_y(UP), matrix.get_y(DOWN)]
    z = matrix.get_z()
    points = np.array([
        [
            interpolate(x_min, x_max, a1),
            interpolate(y_min, y_max, a2),
            z,
        ]
        for a1 in np.linspace(0, alpha_maxes[1], word_shape[1])
        for a2 in np.linspace(0, alpha_maxes[0], word_shape[0])
    ])
    # pointsが空でない場合のみLaggedStartを実行
    if len(points) > 0:
        flying_anim = LaggedStart(
            (data_flying_animation(p, vect=shift_vect, fix_in_frame=fix_in_frame, font_size=font_size)
            for p in points),
            lag_ratio=1 / len(points),
            run_time=run_time
        )
    else:
        flying_anim = Wait(run_time)
    
    return [
        flying_anim,
        RandomizeMatrixEntries(matrix, run_time=run_time),
    ]


def data_modifying_matrix(scene, matrix, *args, **kwargs):
    """
    data_modifying_matrixの使用例
        class DataModifyingMatrixTest(Scene):
            def construct(self):
                matrix = WeightMatrix()
                data_modifying_matrix(self,matrix)
                self.wait()
    """
    anims = get_data_modifying_matrix_anims(matrix, *args, **kwargs)
    scene.play(*anims)


def create_pixels(image_mob:ImageMobject, pixel_width=0.1):
    """
    create_pixelsの使用例
        class CreatePixelTest(Scene):
            def construct(self):
                image_path = "./source/cat.png"
                image_mob = ImageMobject(image_path).scale_to_fit_height(config.frame_height)
                self.wait()
                self.play(FadeIn(image_mob))
                self.wait(2)
                self.play(FadeOut(image_mob))
                self.wait()
                pixels = create_pixels(image_mob, pixel_width=0.5)
                self.play(Create(pixels))
                self.wait(2)
    """
    pixels = PixelsAsSquareColor(image_mob.pixel_array,square_side_length=pixel_width)
    pixels.scale_to_fit_height(config.frame_height)
    return pixels

def get_network_connections(layer1, layer2, max_width=2.0, opacity_exp=1.0):
    """
    get_network_connectionsの使用例
        class GetNetConTest(Scene):
            def construct(self):
                vector = WeightMatrix(shape=(7))
                neurons = VGroup(*[Dot(radius=0.3) for _ in range(10)]).arrange(DOWN,buff=0.3)
                mob = VGroup(vector, neurons).move_to(ORIGIN).arrange(RIGHT,buff=config.frame_width*0.4)
                mob.scale_to_fit_height(config.frame_height*0.9)
                bg = SurroundingRectangle(vector,buff=0.5,stroke_color=BLACK,fill_color=BLACK,fill_opacity=1).set_zet_index(1)
                self.add(bg, mob.set_zet_index(2))
                self.wait()
                con_mob = get_network_connections(vector.get_entries(), neurons)
                self.play(Create(con_mob))
                self.wait()
    """
    radius = layer1[0].width / 2
    return VGroup(
        Line(n1.get_center(), n2.get_center(), buff=radius).set_stroke(
            color=value_to_color(random.uniform(-10, 10)),
            width=max_width * random.random(),
            opacity=random.random()**opacity_exp,
        )
        for n1 in layer1
        for n2 in layer2
    )


def get_vector_pair(angle_in_degrees=90, length=1.0, colors=(BLUE, BLUE)):
    """
    get_vector_pairの使用例
        class GetvecpairTest(Scene):
            def construct(self):
                vector_pair = get_vector_pair(angle_in_degrees=60, length=2.0, colors=(RED, GREEN))
                vector_pair.move_to(ORIGIN).scale(1.5)
                self.add(vector_pair)
    """
    angle = angle_in_degrees * DEGREES
    v1 = Vector(length * RIGHT)
    # v2 = v1.copy().rotate(angle, about_point=ORIGIN)
    v2_dir = rotate_vector(length*RIGHT,angle)
    v2 = Vector(v2_dir)
    v1.set_color(colors[0])
    v2.set_color(colors[1])
    arc = Arc(
        radius=0.5,
        angle=angle,
        color=WHITE,
        stroke_width=2,
    )
    label = MathTex(rf"{angle_in_degrees}^\circ", font_size=24)
    label.next_to(
        arc.point_from_proportion(0.5),
        direction=normalize(arc.point_from_proportion(0.5)),
        buff=SMALL_BUFF,
    )
    return VGroup(v1, v2, arc, label)


class NeuralNetwork(VGroup):
    """
    NeuralNetworkの使用例
        class  NeuralNetworkTest(Scene):
            def construct(self):
                mob = NeuralNetwork().scale_to_fit_height(config.frame_width*0.5)
                mob.move_to(ORIGIN)
                self.play(FadeIn(mob))
                self.wait()
                self.play(mob.animate.randomize_layer_values())
                self.wait()
                self.play(mob.animate.randomize_line_style())
    """
    def __init__(
        self,
        layer_sizes=[6, 12, 6],
        neuron_radius=0.1,
        v_buff_ratio=0.2,
        h_buff_ratio=7.0,
        max_stroke_width=2.0,
        stroke_decay=2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_stroke_width = max_stroke_width
        self.stroke_decay = stroke_decay
        layers = VGroup(*(
            VGroup(*[
                Dot(radius=neuron_radius) for _ in range(n)
            ]).arrange(DOWN, buff=v_buff_ratio)
            for n in layer_sizes
        ))
        layers.arrange(RIGHT, buff=h_buff_ratio * layers[0].width)

        lines = VGroup(*(
            VGroup(*[
                Line(
                    l1[i].get_center(),
                    l2[j].get_center(),
                    buff=neuron_radius,
                )
                for i in range(len(l1))
                for j in range(len(l2))
            ])
            for l1, l2 in zip(layers[:-1], layers[1:])
        ))

        self.add(layers, lines)
        self.layers = layers
        self.lines = lines

        self.randomize_layer_values()
        self.randomize_line_style()

    def randomize_layer_values(self):
        for group in self.lines:
            for line in group:
                line.set_stroke(
                    value_to_color(random.uniform(-10, 10)),
                    self.max_stroke_width * random.random()**self.stroke_decay,
                )
        return self

    def randomize_line_style(self):
        for layer in self.layers:
            for dot in layer:
                dot.set_stroke(WHITE, 1)
                dot.set_fill(WHITE, opacity=random.random())
        return self


def random_bright_color_morewhite() -> ManimColor:
            # random_bright_colorをよりも白に比重を置く
            # ContextAnimationで使用
            curr_rgb = color_to_rgb(random_color())
            new_rgb = 0.25 * curr_rgb + 0.75 * np.ones(3)
            return ManimColor(new_rgb)

class ContextAnimation(AnimationGroup):
    """
    ContextAnimationの使用例
        class ContextAnimTest(Scene):
            def construct(self):
                mob = EmbeddingArray()
                self.add(mob)
                # カッコを除去
                mob = mob[:-2]
                # Dots以外をSourceに
                sources = mob[-2]
                # EmbeddingArrayの最後の要素をTargetに
                target = [*sources][-1]
                self.wait()
                self.play(ContextAnimation(target, sources, run_time=1))
                self.play(ContextAnimation(target, sources, run_time=1))
                self.wait()
    """
    def __init__(
        self,
        target,
        sources,
        direction=UP,
        time_width=2,
        min_stroke_width=1,
        max_stroke_width=6,
        lag_ratio=None,
        strengths=None,
        run_time=3,
        fix_in_frame=False,
        path_arc=PI / 2,
        **kwargs,
    ):
        arcs = VGroup()
        if strengths is None:
            strengths = np.random.random(len(sources))**2

        for source, strength in zip(sources, strengths):
            sign = direction[1] * (-1)**int(source.get_x() < target.get_x())
            arc = ArcBetweenPoints(
                source.get_edge_center(direction),
                target.get_edge_center(direction),
                angle=sign * path_arc,
                stroke_color=random_bright_color_morewhite(),
                stroke_width=interpolate(
                    min_stroke_width,
                    max_stroke_width,
                    strength
                ),
            )
            arcs.add(arc)

        if fix_in_frame:
            arcs.fix_in_frame()

        # arcsが空でない場合のみアニメーションを実行
        if len(arcs) > 0:
            arcs.shuffle()
            lag_ratio = 0.5 / len(arcs) if lag_ratio is None else lag_ratio

            super().__init__(
                *[
                    ShowPassingFlash(arc, time_width=time_width)
                    for arc in arcs
                ],
                lag_ratio=lag_ratio,
                run_time=run_time,
                **kwargs,
            )
        else:
            # arcsが空の場合はWaitアニメーションを実行
            super().__init__(
                Wait(run_time),
                **kwargs,
            )



class TextLabeledArrow(Arrow):
    """
    矢印の終点にテキストラベルが付いたArrowクラス。

    Parameters
    ----------
    args
        Arrowクラスに渡される引数 (始点、終点など)。
    label_text
        矢印に表示するテキスト。
    font_size
        ラベルのフォントサイズ。
    label_buff
        矢印の終点からラベルまでの距離（バッファ）。
    direction
        矢印の終点から見て、どの方向にラベルを配置するかを指定するベクトル。
        指定しない場合は、矢印自体の向きが使われます。
    label_rotation
        ラベルの回転角度。
    kwargs
        Arrowクラスに渡されるその他のキーワード引数。
    """
    """
    TextLabeledArrowの使用例
        class LabeledArrowTest(ThreeDScene):
            def construct(self):
                # 3Dの座標軸とカメラの初期設定
                axes = ThreeDAxes()
                self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES, zoom=0.8)
                self.add(axes)

                # 1. X, Y, Z軸に沿った矢印を生成
                arrow_x = TextLabeledArrow(ORIGIN, 3 * RIGHT, label_text="X-Axis", color=BLUE)
                arrow_y = TextLabeledArrow(ORIGIN, 3 * UP, label_text="Y-Axis", color=GREEN)
                arrow_z = TextLabeledArrow(ORIGIN, 3 * OUT, label_text="Z-Axis", color=YELLOW)

                # 2. 矢印とラベルをそれぞれシーンに追加してアニメーション
                # このクラスはラベルを自動で追加しないため、個別に追加する必要があります。
                self.play(
                    Create(arrow_x),
                    Create(arrow_y),
                    Create(arrow_z),
                    # ラベルも個別に追加する
                    Write(arrow_x.label),
                    Write(arrow_y.label),
                    Write(arrow_z.label),
                    run_time=2
                )
                self.wait(1)

                # 3. カメラを回転させて3Dであることを確認
                self.move_camera(phi=45 * DEGREES, theta=120 * DEGREES, run_time=3)
                self.wait(1)
                self.move_camera(phi=80 * DEGREES, theta=0 * DEGREES, run_time=3)
                self.wait(1)
    """
    def __init__(
        self,
        *args,
        label_text: Optional[str] = None,
        font_size: float = 24,
        label_buff: float = 0.1,
        direction: Optional[Vector3D] = None,
        label_rotation: float = PI / 2,
        **kwargs
    ):
        buff_value = kwargs.pop('buff', 0)
        super().__init__(*args, buff=buff_value,**kwargs)
        if label_text is not None:
            start, end = self.get_start_and_end()
            label = Text(label_text, font_size=font_size)
            label.set_fill(self.get_color())
            label.set_background_stroke()
            label.rotate(label_rotation, RIGHT)            
            if direction is None:
                direction = normalize(end-start)
            label.next_to(self.get_end(), direction, buff=label_buff)
            self.label = label
        else:
            self.label = None


class WeightMatrix(Matrix):
    def __init__(
        self,
        values: Optional[np.ndarray] = None,
        shape: tuple[int, ...] = (6, 6),
        value_range: tuple[float, float] = (-9.9, 9.9),
        num_decimal_places: int = 1,
        bracket_h_buff: float = 0.1,
        decimal_config: dict = {"include_sign": True},
        low_positive_color: ManimColor = BLUE_E,
        high_positive_color: ManimColor = BLUE_B,
        low_negative_color: ManimColor = RED_E,
        high_negative_color: ManimColor = RED_B,
        ellipses_row: Optional[int] = -2,
        ellipses_col: Optional[int] = -2,
        **kwargs,
    ):
        if values is not None:
            values = np.array(values)
            if values.ndim == 1:
                values = values.reshape((-1, 1))  # 1D → 列ベクトル
            shape = values.shape
        else:
            # shapeがintか1要素タプルなら列ベクトルに変換
            if isinstance(shape, int):
                shape = (shape, 1)
            elif isinstance(shape, tuple) and len(shape) == 1:
                shape = (shape[0], 1)
        self.shape = shape
        self.value_range = value_range
        self.low_positive_color = low_positive_color
        self.high_positive_color = high_positive_color
        self.low_negative_color = low_negative_color
        self.high_negative_color = high_negative_color
        self.ellipses_row = ellipses_row
        self.ellipses_col = ellipses_col
        self.num_decimal_places = num_decimal_places
        self.decimal_config = decimal_config

        if values is None:
            values = np.random.uniform(*self.value_range, size=shape)

        self.display_matrix = self._make_display_matrix(values)

        super().__init__(
            self.display_matrix,
            element_to_mobject=self._element_to_mobject,
            element_to_mobject_config={},
            bracket_h_buff=bracket_h_buff,
            element_alignment_corner=ORIGIN,
            **kwargs,
        )

        self.reset_entry_colors()

    def _make_display_matrix(self, values):
        rows, cols = self.shape
        matrix = []
        values = np.array(values)

        target_row = rows + self.ellipses_row if self.ellipses_row is not None else 0
        target_col = cols + self.ellipses_col if self.ellipses_col is not None else 0

        for i in range(rows):
            row = []
            for j in range(cols):
                if (
                    self.ellipses_row is not None
                    and self.ellipses_col is not None
                    and i == target_row
                    and j == target_col
                ):
                    row.append(r"\ddots")
                elif (
                    self.ellipses_row is not None
                    and i == target_row
                    and j < target_col
                ):
                    row.append(r"\vdots")
                elif (
                    self.ellipses_col is not None
                    and j == target_col
                    and i < target_row
                ):
                    row.append(r"\cdots")
                else:
                    row.append(float(values[i][j]))
            matrix.append(row)
        return matrix

    def _element_to_mobject(self, item):
        if isinstance(item, str):
            return MathTex(item)
        else:
            return DecimalNumber(item, num_decimal_places=self.num_decimal_places, **self.decimal_config)

    def reset_entry_colors(self):
        for entry in self.get_entries():
            if isinstance(entry, DecimalNumber):
                value = entry.get_value()
                color = value_to_color(
                    value,
                    self.low_positive_color,
                    self.high_positive_color,
                    self.low_negative_color,
                    self.high_negative_color,
                    0,
                    max(abs(self.value_range[0]), abs(self.value_range[1])),
                )
                entry.set_color(color)
        return self


class NumericEmbedding(WeightMatrix):
    def __init__(
        self,
        values: Optional[np.ndarray] = None,
        shape: Optional[tuple[int, ...]] = None,
        length: int = 7,
        num_decimal_places: int = 1,
        ellipses_row: int = -2,
        ellipses_col: int = -2,
        value_range: tuple[float, float] = (-9.9, 9.9),
        bracket_h_buff: float = 0.1,
        decimal_config=dict(include_sign=True),
        dark_color: ManimColor = GREY_C,
        light_color: ManimColor = WHITE,
        **kwargs,
    ):
        # shape 自動設定または reshape
        if values is not None:
            if len(values.shape) == 1:
                values = values.reshape((values.shape[0], 1))
            shape = values.shape
        if shape is None:
            shape = (length, 1)

        super().__init__(
            values=values,
            shape=shape,
            value_range=value_range,
            num_decimal_places=num_decimal_places,
            bracket_h_buff=bracket_h_buff,
            decimal_config=decimal_config,
            low_positive_color=dark_color,
            high_positive_color=light_color,
            low_negative_color=dark_color,
            high_negative_color=light_color,
            ellipses_row=ellipses_row,
            ellipses_col=ellipses_col,
            **kwargs,
        )

        # 符号付きの 0 のマイナスを非表示にする
        for entry in self.get_entries():
            if isinstance(entry, DecimalNumber) and entry.get_value() == 0:
                # 数値部分の [0] は "-" に相当するパーツ
                if len(entry) > 0:
                    entry[0].set_opacity(0)


class EmbeddingArray(VGroup):
    def __init__(
        self,
        shape=(10, 9),
        height=4,
        dots_index=-4,
        buff_ratio=0.4,
        bracket_color=GREY_B,
        backstroke_width=3,
        add_background_rectangle=False,
    ):
        super().__init__()

        # Embeddings
        embeddings = VGroup(*[
            NumericEmbedding(length=shape[0])
            for _ in range(shape[1])
        ])
        embeddings.height = height
        buff = buff_ratio * embeddings[0].width
        embeddings.arrange(RIGHT, buff=buff)

        # Background rectangle
        if add_background_rectangle:
            for embedding in embeddings:
                embedding.add_background_rectangle()

        # Bracketsを左右別々に作成し、正確に next_to で配置
        left_bracket = MathTex(r"\left[")
        right_bracket = MathTex(r"\right]")

        # 高さ調整（推奨されているやり方）
        target_height = embeddings.height
        left_bracket.height = target_height
        right_bracket.height = target_height

        left_bracket.next_to(embeddings, LEFT, buff=0.1)
        right_bracket.next_to(embeddings, RIGHT, buff=0.1)

        left_bracket.set_color(bracket_color)
        right_bracket.set_color(bracket_color)

        left_bracket.set_stroke(width=backstroke_width)
        right_bracket.set_stroke(width=backstroke_width)


        # dots グループ（あとで追加）
        dots = VGroup()

        # 子要素として追加（重複なし）
        self.add(embeddings, dots, left_bracket, right_bracket)

        # 属性保存
        self.embeddings = embeddings
        self.dots = dots
        self.brackets = VGroup(left_bracket, right_bracket)

        # dots の置き換え
        if dots_index is not None:
            self.swap_embedding_for_dots(dots_index)

    def swap_embedding_for_dots(self, dots_index=-4):
        to_replace = self.embeddings[dots_index]
        dots_tex = MathTex(r"\dots")
        dots_tex.set_width(0.75 * to_replace.width)
        dots_tex.move_to(to_replace)
        self.embeddings.remove(to_replace)
        self.dots.add(dots_tex)
        return self


class RandomizeMatrixEntries(Animation):
    """
    RandomizeMatrixEntriesの使用例
        class MatrixTest(Scene):
            def construct(self):
                mat = WeightMatrix(shape=(4, 4)).scale_to_fit_width(config.frame_width*0.7)
                self.add(mat)
                self.wait(0.5)
                self.play(RandomizeMatrixEntries(mat, run_time=4, lag_ratio=0.1))   #デバッグ済み
                self.wait()
    """
    def __init__(self, matrix, **kwargs):
        self.matrix = matrix
        self.entries = matrix.get_entries()

        self.start_values = [
            entry.get_value() if isinstance(entry, DecimalNumber) else None
            for entry in self.entries
        ]
        self.target_values = [
            np.random.uniform(matrix.value_range[0], matrix.value_range[1])
            if isinstance(entry, DecimalNumber) else None
            for entry in self.entries
        ]

        super().__init__(matrix, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        for i, entry in enumerate(self.entries):
            if not isinstance(entry, DecimalNumber):
                continue

            sub_alpha = self.get_sub_alpha(alpha, i, len(self.entries))

            start = self.start_values[i]
            target = self.target_values[i]
            if start is not None and target is not None:
                entry.set_value(interpolate(start, target, sub_alpha))

        self.matrix.reset_entry_colors()

class AbstractEmbeddingSequence(MobjectMatrix):
    pass


class Needle(Polygon):
    def __init__(self, length=1, width=5, **kwargs):
        # 針の形状を定義する3つの頂点
        points = [
            [0, width / 2, 0],
            [length, 0, 0],
            [0, -width / 2, 0],
        ]
        angle = None
        # Polygonとして初期化
        super().__init__(*points, **kwargs)
        # ストロークは使わず、塗りで色を表現する
        self.set_stroke(width=0)
        self.set_fill(opacity=1.0)
        self.angle = angle

    def get_angle(self):
        return self.angle
    
    def set_angle(self, angle):
        self.angle = angle
    


class Dial(VGroup):
    def __init__(
        self,
        radius=0.5,
        relative_tick_size=0.2,
        value_range=(0, 1, 0.1),
        initial_value=0,
        arc_angle=270 * DEGREES,
        stroke_width=2,
        stroke_color=WHITE,
        needle_color=BLUE,
        needle_stroke_width=5.0,
        value_to_color_config=dict(),
        set_anim_streak_color=TEAL,
        set_anim_streak_width=4,
        set_value_anim_streak_density=6,
        **kwargs
    ):
        # パラメータの保持
        super().__init__(**kwargs)
        self.value_range = value_range
        self.value_to_color_config = value_to_color_config
        self.set_anim_streak_color = set_anim_streak_color
        self.set_anim_streak_width = set_anim_streak_width
        self.set_value_anim_streak_density = set_value_anim_streak_density

        # Main dial
        self.arc = Arc(radius, start_angle=arc_angle / 2+90*DEGREES, angle=-arc_angle)

        low, high, step = value_range
        n_values = int(1 + (high - low) / step)
        tick_points = map(self.arc.point_from_proportion, np.linspace(0, 1, n_values))
        self.ticks = VGroup(*(
            Line((1.0 - relative_tick_size) * point, point)
            for point in tick_points
        ))
        self.bottom_point = VectorizedPoint(radius * DOWN)
        for mob in self.arc, self.ticks:
            mob.set_stroke(stroke_color, stroke_width)

        # 針（Needle）のポリゴン化
        self.needle = Needle(
            length=radius,
            width=needle_stroke_width,
            fill_color=needle_color
        )
        # 針をDialの中心へ移動
        self.needle.move_to(self.arc.get_arc_center(), aligned_edge=LEFT)
        
        # オブジェクトの追加
        self.add(self.arc, self.ticks, self.bottom_point, self.needle)

        # 初期値を設定
        self.set_value(initial_value)

    def value_to_angle(self, value):
        # 値を角度に変換するヘルパー関数
        low, high, step = self.value_range
        alpha = inverse_interpolate(low, high, value)
        alpha = np.clip(alpha, 0, 1)
        start_angle = self.arc.start_angle
        stop_angle = self.arc.start_angle + self.arc.angle
        return interpolate(start_angle, stop_angle, alpha)


    def set_value(self, value):
        # 針（Polygon）の角度と色を直接設定する
        target_angle = self.value_to_angle(value)
        target_color = value_to_color(
            value, min_value=self.value_range[0], max_value=self.value_range[1],
            **self.value_to_color_config
        )
        self.needle.set_angle(target_angle)
        self.needle.set_fill(target_color)

    def animate_set_value(self, value, **kwargs):
        # 針（Polygon）をアニメーションさせる
        target_angle = self.value_to_angle(value)
        target_color = value_to_color(
            value, min_value=self.value_range[0], max_value=self.value_range[1],
            **self.value_to_color_config
        )
        start_angle = self.needle.get_angle()
        

        # streakエフェクト（元のコードのロジックを角度ベースで再現）
        path_arc = target_angle - start_angle
        density = self.set_value_anim_streak_density
        radii = np.linspace(0, 0.5 * self.width, density + 1)[1:]
        diff_arcs = VGroup(*(
            Arc(
                radius, start_angle=start_angle, angle=path_arc,
                arc_center=self.arc.get_arc_center()
            )
            for radius in radii
        ))
        diff_arcs.set_stroke(self.set_anim_streak_color, self.set_anim_streak_width)

        self.needle.set_angle(target_angle)

        return AnimationGroup(
            Rotate(self.needle, path_arc, about_point=self.arc.get_arc_center()),
            *(
            ShowPassingFlash(diff_arc, time_width=1.5, **kwargs)
            for diff_arc in diff_arcs
        ))

    def get_random_value(self):
        low, high, step = self.value_range
        return interpolate(low, high, random.random())



class MachineWithDials(VGroup):
    """
        class MachineDialTest(Scene):
            '''
            MachineWithDialsnの使用例
            '''
            def construct(self):
                dial = MachineWithDials()
                self.add(dial)
                self.play(dial.random_change_animation())
                self.wait()
                self.play(dial.random_change_animation())
                self.wait()
                self.play(dial.rotate_all_dials())
                self.wait()
                self.play(dial.rotate_all_dials())
                self.wait()
    """
    default_dial_config = dict(
        stroke_width=1.0,
        needle_stroke_width=0.1,
        relative_tick_size=0.25,
        set_anim_streak_width=2,
    )
    def __init__(
        self,
        width=5.0,
        height=4.0,
        n_rows=6,
        n_cols=8,
        dial_buff_ratio=0.5,
        stroke_color=WHITE,
        stroke_width=1,
        fill_color=GREY_D,
        fill_opacity=1.0,
        dial_config=dict(),
    ):
        super().__init__()
        box = Rectangle(height=height, width=width)
        box.set_stroke(stroke_color, stroke_width)
        box.set_fill(fill_color, fill_opacity)
        self.box = box

        dial_config = dict(**self.default_dial_config, **dial_config)
        dials = VGroup(*[Dial(**dial_config) for _ in range(n_rows * n_cols)])
        buff = dials[0].width * dial_buff_ratio
        dials.arrange_in_grid(n_rows, n_cols, buff=buff)
        dials.set_width(box.get_width()*0.95)
        dials.move_to(box)
        for dial in dials:
            dial.set_value(dial.get_random_value())
            dial.needle.rotate(dial.needle.angle,about_point=dial.arc.get_arc_center()) #pyright: ignore
        self.dials = dials

        self.add(box, dials)

    def random_change_animation(self, lag_factor=0.5, run_time=3.0, **kwargs):
        # dialsが空でない場合のみLaggedStartを実行
        if len(self.dials) > 0:
            return LaggedStart(
                *(
                    dial.animate_set_value(dial.get_random_value())
                    for dial in self.dials
                ), lag_ratio=lag_factor / len(self.dials),
                run_time=run_time,
                **kwargs
            )
        else:
            return Wait(run_time)

    def rotate_all_dials(self, run_time=2, lag_factor=1.0):
        # dialsが空でない場合のみLaggedStartを実行
        if len(self.dials) > 0:
            shuffled_dials = list(self.dials)
            random.shuffle(shuffled_dials)
            return LaggedStart(
                *(
                    Rotate(dial.needle, TAU, about_point=dial.get_center())
                    for dial in shuffled_dials
                ),
                lag_ratio=lag_factor / len(self.dials)
            )
        else:
            return Wait(run_time)
