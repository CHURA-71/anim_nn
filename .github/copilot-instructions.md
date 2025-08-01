# Deep Learning Visualization with Manim

You are currently developing a high-functionality Manim library for easy visualization of deep learning concepts. The library ensures that each class is defined in a reusable manner, with clear examples of usage placed at the bottom of the program.

## Key Principles:
- Write clean, educational code with precise Manim examples that visualize deep learning concepts.
- Prioritize pedagogical value; animations should make complex topics intuitive and easy to understand.
- Design high-functionality classes and methods to be reusable across multiple projects. The goal is to encapsulate complex animation logic so that it can be invoked with simple, intuitive calls within a `Scene`.
- Follow Manim's coding style and Python's PEP 8 guidelines.
- Use descriptive variable names that reflect the mathematical or deep learning components they represent (e.g., `neuron_layer`, `weight_matrix`, `activation_graph`).
- Structure complex animations into logical, reusable helper functions and classes.

## Manim Development 🎬:
- Use `Scene` classes as the fundamental unit for each animation concept.
- Leverage Manim's core Mobjects (Mathematical Objects) like `Circle`, `Line`, `Arrow`, `Matrix`, `Tex`, and `MathTex` to represent neural network components.
- Employ a variety of `Animations` like `Create`, `FadeIn`, `Transform`, and `MoveTo` to bring concepts to life.
- Use `ValueTracker` and `updater` functions for dynamic animations that respond to changing parameters (e.g., animating training progress).
- Keep the main `construct` method clean by moving complex Mobject creation into separate helper methods.
- Use Manim's configuration system or `set_default` to maintain a consistent visual style (colors, strokes, etc.).

## Deep Learning Visualization 🧠:
- Use NumPy arrays as the primary bridge between PyTorch tensors and Manim's data plotting capabilities.
- Represent neural network layers with `VGroup` of `Circle` (neurons) and `Line` (connections).
- Visualize weights and biases using color gradients (`set_color_by_gradient`) or line thickness.
- Use `Axes` and `plot` to visualize activation functions, loss curves, and data distributions.
- Animate the step-by-step process of forward propagation and backpropagation.
- Use `MathTex` to display mathematical formulas like activation functions or the chain rule beautifully.

## Project Structure and Workflow:
- Separate the deep learning logic (e.g., a simple PyTorch model definition) from the Manim animation code. For instance, have a `models.py` for the network and an `animation.py` for the visualization.
- For modular development, place test `Scenes` directly at the bottom of the file where the corresponding classes or methods are defined. These scenes should serve as a direct demonstration and test case for the module's functionality.
- Begin each animation with a clear storyboard or plan outlining what you intend to visualize.
- Use version control (e.g., git) to manage different versions of your animations and code.
- Use lower-quality render flags (`-l`, `-m`) for rapid prototyping and testing.

## Dependencies:
- manim
- torch
- numpy
- scipy (for specific mathematical functions)
- tqdm (for progress bars in data processing)

## Key Conventions:
1. Focus on one core concept per `Scene` to maintain clarity.
2. Abstract away repetitive animation logic. If you find yourself writing similar animation sequences multiple times, refactor them into a single, configurable method or class.
3. Ensure that the timing and pacing of animations are slow enough for viewers to understand what is happening.
4. Add explanatory text (`Tex` or `Text`) to guide the viewer through the visualization.

## Example Usage:
```python
# Example class for visualizing a neural network layer
class NeuronLayerVisualization(ThreeDScene):
    def construct(self):
        # Create a layer with neurons represented by circles
        layer = VGroup()
        for i in range(5):
            neuron = Circle(radius=0.5, color=BLUE).shift(UP * i)
            layer.add(neuron)
        self.add(layer)
        self.play(FadeIn(layer))

# Place the test Scenes directly at the bottom of the file for demonstration
if __name__ == "__main__":
    scene = NeuronLayerVisualization()
    scene.render()
