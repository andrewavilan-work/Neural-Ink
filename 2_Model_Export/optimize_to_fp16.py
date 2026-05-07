import onnx
from onnxconverter_common import float16
import os

def optimize_model(input_path, output_path):
    """
    Converts an ONNX model from FP32 to FP16 precision.
    Benefits: 50% reduction in VRAM usage and significant inference speedup 
    via AMD DirectML / Unity Sentis.
    """
    print(f"Loading original model: {input_path}")
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file {input_path} not found.")
        return

    model = onnx.load(input_path)
    
    print("Converting model to FP16 and optimizing nodes...")
    # Perform FP16 conversion for GPU-accelerated inference
    model_fp16 = float16.convert_float_to_float16(model)
    
    onnx.save(model_fp16, output_path)
    print(f"[OK] Optimized model saved to: {output_path}")
    
    # Calculate and display size reduction
    size_orig = os.path.getsize(input_path) / (1024 * 1024)
    size_opt = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Size reduction: {size_orig:.2f}MB -> {size_opt:.2f}MB")

if __name__ == "__main__":
    # Define input (from export script) and output (for Unity deployment)
    input_file = "neural_ink_mi300x.onnx"
    output_file = "NeuralInk_V5_FP16.onnx"
    
    optimize_model(input_file, output_file)
