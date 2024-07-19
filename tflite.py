import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image

# 加载 TFLite 模型并分配张量
interpreter = Interpreter(model_path="/mnt/sda/DataSATA/UserData/zengsheng/my_workspace/dl_works/hand_keypiont_onnx_demo/tflite_inference/quantized_tiny_yolo_v2_224_.tflite")
interpreter.allocate_tensors()

# 获取输入和输出张量的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 前处理：加载和预处理输入图像
def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert('RGB')  # 将图像转换为RGB模式
    image = image.resize((input_size[1], input_size[0]))  # 调整大小为 (width, height)
    image_np = np.array(image, dtype=np.uint8)

    flattened_rgb_data = image_np.flatten()
    output_path = './tflite_input_net_img.bin'
    with open(output_path, 'wb') as binary_file:
        binary_file.write(flattened_rgb_data.tobytes())
    print("Preprocessed image has been converted to binary file successfully.")

    print("预处理后的图像形状:", image_np.shape)
    image_np = np.expand_dims(image_np, axis=0)  # 扩展维度为 (1, 224, 224, 3)
    return image_np

# 后处理：解释模型输出
def postprocess_output(output_data):
    predicted_label = np.argmax(output_data)
    return predicted_label

# 输入图像路径
image_path = "/mnt/sda/DataSATA/UserData/zengsheng/my_workspace/dl_works/hand_keypiont_onnx_demo/tflite_inference/hand.png"

# 获取输入张量形状
input_shape = input_details[0]['shape'][1:3]  # (height, width)
print("输入张量形状:", input_shape)

# 预处理图像
preprocessed_image = preprocess_image(image_path, input_shape)

# 将输入数据设置到模型
interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

# 运行推理
interpreter.invoke()

# 获取模型输出
output_data = interpreter.get_tensor(output_details[0]['index'])

print("\n原始 output_data 形状:", output_data.shape)

# 打印原始 output_data 的前8个元素（行优先）
print("\n原始 output_data 行优先前8个元素:")
print(output_data.flatten(order='C')[:8])

# 打印原始 output_data 的前8个元素（列优先）
print("\n原始 output_data 列优先前8个元素:")
print(output_data.flatten(order='F')[:8])

# 交换第二维和第三维
swapped_output_data = np.transpose(output_data, (0, 2, 3, 1))

print("\n交换后 output_data 形状:", swapped_output_data.shape)

# 打印交换后 output_data 的前8个元素（行优先）
print("\n交换后 output_data 行优先前8个元素:")
print(swapped_output_data.flatten(order='C')[:8])

# 打印交换后 output_data 的前8个元素（列优先）
print("\n交换后 output_data 列优先前8个元素:")
print(swapped_output_data.flatten(order='F')[:8])
