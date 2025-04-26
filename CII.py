import torch

# 1. Tạo tensor 4x4 với số nguyên ngẫu nhiên từ 0 đến 9

tensor_4x4 = torch.randint(0, 10, (4, 4))
print("Tensor gốc (4x4):\n", tensor_4x4)

# 2. Trích xuất sub-tensor 2x2 từ góc dưới bên phải

sub_tensor = tensor_4x4[2:, 2:]
print("\nSub-tensor 2x2 (góc dưới phải):\n", sub_tensor)

# 3. Chuyển tensor 4x4 thành 1 chiều

flattened = tensor_4x4.flatten()
print("\nTensor sau khi flatten:\n", flattened)

# 4. Tăng tất cả phần tử ở cột thứ 2 lên 5

tensor_4x4[:, 1] += 5
print("\nTensor sau khi tăng cột 2 lên 5:\n", tensor_4x4)

# 5. Thêm một hàng mới vào tensor

new_row = torch.tensor([[10, 20, 30, 40]])
tensor_extended = torch.cat((tensor_4x4, new_row), dim=0)
print("\nTensor sau khi thêm hàng mới:\n", tensor_extended)
