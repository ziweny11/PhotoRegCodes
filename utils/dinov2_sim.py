import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

# DINOv2
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define the image transformation
transform = trasnsforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the required input size  # Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet statistics
])

# Load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Classify an image
def classify_image(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        output = dinov2_vits14(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

def classify_img2(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = dinov2_vits14(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

# #img need to be nparray
# def img2d2(img):




#problematic:
def img_sim(i1, i2):
    with torch.no_grad():
        # Ensure the inputs are correctly normalized tensors
        if i1.dim() == 3:
            i1 = transform(i1).unsqueeze(0)
        if i2.dim() == 3:
            i2 = transform(i2).unsqueeze(0)
        # Classify images
        iv1 = dinov2_vits14(i1)
        iv2 = dinov2_vits14(i2)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(iv1, iv2)
    
    return cos_sim



# import torch
# import cv2
# from PIL import Image
# from torchvision import transforms
# import torch.nn.functional as F
# import os
# from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
#     read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
# # DINOv2
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')


# # Define the image transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to the required input size
#     transforms.ToTensor(),  # Convert to a PyTorch tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet statistics
# ])

# # Load and preprocess an image
# def load_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# # Classify an image
# def classify_image(image_path):
#     image = load_image(image_path)
#     with torch.no_grad():
#         output = dinov2_vits14(image)
#     probabilities = torch.nn.functional.softmax(output[0], dim=0)
#     return probabilities

# def classify_img2(image):
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         output = dinov2_vits14(image)
#     probabilities = torch.nn.functional.softmax(output[0], dim=0)
#     return probabilities

# def img_sim(i1, i2):
#     iv1 = classify_img2(i1)
#     iv2 = classify_img2(i2)
#     cos_sim = F.cosine_similarity(iv1.unsqueeze(0), iv2.unsqueeze(0))
#     return cos_sim


# # Example usage


# fp1 = 'playground/truckright1/images'
# fp2 = 'playground/truckleft1/images'

# p1array = []
# for filename in os.listdir(fp1):
#     filepath = os.path.join(fp1, filename)
#     p1 = (classify_image(filepath), filename)
#     p1array.append(p1)

# p2array = []
# for filename in os.listdir(fp2):
#     filepath = os.path.join(fp2, filename)
#     p2 = (classify_image(filepath), filename)
#     p2array.append(p2)


# sim_list = []
# for (v1, n1) in p1array:
#     for (v2, n2) in p2array:
#         cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
#         if cos_sim > 0.8:
#             sim_list.append((cos_sim, (n1, n2)))


# _, (name, _) = sim_list[0]
# path = 'playground/truckright1'
# cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
# cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
# cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)


# def find(image_dict, target_name):
#     for image_id, image in image_dict.items():
#         if image.name == target_name:
#             return (image.qvec, image.tvec)
#     return None

# qv, tv = find(cam_extrinsics, name)
# print (qv, tv)
# # image_path1 = 'playground/truckright1/images/000001.jpg'
# # image_path2 = 'playground/truckright1/images/000057.jpg'
# # p1 = classify_image(image_path1)
# # p2 = classify_image(image_path2)

# # cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
# # # To get the top-5 predictions
# # print(cos_sim)


