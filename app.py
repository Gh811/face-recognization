from flask import Flask, request, jsonify, render_template
import cv2
from PIL import Image, UnidentifiedImageError
import numpy as np
from mtcnn import MTCNN
import torch
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
import logging
import json

# ---------------- 初始化 ----------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

detector = MTCNN()
weights = ResNet50_Weights.IMAGENET1K_V1
feature_extractor = resnet50(weights=weights)
feature_extractor.fc = torch.nn.Identity()  # 移除全连接层
feature_extractor.eval()

IMAGES_FILE = Path("images.json")
FEATURES_FILE = Path("features.json")
# 余弦相似度阈值，值越大表示要求越相似（余弦相似度范围是-1到1）
SIMILARITY_THRESHOLD = 0.8 


# ---------------- 余弦相似度计算 ----------------
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 避免除以零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)


# ---------------- JSON 特征文件操作 ----------------
def load_features():
    if FEATURES_FILE.exists() and FEATURES_FILE.stat().st_size > 0:
        with open(FEATURES_FILE, 'r') as f:
            data = json.load(f)
            # 如果读出来不是字典，直接返回空字典
            if isinstance(data, dict):
                return data
    return {}

def save_features(features_dict):
    # 直接写入 JSON
    with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(features_dict, f, ensure_ascii=False, indent=2)


# ---------------- 页面路由 ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/content_page')
def content_page():
    return render_template('content.html')

@app.route('/login_page')
def login_page():
    return render_template('login.html')


@app.route('/get_avatar/<username>')
def get_avatar(username):
    # 加载 features.json 或 images.json
    print(username)
    features_dict = load_features()
    if username in features_dict:
        image_path = features_dict[username]['image']
        return jsonify({"image": image_path})
    else:
        return jsonify({"error": "用户不存在"}), 404


# ---------------- 注册接口 ----------------
@app.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return jsonify({"message": "请上传图片"}), 400

    username = request.form.get('username')
    if not username:
        return jsonify({"message": "请提供用户名"}), 400

    # 加载已有特征
    features_dict = load_features()
    
    # 检查用户名是否已存在
    if username in features_dict:
        return jsonify({"message": f"用户名 '{username}' 已存在，请更换用户名"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert('RGB')
    except UnidentifiedImageError:
        return jsonify({"message": "无效的图片文件"}), 400
    image_save_path =  f"static/images/{username}.jpg"
    image_ = image
    image = np.array(image)
    results = detector.detect_faces(image)
    if not results:
        return jsonify({"message": "未检测到人脸"}), 400
    if len(results) > 1:
        return jsonify({"message": "检测到多张人脸，请上传仅包含一张人脸的图片"}), 400

    x, y, w, h = results[0]['box']
    x, y = max(x, 0), max(y, 0)
    face = image[y:y+h, x:x+w]
    if face.size == 0:
        return jsonify({"message": "人脸提取失败，请上传清晰的图片"}), 400

    face = cv2.resize(face, (224, 224))
    face = np.transpose(face, (2,0,1)) / 255.0
    face_tensor = torch.tensor([face], dtype=torch.float32)
    features = []
    with torch.no_grad():
        features = feature_extractor(face_tensor).numpy().flatten()

    # 检查是否已有相似面部
    for existing_user, data in features_dict.items():
        registered_features = np.array(data['features'], dtype=np.float32)

        # 使用余弦相似度替代欧氏距离
        similarity = cosine_similarity(features, registered_features)

        if similarity > SIMILARITY_THRESHOLD + 0.05:
            print("注册和已有相似度",similarity)
            return jsonify({
                "message": f"您已经注册过哦！！！"
            }), 400

    # 保存新用户特征
    features_dict[username] = {
        "features": features.tolist(),
        "image": image_save_path
    }
    save_features(features_dict)
    image_.save(image_save_path)
    
    logging.info(f"用户 {username} 注册成功")
    return jsonify({"message": "注册成功"})


# ---------------- 登录接口 ----------------
@app.route('/login', methods=['POST','GET'])
def login():
    if 'image' not in request.files:
        return jsonify({"message": "请上传图片"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert('RGB')
    except UnidentifiedImageError:
        return jsonify({"message": "无效的图片文件"}), 400

    image = np.array(image)
    results = detector.detect_faces(image)
    if not results:
        return jsonify({"message": "未检测到人脸"}), 400
    if len(results) > 1:
        return jsonify({"message": "检测到多张人脸，请上传仅包含一张人脸的图片"}), 400

    x, y, w, h = results[0]['box']
    x, y = max(x, 0), max(y, 0)
    face = image[y:y+h, x:x+w]
    if face.size == 0:
        return jsonify({"error": "人脸提取失败，请上传清晰的图片"}), 400

    face = cv2.resize(face, (224, 224))
    face = np.transpose(face, (2,0,1)) / 255.0
    face_tensor = torch.tensor([face], dtype=torch.float32)

    with torch.no_grad():
        features = feature_extractor(face_tensor).numpy().flatten()

    # 加载所有注册特征
    features_dict = load_features()
    username = request.form.get('username')
    
    if username not in features_dict:
        return jsonify({"message": "用户名未注册"}), 401

    registered_features = np.array(features_dict[username]['features'], dtype=np.float32)

    # 使用余弦相似度替代欧氏距离
    similarity = cosine_similarity(features, registered_features)
    
    
    if similarity > SIMILARITY_THRESHOLD + 0.05:
        logging.info(f"用户 {username} 登录成功")
        print(f"{username}登录成功,相似度为{similarity}")
        return jsonify({"message": "登录成功", "username": username})
    else:
        print(f"{username}登录失败,相似度为{similarity}")
        return jsonify({"message": "人脸不匹配"}), 401


# ---------------- 启动服务 ----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7797, debug=True)
