水面垃圾检测数据集、垃圾检测、YOLO模型通用

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3172ff7ea2842b5899681683d815ad5.png)



以下是 **水面垃圾检测（塑料瓶）数据集** 的结构化信息表格：

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28d9a5c0bb1a4d53839c1ce3535f6969.png)

### 📊 水面垃圾检测（塑料瓶）数据集概览表

| **项目**                | **描述**                                                                 |
|------------------------|--------------------------------------------------------------------------|
| **数据集名称**          | 水面垃圾检测数据集（聚焦塑料瓶）                                          |
| **总图像数量**          | 1,200 张                                                                 |
| **数据划分**            | - 训练集（Train）：960 张<br>- 验证集（Val）：120 张<br>- 测试集（Test）：120 张 |
| **类别数量（nc）**      | 1 类                                                                     |
| **目标类别**            | `bottle`（塑料瓶）                                                       |
| **标注框总数**          | 3,248 个                                                                 |
| **每类图片覆盖**        | 所有 1,200 张图像均包含 `bottle` 目标                                     |
| **应用场景**            | 河道/湖泊/海洋垃圾监测、智能环保船、无人机巡检、水体清洁机器人等           |
| **任务类型**            | 目标检测（Object Detection）                                             |

---

### 📈 模型在验证集上的性能对比（mAP@0.5）

| **模型**       | **mAP@0.5** |
|----------------|-------------|
| YOLOv5m        | 0.90641     |
| YOLOv8m        | **0.90834** |
| YOLOv11m       | 0.89696     |

> ✅ **最佳表现**：YOLOv8m（mAP@0.5 = 90.83%），显示该数据集质量高、标注精准，适合训练高精度水面垃圾检测模型。

---

### 📁 推荐目录结构

```
water_bottle_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

### 📄 `dataset.yaml` 示例

```yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 1
names: ['bottle']
```

该数据集可直接用于 **环保 AI 系统开发**、**水域智能治理** 及 **学术研究**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6742236827724177a66eb19b04db8016.png)
于 **水面垃圾检测（塑料瓶）数据集** 的完整 YOLOv8 检测系统实现方案，包含：

✅ 从预训练模型开始的 YOLOv8 训练代码  
✅ 推理脚本（图像/视频/摄像头）  
✅ Web 检测界面（Django + Bootstrap）  
✅ 垃圾计数与环保报告生成  

---

## 📁 一、项目目录结构

```
water_bottle_yolov8/
├── data/
│   ├── images/
│   │   ├── train/    # 960
│   │   ├── val/      # 120
│   │   └── test/     # 120
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml
├── models/
│   └── best.pt       # 训练后模型
├── train.py          # YOLOv8 训练脚本
├── infer.py          # 推理脚本
├── web_app/
│   ├── views.py
│   ├── models.py
│   ├── urls.py
│   └── templates/
│       └── index.html
└── requirements.txt
```

---

## 📦 二、`requirements.txt`

```txt
ultralytics==8.2.0
opencv-python==4.8.0
numpy==1.24.3
pillow==9.5.0
django==4.2.7
matplotlib==3.6.3
```

安装：
```bash
pip install -r requirements.txt
```

---

## 📄 三、`data/dataset.yaml`

```yaml
train: ../data/images/train
val: ../data/images/val
test: ../data/images/test

nc: 1
names: ['bottle']
```

> ✅ 确保 `images/train` 与 `labels/train` 中文件名一一对应（如 `001.jpg` ↔ `001.txt`）

---

## 🚀 四、`train.py` —— YOLOv8 训练代码（从预训练模型微调）

```python
# train.py
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载官方预训练模型（YOLOv8m）
    model = YOLO("yolov8m.pt")  # 可替换为 yolov8n/s/l/x

    # 开始训练
    results = model.train(
        data="data/dataset.yaml",
        epochs=100,
        imgsz=640,               # 水面目标通常较小，640 足够
        batch=16,                # 根据 GPU 调整（RTX 3060 可用 16）
        device=0,                # 使用 GPU；无 GPU 改为 "cpu"
        name="water_bottle_yolov8m",
        project="runs/train",
        patience=20,             # 早停机制
        save=True,
        exist_ok=True
    )

    print(f"✅ 训练完成！最佳模型路径: {results.save_dir}/weights/best.pt")
```

运行训练：
```bash
python train.py
```

> 💡 预期 mAP@0.5 ≈ 0.908（与你提供的验证结果一致）

---

## 🔍 五、`infer.py` —— 推理与塑料瓶计数

```python
# infer.py
from ultralytics import YOLO
import cv2
import os

# 加载训练好的模型
model = YOLO("runs/train/water_bottle_yolov8m/weights/best.pt")

def detect_bottles(image_path, conf=0.4):
    results = model(image_path, conf=conf)
    annotated_img = results[0].plot()

    # 统计塑料瓶数量
    bottle_count = len(results[0].boxes)

    # 保存结果
    output_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, annotated_img)

    print(f"✅ 检测到 {bottle_count} 个塑料瓶，结果保存至: {output_path}")
    return bottle_count, output_path

# 示例：单张图像
detect_bottles("river_scene.jpg")

# 视频推理（可选）
# model.predict("lake_video.mp4", save=True, conf=0.4)

# 摄像头实时检测
# model.predict(source=0, show=True, conf=0.4)
```

---

## 🌐 六、Django Web 检测系统

### 1. `web_app/models.py`

```python
from django.db import models

class BottleDetection(models.Model):
    image = models.ImageField(upload_to='uploads/')
    result_image = models.ImageField(upload_to='results/', null=True)
    bottle_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)
    location = models.CharField(max_length=100, default="河道A段")
```

### 2. `web_app/views.py`

```python
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from .models import BottleDetection
from ultralytics import YOLO
import cv2
import os

# 全局加载模型（仅加载一次）
model = YOLO("runs/train/water_bottle_yolov8m/weights/best.pt")

def index(request):
    return render(request, 'index.html')

def detect_bottle_web(request):
    if request.method == "POST" and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        input_path = fs.path(filename)
        output_dir = os.path.join(fs.location, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{filename}")

        # 推理
        results = model(input_path, conf=0.4)
        annotated_img = results[0].plot()
        cv2.imwrite(output_path, annotated_img)

        bottle_count = len(results[0].boxes)

        # 保存记录
        record = BottleDetection.objects.create(
            image=uploaded_file,
            result_image=f'results/result_{filename}',
            bottle_count=bottle_count
        )

        return JsonResponse({
            'result_url': fs.url(f'results/result_{filename}'),
            'bottle_count': bottle_count
        })
    return JsonResponse({'error': '请上传图像'})
```

### 3. `templates/index.html`

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>🌊 水面塑料瓶智能检测系统</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-blue-50">
<div class="max-w-4xl mx-auto p-6">
    <h1 class="text-2xl font-bold text-center mb-6">🌊 水面垃圾（塑料瓶）AI检测系统</h1>
    <form id="uploadForm" enctype="multipart/form-data" class="mb-6">
        <input type="file" name="image" accept=".jpg,.png" class="block w-full mb-4">
        <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded">上传并检测</button>
    </form>

    <div id="result" class="hidden">
        <img id="resultImg" class="max-w-full border rounded shadow">
        <div class="mt-4 p-4 bg-white rounded">
            <p class="text-lg">检测到塑料瓶数量：<span id="count" class="font-bold text-red-600"></span> 个</p>
            <p class="text-sm text-gray-500">可用于环保评估与清洁任务调度</p>
        </div>
    </div>
</div>

<script>
document.getElementById('uploadForm').onsubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const res = await fetch('/detect/', {method: 'POST', body: formData});
    const data = await res.json();
    if (data.result_url) {
        document.getElementById('resultImg').src = data.result_url;
        document.getElementById('count').textContent = data.bottle_count;
        document.getElementById('result').classList.remove('hidden');
    }
};
</script>
</body>
</html>
```

### 4. 启动 Web 系统

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

访问：http://127.0.0.1:8000

---

## 🌍 七、应用场景

| 应用方向 | 说明 |
|--------|------|
| **河道/湖泊巡检** | 无人机或无人船搭载模型，自动识别水面漂浮塑料瓶 |
| **环保执法取证** | 快速生成污染证据图，辅助城管或水务部门 |
| **清洁机器人导航** | 为水面垃圾打捞机器人提供目标定位 |
| **公众环保教育** | Web 系统开放给公众上传照片，提升环保意识 |

---


