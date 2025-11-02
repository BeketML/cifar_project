import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Конфигурация страницы
st.set_page_config(
    page_title="Классификация CIFAR-10",
    page_icon="🖼️",
    layout="wide"
)

# Заголовок приложения
st.title("🖼️ Классификация изображений CIFAR-10")
st.markdown("""
Это приложение использует обученную сверточную нейронную сеть (CNN) для классификации изображений 
на 10 классов набора данных CIFAR-10. Загрузите изображение и получите предсказание!
""")

# Определение классов
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Определение архитектуры модели (должна совпадать с обученной)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Кеширование загрузки модели
@st.cache_resource
def load_pytorch_model(model_path='./cifar_net.pth'):
    """Загружает обученную модель PyTorch"""
    try:
        model = Net()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

# Загрузка модели
model = load_pytorch_model()

if model is None:
    st.stop()

# Определение трансформаций (должны совпадать с обучением)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Функция для предсказания
def predict_image(model, image):
    """Выполняет предсказание на изображении"""
    # Преобразование изображения
    img_tensor = transform(image).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        probs, indices = torch.topk(probabilities, k=10)
    
    return probs[0].numpy(), indices[0].numpy()

# Интерфейс загрузки файла
uploaded_file = st.file_uploader(
    "Выберите изображение для классификации",
    type=['png', 'jpg', 'jpeg'],
    help="Поддерживаются форматы: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    # Загрузка и отображение изображения
    image = Image.open(uploaded_file)
    
    # Конвертация в RGB, если нужно
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Загруженное изображение")
        st.image(image, caption="Ваше изображение", use_container_width=True)
        
        # Изменение размера для показа (оригинальный размер для предсказания)
        image_resized = image.copy()
    
    with col2:
        st.subheader("🔮 Результат классификации")
        
        # Предсказание
        probs, indices = predict_image(model, image)
        
        # Получение предсказанного класса
        predicted_class_idx = int(indices[0])
        predicted_class = classes[predicted_class_idx]
        confidence = float(probs[0] * 100)
        
        # Отображение результата
        st.success(f"**Предсказанный класс:** {predicted_class}")
        st.metric("Уверенность", f"{confidence:.2f}%")
        
        # Топ-3 предсказания
        st.markdown("### Топ-3 предсказания:")
        for i in range(3):
            idx = int(indices[i])
            prob = float(probs[i] * 100)
            st.progress(float(prob / 100))
            st.text(f"{classes[idx]}: {prob:.2f}%")
    
    # Визуализация вероятностей для всех классов
    st.subheader("📊 Вероятности для всех классов")
    
    # Создание словаря с вероятностями
    class_probs = {classes[int(idx)]: float(probs[i] * 100) for i, idx in enumerate(indices)}
    
    # Сортировка по убыванию вероятности
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Отображение в виде столбчатой диаграммы
    chart_data = {
        'Класс': [item[0] for item in sorted_probs],
        'Вероятность (%)': [item[1] for item in sorted_probs]
    }
    
    st.bar_chart(chart_data, x='Класс', y='Вероятность (%)')
    
    # Детальная таблица
    with st.expander("📋 Детальная информация о всех классах"):
        import pandas as pd
        df = pd.DataFrame(sorted_probs, columns=['Класс', 'Вероятность (%)'])
        st.dataframe(df, use_container_width=True)

else:
    st.info("👆 Пожалуйста, загрузите изображение для классификации")
    
    # Информация о классах
    st.subheader("📚 Классы CIFAR-10")
    st.markdown("""
    Модель может классифицировать изображения на следующие 10 классов:
    """)
    
    cols = st.columns(5)
    for i, class_name in enumerate(classes):
        with cols[i % 5]:
            st.markdown(f"- **{class_name}**")
    
    st.markdown("""
    ---
    ### 💡 Совет
    Загрузите изображение одного из объектов выше. Лучшие результаты будут на изображениях 
    размером около 32x32 пикселей, но модель автоматически изменит размер загруженного изображения.
    """)

# Информация о модели
with st.sidebar:
    st.header("ℹ️ О модели")
    st.markdown("""
    **Архитектура:** CNN с двумя сверточными слоями и тремя полносвязными слоями
    
    **Точность:** ~57% на тестовом наборе CIFAR-10
    
    **Обучение:** 5 эпох на наборе данных CIFAR-10
    
    **Классов:** 10
    """)
    
    st.markdown("---")
    st.markdown("**Размер входного изображения:** 32x32 пикселя")
    st.markdown("**Форматы:** PNG, JPG, JPEG")

