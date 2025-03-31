### Рубежный контроль №1 (весна 2025 года)

### Тема: Методы обработки данных

### **1.  Варианты заданий**

- Номер варианта: 16
- Номер задачи №1: 16
- Номер задачи №2: 36

### **2.  Условия задач**

- Для набора данных проведите нормализацию для одного (произвольного) числового признака с использованием преобразования Бокса-Кокса (Box-Cox transformation).
- Для набора данных проведите процедуру отбора признаков (feature selection). Используйте метод вложений (embedded method). Используйте подход на основе дерева решений.
- Для произвольной колонки данных построить график "Скрипичная диаграмма (violin plot)".

### **3.  Выход**

### **3.1.  Преобразование Бокса-Кокса**

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 生成示例数据
file_path = os.path.abspath('C:/Users/xue_j/Desktop/2024-2025-2/MMO/PK1/data1.csv')  # Windows
data = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head(10))
``` 

**Вывести первые десять строк данных：**

> ![image](https://github.com/user-attachments/assets/80dbda81-4c68-4ae7-9242-0e77c8a5473f)


**Выберите столбец «FFMC» для преобразования boxcox и выведите оптимальные параметры:**

```python
# 应用Box-Cox变换
data['FFMC_boxcox'], lambda_param = stats.boxcox(data['FFMC'])
print(f"best_λ: {lambda_param:.3f}")
``` 
> ![image](https://github.com/user-attachments/assets/1c0d2c44-688c-434e-a084-44e07c32ace3)


**Сравните изменения данных «FFMC» до и после нормализации (гистограмма):**

```python
# 变换前的分布检查
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data['FFMC'], bins=30, color='blue', alpha=0.7)
plt.title('generation')

# 变换后的分布检查
plt.figure(1)
plt.subplot(1, 2, 2)
plt.hist(data['FFMC_boxcox'], bins=30, color='green', alpha=0.7)
plt.title('Box-Cox')
```
> ![image](https://github.com/user-attachments/assets/b0aeed7a-1e7f-4630-85c5-9d6fec0a66c9)
> Рис. 1  Сравнение гистограмм


**Сравните изменения данных «FFMC» до и после нормализации (violin plot график):**

```python
# 垂直小提琴图（针对'数值列'）
plt.figure(2)
sns.violinplot(data=data, y='FFMC')
plt.title('小提琴图 (Violin Plot)_generation')
plt.ylabel('FFMC')

plt.figure(3)
sns.violinplot(data=data, y='FFMC_boxcox')
plt.title('小提琴图 (Violin Plot)_boxcox')
plt.ylabel('FFMC_boxcox')
```

> ![image](https://github.com/user-attachments/assets/0b7fab0c-d947-4cde-a7e2-c227f092d061)
> Рис. 2  "FFMC"violin plot

> ![image](https://github.com/user-attachments/assets/6482fb30-1ac9-4e3d-9cca-6a3d1d0ab267)
> Рис. 3  "FFMC_boxcox"violin plot


**Начертите график скрипки по столбцу классификации «классы»:**

```python
#按类划分
plt.figure(4)
sns.violinplot(data=data, y='FFMC',x='Classes')
plt.title('小提琴图 (Violin Plot)_classes')
plt.ylabel('FFMC')
plt.xlabel('Classes')
plt.figure(5)
sns.violinplot(data=data, y='FFMC_boxcox',x='Classes')
plt.title('小提琴图 (Violin Plot)_boxcox_classes')
plt.xlabel('Classes')
plt.ylabel('FFMC_boxcox')
plt.show()
``` 

> ![image](https://github.com/user-attachments/assets/604f48df-be9b-448f-81e0-54a4839c9c38)
> Рис. 4  "FFMC"violin plot(classes)

> ![image](https://github.com/user-attachments/assets/af14ef6c-d38c-42de-8a57-ed5dcd4ce42c)
> Рис. 5  "FFMC_boxcox"violin plot(classes)


### **3.2.  Процедура отбора признаков**

**Используем метод SelectKBest с mutual_info_classif, чтобы выбрать 5 наиболее информативных признаков для предсказания класса (Classes — fine/notfine). Данные содержат метеорологические показатели и индексы пожароопасности.**

**(1) Подготовка данных:**

```python
# Преобразуем целевой признак в числовой формат (0 для 'notfine', 1 для 'fine')
data['Classes'] = data['Classes'].map({'notfire': 0, 'fire': 1})

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data.drop(['day', 'month', 'year', 'Classes'], axis=1)  # Исключаем даты и целевой признак
y = data['Classes']
```


**(2) Отбор 5 лучших признаков：**

```python
# Инициализация SelectKBest с mutual_info_classif
selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Получение имен выбранных признаков
selected_features = X.columns[selector.get_support()]
print("Лучшие 5 признаков:\n", selected_features.tolist())
```

**Выход：**

> ![image](https://github.com/user-attachments/assets/cef98746-95e6-4f7f-8612-e6cc86a78e9b)


**(3) Визуализация значимости признаков：**

```python
# Оценки важности всех признаков
scores = selector.scores_

plt.figure(figsize=(10, 5))
plt.bar(X.columns, scores, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Важность признаков (Mutual Information)")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

**Выход：**
> ![image](https://github.com/user-attachments/assets/fa1e370a-6679-40fc-90fa-11adc587dbda)
> Рис. 6  Оценки важности всех признаков

**Интерпретация:**
Признаки FFMC, DMC, DC, ISI, FWI (индексы пожароопасности) имеют наибольшую взаимную информацию с целевой переменной Classes.


**(4) Проверка отобранных данных：**

```python
print("Исходные признаки:\n", X.columns.tolist())
print("\nОтобранные признаки:\n", selected_features.tolist())
print("\nПример преобразованных данных (первые 5 строк):\n", X_selected[:5])
```

**Выход：**

> ![image](https://github.com/user-attachments/assets/e6acdfe9-53fb-4767-b9ac-1995f87b886b)
> Рис. 7  Проверка отобранных данных




