# Машинное обучение. Лабораторная работа №4
## Задание 1
*Реализуйте глубокую нейронную сеть (полносвязную или сверточную) и обучите ее на синтетических данных (например, наборы MNIST (http://yann.lecun.com/exdb/mnist/) или notMNIST).*

В качестве синтетических данных был выбран датасет MNIST.

Модель нейронной сети:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu',
        input_shape=(28, 28, 1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
])
```

Для борьбы с переобучением использовались регуляризация L2 и случайное "отключение" части нейронов в слоях на каждом этапе обучения (Dropout). Также во время обучения модели иcпользовался один из популярных оптимизаторов `Adam` с динамически изменяемой скоростью обучения.

**Точность данной модели на тестовой выборке составила: 98,7%.**

Кривые обучения данной модели для набора MNIST:
![Learning curves](https://github.com/chiki-takipavel/ML_LW4_Back/assets/55394253/75564fe3-8448-4c25-8d39-3fe0ff5c4317)

## Задание 2
*После уточнения модели на синтетических данных попробуйте обучить ее на реальных данных (набор Google Street View). Что изменилось в модели?*

Модель нейронной сети:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu',
        input_shape=(32, 32, 1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
])
```

Используемый формат данных - изображения размером 32 × 32, содержащие одну цифру.

В данном наборе изображения, содержащие 0, имели метку "10", поэтому все метки "10" были заменены на "0". Также данные хранятся в порядке `(ширина, длина, количество каналов, количество изображений). Для удобного использования в модели порядок был переопределён на `(количество изображений, ширина, высота, количество каналов`).

Отличия от модели для набора MNIST:
- размер входных данных 32x32 (вместо 28x28);
- было добавлено ещё 2 скрытых слоя;
   ```python
   tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
   tf.keras.layers.MaxPooling2D((2, 2))
   ```
- изображения в наборе SVHN 3-канальные, поэтому в предобработку изображений добавилась функция по переводу 3-канальных RGB изображений в 1-канальные градации серого.
   ```python
   def rgb2gray(rgb):
       return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
   ```
Для последующей работы в Android-приложении `Keras`-модель была преобразована в `TFLite`-модель и сохранена в файл `model.tflite`.

**Точность данной модели на тестовой выборке составила: 90,5%.**

Кривые обучения данной модели для набора SVHN:
![Learning curves](https://github.com/chiki-takipavel/ML_LW4_Back/assets/55394253/9aea0fba-3ff2-44ee-b5df-5c8adfe45e59)

## Задание 3-4
*Реализуйте приложение для ОС Android, которое может распознавать цифры в номерах домов, используя разработанный ранее классификатор.*

Было разработано Android-приложение на языке программирования `Kotlin` с использованием библиотеки `TFLite`. Оно позволяет:
- загрузить фотографию из галереи, нажав на кнопку "Choose a photo";
- сделать фотографию с помощью камеры телефона, нажав на кнопку c иконкой фотоаппарата.

После загрузки изображения появляется всплывающее окно с результатом распознавания. Если цифры номера дома на изображении не найдено - выведется "Распознанная цифра: не найдено.".

Ранее сохранённая модель хранится в папке `assets` и загружается с помощью следующего метода:
```kotlin
private fun loadModel(context: Context): Interpreter {
    val assetManager = context.assets
    val fileDescriptor: AssetFileDescriptor = assetManager.openFd(MODEL_FILE_NAME)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel: FileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength

    return Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength))
}
```



Полный метод для распознавания цифры на изображении:
```kotlin
private fun recognizeImage(bitmap: Bitmap): Int {
    val convertedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

    val inputProcessor = ImageProcessor.Builder()
        .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .add(TransformToGrayscaleOp())
        .add(NormalizeOp(0f, 255f))
        .build()
    val tensorImage = TensorImage(DataType.FLOAT32)
    tensorImage.load(convertedBitmap)

    val inputBuffer = inputProcessor.process(tensorImage)
    val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 10), DataType.FLOAT32)

    tflite.run(inputBuffer.buffer, outputBuffer.buffer)

    return selectClass(outputBuffer.floatArray)
}
```

Результаты работы программы:
![Report](https://github.com/chiki-takipavel/ML_LW4_Back/assets/55394253/bc3f4175-e80b-46ea-bb99-11e58866e52e)
