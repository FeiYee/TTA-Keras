# keras_tta
这是一个TTA流程案例，可以自己指定TTA增强数量


### 使用方式:

```python
# 数据增强
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# 传入部分数据
datagen.fit(Train——Immage)

tta_model = TTA_ModelWrapper(model,datagen )

predictions = tta_model.predict(X_test,4)
```
