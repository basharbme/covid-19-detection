import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # resize the heatmap to oringnal X-Ray image size
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

st.title("Respiratory Disease Detection")

image = st.file_uploader("Choose an image...")

if image is not None:
    pil_img = Image.open(image)
    ocv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    st.image(pil_img.resize((299, 299)), caption='Uploaded Image.', use_column_width=True)
    st.write('Classifying...')

    resized = cv2.resize(ocv_img, (299, 299))
    scaled = np.array(resized) / 255.0
    data = np.expand_dims(scaled, axis=0)

    model = tf.keras.models.load_model('model.h5')

    output = model.predict(data)
    prediction = np.argmax(output)
    accuracy = output[0][prediction]

    cam = GradCAM(model=model, classIdx=prediction, layerName='block14_sepconv2_act') # find the last 4d shape "mixed10" in this case
    heatmap = cam.compute_heatmap(data)

    heatmap = cv2.resize(heatmap, (resized.shape[1], resized.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)  # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT
    result = cv2.addWeighted(heatmap, 0.5, resized, 1.0, 0)

    result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    st.image(result, caption='Result.', use_column_width=True)
    classes = {0:'COVID-19', 1: 'HEALTHY', 2: 'PNEUMONIA/OTHER'}
    st.write(f'Status: {classes[prediction]}')
    st.write(f'Accuracy: {accuracy:.2%}')


