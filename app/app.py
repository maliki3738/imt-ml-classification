import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd


st.set_page_config(
    page_title="Classification de Déchets",
    layout="centered",
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/model_dechets.keras")
    return model


model = load_model()

class_names = ["organique", "papier", "plastique"]
img_size = (224, 224)

st.markdown(
    """
<div style="text-align: center; margin-bottom: 25px;">
    <h1 style="font-size: 38px; color: #EDEDED;">Classification d'Images de Déchets</h1>
    <p style="font-size: 18px; color: #AAAAAA;">
        Téléversez une image et découvrez si elle appartient à la catégorie
        <b>organique</b>, <b>papier</b> ou <b>plastique</b>.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Choisissez une image (jpg/png/jpeg)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image téléversée", width=350)

    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    probas = preds[0]
    idx = np.argmax(probas)
    confidence = probas[idx]
    predicted_class = class_names[idx]

    st.markdown("---")

    st.markdown(
        f"""
    <div style="padding: 20px; border-radius: 12px; background-color: #1E1E1E; text-align: center;">
        <h2 style="color: #FFFFFF;">Résultat de la prédiction</h2>
        <p style="font-size: 20px; margin-top: 10px;">
            Catégorie prédite :
            <span style="background-color: #4CAF50; padding: 6px 12px; border-radius: 8px; color: white;">
                {predicted_class}
            </span>
        </p>
        <p style="font-size: 18px; color: #CCCCCC;">Confiance : {confidence*100:.2f}%</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("Probabilités par classe")

    proba_df = pd.DataFrame(
        {
            "Classe": class_names,
            "Probabilité (%)": probas * 100,
        }
    ).set_index("Classe")

    st.bar_chart(proba_df)

    st.markdown(
        """
    <small style="color:#777;">
    *Astuce : les prédictions sont plus fiables si l’objet occupe bien la photo
    et que l’image n’est pas trop sombre ou floue.*
    </small>
    """,
        unsafe_allow_html=True,
    )
