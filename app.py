import streamlit as st
import plotly.express as px
from fastai.vision.all import *

# CSS orqali chiroyli stil qo'shish
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(to right, #1f4037, #99f2c8);
            color: white;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
        }
        .stButton > button:hover {
            background-color: #ff1e1e;
        }
        .stFileUploader > div {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sarlavha va izoh
st.title("⚽🏀 Sport turlarini klassifikatsiya qiluvchi model")
st.subheader("Football, Tennis Ball va Volleyball rasmlarini bashorat qiluvchi web ilova")
st.markdown("### Iltimos, rasm yuklang va natijani kuting")

# Foydalanuvchi fayl yuklaydi
file = st.file_uploader("Rasm yuklash", type=['png', 'jpeg', 'jpg', 'gif', 'svg'])

if file:
    st.image(file, caption="Yuklangan rasm", use_column_width=True)
    
    # PIL konvertatsiya
    img = PILImage.create(file)

    # Modelni yuklash
    model = load_learner('sports_classifier.pkl')
    
    # Natijani bashorat qilish
    pred, pred_id, probs = model.predict(img)
    
    # Natijalarni chiroyli ko'rsatish
    st.success(f"🧐 Bashorat: **{pred}**")
    st.info(f'📊 Ehtimollik: **{probs[pred_id]*100:.1f}%**')

    # Bashorat ehtimolliklarini grafikda chiqarish
    fig = px.bar(x=probs*100, y=model.dls.vocab, labels={'x':'Ehtimollik (%)', 'y':'Sport turi'}, 
                 title="📊 Bashorat ehtimolliklari", color=model.dls.vocab, text=probs*100)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig)

# Footer
st.markdown("""<hr>
<div style='text-align: center;'>
    <p>📌 Dastur Streamlit va FastAI yordamida ishlab chiqildi.</p>
    <p>💡 Developer: <b>Dilmurod</b></p>
</div>""", unsafe_allow_html=True)
