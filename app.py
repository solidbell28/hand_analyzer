import streamlit as st
import cv2
import tempfile

from detection import parse_img, hands

st.title('Определение "позы" ладоней')
uploaded = st.file_uploader(
    'Загрузите изображение (.tif, .jpg, .png)',
    type=['tif', 'jpg', 'png']
)

if uploaded:
    tmp_img = tempfile.NamedTemporaryFile(
        delete=False, suffix=uploaded.name
    )
    tmp_img.write(uploaded.getvalue())
    tmp_img.flush()

    tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    tmp_txt.close()

    try:
        annotated = parse_img(
            tmp_img.name, tmp_txt.name,
            uploaded.name
        )
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(
            annotated, caption='Размеченное изображение',
            use_container_width=True
        )

        with open(tmp_txt.name, 'r') as f:
            txt_data = f.read()
        st.download_button(
            'Скачать результаты (txt)', txt_data,
            file_name='results.txt',

        )
    except Exception as e:
        st.error(f'Ошибка: {e}')

# hands.close()
