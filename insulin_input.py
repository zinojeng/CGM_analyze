import streamlit as st
from datetime import time

def get_insulin_info():
    st.sidebar.header("胰島素資訊")

    insulin_info = {}

    # 長效胰島素
    with st.sidebar.expander("長效胰島素"):
        long_acting_options = [
            "無",
            "蘭德仕（Lantus）",
            "糖德仕（Toujeo）",
            "瑞和密爾（Levemir）",
            "諾胰保（Tresiba）"
        ]
        selected_long_acting = st.selectbox("選擇長效胰島素", options=long_acting_options, key="long_acting_select")
        if selected_long_acting != "無":
            doses = st.columns(4)
            morning_dose = doses[0].number_input("早上劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="long_acting_morning")
            noon_dose = doses[1].number_input("中午劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="long_acting_noon")
            evening_dose = doses[2].number_input("晚上劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="long_acting_evening")
            bedtime_dose = doses[3].number_input("睡前劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="long_acting_bedtime")
            insulin_info['長效胰島素'] = {
                '種類': selected_long_acting,
                '劑量': {
                    '早上': morning_dose,
                    '中午': noon_dose,
                    '晚上': evening_dose,
                    '睡前': bedtime_dose
                }
            }

    # 短效/速效胰島素
    with st.sidebar.expander("短效/速效胰島素"):
        short_acting_options = [
            "無",
            "諾和瑞（Novorapid）",
            "愛速基因人體胰島素（Actrapid HM）"
        ]
        selected_short_acting = st.selectbox("選擇短效/速效胰島素", options=short_acting_options, key="short_acting_select")
        if selected_short_acting != "無":
            doses = st.columns(4)
            morning_dose = doses[0].number_input("早上劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="short_acting_morning")
            noon_dose = doses[1].number_input("中午劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="short_acting_noon")
            evening_dose = doses[2].number_input("晚上劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="short_acting_evening")
            bedtime_dose = doses[3].number_input("睡前劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="short_acting_bedtime")
            insulin_info['短效/速效胰島素'] = {
                '種類': selected_short_acting,
                '劑量': {
                    '早上': morning_dose,
                    '中午': noon_dose,
                    '晚上': evening_dose,
                    '睡前': bedtime_dose
                }
            }

    # 預混型胰島素
    with st.sidebar.expander("預混型胰島素"):
        premixed_options = [
            "無",
            "諾和密斯30（Novomix 30）",
            "諾和密斯50（Novomix 50）",
            "優泌樂筆25（Humalog mix 25）",
            "優泌樂筆50（Humalog mix 50）",
            "諾胰得諾特筆（Ryzodeg FlexTouch）",
            "爽胰達（Soliqua）"
        ]
        selected_premixed = st.selectbox("選擇預混型胰島素", options=premixed_options, key="premixed_select")
        if selected_premixed != "無":
            doses = st.columns(4)
            morning_dose = doses[0].number_input("早上劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="premixed_morning")
            noon_dose = doses[1].number_input("中午劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="premixed_noon")
            evening_dose = doses[2].number_input("晚上劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="premixed_evening")
            bedtime_dose = doses[3].number_input("睡前劑量 (單位)", min_value=0.0, step=0.1, format="%.1f", key="premixed_bedtime")
            insulin_info['預混型胰島素'] = {
                '種類': selected_premixed,
                '劑量': {
                    '早上': morning_dose,
                    '中午': noon_dose,
                    '晚上': evening_dose,
                    '睡前': bedtime_dose
                }
            }

    return insulin_info